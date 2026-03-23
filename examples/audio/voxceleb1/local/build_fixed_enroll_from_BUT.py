#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_speech_json(path):
    """
    Load speech.json as:
      spk_id -> list of {utt_id, path}
    """
    with open(path, "r", encoding="utf-8") as f:
        spk2items = json.load(f)
    return spk2items


def load_mixture2enrollment(path):
    """
    mixture2enrollment format (space separated):
      mix_key target_field enroll_relpath

    Returns:
      list of (mix_key, target_field, enroll_relpath)
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"[line {lineno}] Bad line: {line}")

            mix_key = parts[0]
            target_field = parts[1]
            enroll_relpath = parts[2]

            rows.append((mix_key, target_field, enroll_relpath))

    return rows


def parse_target_spk_from_field(target_field: str) -> str:
    """
    target_field examples:
      4077
      4077-13754-0001
      mixkey::4077

    We only care about speaker id.
    """
    if "::" in target_field:
        target_field = target_field.split("::")[-1]
    return target_field.split("-")[0]


def parse_enroll_uid_from_relpath(enroll_relpath: str, target_spk: str) -> str:
    """
    Example:
      enroll_relpath = s2/4992-41797-0018_6930-75918-0017.wav
      target_spk     = 4992

    Return:
      4992-41797-0018

    Logic:
      - strip extension
      - split by "_"
      - pick the one whose speaker id matches target_spk
    """
    name = Path(enroll_relpath).name
    stem = name[:-4] if name.endswith(".wav") else name

    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad enroll relpath: {enroll_relpath}")

    for utt in parts:
        spk = utt.split("-")[0]
        if spk == target_spk:
            return utt

    raise KeyError(f"Target speaker not found in relpath: "
                   f"target_spk={target_spk}, relpath={enroll_relpath}")


def normalize_relpath(p: str) -> str:
    """
    Make relpath comparable:
      - remove leading slashes
      - drop .wav extension
    """
    p = p.lstrip("/")
    if p.endswith(".wav"):
        p = p[:-4]
    return p


def find_enroll_path(spk_items, enroll_uid, enroll_relpath):
    """
    From a speaker's speech.json items, find the one that:
      - utt_id == enroll_uid
      - path endswith enroll_relpath

    Returns:
      absolute wav path (str)

    Raises if not found or ambiguous.
    """
    want = normalize_relpath(enroll_relpath)

    candidates = []

    for item in spk_items:
        if item["utt_id"] != enroll_uid:
            continue

        path = item["path"]
        got = normalize_relpath(path)

        if got.endswith(want):
            candidates.append(path)

    if len(candidates) == 0:
        raise KeyError(f"Enroll not found: utt_id={enroll_uid}, "
                       f"relpath={enroll_relpath}")

    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous enroll: utt_id={enroll_uid}, "
                           f"relpath={enroll_relpath}, "
                           f"matches={candidates}")

    return candidates[0]


def build_fixed_enroll(spk2items, mix2enroll):
    """
    Build:
      fixed_enroll["mix_key::spk_id"] = [
        {"utt_id": enroll_uid, "path": wav_path}
      ]
    """
    fixed = {}

    for mix_key, target_field, enroll_relpath in mix2enroll:
        # 1) target speaker: ONLY from field 2
        target_spk = parse_target_spk_from_field(target_field)

        if target_spk not in spk2items:
            raise KeyError(f"Speaker {target_spk} not found in speech.json "
                           f"(from target_field={target_field})")

        # 2) enroll utt_id: ONLY from relpath + target_spk
        enroll_uid = parse_enroll_uid_from_relpath(
            enroll_relpath,
            target_spk,
        )

        cue_key = f"{mix_key}::{target_spk}"

        # 3) search only inside this speaker's pool
        spk_items = spk2items[target_spk]

        wav_path = find_enroll_path(
            spk_items,
            enroll_uid,
            enroll_relpath,
        )

        fixed[cue_key] = [{
            "utt_id": enroll_uid,
            "path": wav_path,
        }]

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speech_json",
        type=str,
        required=True,
        help="Path to resources/speech.json",
    )
    parser.add_argument(
        "--mixture2enrollment",
        type=str,
        required=True,
        help="Path to mixture2enrollment file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to output fixed_enroll.json",
    )

    args = parser.parse_args()

    spk2items = load_speech_json(args.speech_json)
    mix2enroll = load_mixture2enrollment(args.mixture2enrollment)

    fixed = build_fixed_enroll(spk2items, mix2enroll)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote fixed_enroll.json to: {out}")
    print(f"[OK] #entries = {len(fixed)}")


if __name__ == "__main__":
    main()
