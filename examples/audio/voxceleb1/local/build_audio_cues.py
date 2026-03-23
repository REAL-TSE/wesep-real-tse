#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_samples_jsonl(jsonl_path):
    """
    samples.jsonl line format (aligned to your current schema):

    {
      "key": "...",
      "spk": ["7059", "6385", ...],
      "mix": {
        "default": [".../mix.wav"]
      },
      "src": {
        "7059": [".../s1.wav"],
        "6385": [".../s2.wav"]
      }
    }

    Returns:
      list of (mix_key, spk_ids, src_map)
        where src_map: spk_id -> list[path]
    """
    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            mix_key = obj["key"]
            spk_ids = obj["spk"]
            src_map = obj["src"]

            # sanity checks
            if not isinstance(src_map, dict):
                raise ValueError(f"`src` must be dict in sample {mix_key}")

            for spk in spk_ids:
                if spk not in src_map:
                    raise ValueError(
                        f"Speaker {spk} missing in src map for sample {mix_key}"
                    )

            samples.append((mix_key, spk_ids, src_map))

    return samples


def build_spk2utt_from_librimix(samples):
    """
    Build:
      spk_id -> list of {utt_id, path}

    - utt_id is per-speaker utterance id
      (e.g., 831-130739-0007)
    - one speaker may appear multiple times
    """
    spk2utt = defaultdict(list)

    for mix_key, spk_ids, src_map in samples:
        # mix_key example:
        # 831-130739-0007_2952-408-0008
        utt_fields = mix_key.split("_")

        if len(utt_fields) != len(spk_ids):
            raise ValueError(
                f"Mismatch #utt_fields and #spk in key {mix_key}: "
                f"{len(utt_fields)} vs {len(spk_ids)}")

        for i, spk in enumerate(spk_ids):
            uid = utt_fields[i]  # e.g., 831-130739-0007
            wav_list = src_map[spk]

            if not wav_list:
                raise ValueError(
                    f"Empty src list for speaker {spk} in sample {mix_key}")

            # for now: take the first channel/view
            # (your schema already allows multi-channel later)
            src_path = wav_list[0]

            spk2utt[spk].append({"utt_id": uid, "path": src_path})

    return spk2utt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_jsonl",
                        type=str,
                        required=True,
                        help="Path to samples.jsonl")
    parser.add_argument("--outfile",
                        type=str,
                        required=True,
                        help="Path to output resources/speech.json")
    args = parser.parse_args()

    samples_jsonl = Path(args.samples_jsonl)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    samples = parse_samples_jsonl(samples_jsonl)
    spk2utt = build_spk2utt_from_librimix(samples)

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote speech resource to: {outfile}")
    print(f"[OK] #speakers = {len(spk2utt)}")
    print(f"[OK] #samples = {len(samples)}")


if __name__ == "__main__":
    main()
