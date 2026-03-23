#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_samples_jsonl(jsonl_path):
    """
    VoxCeleb1 samples.jsonl line format:

    {
      "key": "...",
      "spk": ["id10001"],
      "src": {
        "id10001": ["/abs/path/.../00001.wav"]
      }
    }

    Returns:
      list of (key, spk_ids, src_map)
    """
    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            key = obj["key"]
            spk_ids = obj["spk"]
            src_map = obj["src"]

            if not isinstance(spk_ids, list):
                raise ValueError(f"`spk` must be list in sample {key}")
            if not isinstance(src_map, dict):
                raise ValueError(f"`src` must be dict in sample {key}")

            for spk in spk_ids:
                if spk not in src_map:
                    raise ValueError(
                        f"Speaker {spk} missing in src map for sample {key}")

            samples.append((key, spk_ids, src_map))

    return samples


def build_spk2utt_from_vox(samples):
    """
    Build:
      spk_id -> list of {utt_id, path}

    utt_id is derived from wav filename (without .wav).
    """
    spk2utt = defaultdict(list)

    for _, spk_ids, src_map in samples:
        for spk in spk_ids:
            wav_list = src_map[spk]

            if not wav_list:
                raise ValueError(f"Empty src list for speaker {spk}")

            # VoxCeleb1: single wav per entry
            wav_path = Path(wav_list[0])

            utt_id = f"{spk}/{wav_path.parent.name}/{wav_path.stem}"

            spk2utt[spk].append({"utt_id": utt_id, "path": str(wav_path)})

    return spk2utt


def main():
    parser = argparse.ArgumentParser(
        description="Build audio cues for VoxCeleb1 from samples.jsonl")
    parser.add_argument("--samples_jsonl",
                        type=str,
                        required=True,
                        help="Path to samples.jsonl")
    parser.add_argument("--outfile",
                        type=str,
                        required=True,
                        help="Path to output audio.json")
    args = parser.parse_args()

    samples_jsonl = Path(args.samples_jsonl)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    samples = parse_samples_jsonl(samples_jsonl)
    spk2utt = build_spk2utt_from_vox(samples)

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote audio cues to: {outfile}")
    print(f"[OK] #speakers = {len(spk2utt)}")
    print(f"[OK] #samples = {len(samples)}")


if __name__ == "__main__":
    main()
