#!/usr/bin/env python3
# Copyright (c) 2026
# Author: Ke Zhang

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Scan VoxCeleb1 single-speaker wavs and generate samples.jsonl "
        "for online mixing (Libri2Mix-compatible schema)."), )
    parser.add_argument(
        "wav_root",
        type=str,
        help=(
            "Root directory of VoxCeleb1 wavs, e.g. wav/id10001/xxxx/00001.wav"
        ),
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output samples.jsonl file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    wav_root = Path(args.wav_root).resolve()
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_root.rglob("*.wav"))

    if len(wav_files) == 0:
        raise RuntimeError(f"No wav files found under {wav_root}")

    with outfile.open("w", encoding="utf-8") as f:
        for wav_path in wav_files:
            # Expected structure:
            # wav_root/id10001/1zcIwhmdeo4/00001.wav
            try:
                spk_id = wav_path.parts[-3]  # id10001
            except IndexError:
                raise RuntimeError(f"Unexpected path structure: {wav_path}")

            # key: relative path without suffix
            # id10001/1zcIwhmdeo4/00001
            key = str(wav_path.relative_to(wav_root)).replace(".wav", "")

            sample = {
                "key": key,
                "spk": [spk_id],
                "src": {
                    spk_id: [str(wav_path)]
                }
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Scanned {len(wav_files)} wavs")
    print(f"samples.jsonl written to: {outfile}")


if __name__ == "__main__":
    main()
