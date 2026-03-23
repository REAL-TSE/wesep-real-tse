#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_spk_from_key(key: str):
    # key: 7059-77900-0042_6385-34655-0034
    parts = key.split("_")
    spk = []
    for p in parts:
        spk_id = p.split("-")[0]
        spk.append(spk_id)
    return spk


def scan_librimix(dataset_path: Path):
    """
    Scan mix_clean (or mix_{noise_type}) directory and yield samples.
    """
    mix_wavs = sorted(dataset_path.glob("*.wav"))

    for mix_wav in mix_wavs:
        key = mix_wav.stem  # without .wav

        # infer speakers from key
        spk = parse_spk_from_key(key)

        # infer source wavs
        s1 = dataset_path.parent / "s1" / mix_wav.name
        s2 = dataset_path.parent / "s2" / mix_wav.name

        sample = {
            "key": key,
            "spk": spk,
            "mix": {
                "default": [str(mix_wav)]
            },
            "src": {
                spk[0]: [str(s1)],
                spk[1]: [str(s2)],
            },
        }

        yield sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to mix_{noise_type} directory",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output samples.jsonl",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for sample in scan_librimix(dataset_path):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved samples to {out}")


if __name__ == "__main__":
    main()
