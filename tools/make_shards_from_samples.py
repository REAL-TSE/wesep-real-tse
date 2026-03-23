# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import json
import logging
import multiprocessing
import os
import random
import tarfile

AUDIO_FORMAT_SETS = {'wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'wma'}


def write_tar_file(data_list, tar_file, index=0, total=1):
    logging.info(f'Processing {tar_file} {index}/{total}')
    with tarfile.open(tar_file, "w") as tar:
        for sample in data_list:
            key = sample["key"]
            spks = sample["spk"]

            # ---- write spk files ----
            for i, spk in enumerate(spks, 1):
                name = f"{key}.spk{i}"
                data = spk.encode("utf8")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            # ---- write mix wavs ----
            for wav in sample["mix"]["default"]:
                suffix = wav.split('.')[-1]
                assert suffix in AUDIO_FORMAT_SETS
                with open(wav, "rb") as f:
                    data = f.read()
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            # ---- write src wavs ----
            for i, spk in enumerate(spks, 1):
                for wav in sample["src"][spk]:
                    suffix = wav.split('.')[-1]
                    assert suffix in AUDIO_FORMAT_SETS
                    with open(wav, "rb") as f:
                        data = f.read()
                    info = tarfile.TarInfo(f"{key}_spk{i}.{suffix}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="samples.jsonl")
    parser.add_argument("--num_utts_per_shard", type=int, default=1000)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--prefix", default="shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("shards_dir")
    parser.add_argument("shards_list")
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # ---- load samples ----
    data = []
    with open(args.samples, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))

    if args.shuffle:
        random.shuffle(data)

    chunks = [
        data[i:i + args.num_utts_per_shard]
        for i in range(0, len(data), args.num_utts_per_shard)
    ]

    os.makedirs(args.shards_dir, exist_ok=True)

    pool = multiprocessing.Pool(args.num_threads)
    shard_paths = []

    for i, chunk in enumerate(chunks):
        tar_path = os.path.join(args.shards_dir, f"{args.prefix}_{i:09d}.tar")
        shard_paths.append(tar_path)
        pool.apply_async(write_tar_file, (chunk, tar_path, i, len(chunks)))

    pool.close()
    pool.join()

    with open(args.shards_list, "w", encoding="utf8") as f:
        for p in shard_paths:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
