import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import random
import tarfile

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def write_tar_file(samples, tar_path, index=0, total=1):
    logging.info(f'Processing {tar_path} {index}/{total}')

    with tarfile.open(tar_path, "w") as tar:
        for sample in samples:
            key = sample["key"]
            spk = sample["spk"][0]

            assert spk in sample["src"]
            wav_path = sample["src"][spk][0]

            suffix = wav_path.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS

            # ---------- write spk ----------
            spk_bytes = spk.encode("utf8")
            spk_info = tarfile.TarInfo(name=f"{key}.spk")
            spk_info.size = len(spk_bytes)
            tar.addfile(spk_info, io.BytesIO(spk_bytes))

            # ---------- write wav ----------
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

            wav_info = tarfile.TarInfo(name=f"{key}.{suffix}")
            wav_info.size = len(wav_bytes)
            tar.addfile(wav_info, io.BytesIO(wav_bytes))


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

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # ---------- load samples.jsonl ----------
    samples = []
    with open(args.samples, "r", encoding="utf8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.shuffle:
        random.shuffle(samples)

    # ---------- chunk ----------
    num = args.num_utts_per_shard
    chunks = [samples[i:i + num] for i in range(0, len(samples), num)]

    os.makedirs(args.shards_dir, exist_ok=True)

    pool = mp.Pool(processes=args.num_threads)
    shard_paths = []

    for i, chunk in enumerate(chunks):
        tar_path = os.path.join(args.shards_dir, f"{args.prefix}_{i:09d}.tar")
        shard_paths.append(tar_path)
        pool.apply_async(write_tar_file,
                         args=(chunk, tar_path, i, len(chunks)))

    pool.close()
    pool.join()

    with open(args.shards_list, "w", encoding="utf8") as f:
        for p in shard_paths:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
