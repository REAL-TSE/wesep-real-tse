import argparse
import soundfile as sf
import wesep


def evaluate(
    pretrain_path,
    mixture_path,
    enroll_path,
    output_path,
    device="cpu",
    resample_rate=16000,
    vad=False,
    output_norm=True,
):
    """
    Run target speech extraction on a single mixture/enroll pair.

    Args:
        pretrain_path (str): path to pretrained model
        mixture_path (str): path to mixture audio
        enroll_path (str): path to enrollment audio
        output_path (str): path to save extracted speech
    """

    # 1. load model
    model = wesep.load_model_local(pretrain_path)
    model.set_resample_rate(resample_rate)
    model.set_vad(vad)
    model.set_device(device)
    model.set_output_norm(output_norm)

    # 2. inference
    speech = model.extract_speech(mixture_path, enroll_path)

    if speech is None:
        print("Failed to extract target speech")
        return

    # 3. save
    sf.write(output_path, speech[0], resample_rate)

    print(f"Success! Output saved to: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="WeSep single audio evaluation")

    parser.add_argument(
        "--pretrain",
        required=True,
        help="Path to pretrained model, containing avg_model.pt and config.yaml"
    )
    parser.add_argument("--mixture",
                        required=True,
                        help="Path to mixture audio")
    parser.add_argument("--enroll",
                        required=True,
                        help="Path to enrollment audio")
    parser.add_argument("--output", required=True, help="Path to output audio")

    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--resample_rate", type=int, default=16000)
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--no_output_norm", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    evaluate(
        pretrain_path=args.pretrain,
        mixture_path=args.mixture,
        enroll_path=args.enroll,
        output_path=args.output,
        device=args.device,
        resample_rate=args.resample_rate,
        vad=args.vad,
        output_norm=not args.no_output_norm,
    )


if __name__ == "__main__":
    main()
