from src.inference import run_inference, parse_args


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
