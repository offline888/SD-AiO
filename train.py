from src.training import Trainer
from src.training.arguments import parse_args


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
