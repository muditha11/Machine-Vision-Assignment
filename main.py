from argparse import ArgumentParser
import logging
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s:%(levelname)s: %(message)s"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config-path", type=str)
    parser.add_argument("-o", "--out-root", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, default=0)
    parser.add_argument("-m", "--mock-batch-count", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.config_path, args.device, args.mock_batch_count)
    trainer.fit(args.out_root)
