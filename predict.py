import os
import sys
import argparse

import yaml

from utils.trainer import Trainer


DESCRIPTION = """Make prediction."""


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config["action"] = "predict"
    config["config_path"] = args.config_path
    config["load_from"] = args.load_from
    config["resume_from"] = None

    # Initializer trainer
    trainer = Trainer(config)

    # Start training
    while True:
        text = input("Enter a review:")
        trainer.predict(text)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config that the model is trained with.')
    parser.add_argument(
        '-m', '--load-from', type=str, required=True, help='Path to trained model.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
