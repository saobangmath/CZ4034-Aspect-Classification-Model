import sys
import argparse

import yaml

from utils.trainer import Trainer


DESCRIPTION = """Make prediction."""


def main(args):
    # Initializer trainer; initialize only one cos very costly
    trainer = load_trainer(args.config_path, args.load_from)

    # Start training
    while True:
        text = input("Enter a review:")
        predict_aspects(trainer, text)


# call this function one to load the trainer;
"""
    config_path: the link to config.yaml file
    load_from: load from the pretrained model 
"""
def load_trainer(config_path, load_from):
    with open(config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config["action"] = "predict"
    config["config_path"] = config_path
    config["load_from"] = load_from
    config["resume_from"] = None
    trainer = Trainer(config)
    return trainer

"""
    return the aspects prediction for a text in the form: 
        {
            "food_score_preds" : [0.5, 3.5],
            "food_existence_preds" : [0, 1],
            "service_score_preds" : [0.5, 3.5],
            "service_existence_preds" : [0, 1],
            "price_score_preds" : [0.5, 3.5],
            "price_existence_preds" : [0, 1],
        }
"""
def predict_aspects(trainer, text):
    return trainer.predict(text)


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
