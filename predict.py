import yaml
import pathlib

# Need "." to import the models from the search engine
from .utils.trainer import Trainer

DESCRIPTION = """Make prediction."""

current_dir = pathlib.Path(__file__).parent.absolute()
config_path = str(current_dir.joinpath("work_dirs/config_cpu.yaml"))
load_from = str(current_dir.joinpath("work_dirs/checkpoint_best.pth"))

with open(config_path, "r") as conf:
    config = yaml.load(conf, Loader=yaml.FullLoader)
config["action"] = "predict"
config["config_path"] = config_path
config["load_from"] = load_from
config["resume_from"] = None
trainer = Trainer(config)

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


def predict_aspects(text):
    return trainer.predict(text)


def main():
    while True:
        text = input("Enter a review:")
        predict_aspects(text)


if __name__ == '__main__':
    main()
