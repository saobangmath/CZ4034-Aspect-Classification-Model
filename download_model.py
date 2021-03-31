import gdown
import pathlib

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.absolute()
MODEL_DIR = CURRENT_FILE_PATH.joinpath("work_dirs/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR.joinpath("checkpoint_best.pth")

gdown.download(
    "https://drive.google.com/uc?id=1-RAPEW5n8PKzS2cetkekwc5XN_4M49RC",
    output=str(MODEL_PATH),
)
