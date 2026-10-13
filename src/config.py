from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
IMAGE_DIR = DATA_ROOT / "images"
LABEL_CSV = DATA_ROOT / "stage_2_train_labels.csv"
PROCESSED_ROOT = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

IMAGE_SIZE = 128
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 20
