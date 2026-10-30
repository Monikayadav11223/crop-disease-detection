"""
project_config.py - Centralized configuration and constants
HCLTech Career Shaper Programme | Project 35: Crop Disease Detection
"""

from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_IMAGES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATASET_CSV = BASE_DIR / "dataset.csv"
CLEANED_DATASET_CSV = BASE_DIR / "cleaned_dataset.csv"
DATA_DICTIONARY_CSV = BASE_DIR / "data_dictionary.csv"
CLEANING_REPORT_JSON = PROCESSED_DATA_DIR / "data_cleaning_report.json"
MODEL_SAVE_PATH = MODELS_DIR / "mobilenetv2_crop_disease.keras"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"
DISEASE_INFO_PATH = MODELS_DIR / "disease_info.json"
METRICS_JSON_PATH = MODELS_DIR / "evaluation_metrics.json"

# Image Specifications & Training Hyperparameters
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
RANDOM_SEED = 42

# Train / Validation / Test Splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Target PlantVillage Disease Classes (Representative multi-crop subset)
CLASS_NAMES = [
    "Apple___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Corn___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold"
]

NUM_CLASSES = len(CLASS_NAMES)
