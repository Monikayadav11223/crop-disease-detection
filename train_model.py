"""
train_model.py - CNN Training Pipeline with Transfer Learning
HCLTech Career Shaper Programme | Project 35: Crop Disease Detection
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

from project_config import (
    MODEL_SAVE_PATH,
    CLASS_LABELS_PATH,
    METRICS_JSON_PATH,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_transfer_learning_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    architecture: str = "mobilenetv2"
):
    """
    Constructs a transfer learning model with MobileNetV2 (or ResNet50) backbone
    and custom regularized classification head.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    logger.info(f"Building {architecture.upper()} transfer learning model with {num_classes} output classes...")

    # Base pretrained feature extractor
    if architecture.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

    # Freeze base model parameters for Stage 1
    base_model.trainable = False

    # Data Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
        layers.RandomRotation(0.15, seed=RANDOM_SEED),
        layers.RandomZoom(0.12, seed=RANDOM_SEED),
        layers.RandomTranslation(0.08, 0.08, seed=RANDOM_SEED)
    ], name="data_augmentation")

    # Assemble Architecture
    inputs = tf.keras.Input(shape=input_shape, name="input_leaf_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_crop_disease_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model, base_model


def fine_tune_model(model, base_model, fine_tune_at: int = 100):
    """
    Unfreezes the top layers of the base model for Stage 2 fine-tuning
    with a conservative learning rate.
    """
    import tensorflow as tf

    logger.info(f"Unfreezing base model layers from index {fine_tune_at} onwards for fine-tuning...")
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model


def train_pipeline(
    data_dir: Path,
    epochs: int = INITIAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    architecture: str = "mobilenetv2"
):
    """
    Executes the end-to-end training and fine-tuning pipeline.
    """
    import tensorflow as tf

    logger.info(f"Checking dataset directory at {data_dir}...")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning(
            f"Image dataset path '{data_dir}' is empty or not yet mounted.\n"
            "To train on the complete PlantVillage dataset:\n"
            "1. Download PlantVillage from Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
            "2. Extract into data/raw/plantvillage/\n"
            f"3. Run: python train_model.py --data_dir data/raw/plantvillage --epochs {epochs}"
        )
        return None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Optimization with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    num_classes = len(train_ds.class_names)
    model, base_model = build_transfer_learning_model(
        num_classes=num_classes,
        architecture=architecture
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(MODEL_SAVE_PATH), save_best_only=True)
    ]

    # Stage 1: Train Head
    logger.info("Executing Stage 1: Feature Extraction Training...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Stage 2: Fine-Tuning
    logger.info("Executing Stage 2: Fine-Tuning Top Convolutional Blocks...")
    model = fine_tune_model(model, base_model)
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs + FINE_TUNE_EPOCHS,
        initial_epoch=epochs,
        callbacks=callbacks
    )

    model.save(str(MODEL_SAVE_PATH))
    logger.info(f"Trained model exported successfully to {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Crop Disease Detection")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--architecture", type=str, default="mobilenetv2", choices=["mobilenetv2", "resnet50"])
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        architecture=args.architecture
    )
