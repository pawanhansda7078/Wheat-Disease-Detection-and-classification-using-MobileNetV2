import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import argparse
import logging
import matplotlib.pyplot as plt
import random
import numpy as np
import json

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 model for wheat disease classification.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for initial training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--fine_tune_lr', type=float, default=0.00001, help='Learning rate for fine-tuning.')
    return parser.parse_args()

def check_data_distribution(data_path):
    """
    Checks and logs the distribution of classes in the dataset.
    Returns class_counts for further processing.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist.")
        return {}
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    class_counts = {}
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        class_counts[cls] = count
    logger.info(f"Data distribution in {data_path}: {class_counts}")
    total = sum(class_counts.values())
    logger.info(f"Total images: {total}")
    if len(set(class_counts.values())) > 1:
        logger.warning("Class imbalance detected. Consider data augmentation or resampling.")
    return class_counts

def load_data(args):
    """
    Loads and prepares the training and validation data generators.
    Computes class weights to handle imbalance.
    """
    logger.info("Loading and augmenting data.")
    train_class_counts = check_data_distribution(DATASET_PATH)
    check_data_distribution(VALIDATION_PATH)

    # Create data generators with augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest'
    )

    # Validation data generator (no augmentation, just rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    # Validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_PATH,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    train_samples = train_generator.samples
    val_samples = validation_generator.samples

    # Calculate steps per epoch properly
    steps_per_epoch = max(1, train_samples // args.batch_size)
    validation_steps = max(1, val_samples // args.batch_size)
    print(f"Found {train_samples} training images belonging to {num_classes} classes.")
    print(f"Found {val_samples} validation images.")
    print("Class Indices:", train_generator.class_indices)

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))
    logger.info(f"Computed class weights: {class_weights}")

    return train_generator, validation_generator, train_samples, val_samples, num_classes, class_weights

# --- CONFIGURATION ---
DATASET_PATH = "dataset/train"  # Directory containing your training images
VALIDATION_PATH = "dataset/validation"  # Directory containing validation images
MODEL_SAVE_PATH = "wheat_mobilenet_model_v3.h5"
IMAGE_SIZE = (224, 224)

def build_model(num_classes):
    """
    Builds the MobileNetV2 model for wheat disease classification.
    """
    # Load the base MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Freeze the layers of the base model so they are not trained
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)  # Final classification layer

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

def train_model(model, base_model, train_generator, validation_generator, args, class_weights):
    """
    Trains the model with initial training and fine-tuning phases.
    Uses class weights to handle imbalance.
    """
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    # 3. Compile the Model
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])

    print("\nModel Summary:")
    model.summary()

    # 4. Train the Model (Initial Phase)
    print("\nStarting model training (classifier only)...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 5. Fine-Tuning Phase
    print("\nStarting fine-tuning...")

    # Callbacks for fine-tuning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    # Unfreeze the top layers of the base model
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 120

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=args.fine_tune_lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])

    print("\nModel Summary (after unfreezing for fine-tuning):")
    model.summary()

    # Continue training
    total_epochs = args.epochs + args.fine_tune_epochs

    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return history, history_fine

def evaluate_and_save(model, validation_generator, args, history, history_fine):
    """
    Evaluates the model, saves history, plots, and saves the model.
    """
    # 6. Evaluate the Model
    print("\nEvaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # 7. Save Training History
    history_combined = {
        'epochs': list(range(1, len(history.history['accuracy']) + len(history_fine.history['accuracy']) + 1)),
        'accuracy': history.history['accuracy'] + history_fine.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'] + history_fine.history['val_accuracy'],
        'loss': history.history['loss'] + history_fine.history['loss'],
        'val_loss': history.history['val_loss'] + history_fine.history['val_loss']
    }
    with open('training_history.json', 'w') as f:
        json.dump(history_combined, f)
    print("Training history saved to 'training_history.json'")

    # 8. Plot Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_combined['epochs'], history_combined['accuracy'], label='Train Accuracy')
    plt.plot(history_combined['epochs'], history_combined['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_combined['epochs'], history_combined['loss'], label='Train Loss')
    plt.plot(history_combined['epochs'], history_combined['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")

    # 9. Save the Trained Model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training complete! Model saved to '{MODEL_SAVE_PATH}'")

def build_and_train_model(args):
    """
    Builds, compiles, and trains the MobileNetV2 model for wheat disease classification.
    """
    # 1. Load Data
    train_generator, validation_generator, train_samples, val_samples, num_classes, class_weights = load_data(args)

    # 2. Build Model
    model, base_model = build_model(num_classes)

    # 3. Train Model
    history, history_fine = train_model(model, base_model, train_generator, validation_generator, args, class_weights)

    # 4. Evaluate and Save
    evaluate_and_save(model, validation_generator, args, history, history_fine)

if __name__ == "__main__":
    try:
        args = parse_arguments()
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset directory not found at '{DATASET_PATH}'")
            print("Please create the directory and organize your images into subfolders named by class.")
            exit(1)
        build_and_train_model(args)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        exit(1)
