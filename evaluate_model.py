import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuration
MODEL_PATH = "wheat_mobilenet_finetuned.h5"
VALIDATION_PATH = "dataset/validation"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['Brown Rust', 'Healthy', 'Leaf Blight', 'Mildew', 'Smut', 'Yellow Rust']

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model, validation_path):
    # Validation data generator (no augmentation, just rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )

    print(f"Found {validation_generator.samples} validation images belonging to {len(validation_generator.class_indices)} classes.")
    print("Class Indices:", validation_generator.class_indices)

    # Evaluate the model
    print("\nEvaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Get predictions
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes

    # Classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)

    return val_accuracy, report, cm, true_classes, predicted_classes

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    if model is None:
        exit(1)

    if not os.path.exists(VALIDATION_PATH):
        print(f"Validation path {VALIDATION_PATH} does not exist.")
        exit(1)

    accuracy, report, cm, true_classes, predicted_classes = evaluate_model(model, VALIDATION_PATH)

    # Save results to file
    with open('model_evaluation_results.txt', 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\nResults saved to model_evaluation_results.txt")
