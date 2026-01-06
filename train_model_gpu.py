import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import json
import os

print("\n" + "="*70)
print("GPU VERIFICATION")
print("="*70)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU DETECTED: {gpus[0].name}")
        print(f"   Device: {tf.test.gpu_device_name()}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("❌ NO GPU DETECTED - Will use CPU")
print("="*70)

def build_model(num_classes, input_shape=(224, 224, 3)):
    """Build MobileNetV2 model optimized for wheat disease detection"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def create_data_generators(train_dir, val_dir, img_size=224, batch_size=32):
    """Create data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    return train_gen, val_gen

def calculate_class_weights(generator):
    """Calculate class weights to handle imbalance"""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    class_counts = {}
    for class_name, class_idx in generator.class_indices.items():
        class_counts[class_idx] = sum(generator.labels == class_idx)
    
    classes = list(class_counts.keys())
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=generator.labels
    )
    
    class_weight_dict = dict(zip(classes, weights))
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    for class_name, class_idx in sorted(generator.class_indices.items(), key=lambda x: x[1]):
        print(f"  {class_name:<20} {class_counts[class_idx]:>6} images (weight: {class_weight_dict[class_idx]:.3f})")
    print("="*70)
    
    return class_weight_dict

def train_model(train_dir='dataset/train', 
                val_dir='dataset/validation',
                epochs_phase1=15,
                epochs_phase2=20,
                batch_size=32,
                img_size=224):
    """Two-phase training for wheat disease detection"""
    
    print("\n" + "="*70)
    print("WHEAT DISEASE DETECTION - GPU TRAINING")
    print("="*70)
    print(f"Training will complete in approximately 30-45 minutes")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir, img_size, batch_size)
    num_classes = len(train_gen.class_indices)
    
    print(f"\n✓ Classes ({num_classes}): {list(train_gen.class_indices.keys())}")
    print(f"✓ Training samples: {train_gen.samples:,}")
    print(f"✓ Validation samples: {val_gen.samples:,}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Steps per epoch: {train_gen.samples // batch_size}")
    
    # Save class indices
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f, indent=4)
    print("\n✓ Saved class_indices.json")
    
    # Calculate class weights
    print("\n[2/5] Calculating class weights...")
    class_weights = calculate_class_weights(train_gen)
    
    # Build model
    print("\n[3/5] Building MobileNetV2 model...")
    model, base_model = build_model(num_classes, input_shape=(img_size, img_size, 3))
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters (Phase 1): {trainable_params:,}")
    
    # Callbacks Phase 1
    checkpoint = ModelCheckpoint(
        'wheat_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
    
    # PHASE 1: Train top layers
    print("\n" + "="*70)
    print("[4/5] PHASE 1: TRAINING TOP LAYERS (BASE MODEL FROZEN)")
    print("="*70)
    print(f"Epochs: {epochs_phase1} | Learning Rate: 0.001 | Batch Size: {batch_size}")
    print("-"*70)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("\n✓ Phase 1 completed!")
    
    # PHASE 2: Fine-tune
    print("\n" + "="*70)
    print("[5/5] PHASE 2: FINE-TUNING (UNFREEZING LAST 50 LAYERS)")
    print("="*70)
    
    base_model.trainable = True
    fine_tune_at = 104
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in model.layers if layer.trainable])
    trainable_params_phase2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"✓ Unfreezing from layer: {fine_tune_at}")
    print(f"✓ Trainable layers: {trainable_layers}")
    print(f"✓ Trainable parameters: {trainable_params_phase2:,}")
    print(f"Epochs: {epochs_phase2} | Learning Rate: 0.0001 | Batch Size: {batch_size}")
    print("-"*70)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint2 = ModelCheckpoint(
        'wheat_model_finetuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop2 = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr2 = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
    
    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase2,
        class_weight=class_weights,
        callbacks=[checkpoint2, early_stop2, reduce_lr2],
        verbose=1
    )
    
    # Save final model
    model.save('wheat_model_final.h5')
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nSaved models:")
    print("  • wheat_model_best.h5 (best from Phase 1)")
    print("  • wheat_model_finetuned.h5 (best from Phase 2) ⭐ Use this")
    print("  • wheat_model_final.h5 (final model)")
    print("  • class_indices.json (class mapping)")
    print("="*70)
    
    # Final metrics
    final_train_acc = history_phase2.history['accuracy'][-1]
    final_val_acc = history_phase2.history['val_accuracy'][-1]
    final_train_loss = history_phase2.history['loss'][-1]
    final_val_loss = history_phase2.history['val_loss'][-1]
    
    print("\nFinal Performance:")
    print(f"  Training Accuracy:   {final_train_acc*100:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"  Training Loss:       {final_train_loss:.4f}")
    print(f"  Validation Loss:     {final_val_loss:.4f}")
    print("="*70 + "\n")
    
    return model, history_phase1, history_phase2

if __name__ == "__main__":
    # Start GPU training
    model, hist1, hist2 = train_model(
        train_dir='dataset/train',
        val_dir='dataset/validation',
        epochs_phase1=15,
        epochs_phase2=20,
        batch_size=32,
        img_size=224
    )
