#!/usr/bin/env python3
# Lightning-Fast Plant Disease Detection - Maximum Speed Training
# Target: 87-90% accuracy with 5-10x speed improvement

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import argparse
import json
import cv2
from PIL import Image
import time
from sklearn.utils.class_weight import compute_class_weight

# LIGHTNING-FAST CONFIGURATION
IMG_SIZE = 128  # Reduced from 224 for 3x speed boost
BATCH_SIZE = 256  # Maximum batch size for GPU utilization
EPOCHS = 8  # Minimal epochs for fast convergence
LEARNING_RATE = 0.003  # Higher LR for faster convergence
MODEL_PATH = 'plant_disease_model_lightning.h5'

# Extreme optimization settings
USE_MIXED_PRECISION = True
USE_XLA_COMPILATION = True
PREFETCH_BUFFER = tf.data.AUTOTUNE
CACHE_DATASET = True
MINIMAL_AUGMENTATION = True  # Reduce augmentation for speed

# Configure TensorFlow for maximum speed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Set memory limit for stability with large batch sizes
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # 6GB limit
        )
    except:
        pass

# Enable all performance optimizations
if USE_MIXED_PRECISION:
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('⚡ Mixed precision enabled - 2x speed boost')
    except Exception as e:
        print(f'Mixed precision not available: {e}')

if USE_XLA_COMPILATION:
    tf.config.optimizer.set_jit(True)
    print('⚡ XLA compilation enabled - additional 30% speed boost')

# Simplified disease database for speed
DISEASE_REMEDIES = {
    "Apple___Apple_scab": {
        "description": "Fungal infection causing olive-green to brown spots on leaves and fruit.",
        "remedies": ["Apply fungicides", "Improve air circulation", "Remove infected debris"],
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "description": "Fungal infection affecting leaves, fruit, and bark with dark lesions.",
        "remedies": ["Prune diseased wood", "Apply fungicides", "Improve sanitation"],
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "description": "Creates orange-yellow spots, requires cedar tree proximity.",
        "remedies": ["Remove cedar trees", "Apply fungicides", "Use resistant varieties"],
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "description": "Healthy apple plant with no disease symptoms.",
        "remedies": ["Continue maintenance", "Monitor regularly"],
        "severity": "None"
    },
    "Tomato___Early_blight": {
        "description": "Concentric ring spots caused by Alternaria solani.",
        "remedies": ["Remove infected leaves", "Apply fungicides", "Improve spacing"],
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "description": "Highly destructive disease with water-soaked spots.",
        "remedies": ["Apply copper fungicides", "Remove infected plants", "Improve drainage"],
        "severity": "Very High"
    },
    "Tomato___healthy": {
        "description": "Healthy tomato plant with no disease symptoms.",
        "remedies": ["Continue maintenance", "Regular monitoring"],
        "severity": "None"
    }
}

DEFAULT_REMEDY = {
    "description": "Unknown plant disease detected.",
    "remedies": ["Consult experts", "Isolate plants", "Apply broad-spectrum treatment"],
    "severity": "Unknown"
}

def create_lightning_data_pipeline(data_dir, validation_split=0.15):
    """Create extremely optimized data pipeline for maximum speed"""
    
    if MINIMAL_AUGMENTATION:
        # Minimal augmentation for speed
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Reduced from 25
            width_shift_range=0.1,  # Reduced from 0.15
            height_shift_range=0.1,  # Reduced from 0.15
            horizontal_flip=True,
            validation_split=validation_split
        )
    else:
        # Standard augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            validation_split=validation_split
        )
    
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators with maximum batch size
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Convert to tf.data for maximum optimization
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
        )
    )
    
    valid_dataset = tf.data.Dataset.from_generator(
        lambda: valid_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, valid_generator.num_classes), dtype=tf.float32)
        )
    )
    
    # Apply all optimizations
    if CACHE_DATASET:
        train_dataset = train_dataset.cache()
        valid_dataset = valid_dataset.cache()
    
    train_dataset = train_dataset.prefetch(PREFETCH_BUFFER)
    valid_dataset = valid_dataset.prefetch(PREFETCH_BUFFER)
    
    # Calculate class weights for balanced training
    labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    # Save class names
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    with open('class_names_lightning.json', 'w') as f:
        json.dump(class_names, f)
    
    return train_dataset, valid_dataset, train_generator, valid_generator, class_weights_dict

def create_lightning_model(num_classes, architecture='mobilenet'):
    """Create lightning-fast model optimized for speed"""
    
    if architecture.lower() == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model_name = "EfficientNetB0-Lightning"
    else:  # Default to MobileNet for maximum speed
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model_name = "MobileNetV2-Lightning"
    
    # Ultra-lightweight head for maximum speed
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)  # Reduced from 256 for speed
    x = Dropout(0.2)(x)  # Reduced dropout for faster training
    
    # Output layer
    if USE_MIXED_PRECISION:
        predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze most layers for speed - only train last few
    for layer in base_model.layers[:-5]:  # Freeze all but last 5 layers
        layer.trainable = False
    
    # Compile with aggressive settings for speed
    optimizer = Adam(learning_rate=LEARNING_RATE, epsilon=1e-3)  # Larger epsilon for speed
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=USE_XLA_COMPILATION
    )
    
    print(f"Created {model_name} model with {model.count_params():,} parameters")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,} (only {trainable_params/model.count_params()*100:.1f}%)")
    
    return model

def train_lightning_fast(data_dir, architecture='mobilenet'):
    """Lightning-fast training for maximum speed"""
    
    print("⚡ Starting lightning-fast training...")
    start_time = time.time()
    
    # Create optimized data pipeline
    train_dataset, valid_dataset, train_gen, valid_gen, class_weights = create_lightning_data_pipeline(data_dir)
    
    # Create model
    num_classes = train_gen.num_classes
    model = create_lightning_model(num_classes, architecture)
    
    # Minimal callbacks for speed
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=2,  # Very aggressive early stopping
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive LR reduction
            patience=1,  # Immediate response
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Calculate steps
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, valid_gen.samples // BATCH_SIZE)
    
    print(f"Lightning training setup:")
    print(f"- Classes: {num_classes}")
    print(f"- Training samples: {train_gen.samples}")
    print(f"- Validation samples: {valid_gen.samples}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Maximum epochs: {EPOCHS}")
    
    # Single-phase training for maximum speed
    print("\n⚡ Lightning training (single phase)")
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_PATH)
    
    total_time = time.time() - start_time
    print(f"\n⚡ Lightning training completed in {total_time/60:.1f} minutes!")
    print(f"Model saved to {MODEL_PATH}")
    
    # Get final accuracy
    final_acc = max(history.history['val_accuracy']) if history.history['val_accuracy'] else 0
    print(f"Final validation accuracy: {final_acc:.1%}")
    
    # Quick training plot
    plot_lightning_history(history)
    
    return model

def plot_lightning_history(history):
    """Quick training history plot"""
    plt.figure(figsize=(10, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    plt.title('Lightning Training - Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'b-', label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    plt.title('Lightning Training - Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lightning_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history saved to lightning_training_history.png")

def preprocess_image_fast(image_path):
    """Fast image preprocessing"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease_fast(model, image_path):
    """Fast disease prediction"""
    with open('class_names_lightning.json', 'r') as f:
        class_names = json.load(f)
    
    img = preprocess_image_fast(image_path)
    preds = model.predict(img, verbose=0)
    
    pred_idx = np.argmax(preds[0])
    pred_class = class_names[str(pred_idx)]
    confidence = float(preds[0][pred_idx])
    
    return pred_class, confidence

def make_gradcam_heatmap_fast(img_array, model, last_conv_layer_name, pred_index=None):
    """Fast Grad-CAM generation"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_last_conv_layer_name_fast(model):
    """Fast last conv layer detection"""
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and any(arch in layer.name.lower() 
                                         for arch in ['mobilenet', 'efficientnet']):
            base_model = layer
            break
    
    if base_model is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    for layer in reversed(base_model.layers):
        if 'conv' in layer.name.lower() and 'add' not in layer.name.lower():
            return layer.name
    
    return None

def create_gradcam_visualization_fast(image_path, model, predicted_class, confidence):
    """Fast Grad-CAM visualization"""
    img_array = preprocess_image_fast(image_path)
    last_conv_layer_name = get_last_conv_layer_name_fast(model)
    
    if last_conv_layer_name is None:
        print("Warning: Could not find last convolutional layer for Grad-CAM")
        return None
    
    heatmap = make_gradcam_heatmap_fast(img_array, model, last_conv_layer_name)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    superimposed_img = heatmap_colored * 0.4 + img * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Heatmap', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'{predicted_class.replace("___", " - ").replace("_", " ")}\n{confidence:.1%}', fontsize=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    gradcam_path = 'gradcam_lightning.png'
    plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return gradcam_path

def generate_lightning_report(image_path, disease_name, confidence, output_pdf, gradcam_path=None):
    """Generate fast, simplified PDF report"""
    disease_info = DISEASE_REMEDIES.get(disease_name, DEFAULT_REMEDY)
    
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'], fontSize=20, textColor=colors.darkgreen,
        spaceAfter=15, alignment=1
    )
    story.append(Paragraph("⚡ Lightning Plant Disease Detection", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Quick diagnosis
    disease_display = disease_name.replace('___', ' - ').replace('_', ' ')
    diagnosis_data = [
        ['Disease:', disease_display],
        ['Confidence:', f'{confidence:.1%}'],
        ['Severity:', disease_info.get('severity', 'Unknown')]
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[1.5*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(diagnosis_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Add Grad-CAM if available
    if gradcam_path and os.path.exists(gradcam_path):
        story.append(Paragraph("🧠 AI Analysis", styles['Heading2']))
        gradcam_img = ReportImage(gradcam_path, width=5*inch, height=1.7*inch)
        story.append(gradcam_img)
        story.append(Spacer(1, 0.2*inch))
    
    # Quick treatment
    story.append(Paragraph("💊 Treatment", styles['Heading2']))
    for remedy in disease_info['remedies']:
        story.append(Paragraph(f"• {remedy}", styles['Normal']))
    
    doc.build(story)
    print(f"Lightning report generated: {output_pdf}")

def main():
    """Main function with lightning-fast options"""
    parser = argparse.ArgumentParser(description='Lightning-Fast Plant Disease Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Lightning-fast training')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Directory containing the plant disease dataset')
    train_parser.add_argument('--architecture', type=str, choices=['mobilenet', 'efficientnet'],
                             default='mobilenet', help='Model architecture (default: mobilenet)')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                             help=f'Batch size (default: {BATCH_SIZE})')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Fast prediction')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    predict_parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    predict_parser.add_argument('--output', type=str, default='lightning_report.pdf', help='Output PDF')
    predict_parser.add_argument('--no_gradcam', action='store_true', help='Disable Grad-CAM for speed')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Update batch size if specified
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
        
        train_lightning_fast(args.data_dir, args.architecture)
    
    elif args.command == 'predict':
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found. Train the model first.")
            return
        
        print(f"Loading lightning model from {args.model}...")
        model = load_model(args.model)
        
        print(f"Analyzing image {args.image}...")
        disease_name, confidence = predict_disease_fast(model, args.image)
        
        print(f"⚡ Detected: {disease_name}")
        print(f"⚡ Confidence: {confidence:.1%}")
        
        gradcam_path = None
        if not args.no_gradcam:
            print("Generating fast Grad-CAM...")
            try:
                gradcam_path = create_gradcam_visualization_fast(args.image, model, disease_name, confidence)
                if gradcam_path:
                    print(f"Grad-CAM saved: {gradcam_path}")
            except Exception as e:
                print(f"Warning: Grad-CAM failed: {e}")
        
        print("Generating lightning report...")
        generate_lightning_report(args.image, disease_name, confidence, args.output, gradcam_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 