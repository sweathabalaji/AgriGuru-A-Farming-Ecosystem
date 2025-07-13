# Ultra-Fast Plant Disease Detection using ResNet50 and TensorFlow with Explainable AI
# Optimized for maximum training speed while maintaining accuracy

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
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
from PIL import Image, ImageDraw, ImageFont
import time
from sklearn.utils.class_weight import compute_class_weight
import math  # Add this at the top with other imports

# SPEED OPTIMIZATION CONSTANTS
IMG_SIZE = 224  # Keep standard ResNet input size
BATCH_SIZE = 128  # Increased from 64 for better GPU utilization
EPOCHS = 15  # Reduced from 20 - early stopping will handle convergence
LEARNING_RATE = 0.001  # Increased for faster convergence
MODEL_PATH = 'plant_disease_model_fast.h5'

# Advanced optimization settings
USE_MIXED_PRECISION = False
USE_XLA_COMPILATION = True  # XLA compilation for faster execution
PREFETCH_BUFFER = tf.data.AUTOTUNE
CACHE_DATASET = True

# Configure TensorFlow for maximum performance
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU acceleration enabled: {physical_devices[0]}")

# Enable mixed precision and XLA
if USE_MIXED_PRECISION:
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled - up to 2x speed boost')
    except Exception as e:
        print(f'Mixed precision not available: {e}')

if USE_XLA_COMPILATION:
    tf.config.optimizer.set_jit(True)
    print('XLA compilation enabled - additional 10-30% speed boost')

# Enhanced disease information (same as before but condensed for faster loading)
DISEASE_REMEDIES = {
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust of corn is caused by the fungus Puccinia sorghi. It is one of the most frequently occurring foliar diseases of corn, capable of causing significant yield losses in susceptible hybrids when conditions are favorable for disease development.",
        "causes": [
            "Infection by Puccinia sorghi fungus",
            "Cool temperatures (60-77°F)",
            "High humidity (>95%) or extended dew periods",
            "Frequent rainfall or overhead irrigation",
            "Wind-blown spores from infected plants",
            "Presence of alternate host plants",
            "Susceptible corn hybrids"
        ],
        "symptoms": [
            "Small, circular to elongated cinnamon-brown pustules on leaves",
            "Pustules appear on both leaf surfaces",
            "Dark brown to black spores as pustules mature",
            "Chlorotic (yellow) areas around pustules",
            "Severe infection can cause leaf death",
            "Reduced photosynthesis and plant vigor"
        ],
        "remedies": [
            "Apply appropriate fungicides at first sign of disease",
            "Remove and destroy infected plant debris",
            "Rotate crops with non-host plants",
            "Plant resistant corn hybrids",
            "Avoid overhead irrigation",
            "Improve air circulation between plants"
        ],
        "maintenance": [
            "Regular field scouting for early detection",
            "Monitor weather conditions",
            "Maintain proper plant spacing",
            "Control weeds that may serve as alternate hosts",
            "Ensure balanced soil fertility",
            "Document disease occurrence for future planning"
        ],
        "severity": "Moderate to High"
    },
    "Apple___Apple_scab": {
        "description": "Apple scab is a disease caused by the fungus Venturia inaequalis. It manifests as olive-green to brown or black spots on leaves and fruit.",
        "causes": ["Fungal infection by Venturia inaequalis", "Humid weather", "Poor air circulation", "Infected debris"],
        "symptoms": ["Olive-green to brown spots", "Black fruit spots", "Leaf yellowing", "Premature drop"],
        "remedies": ["Apply fungicides early", "Prune for air circulation", "Remove infected debris", "Use resistant varieties"],
        "maintenance": ["Regular inspection", "Proper spacing", "Avoid overhead watering", "Preventive spraying"],
        "severity": "Moderate to High"
    },
    "Apple___Black_rot": {
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa affecting leaves, fruit, and bark.",
        "causes": ["Fungal infection", "Drought stress", "Bark wounds", "Poor sanitation"],
        "symptoms": ["Dark lesions on leaves", "Mummified fruit", "Branch cankers", "Premature drop"],
        "remedies": ["Prune diseased wood", "Remove mummified fruit", "Apply fungicides", "Improve sanitation"],
        "maintenance": ["Regular pruning", "Proper irrigation", "Wound protection", "Debris removal"],
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust requires both apple and cedar hosts, creating orange-yellow spots.",
        "causes": ["Dual host pathogen", "Cedar tree proximity", "Wet spring weather", "Spore dispersal"],
        "symptoms": ["Orange-yellow spots", "Orange pustules", "Premature leaf drop", "Reduced photosynthesis"],
        "remedies": ["Remove cedar trees", "Spring fungicides", "Resistant varieties", "Improve air flow"],
        "maintenance": ["Monitor cedar proximity", "Weather-based spraying", "Early detection", "Tree vigor"],
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "description": "Healthy apple plant with no visible disease symptoms.",
        "causes": ["No disease detected"], "symptoms": ["No visible symptoms"],
        "remedies": ["Continue maintenance", "Monitor regularly", "Proper nutrition", "Good sanitation"],
        "maintenance": ["Health monitoring", "Nutrition management", "Water supply", "Pest management"],
        "severity": "None"
    },
    "Tomato___Early_blight": {
        "description": "Early blight caused by Alternaria solani with characteristic concentric ring spots.",
        "causes": ["Alternaria solani", "High humidity", "Poor air circulation", "Plant stress"],
        "symptoms": ["Concentric ring spots", "Yellowing around spots", "Lower leaf progression", "Defoliation"],
        "remedies": ["Remove infected leaves", "Apply fungicides", "Improve spacing", "Use mulch"],
        "maintenance": ["Ground-level watering", "Proper spacing", "Remove ground contact", "Crop rotation"],
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "description": "Late blight caused by Phytophthora infestans - highly destructive disease.",
        "causes": ["Phytophthora infestans", "Cool wet weather", "High humidity >90%", "Infected materials"],
        "symptoms": ["Water-soaked spots", "White fuzzy growth", "Brown stem lesions", "Rapid plant death"],
        "remedies": ["Copper fungicides", "Remove infected plants", "Avoid overhead water", "Resistant varieties"],
        "maintenance": ["Weather monitoring", "Excellent drainage", "Air circulation", "Certified seeds"],
        "severity": "Very High"
    },
    "Tomato___healthy": {
        "description": "Healthy tomato plant with no visible disease symptoms.",
        "causes": ["No disease detected"], "symptoms": ["No visible symptoms"],
        "remedies": ["Continue maintenance", "Regular monitoring", "Proper watering", "Crop rotation"],
        "maintenance": ["Plant inspection", "Watering techniques", "Balanced fertilization", "Pest monitoring"],
        "severity": "None"
    }
}

DEFAULT_REMEDY = {
    "description": "Unknown plant disease detected.",
    "causes": ["Unknown pathogen"], "symptoms": ["Various symptoms"],
    "remedies": ["Consult experts", "Isolate plants", "Apply broad-spectrum treatment", "Improve conditions"],
    "maintenance": ["Regular monitoring", "Good sanitation", "Proper nutrition", "Adequate watering"],
    "severity": "Unknown"
}

def create_fast_data_pipeline(data_dir, validation_split=0.2):
    """Create an optimized data pipeline for maximum speed"""
    
    # Aggressive but effective augmentations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators with optimized settings
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
    
    # Calculate class weights
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
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    return train_generator, valid_generator, train_generator, valid_generator, class_weights_dict

def create_ultra_fast_model(num_classes, architecture='resnet50'):
    """Create an optimized model for fast training"""
    
    if architecture.lower() == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model_name = "EfficientNetB0"
    elif architecture.lower() == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model_name = "MobileNetV2"
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model_name = "ResNet50"
    
    # Streamlined head for faster training
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)  # Reduced from 512 for speed
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    if USE_MIXED_PRECISION:
        predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile with optimized settings
    optimizer = Adam(learning_rate=LEARNING_RATE, epsilon=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=USE_XLA_COMPILATION  # Enable XLA compilation
    )
    
    print(f"Created {model_name} model with {model.count_params():,} parameters")
    return model

def smart_fine_tuning(model, architecture='resnet50'):
    """Intelligent fine-tuning that unfreezes layers strategically"""
    
    # Find base model
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and any(arch in layer.name.lower() 
                                         for arch in ['resnet', 'mobilenet', 'efficientnet']):
            base_model = layer
            break
    
    if base_model is None:
        print("Base model not found, skipping fine-tuning")
        return model
    
    # Unfreeze strategy based on architecture
    if 'efficientnet' in architecture.lower():
        # EfficientNet: unfreeze last 20 layers
        layers_to_unfreeze = 20
    elif 'mobilenet' in architecture.lower():
        # MobileNet: unfreeze last 15 layers
        layers_to_unfreeze = 15
    else:
        # ResNet: unfreeze last 10 layers
        layers_to_unfreeze = 10
    
    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE/10, epsilon=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=USE_XLA_COMPILATION
    )
    
    print(f"Fine-tuning enabled: unfroze last {layers_to_unfreeze} layers")
    return model

def train_ultra_fast(data_dir, architecture='resnet50'):
    """Ultra-fast training with all optimizations"""
    
    print("🚀 Starting ultra-fast training...")
    start_time = time.time()
    
    # Create optimized data pipeline
    train_generator, valid_generator, _, _, class_weights = create_fast_data_pipeline(data_dir)
    
    # Create model
    num_classes = train_generator.num_classes
    model = create_ultra_fast_model(num_classes, architecture)
    
    # Optimized callbacks
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
            patience=3,  # Reduced patience for faster training
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reduced patience
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Calculate steps correctly using math.ceil
    steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
    validation_steps = math.ceil(valid_generator.samples / BATCH_SIZE)
    
    print(f"Training setup:")
    print(f"- Classes: {num_classes}")
    print(f"- Training samples: {train_generator.samples}")
    print(f"- Validation samples: {valid_generator.samples}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Validation steps: {validation_steps}")
    
    # Optional: Add sanity check for overfitting test
    if os.environ.get('SANITY_CHECK'):
        print("\n🔍 Running sanity check (overfitting test)...")
        small_steps = 1  # Test with just one batch
        history_sanity = model.fit(
            train_generator,
            steps_per_epoch=small_steps,
            validation_data=valid_generator,
            validation_steps=1,
            epochs=5,
            callbacks=[],  # No callbacks for sanity check
            verbose=1
        )
        if max(history_sanity.history['accuracy']) < 0.5:
            print("⚠️ Warning: Model failed to overfit single batch. Architecture may need review.")
            return None
    
    # Phase 1: Initial training (frozen base)
    print("\n📚 Phase 1: Initial training (frozen base model)")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        epochs=3,  # Quick initial training
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n🔧 Phase 2: Fine-tuning")
    model = smart_fine_tuning(model, architecture)
    
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        initial_epoch=3,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_PATH)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes!")
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training history
    plot_training_history(history1, history2)
    
    return model

def plot_training_history(history1, history2=None):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Phase 1 Train')
    plt.plot(history1.history['val_accuracy'], label='Phase 1 Val')
    
    if history2:
        offset = len(history1.history['accuracy'])
        epochs = [i + offset for i in range(len(history2.history['accuracy']))]
        plt.plot(epochs, history2.history['accuracy'], label='Phase 2 Train')
        plt.plot(epochs, history2.history['val_accuracy'], label='Phase 2 Val')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label='Phase 1 Train')
    plt.plot(history1.history['val_loss'], label='Phase 1 Val')
    
    if history2:
        plt.plot(epochs, history2.history['loss'], label='Phase 2 Train')
        plt.plot(epochs, history2.history['val_loss'], label='Phase 2 Val')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fast_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history saved to fast_training_history.png")

# Include all the previous prediction and XAI functions (same as before)
def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(model, image_path):
    """Predict the disease class for a given image"""
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    img = preprocess_image(image_path)
    preds = model.predict(img)
    
    pred_idx = np.argmax(preds[0])
    pred_class = class_names[str(pred_idx)]
    confidence = float(preds[0][pred_idx])
    
    return pred_class, confidence

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap with improved accuracy"""
    # Create a model that maps the input image to:
    # 1. The last conv layer's activations
    # 2. The final class predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by corresponding gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Apply ReLU to focus on features that have a positive influence on the class
    heatmap = tf.nn.relu(heatmap)
    
    # Normalize again after ReLU
    if tf.math.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def get_last_conv_layer_name(model):
    """Find the name of the last convolutional layer that's most suitable for Grad-CAM"""
    # First try to find the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Check if it's a nested model
            base_model = layer
            break
    
    target_layers = []
    # If we found a base model, search in its layers
    if base_model:
        for layer in base_model.layers:
            # Look for conv layers, but exclude pointwise/depthwise convs
            if ('conv' in layer.name.lower() and 
                'pointwise' not in layer.name.lower() and 
                'depthwise' not in layer.name.lower()):
                target_layers.append(layer.name)
    else:
        # If no base model, search in the main model
        for layer in model.layers:
            if ('conv' in layer.name.lower() and 
                'pointwise' not in layer.name.lower() and 
                'depthwise' not in layer.name.lower()):
                target_layers.append(layer.name)
    
    # Return the last conv layer found
    if target_layers:
        return target_layers[-1]
    
    # Fallback to any layer with 'conv' in the name
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    
    return None

def create_gradcam_visualization(image_path, model, predicted_class, confidence):
    """Create Grad-CAM visualization"""
    img_array = preprocess_image(image_path)
    last_conv_layer_name = get_last_conv_layer_name(model)
    
    if last_conv_layer_name is None:
        print("Warning: Could not find last convolutional layer for Grad-CAM")
        return None
    
    print(f"Using layer '{last_conv_layer_name}' for Grad-CAM")
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Read and resize original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB (3 channels)
    heatmap_colored = np.uint8(255 * cm.jet(heatmap_resized)[..., :3])
    
    # Ensure both images are float32 and same shape
    img = img.astype(np.float32)
    heatmap_colored = heatmap_colored.astype(np.float32)
    
    # Create overlay
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    superimposed_img = np.uint8(superimposed_img)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img.astype(np.uint8))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Grad-CAM Overlay\nPredicted: {predicted_class.replace("___", " - ").replace("_", " ")}\nConfidence: {confidence:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    gradcam_path = 'gradcam_visualization_fast.png'
    plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return gradcam_path

def generate_report(image_path, disease_name, confidence, output_pdf, gradcam_path=None):
    """Generate comprehensive PDF report"""
    disease_info = DISEASE_REMEDIES.get(disease_name, DEFAULT_REMEDY)
    
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'], fontSize=24, textColor=colors.darkgreen,
        spaceAfter=20, alignment=1
    )
    story.append(Paragraph("Ultra-Fast Plant Disease Detection Report", title_style))
    story.append(Paragraph("with Explainable AI Analysis", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    date_style = ParagraphStyle('Date', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {current_time}", date_style))
    story.append(Paragraph(f"Image: {os.path.basename(image_path)}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'], fontSize=16, 
        textColor=colors.darkblue, spaceAfter=10
    )
    story.append(Paragraph("🔍 Diagnosis Results", section_style))
    
    disease_display = disease_name.replace('___', ' - ').replace('_', ' ')
    diagnosis_data = [
        ['Detected Disease:', disease_display],
        ['Confidence Score:', f'{confidence:.2%}'],
        ['Severity Level:', disease_info.get('severity', 'Unknown')]
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(diagnosis_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Add Grad-CAM if available
    if gradcam_path and os.path.exists(gradcam_path):
        story.append(Paragraph("🧠 AI Explanation - Grad-CAM Visualization", section_style))
        story.append(Paragraph(
            "The visualization shows which parts of the leaf the AI focused on. "
            "Red/yellow areas indicate regions that most influenced the diagnosis.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.1*inch))
        
        gradcam_img = ReportImage(gradcam_path, width=6*inch, height=2*inch)
        story.append(gradcam_img)
        story.append(Spacer(1, 0.2*inch))
    
    # Disease information sections
    for section_name, section_key in [
        ("📋 Disease Description", "description"),
        ("🔬 Causes", "causes"),
        ("🔍 Symptoms", "symptoms"),
        ("💊 Treatment & Remedies", "remedies"),
        ("🌱 Prevention & Maintenance", "maintenance")
    ]:
        story.append(Paragraph(section_name, section_style))
        
        if section_key == "description":
            story.append(Paragraph(disease_info[section_key], styles["Normal"]))
        else:
            for item in disease_info[section_key]:
                bullet_style = ParagraphStyle(
                    'Bullet', parent=styles['Normal'], leftIndent=20, firstLineIndent=-15
                )
                story.append(Paragraph(f"• {item}", bullet_style))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Italic'], fontSize=8, textColor=colors.grey,
        borderWidth=1, borderColor=colors.grey, borderPadding=10
    )
    disclaimer_text = (
        "⚠️ IMPORTANT DISCLAIMER: This report is generated by an automated AI system. "
        "The Grad-CAM visualization shows AI attention areas but doesn't guarantee accuracy. "
        "Please consult with professional plant pathologists for confirmation and detailed treatment plans."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    doc.build(story)
    print(f"Ultra-fast report generated: {output_pdf}")

def main():
    """Main function with ultra-fast options"""
    parser = argparse.ArgumentParser(description='Ultra-Fast Plant Disease Detection with XAI')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Ultra-fast training')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Directory containing the plant disease dataset')
    train_parser.add_argument('--architecture', type=str, choices=['resnet50', 'mobilenet', 'efficientnet'],
                             default='resnet50', help='Model architecture')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict with XAI')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    predict_parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    predict_parser.add_argument('--output', type=str, default='ultra_fast_report.pdf', help='Output PDF')
    predict_parser.add_argument('--no_gradcam', action='store_true', help='Disable Grad-CAM')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_ultra_fast(args.data_dir, args.architecture)
    
    elif args.command == 'predict':
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found. Train the model first.")
            return
        
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        
        print(f"Analyzing image {args.image}...")
        disease_name, confidence = predict_disease(model, args.image)
        
        print(f"Detected: {disease_name}")
        print(f"Confidence: {confidence:.2%}")
        
        gradcam_path = None
        if not args.no_gradcam:
            print("Generating Grad-CAM visualization...")
            try:
                gradcam_path = create_gradcam_visualization(args.image, model, disease_name, confidence)
                if gradcam_path:
                    print(f"Grad-CAM saved: {gradcam_path}")
            except Exception as e:
                print(f"Warning: Grad-CAM failed: {e}")
        
        print("Generating report...")
        generate_report(args.image, disease_name, confidence, args.output, gradcam_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
