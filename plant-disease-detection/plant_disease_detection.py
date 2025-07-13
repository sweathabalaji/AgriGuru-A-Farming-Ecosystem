#!/usr/bin/env python3
# Plant Disease Detection using ResNet50 and TensorFlow with Explainable AI
# This script provides a complete implementation for:
# 1. Loading and preprocessing the PlantVillage dataset
# 2. Training a ResNet50 model for plant disease classification
# 3. Generating PDF reports with disease information and remedies
# 4. Explainable AI using Grad-CAM for visual interpretation

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
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

# Define constants
IMG_SIZE = 224  # ResNet50 default input size
BATCH_SIZE = 64  # Increased from 32 for faster training
EPOCHS = 20
LEARNING_RATE = 0.0001
MODEL_PATH = 'plant_disease_model.h5'
USE_MIXED_PRECISION = True  # Enable mixed precision training for speed

# Fix mixed precision API - use only stable API
if USE_MIXED_PRECISION:
    try:
        # Use the stable API
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')
    except Exception as e:
        # If any error occurs, disable mixed precision
        print(f'Mixed precision could not be enabled: {e}')
        print('Training will continue with default precision')

# Enhanced dictionary mapping diseases to detailed information
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
            "Small, circular to elongated cinnamon-brown pustules on both leaf surfaces",
            "Pustules turn dark brown to black as they mature",
            "Pustules can appear on stalks and husks in severe cases",
            "Leaves may turn yellow and die in severe infections",
            "Pustules rupture leaf surface, releasing powdery rust spores",
            "Initial symptoms appear as flecks, then develop into pustules",
            "Pustules are scattered across leaf surface rather than clustered"
        ],
        "remedies": [
            "Apply appropriate fungicides at first sign of disease",
            "Use systemic fungicides for better control",
            "Time fungicide applications based on disease forecasts",
            "Remove alternate host plants from field edges",
            "Improve air circulation in the field",
            "Monitor fields regularly during growing season",
            "Consider fungicide application at tasseling stage if disease pressure is high"
        ],
        "maintenance": [
            "Plant resistant hybrids when available",
            "Avoid highly susceptible varieties in high-risk areas",
            "Scout fields regularly, especially in humid conditions",
            "Maintain balanced soil fertility",
            "Avoid excessive nitrogen fertilization",
            "Practice good field hygiene",
            "Rotate crops with non-host plants",
            "Time planting to avoid peak disease periods"
        ],
        "severity": "Moderate to High"
    }
}

# Default remedy information for unknown diseases
DEFAULT_REMEDY = {
    "description": "Unknown plant disease detected.",
    "causes": ["Unknown pathogen or condition"],
    "symptoms": ["Various symptoms may be present"],
    "remedies": [
        "Consult with a local agricultural extension office",
        "Remove and isolate affected plants",
        "Apply broad-spectrum fungicide if appropriate",
        "Improve plant growing conditions"
    ],
    "maintenance": [
        "Regular monitoring",
        "Good sanitation practices",
        "Proper plant nutrition",
        "Adequate watering"
    ],
    "severity": "Unknown"
}

def create_model(num_classes):
    """Create and compile a ResNet50 model for plant disease classification"""
    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  # Add BatchNormalization for better training stability
    x = Dense(512, activation='relu')(x)  # Reduced from 1024 to 512 for faster training
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Increased dropout for better generalization
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_mobilenet_model(num_classes):
    """Create and compile a MobileNetV2 model for plant disease classification
       This is a faster alternative to ResNet50"""
    # Load the MobileNetV2 model with pre-trained ImageNet weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=3):
    """Unfreeze some layers of the base model for fine-tuning
       Only unfreeze the last few layers to preserve learned features"""
    # Find the base model
    base_model = None
    
    # Try different approaches to find the base model
    if hasattr(model, 'layers') and len(model.layers) > 0:
        # Check if the first layer is the base model
        if hasattr(model.layers[0], 'name'):
            if model.layers[0].name in ['resnet50', 'mobilenetv2']:
                base_model = model.layers[0]
        
        # If not found, search through all layers
        if base_model is None:
            for layer in model.layers:
                if hasattr(layer, 'name') and layer.name in ['resnet50', 'mobilenetv2']:
                    base_model = layer
                    break
    
    # If base model still not found, use a simpler approach
    if base_model is None:
        # Just unfreeze the last few layers of the entire model
        trainable_layers = [layer for layer in model.layers if hasattr(layer, 'trainable')]
        for layer in trainable_layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
        print(f"Unfreezing last {num_layers_to_unfreeze} trainable layers")
    else:
        # Unfreeze the last few layers of the found base model
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
        print(f"Unfreezing last {num_layers_to_unfreeze} layers of {base_model.name}")
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the given image and model"""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_last_conv_layer_name(model):
    """Find the name of the last convolutional layer in the model"""
    # Look for the base model (ResNet50 or MobileNetV2)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and ('resnet50' in layer.name.lower() or 'mobilenet' in layer.name.lower()):
            base_model = layer
            break
    
    if base_model is None:
        # Fallback: look for any convolutional layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    # Find the last convolutional layer in the base model
    for layer in reversed(base_model.layers):
        if 'conv' in layer.name.lower() and 'add' not in layer.name.lower():
            return layer.name
    
    return None

def create_gradcam_visualization(image_path, model, predicted_class, confidence):
    """Create Grad-CAM visualization and save it"""
    # Load and preprocess the image
    img_array = preprocess_image(image_path)
    
    # Get the last convolutional layer name
    last_conv_layer_name = get_last_conv_layer_name(model)
    
    if last_conv_layer_name is None:
        print("Warning: Could not find last convolutional layer for Grad-CAM")
        return None
    
    print(f"Using layer '{last_conv_layer_name}' for Grad-CAM")
    
    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Create the Grad-CAM visualization
    heatmap_colored = cm.jet(heatmap_resized)
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * 0.4 + img * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Superimposed
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Grad-CAM Overlay\nPredicted: {predicted_class.replace("___", " - ").replace("_", " ")}\nConfidence: {confidence:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    gradcam_path = 'gradcam_visualization.png'
    plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return gradcam_path

def prepare_data(data_dir):
    """Prepare data generators for training and validation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased from 20
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip
        brightness_range=[0.8, 1.2],  # Added brightness variation
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation data generator
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save the class indices for later use
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Save class names to a JSON file
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print(f"Found {len(class_names)} classes: {list(class_names.values())}")
    
    # Calculate class weights to handle imbalanced data
    labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    return train_generator, valid_generator, len(class_names), class_weights_dict

def train_model(data_dir, use_mobilenet=False):
    """Train the model on the plant disease dataset"""
    # Prepare data
    train_generator, valid_generator, num_classes, class_weights = prepare_data(data_dir)
    
    # Create model
    if use_mobilenet:
        model = create_mobilenet_model(num_classes)
        print("MobileNetV2 model created. Starting initial training...")
    else:
        model = create_model(num_classes)
        print("ResNet50 model created. Starting initial training...")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Initial training with frozen base model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=5,  # Reduced from 10 for faster initial training
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )
    
    # Fine-tuning
    print("Starting fine-tuning...")
    model = fine_tune_model(model, num_layers_to_unfreeze=3)
    
    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        initial_epoch=history.epoch[-1] + 1,
        class_weight=class_weights
    )
    
    # Save the final model
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    
    # Plot and save training history
    plot_training_history(history, history_fine_tune)
    
    return model

def plot_training_history(history1, history2=None):
    """Plot and save the training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Initial Training')
    plt.plot(history1.history['val_accuracy'], label='Initial Validation')
    
    if history2:
        # Combine epochs
        offset = len(history1.history['accuracy'])
        epochs_fine = [i + offset for i in range(len(history2.history['accuracy']))]
        plt.plot(epochs_fine, history2.history['accuracy'], label='Fine-tuning Training')
        plt.plot(epochs_fine, history2.history['val_accuracy'], label='Fine-tuning Validation')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label='Initial Training')
    plt.plot(history1.history['val_loss'], label='Initial Validation')
    
    if history2:
        plt.plot(epochs_fine, history2.history['loss'], label='Fine-tuning Training')
        plt.plot(epochs_fine, history2.history['val_loss'], label='Fine-tuning Validation')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Load and resize the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_disease(model, image_path):
    """Predict the disease class for a given image"""
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make prediction
    preds = model.predict(img)
    
    # Get the predicted class index and name
    pred_idx = np.argmax(preds[0])
    pred_class = class_names[str(pred_idx)]
    confidence = float(preds[0][pred_idx])
    
    return pred_class, confidence

def generate_report(image_path, disease_name, confidence, output_pdf, gradcam_path=None):
    """Generate a comprehensive PDF report with disease information and XAI visualization"""
    # Get disease information
    disease_info = DISEASE_REMEDIES.get(disease_name, DEFAULT_REMEDY)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.darkgreen,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Plant Disease Detection Report", title_style))
    story.append(Paragraph("with Explainable AI Analysis", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date and metadata
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey
    )
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {current_time}", date_style))
    story.append(Paragraph(f"Image: {os.path.basename(image_path)}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis section
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceAfter=10
    )
    story.append(Paragraph("🔍 Diagnosis Results", section_style))
    
    # Disease information table
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
    
    # Add original image if Grad-CAM is available
    if gradcam_path and os.path.exists(gradcam_path):
        story.append(Paragraph("🧠 AI Explanation - Grad-CAM Visualization", section_style))
        story.append(Paragraph(
            "The visualization below shows which parts of the leaf the AI model focused on to make its prediction. "
            "Red/yellow areas indicate regions that most strongly influenced the diagnosis.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Add Grad-CAM visualization
        gradcam_img = ReportImage(gradcam_path, width=6*inch, height=2*inch)
        story.append(gradcam_img)
        story.append(Spacer(1, 0.2*inch))
    
    # Disease description
    story.append(Paragraph("📋 Disease Description", section_style))
    story.append(Paragraph(disease_info["description"], styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))
    
    # Causes
    story.append(Paragraph("🔬 Causes", section_style))
    for cause in disease_info["causes"]:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-15
        )
        story.append(Paragraph(f"• {cause}", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Symptoms
    story.append(Paragraph("🔍 Symptoms", section_style))
    for symptom in disease_info["symptoms"]:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-15
        )
        story.append(Paragraph(f"• {symptom}", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Treatment remedies
    story.append(Paragraph("💊 Treatment & Remedies", section_style))
    for remedy in disease_info["remedies"]:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-15
        )
        story.append(Paragraph(f"• {remedy}", bullet_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Maintenance
    story.append(Paragraph("🌱 Prevention & Maintenance", section_style))
    for maintenance in disease_info["maintenance"]:
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            firstLineIndent=-15
        )
        story.append(Paragraph(f"• {maintenance}", bullet_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=8,
        textColor=colors.grey,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=10
    )
    disclaimer_text = (
        "⚠️ IMPORTANT DISCLAIMER: This report is generated by an automated AI system and should be used for "
        "informational purposes only. The Grad-CAM visualization shows which parts of the image influenced "
        "the AI's decision but does not guarantee accuracy. Please consult with a professional plant pathologist "
        "or agricultural expert for confirmation and detailed treatment plans before taking any action."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build the PDF
    doc.build(story)
    print(f"Comprehensive report with XAI analysis generated and saved to {output_pdf}")

def main():
    """Main function to parse arguments and run the appropriate action"""
    parser = argparse.ArgumentParser(description='Plant Disease Detection with Explainable AI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Directory containing the plant disease dataset')
    train_parser.add_argument('--use_mobilenet', action='store_true',
                             help='Use MobileNetV2 instead of ResNet50 for faster training')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict disease from an image')
    predict_parser.add_argument('--image', type=str, required=True,
                               help='Path to the input image')
    predict_parser.add_argument('--model', type=str, default=MODEL_PATH,
                               help='Path to the trained model')
    predict_parser.add_argument('--output', type=str, default='plant_disease_report.pdf',
                               help='Path for the output PDF report')
    predict_parser.add_argument('--no_gradcam', action='store_true',
                               help='Disable Grad-CAM visualization')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir, args.use_mobilenet)
    
    elif args.command == 'predict':
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found. Please train the model first.")
            return
        
        # Load the model
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        
        # Make prediction
        print(f"Analyzing image {args.image}...")
        disease_name, confidence = predict_disease(model, args.image)
        
        # Print results
        print(f"Detected: {disease_name}")
        print(f"Confidence: {confidence:.2%}")
        
        # Generate Grad-CAM visualization if not disabled
        gradcam_path = None
        if not args.no_gradcam:
            print("Generating Grad-CAM visualization...")
            try:
                gradcam_path = create_gradcam_visualization(args.image, model, disease_name, confidence)
                if gradcam_path:
                    print(f"Grad-CAM visualization saved to {gradcam_path}")
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM visualization: {e}")
        
        # Generate comprehensive report
        print(f"Generating comprehensive PDF report...")
        generate_report(args.image, disease_name, confidence, args.output, gradcam_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 