#!/usr/bin/env python3
"""
Simple test script to demonstrate how to use the plant disease detection model.
This script loads a trained model and makes a prediction on a sample image.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from plant_disease_detection import predict_disease, generate_report, preprocess_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Plant Disease Detection Model')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the test image')
    parser.add_argument('--model', type=str, default='plant_disease_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--output', type=str, default='test_report.pdf',
                        help='Path for the output PDF report')
    parser.add_argument('--show_image', action='store_true',
                        help='Display the image with prediction')
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Please train the model first or provide the correct path.")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    try:
        model = load_model(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    print(f"Analyzing image: {args.image}")
    try:
        disease_name, confidence = predict_disease(model, args.image)
        
        # Print results
        print(f"\nResults:")
        print(f"Detected Disease: {disease_name.replace('___', ' - ').replace('_', ' ')}")
        print(f"Confidence: {confidence:.2%}")
        
        # Generate report
        print(f"\nGenerating PDF report to {args.output}...")
        generate_report(args.image, disease_name, confidence, args.output)
        print(f"Report generated successfully!")
        
        # Optionally display the image with prediction
        if args.show_image:
            # Load and preprocess image for display
            import cv2
            import numpy as np
            
            img = cv2.imread(args.image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Detected: {disease_name.replace('___', ' - ').replace('_', ' ')}\nConfidence: {confidence:.2%}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

if __name__ == "__main__":
    main() 