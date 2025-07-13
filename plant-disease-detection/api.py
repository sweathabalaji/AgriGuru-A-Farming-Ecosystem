from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import uvicorn
import shutil
import os
from datetime import datetime
from PIL import Image
import io
import json
import base64
import cv2
import matplotlib.pyplot as plt

# Import the prediction functions from our fast model
from plant_disease_detection_fast import (
    predict_disease,
    make_gradcam_heatmap,
    generate_report,
    DISEASE_REMEDIES
)

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases from images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing files
TEMP_DIR = "temp"
RESULTS_DIR = "results"
for dir_path in [TEMP_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Store prediction results for download
PREDICTIONS = {}

# Load the model at startup
MODEL = None
try:
    MODEL = tf.keras.models.load_model("plant_disease_model_fast.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names
try:
    with open('class_names.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except Exception as e:
    print(f"Error loading class names: {e}")
    CLASS_NAMES = {}

def save_uploaded_image(file: UploadFile, target_path: str):
    """Safely save an uploaded file"""
    try:
        # Read the content first to verify it's a valid image
        content = file.file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image file")
            
        # Save the image
        cv2.imwrite(target_path, img)
        
        # Reset file pointer for potential reuse
        file.file.seek(0)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def create_gradcam_visualization(image_path, model, disease_name, confidence, output_path):
    """Create and save Grad-CAM visualization"""
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img.astype('float32') / 255, axis=0)
        
        # Generate Grad-CAM heatmap
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
        
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model")
            
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(superimposed_img)
        plt.title(f"Disease: {disease_name}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error creating Grad-CAM visualization: {e}")
        return False

def cleanup_temp_files(files: list):
    """Safely cleanup temporary files"""
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for complete disease analysis"
        }
    }

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Upload an image and get complete disease analysis including:
    - Disease prediction
    - Confidence score
    - PDF report (base64 encoded)
    - Visualization image (base64 encoded)
    - Detailed disease information
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Initialize temp file paths
    temp_files = []
    try:
        # Create unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_id = f"{timestamp}_{os.path.splitext(file.filename)[0]}"
        
        # Create file paths
        temp_image_path = os.path.join(TEMP_DIR, f"{prediction_id}_input.jpg")
        report_path = os.path.join(RESULTS_DIR, f"{prediction_id}_report.pdf")
        viz_path = os.path.join(RESULTS_DIR, f"{prediction_id}_gradcam.png")
        
        temp_files = [temp_image_path]
        
        # Save uploaded file
        if not save_uploaded_image(file, temp_image_path):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        if MODEL is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run prediction
        disease_name, confidence = predict_disease(MODEL, temp_image_path)
        
        # Create visualization directory if it doesn't exist
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        
        # Generate visualization
        if not create_gradcam_visualization(temp_image_path, MODEL, disease_name, confidence, viz_path):
            raise HTTPException(status_code=500, detail="Failed to generate visualization")
        
        # Generate report
        generate_report(temp_image_path, disease_name, confidence, report_path, viz_path)
        
        # Verify files exist
        if not all(os.path.exists(f) for f in [report_path, viz_path]):
            raise HTTPException(status_code=500, detail="Failed to generate report or visualization")
            
        # Get disease information
        disease_info = DISEASE_REMEDIES.get(disease_name, {
            "description": "Information not available",
            "causes": [],
            "symptoms": [],
            "remedies": [],
            "maintenance": [],
            "severity": "Unknown"
        })
        
        # Store prediction info for downloads
        PREDICTIONS[prediction_id] = {
            "report_path": report_path,
            "visualization_path": viz_path,
            "timestamp": timestamp,
            "disease_name": disease_name,
            "confidence": confidence
        }
        
        # Clean up temporary input file
        cleanup_temp_files(temp_files)
        
        return JSONResponse({
            "status": "success",
            "prediction": {
                "disease_name": disease_name.replace('___', ' - ').replace('_', ' '),
                "confidence": f"{confidence:.2%}",
                "prediction_id": prediction_id
            },
            "disease_info": disease_info,
            "download_urls": {
                "report": f"/download/report/{prediction_id}",
                "visualization": f"/download/visualization/{prediction_id}"
            },
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "timestamp": timestamp
            }
        })
        
    except Exception as e:
        # Clean up any remaining temporary files
        cleanup_temp_files(temp_files)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/report/{prediction_id}")
async def download_report(prediction_id: str):
    """Download the PDF report for a prediction"""
    if prediction_id not in PREDICTIONS:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report_path = PREDICTIONS[prediction_id]["report_path"]
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"plant_disease_report_{prediction_id}.pdf"
    )

@app.get("/download/visualization/{prediction_id}")
async def download_visualization(prediction_id: str):
    """Download the Grad-CAM visualization for a prediction"""
    if prediction_id not in PREDICTIONS:
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    viz_path = PREDICTIONS[prediction_id]["visualization_path"]
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Visualization file not found")
    
    return FileResponse(
        viz_path,
        media_type="image/png",
        filename=f"disease_visualization_{prediction_id}.png"
    )

@app.get("/predictions")
async def list_predictions():
    """List all available predictions"""
    return {
        "predictions": [
            {
                "id": pred_id,
                "disease_name": info["disease_name"].replace('___', ' - ').replace('_', ' '),
                "confidence": f"{info['confidence']:.2%}",
                "timestamp": info["timestamp"],
                "download_urls": {
                    "report": f"/download/report/{pred_id}",
                    "visualization": f"/download/visualization/{pred_id}"
                }
            }
            for pred_id, info in PREDICTIONS.items()
        ]
    }

if __name__ == "__main__":
    # Try different ports if default port is unavailable
    ports = [8080, 5000, 3000]  # Common alternative ports
    host = "127.0.0.1"  # Use localhost instead of 0.0.0.0 for Windows
    
    for port in ports:
        try:
            print(f"Attempting to start server on port {port}...")
            uvicorn.run("api:app", host=host, port=port, reload=True)
            break
        except Exception as e:
            print(f"Failed to bind to port {port}: {e}")
            if port == ports[-1]:
                print("All ports failed. Please ensure no other services are using these ports.")
                raise 