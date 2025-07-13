import requests

def test_plant_disease_api(image_path):
    # API endpoint
    url = "http://127.0.0.1:8080/predict"
    
    try:
        # Open and send the image
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            print("Sending request...")
            response = requests.post(url, files=files)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Print prediction results
            print("\nResults:")
            print(f"Disease: {result['prediction']['disease_name']}")
            print(f"Confidence: {result['prediction']['confidence']}")
            
            # Save the files
            import base64
            
            # Save report
            report_path = "disease_report.pdf"
            with open(report_path, "wb") as f:
                f.write(base64.b64decode(result["report_pdf_base64"]))
            print(f"\nSaved report to: {report_path}")
            
            # Save visualization
            viz_path = "disease_visualization.png"
            with open(viz_path, "wb") as f:
                f.write(base64.b64decode(result["visualization_image_base64"]))
            print(f"Saved visualization to: {viz_path}")
            
            # Print disease information
            print("\nDisease Information:")
            print(f"Description: {result['disease_info']['description']}")
            print("\nRemedies:")
            for remedy in result['disease_info']['remedies']:
                print(f"- {remedy}")
                
            return True
        else:
            print(f"\nError: {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Plant Disease Detection API')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    
    args = parser.parse_args()
    test_plant_disease_api(args.image_path) 