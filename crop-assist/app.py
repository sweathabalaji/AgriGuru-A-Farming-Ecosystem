from flask import Flask, jsonify, request
import joblib
from ai_agent import CropPlanningAgent
from translator import TamilTranslator
from dotenv import load_dotenv
import pandas as pd
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load ML model and encoders
model = joblib.load('crop_predictor_model.pkl')
state_encoder = joblib.load('state_encoder.pkl')
district_encoder = joblib.load('district_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')

# Initialize AI Agent and Translator
ai_agent = CropPlanningAgent()
tamil_translator = TamilTranslator()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        print("➡️ /api/predict received:", data)

        state = data['state']
        district = data['district']
        investment = float(data['investment'])
        soil_type = data.get('soil_type', 'Loamy')

        state_code = state_encoder.transform([state])[0]
        district_code = district_encoder.transform([district])[0]
        soil_code = soil_encoder.transform([soil_type])[0]

        input_data = pd.DataFrame([[state_code, district_code, investment, soil_code]], 
                                   columns=['State_Code', 'District_Code', 'Investment_Price', 'Soil_Code'])
        prediction = model.predict(input_data)
        crop = crop_encoder.inverse_transform(prediction)[0]

        print(f"✅ Predicted crop: {crop}")
        return jsonify({"success": True, "crop": crop})
    except Exception as e:
        print(f"❌ Error in /api/predict: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/plan', methods=['POST'])
def api_plan():
    try:
        data = request.get_json()
        print("➡️ /api/plan received:", data)

        state = data['state']
        district = data['district']
        investment = float(data['investment'])
        duration = int(data['duration'])
        acres = float(data.get('acres', 1))
        soil_type = data.get('soil_type', "Loamy")

        state_code = state_encoder.transform([state])[0]
        district_code = district_encoder.transform([district])[0]
        soil_code = soil_encoder.transform([soil_type])[0]

        input_data = pd.DataFrame([[state_code, district_code, investment, soil_code]], 
                                   columns=['State_Code', 'District_Code', 'Investment_Price', 'Soil_Code'])
        prediction = model.predict(input_data)
        crop = crop_encoder.inverse_transform(prediction)[0]

        location = f"{district}, {state}"
        crop_plan = ai_agent.create_crop_plan(crop, duration, location, investment, soil_type, acres)

        print(f"✅ Generated crop plan for {crop} in {location}")
        return jsonify({
            "success": True,
            "crop": crop,
            "location": location,
            "soil_type": soil_type,
            "overview": crop_plan.get("overview"),
            "estimated_yield": crop_plan.get("estimated_yield"),
            "yield_roi": crop_plan.get("yield_roi"),
            "plan": crop_plan.get("plan")
        })
    except Exception as e:
        print(f"❌ Error in /api/plan: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Flask API on http://localhost:5000")
    app.run(debug=True)
