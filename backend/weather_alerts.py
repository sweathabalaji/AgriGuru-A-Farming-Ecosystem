import os
from fastapi import FastAPI, HTTPException
import requests
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Weatherbit API Configuration
WEATHERBIT_API_KEY = "2def20a210ad4d459fd6c52818694218"
WEATHERBIT_BASE_URL = "https://api.weatherbit.io/v2.0"

def get_weather_alerts(lat: float, lon: float):
    """Fetch weather alerts from Weatherbit API"""
    try:
        # Get current weather data
        current_weather_url = f"{WEATHERBIT_BASE_URL}/current"
        params = {
            "lat": lat,
            "lon": lon,
            "key": WEATHERBIT_API_KEY,
            "include": "alerts"
        }

        print(f"Fetching weather data from Weatherbit...")
        response = requests.get(current_weather_url, params=params)
        print(f"Response Status Code: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            raise HTTPException(status_code=500, detail="No weather data available")

        # Extract current conditions
        current = data["data"][0]
        
        # Get location name
        location_name = current.get("city_name", "Your Location")
        
        # Initialize alerts list
        alerts = []

        # Check temperature conditions
        temp = current.get("temp", 0)  # Temperature in Celsius
        if temp > 35:
            alerts.append(
                "High Temperature Alert (35°C+) - Ensure proper irrigation. Recommended actions: 1) Increase watering frequency, especially during morning and evening 2) Apply mulching to retain soil moisture 3) Consider temporary shade structures for sensitive crops 4) Monitor plants for heat stress signs / "
                "அதிக வெப்பநிலை எச்சரிக்கை (35°C+) - சரியான பாசனத்தை உறுதி செய்யவும். பரிந்துரைக்கப்படும் நடவடிக்கைகள்: 1) காலை மற்றும் மாலை நேரங்களில் நீர் பாய்ச்சும் அதிர்வெண்ணை அதிகரிக்கவும் 2) மண் ஈரப்பதத்தை தக்க வைக்க மல்ச்சிங் செய்யவும் 3) உணர்திறன் பயிர்களுக்கு தற்காலிக நிழல் அமைப்புகளை பரிசீலிக்கவும் 4) வெப்ப அழுத்த அறிகுறிகளுக்கு பயிர்களை கண்காணிக்கவும்"
            )

        # Check cloud cover and temperature combination
        clouds = current.get("clouds", 0)  # Cloud coverage percentage
        if clouds < 20 and temp > 30:
            alerts.append(
                "Clear Sky with High Temperature - Critical farm advisory: 1) Create temporary shade using shade nets or natural covers 2) Focus irrigation on root zones 3) Avoid midday farm operations 4) Protect workers and livestock from direct sunlight 5) Monitor leaf burn in sensitive crops / "
                "தெளிவான வானம் மற்றும் அதிக வெப்பநிலை - முக்கிய விவசாய ஆலோசனை: 1) ஷேட் நெட் அல்லது இயற்கை மூடிகளைப் பயன்படுத்தி தற்காலிக நிழலை உருவாக்கவும் 2) வேர் மண்டலங்களில் பாசனத்தை கவனிக்கவும் 3) நண்பகல் நேர விவசாய செயல்பாடுகளை தவிர்க்கவும் 4) தொழிலாளர்கள் மற்றும் கால்நடைகளை நேரடி சூரிய ஒளியிலிருந்து பாதுகாக்கவும் 5) உணர்திறன் பயிர்களில் இலை எரிவை கண்காணிக்கவும்"
            )

        # Check precipitation
        precip = current.get("precip", 0)  # Precipitation in mm
        if precip > 0:
            if precip > 10:
                alerts.append("Heavy rain expected - Protect harvested crops and avoid field work / கனமழை எதிர்பார்க்கப்படுகிறது - அறுவடை செய்த பயிர்களை பாதுகாத்து வயல் வேலைகளை தவிர்க்கவும்")
            else:
                alerts.append("Light rain expected - Plan field work accordingly / லேசான மழை எதிர்பார்க்கப்படுகிறது - வயல் வேலைகளை திட்டமிட்டு செய்யவும்")

        # Check humidity
        rh = current.get("rh", 0)  # Relative humidity
        if rh > 80:
            alerts.append("High humidity conditions - Monitor for potential fungal diseases / அதிக ஈரப்பதம் நிலைமைகள் - பூஞ்சை நோய்களை கவனிக்கவும்")
        elif rh < 30:
            alerts.append("Low humidity alert - Increase irrigation frequency / குறைந்த ஈரப்பதம் எச்சரிக்கை - பாசன அதிர்வெண்ணை அதிகரிக்கவும்")

        # Check wind conditions
        wind_spd = current.get("wind_spd", 0)  # Wind speed in m/s
        if wind_spd > 10:
            alerts.append("Strong winds expected - Secure young plants and delay spraying operations / பலத்த காற்று எதிர்பார்க்கப்படுகிறது - இளம் செடிகளை பாதுகாத்து தெளிப்பு பணிகளை தள்ளி வைக்கவும்")

        # Check cloud cover
        clouds = current.get("clouds", 0)  # Cloud coverage percentage
        if clouds > 80:
            alerts.append("Heavily cloudy conditions - Monitor light exposure for sensitive crops / அதிக மேக மூட்டம் - ஒளி வெளிப்பாட்டை கண்காணிக்கவும்")
        elif clouds < 20 and temp > 30:
            alerts.append("Clear sky with high temperature - Ensure adequate shade for sensitive crops / தெளிவான வானம் மற்றும் அதிக வெப்பநிலை - போதுமான நிழலை உறுதி செய்யவும்")

        # Check UV index for sun exposure
        uv = current.get("uv", 0)
        if uv > 8:
            alerts.append("High UV levels - Protect workers and sensitive crops from sun exposure / அதிக புற ஊதா கதிர்வீச்சு - தொழிலாளர்கள் மற்றும் உணர்திறன் பயிர்களை பாதுகாக்கவும்")

        # Check soil moisture if available
        soil = current.get("soil_moisture", None)
        if soil is not None:
            if soil < 0.2:  # Example threshold
                alerts.append("Low soil moisture detected - Consider irrigation / குறைந்த மண் ஈரப்பதம் கண்டுபிடிக்கப்பட்டால் - நீர்ப்பாசனத்தை பரிசீலிக்கவும்")

        # If no specific alerts, add default message
        if not alerts:
            alerts = ["Normal weather conditions - Good for regular farming activities / சாதாரண வானிலை நிலைமைகள் - வழக்கமான விவசாய நடவடிக்கைகளுக்கு ஏற்றது"]

        # Translate alerts to include Tamil
        translated_alerts = [get_basic_tamil_translation(alert) for alert in alerts]

        return {
            "location": location_name,
            "alerts": translated_alerts,
            "current_conditions": {
                "temperature": f"{temp}°C",
                "humidity": f"{rh}%",
                "wind_speed": f"{wind_spd} m/s",
                "precipitation": f"{precip} mm",
                "cloud_cover": f"{clouds}%",
                "uv_index": uv
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

def get_basic_tamil_translation(text: str) -> str:
    """Basic Tamil translations for weather alerts"""
    translations = {
        "Thunderstorm expected": "இடியுடன் கூடிய மழை எதிர்பார்க்கப்படுகிறது",
        "Heavy rain expected": "கனமழை எதிர்பார்க்கப்படுகிறது",
        "Light rain expected": "லேசான மழை எதிர்பார்க்கப்படுகிறது",
        "Rain expected": "மழை எதிர்பார்க்கப்படுகிறது",
        "High temperature alert": "அதிக வெப்பநிலை எச்சரிக்கை",
        "High humidity conditions": "அதிக ஈரப்பதம் நிலைமைகள்",
        "Cloudy conditions": "மேகமூட்டம் நிலைமைகள்",
        "Strong winds expected": "பலத்த காற்று எதிர்பார்க்கப்படுகிறது",
        "Low temperature alert": "குறைந்த வெப்பநிலை எச்சரிக்கை",
        "Extreme heat alert": "கடுமையான வெப்ப எச்சரிக்கை",
        "Low humidity alert": "குறைந்த ஈரப்பதம் எச்சரிக்கை",
        "Very high humidity": "மிக அதிக ஈரப்பதம்",
        "Normal weather conditions": "சாதாரண வானிலை நிலைமைகள்",
        "Heavily cloudy conditions": "அதிக மேக மூட்டம்",
        "Clear sky": "தெளிவான வானம்",
        "High UV levels": "அதிக புற ஊதா கதிர்வீச்சு",
        "Low soil moisture": "குறைந்த மண் ஈரப்பதம்"
    }
    
    # Add farming advice translations
    farming_translations = {
        "Avoid pesticide spraying and field work": "பூச்சிக்கொல்லி தெளிப்பதையும் வயல் வேலைகளையும் தவிர்க்கவும்",
        "Protect harvested crops and avoid field work": "அறுவடை செய்த பயிர்களை பாதுகாத்து வயல் வேலைகளை தவிர்க்கவும்",
        "Plan field work accordingly": "வயல் வேலைகளை திட்டமிட்டு செய்யவும்",
        "Ensure proper irrigation": "சரியான பாசனத்தை உறுதி செய்யவும்",
        "Monitor for potential fungal diseases": "பூஞ்சை நோய்களை கவனிக்கவும்",
        "Good for general farming activities": "பொதுவான விவசாய நடவடிக்கைகளுக்கு ஏற்றது",
        "Secure young plants and delay spraying operations": "இளம் செடிகளை பாதுகாத்து தெளிப்பு பணிகளை தள்ளி வைக்கவும்",
        "Protect sensitive crops from cold": "குளிரில் இருந்து உணர்திறன் பயிர்களை பாதுகாக்கவும்",
        "Ensure adequate irrigation and shade": "போதுமான பாசனம் மற்றும் நிழலை உறுதி செய்யவும்",
        "Increase irrigation frequency": "பாசன அதிர்வெண்ணை அதிகரிக்கவும்",
        "Watch for disease development": "நோய் வளர்ச்சியை கவனிக்கவும்",
        "Good for regular farming activities": "வழக்கமான விவசாய நடவடிக்கைகளுக்கு ஏற்றது",
        "Monitor light exposure": "ஒளி வெளிப்பாட்டை கண்காணிக்கவும்",
        "Ensure adequate shade": "போதுமான நிழலை உறுதி செய்யவும்",
        "Protect workers and sensitive crops": "தொழிலாளர்கள் மற்றும் உணர்திறன் பயிர்களை பாதுகாக்கவும்",
        "Consider irrigation": "நீர்ப்பாசனத்தை பரிசீலிக்கவும்"
    }

    # Combine both translation dictionaries
    all_translations = {**translations, **farming_translations}
    
    # Try to find direct matches first
    if text in all_translations:
        return f"{text} / {all_translations[text]}"
    
    # If no direct match, try to find partial matches
    for eng, tamil in all_translations.items():
        if eng in text:
            return f"{text} / {tamil}"
    
    # If no match found, return original text
    return text

# Add the FastAPI route handler
async def weather_alerts_handler(lat: float, lon: float):
    """FastAPI route handler for weather alerts"""
    return get_weather_alerts(lat, lon) 