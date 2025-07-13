import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
import google.generativeai as genai

# Set your direct API key here
genai.configure(api_key="AIzaSyDB9Qtf3qr238R4ruHbkcXl9OTnKqnoRx8")


class QueryProcessor:
    def __init__(self):
        self.query_handlers = {
            'crop_planning': self._handle_crop_planning,
            'weather_info': self._handle_weather_info,
            'market_analysis': self._handle_market_analysis,
            'pest_management': self._handle_pest_management,
            'fertilizer_recommendation': self._handle_fertilizer_recommendation
        }

    def process_query(self, query_type: str, **kwargs) -> Dict[str, Any]:
        handler = self.query_handlers.get(query_type)
        if handler:
            return handler(**kwargs)
        return {"error": f"Unknown query type: {query_type}"}

    def _handle_crop_planning(self, crop, duration, location, investment, soil_type="loamy") -> Dict[str, Any]:
        return {
            "type": "crop_planning",
            "crop": crop,
            "duration": duration,
            "location": location,
            "investment": investment,
            "soil_type": soil_type,
            "plan": self._generate_crop_plan(crop, duration, location, investment, soil_type)
        }

    def _handle_weather_info(self, location):
        return {"type": "weather_info", "location": location, "info": "Weather info"}

    def _handle_market_analysis(self, crop):
        return {"type": "market_analysis", "crop": crop, "analysis": f"Market analysis for {crop}"}

    def _handle_pest_management(self, crop):
        return {"type": "pest_management", "crop": crop, "recommendations": f"Pest management for {crop}"}

    def _handle_fertilizer_recommendation(self, crop, soil_type="loamy"):
        return {"type": "fertilizer_recommendation", "crop": crop, "soil_type": soil_type,
                "recommendations": f"Fertilizer for {crop} in {soil_type}"}

    def _generate_crop_plan(self, crop, duration, location, investment, soil_type="loamy", acres=1):
        plan = []
        start_date = datetime.now()

        for day in range(1, duration + 1):
            current_date = start_date + timedelta(days=day - 1)
            phase = ("Land Preparation & Sowing" if day <= 30 else
                     "Vegetative Growth" if day <= 60 else
                     "Flowering & Fruiting" if day <= 90 else "Maturity & Harvest")

            prompt = f"""
You are a JSON-only machine. No explanations. Strictly output valid JSON:
{{ 
  "activities": ["activity1", "activity2"],
  "water_requirement": "...",
  "fertilizer_application": "...",
  "disease_monitoring": "...",
  "weather_consideration": "..."
}}
for Day {day} of {crop} cultivation in {location} ({acres} acres, {soil_type} soil). Phase: {phase}.
"""

            try:
                json_result = self._call_gemini(prompt)
                data = json.loads(json_result)
                # sanitize
                data = self._validate_day_plan(data, day, current_date, phase)
            except Exception as e:
                print(f"⚠️ Gemini failed for day {day}, fallback used: {e}")
                data = self._generate_fallback_day_plan(day, phase, current_date)

            plan.append(data)

        if not plan:
            plan = [self._generate_fallback_day_plan(1, "General", datetime.now())]
        return plan

    def _validate_day_plan(self, data, day, current_date, phase):
        # ensure all expected keys exist
        return {
            "day": day,
            "date": current_date.strftime('%Y-%m-%d'),
            "phase": phase,
            "activities": data.get("activities") if isinstance(data.get("activities"), list) else ["General field work"],
            "water_requirement": data.get("water_requirement", "Standard irrigation."),
            "fertilizer_application": data.get("fertilizer_application", "Follow local recommendations."),
            "disease_monitoring": data.get("disease_monitoring", "Inspect for common diseases."),
            "weather_consideration": data.get("weather_consideration", "Adjust for local weather.")
        }

    def _call_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Strictly output valid JSON. No text, no explanation.",
            prompt
        ])
        text = response.text
        print(f"📝 Raw Gemini output: {text}")

        # extract JSON object
        match = re.search(r'({[\s\S]*})', text)
        if match:
            return match.group(1)
        try:
            json.loads(text)
            return text
        except Exception:
            raise ValueError("Invalid JSON from Gemini")

    def _generate_fallback_day_plan(self, day, phase, date):
        return {
            "day": day,
            "date": date.strftime('%Y-%m-%d'),
            "phase": phase,
            "activities": ["Monitor growth", "Check moisture", "Inspect pests", "Record observations"],
            "water_requirement": "Routine irrigation.",
            "fertilizer_application": "Balanced application.",
            "disease_monitoring": "Visual check.",
            "weather_consideration": "Adjust for rain."
        }


class CropPlanningAgent:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.soil_types_for_crops = {
            "default": ["Loamy", "Sandy Loam", "Clay Loam", "Silt Loam"],
            "rice": ["Clay", "Clay Loam", "Silty Clay"],
            "wheat": ["Loamy", "Clay Loam", "Silt Loam"],
            "cotton": ["Black Cotton Soil", "Clay Loam", "Sandy Loam"],
            "sugarcane": ["Loamy", "Clay Loam", "Alluvial"]
        }

    def create_crop_plan(self, crop, duration, location, investment, soil_type=None, acres=1):
        if not soil_type:
            soil_type = self.soil_types_for_crops.get(crop.lower(), self.soil_types_for_crops["default"])[0]

        plan = self.query_processor._generate_crop_plan(crop, duration, location, investment, soil_type, acres)

        if not isinstance(plan, list) or not plan:
            plan = [self.query_processor._generate_fallback_day_plan(1, "General", datetime.now())]

        return {
            "crop": crop or "Unknown Crop",
            "location": location or "Unknown Location",
            "duration": duration or 0,
            "investment": investment or 0,
            "soil_type": soil_type or "Loamy",
            "acres": acres or 1,
            "overview": f"{duration}-day plan for {crop} in {location} on {acres} acres of {soil_type} soil.",
            "soil_preparation": f"Prepare {soil_type} soil for {crop}.",
            "irrigation": f"Routine irrigation for {crop}.",
            "fertilizer": f"Balanced fertilizer for {crop}.",
            "pest_management": f"Pest checks for {crop}.",
            "harvest": f"Harvest {crop} at maturity.",
            "plan": plan,
            "estimated_yield": self._estimate_yield(crop, investment, acres),
            "yield_roi": "20-25%"
        }

    def _estimate_yield(self, crop, investment, acres):
        base_yields = {"rice": "2.5-3 tons", "wheat": "2-2.5 tons", "cotton": "1.5-2 bales",
                       "sugarcane": "40-50 tons", "default": "2-3 tons"}
        y = base_yields.get(crop.lower(), base_yields["default"])
        return f"{y} per acre"


