import json
import requests
from typing import Dict, Any, List

class TamilTranslator:
    """Utility class for translating crop planning content to Tamil"""
    
    def __init__(self):
        self.cache = {}  # Simple cache to avoid repeated translations
        self.common_translations = self._get_common_translations()
    
    def _get_common_translations(self) -> Dict[str, str]:
        """Get common translations for agricultural terms"""
        return {
            # Major Crops
            "rice": "அரிசி",
            "wheat": "கோதுமை",
            "maize": "மக்காச்சோளம்",
            "cotton": "பருத்தி",
            "cotton (lint)": "பருத்தி (நூல்)",
            
            # Cereals
            "bajra": "கம்பு",
            "jowar": "சோளம்",
            "ragi": "கேழ்வரகு",
            "sugarcane": "கரும்பு",
            
            # Pulses
            "arhar/tur": "துவரம் பருப்பு",
            "moong (green gram)": "பாசிப்பயறு",
            "urad": "உளுந்து",
            "gram": "கடலை",
            "pulses total": "பருப்பு வகைகள் மொத்தம்",
            "horse-gram": "கொள்ளு",
            
            # Oilseeds
            "groundnut": "நிலக்கடலை",
            "sunflower": "சூரியகாந்தி",
            "sesamum": "எள்ளு",
            "rapeseed & mustard": "கடுகு & கடுகு விதை",
            "soyabean": "சோயா பீன்",
            "niger seed": "கருங்கடலை",
            
            # Commercial Crops
            "tobacco": "புகையிலை",
            "jute": "சணல்",
            "mesta": "மெஸ்டா",
            
            # Spices
            "turmeric": "மஞ்சள்",
            "black pepper": "மிளகு",
            "cardamom": "ஏலக்காய்",
            "dry chillies": "உலர் மிளகாய்",
            "dry ginger": "உலர் இஞ்சி",
            
            # Fruits
            "banana": "வாழைப்பழம்",
            "mango": "மாம்பழம்",
            "orange": "ஆரஞ்சு",
            "grapes": "திராட்சை",
            "papaya": "பப்பாளி",
            "pineapple": "அன்னாசி",
            "apple": "ஆப்பிள்",
            "peach": "பீச்",
            "pear": "பேரிக்காய்",
            "plums": "பிளம்",
            "litchi": "லிச்சி",
            "jack fruit": "பலா",
            "citrus fruit": "எலுமிச்சை பழங்கள்",
            "other citrus fruit": "பிற எலுமிச்சை பழங்கள்",
            "other fresh fruits": "பிற புதிய பழங்கள்",
            "pome fruit": "ஆப்பிள் வகை பழங்கள்",
            "pome granet": "மாதுளை",
            
            # Vegetables
            "onion": "வெங்காயம்",
            "tomato": "தக்காளி",
            "potato": "உருளைக்கிழங்கு",
            "brinjal": "கத்தரிக்காய்",
            "cauliflower": "காலிஃபிளவர்",
            "cabbage": "முட்டைக்கோஸ்",
            "carrot": "கேரட்",
            "garlic": "பூண்டு",
            "beet root": "பீட்ரூட்",
            "bhindi": "வெண்டைக்காய்",
            "cucumber": "வெள்ளரி",
            "pump kin": "பூசணிக்காய்",
            "water melon": "தர்பூசணி",
            "turnip": "டர்னிப்",
            "bitter gourd": "பாகற்காய்",
            "bottle gourd": "சுரைக்காய்",
            "snak guard": "புடலங்காய்",
            "ribed guard": "ரிப்பட் கார்டு",
            "redish": "முள்ளங்கி",
            "ash gourd": "நீர்மட்டி",
            "beans & mutter (vegetable)": "பீன்ஸ் & மட்டர் (காய்கறி)",
            "lab-lab": "அவரைக்காய்",
            "drum stick": "முருங்கைக்காய்",
            "other vegetables": "பிற காய்கறிகள்",
            "yam": "சேனைக்கிழங்கு",
            
            # Others
            "coconut": "தேங்காய்",
            "cashewnut": "முந்திரிப்பருப்பு",
            "castor seed": "ஆமணக்கு விதை",
            "coriander": "கொத்தமல்லி",
            "sweet potato": "சர்க்கரைவள்ளிக்கிழங்கு",
            "tapioca": "மரவள்ளிக்கிழங்கு",
            "small millets": "சிறு தானியங்கள்",
            "sannhamp": "சண்ணம்பு",
            "korra": "கொர்ரா",
            "samai": "சாமை",
            "guar seed": "குவார் விதை",
            "varagu": "வரகு",
            "arecanut": "பாக்கு",
            "other cereals & millets": "பிற தானியங்கள் & சிறுதானியங்கள்",
            "other kharif pulses": "பிற காரீஃப் பருப்புகள்",
            "other foodgrain": "பிற உணவு தானியங்கள்",
            "total foodgrain": "மொத்த உணவு தானியங்கள்",
            
            # Activities
            "Land preparation": "நிலம் தயாரித்தல்",
            "Seed selection": "விதை தேர்வு",
            "Nursery preparation": "நாற்று தயாரித்தல்",
            "Transplanting": "நாற்று நடுதல்",
            "Water management": "நீர் மேலாண்மை",
            "Fertilizer application": "உரம் பயன்படுத்துதல்",
            "Weeding": "களை நீக்குதல்",
            "Pest control": "பூச்சி கட்டுப்பாடு",
            "Irrigation": "நீர்ப்பாசனம்",
            "Harvesting": "அறுவடை",
            "Threshing": "கதிரடித்தல்",
            "Storage preparation": "சேமிப்பு தயாரித்தல்",
            "Soil testing": "மண் சோதனை",
            "Seed treatment": "விதை சிகிச்சை",
            "Sowing": "விதைப்பு",
            "Disease control": "நோய் கட்டுப்பாடு",
            "Irrigation management": "நீர்ப்பாசன மேலாண்மை",
            "Drying": "உலர்த்துதல்",
            "Storage": "சேமிப்பு",
            "Picking": "பறித்தல்",
            "Ginning": "பருத்தி பிரித்தல்",
            "Pruning": "கிளை வெட்டுதல்",
            
            # Weather
            "Monitor rainfall patterns": "மழை முறைகளை கண்காணிக்கவும்",
            "Check temperature conditions": "வெப்பநிலை நிலைமைகளை சரிபார்க்கவும்",
            "Ensure proper irrigation": "சரியான நீர்ப்பாசனத்தை உறுதிசெய்யவும்",
            "Protect from extreme weather": "கடுமையான வானிலையில் இருந்து பாதுகாக்கவும்",
            "Optimize watering schedule": "நீர்ப்பாசன அட்டவணையை உகந்தமயமாக்கவும்",
            
            # Tips
            "Start early in the morning": "காலையில் முன்கூட்டியே தொடங்குங்கள்",
            "Keep detailed records": "விரிவான பதிவுகளை வைத்திருங்கள்",
            "Monitor soil moisture levels": "மண் ஈரப்பத அளவுகளை கண்காணிக்கவும்",
            "Prepare backup plans": "காப்பு திட்டங்களை தயாரிக்கவும்",
            "Consult with local agricultural experts": "உள்ளூர் வேளாண்மை நிபுணர்களை ஆலோசிக்கவும்",
            
            # UI Elements
            "Today's Activities": "இன்றைய செயல்பாடுகள்",
            "Weather Considerations": "வானிலை கருத்துகள்",
            "Estimated Daily Cost": "மதிப்பிடப்பட்ட தினசரி செலவு",
            "Pro Tips": "நிபுணர் குறிப்புகள்",
            "AI Insights": "செயற்கை நுண்ணறிவு நுண்ணறிவுகள்",
            "Key Recommendations": "முக்கிய பரிந்துரைகள்",
            "Risk Assessment": "ஆபத்து மதிப்பீடு",
            "Previous Day": "முந்தைய நாள்",
            "Next Day": "அடுத்த நாள்",
            "Day": "நாள்",
            "of": "இல்",
            "Back to Predictor": "கணிப்பாளருக்கு திரும்பு",
            "Create New Plan": "புதிய திட்டத்தை உருவாக்கு",
            "Project initiation and planning": "திட்ட தொடக்கம் மற்றும் திட்டமிடல்",
            "Final inspection and documentation": "இறுதி ஆய்வு மற்றும் ஆவணப்படுத்தல்",
            
            # Risk Assessment
            "Diversify crops": "பயிர்களை பல்வகைப்படுத்தவும்",
            "Insurance coverage": "காப்பீட்டு பாதுகாப்பு",
            "Market monitoring": "சந்தை கண்காணிப்பு",
            "Proper pest management": "சரியான பூச்சி மேலாண்மை",
            
            # AI Insights
            "Based on agricultural best practices": "வேளாண்மை சிறந்த நடைமுறைகளின் அடிப்படையில்",
            "ensure proper soil preparation": "சரியான மண் தயாரிப்பை உறுதிசெய்யவும்",
            "timely irrigation": "சரியான நேரத்தில் நீர்ப்பாசனம்",
            "pest management": "பூச்சி மேலாண்மை",
            "optimal yield": "உகந்த மகசூல்",
            "Maintain proper soil moisture levels": "சரியான மண் ஈரப்பத அளவுகளை பராமரிக்கவும்",
            "Monitor for pests and diseases regularly": "பூச்சிகள் மற்றும் நோய்களை தவறாமல் கண்காணிக்கவும்",
            "Apply fertilizers at recommended intervals": "பரிந்துரைக்கப்பட்ட இடைவெளிகளில் உரங்களை பயன்படுத்தவும்",
            "Follow crop rotation practices": "பயிர் சுழற்சி நடைமுறைகளை பின்பற்றவும்"
        }
    
    def translate_text(self, text: str) -> str:
        """Translate a single text to Tamil"""
        if not text or text.strip() == "":
            return text
        
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Check common translations first
        text_lower = text.lower()
        for english, tamil in self.common_translations.items():
            if english.lower() in text_lower:
                text = text.replace(english, tamil)
        
        # Cache the result
        self.cache[text] = text
        return text
    
    def translate_list(self, items: List[str]) -> List[str]:
        """Translate a list of strings to Tamil"""
        return [self.translate_text(item) for item in items]
    
    def translate_crop_plan(self, crop_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Translate the entire crop plan to Tamil"""
        translated_plan = crop_plan.copy()
        
        # Translate crop name
        if 'crop' in translated_plan:
            translated_plan['crop'] = self.translate_text(translated_plan['crop'])
        
        # Translate AI insights
        if 'ai_insights' in translated_plan:
            translated_plan['ai_insights'] = self.translate_text(translated_plan['ai_insights'])
        
        # Translate recommendations
        if 'recommendations' in translated_plan:
            translated_plan['recommendations'] = self.translate_list(translated_plan['recommendations'])
        
        # Translate risk assessment
        if 'risk_assessment' in translated_plan:
            risk_assessment = translated_plan['risk_assessment'].copy()
            if 'mitigation_strategies' in risk_assessment:
                risk_assessment['mitigation_strategies'] = self.translate_list(risk_assessment['mitigation_strategies'])
            translated_plan['risk_assessment'] = risk_assessment
        
        # Translate daily plan activities
        if 'plan' in translated_plan:
            translated_daily_plans = []
            for day_plan in translated_plan['plan']:
                translated_day = day_plan.copy()
                
                # Translate activities
                if 'activities' in translated_day:
                    translated_day['activities'] = self.translate_list(translated_day['activities'])
                
                # Translate weather consideration
                if 'weather_consideration' in translated_day:
                    translated_day['weather_consideration'] = self.translate_text(translated_day['weather_consideration'])
                
                translated_daily_plans.append(translated_day)
            
            translated_plan['plan'] = translated_daily_plans
        
        return translated_plan
    
    def translate_common_phrases(self) -> Dict[str, str]:
        """Translate common UI phrases to Tamil"""
        phrases = {
            "Crop Prediction": "பயிர் கணிப்பு",
            "AI Crop Planning": "செயற்கை நுண்ணறிவு பயிர் திட்டமிடல்",
            "State": "மாநிலம்",
            "District": "மாவட்டம்",
            "Investment": "முதலீடு",
            "Duration": "கால அளவு",
            "Days": "நாட்கள்",
            "Generate Plan": "திட்டத்தை உருவாக்கு",
            "Today's Activities": "இன்றைய செயல்பாடுகள்",
            "Weather Considerations": "வானிலை கருத்துகள்",
            "Estimated Daily Cost": "மதிப்பிடப்பட்ட தினசரி செலவு",
            "Pro Tips": "நிபுணர் குறிப்புகள்",
            "AI Insights": "செயற்கை நுண்ணறிவு நுண்ணறிவுகள்",
            "Key Recommendations": "முக்கிய பரிந்துரைகள்",
            "Risk Assessment": "ஆபத்து மதிப்பீடு",
            "Previous Day": "முந்தைய நாள்",
            "Next Day": "அடுத்த நாள்",
            "Day": "நாள்",
            "of": "இல்",
            "Back to Predictor": "கணிப்பாளருக்கு திரும்பு",
            "Create New Plan": "புதிய திட்டத்தை உருவாக்கு",
            "Start early in the morning": "காலையில் முன்கூட்டியே தொடங்குங்கள்",
            "Keep detailed records": "விரிவான பதிவுகளை வைத்திருங்கள்",
            "Monitor soil moisture": "மண் ஈரப்பதத்தை கண்காணிக்கவும்",
            "Prepare backup plans": "காப்பு திட்டங்களை தயாரிக்கவும்",
            "Consult experts": "நிபுணர்களை ஆலோசிக்கவும்"
        }
        
        return phrases 