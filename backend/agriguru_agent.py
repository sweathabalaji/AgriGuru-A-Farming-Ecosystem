from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class ChatMessage(BaseModel):
    content: str
    language: str
    voice_input: bool = False
    voice_output: bool = False

class ChatResponse(BaseModel):
    text_response: str
    audio_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# System prompts for different languages
SYSTEM_PROMPTS = {
    "english": """You are AgriGuru, an expert AI farming assistant powered by Gemma. You help farmers with all aspects of agriculture including:
- Crop selection and rotation
- Pest and disease management
- Soil health and fertilization
- Weather-based farming decisions
- Modern farming techniques
- Sustainable agriculture practices
- Market trends and pricing
- Government schemes and subsidies

Provide detailed, practical advice that farmers can implement. Use simple language and clear explanations. Keep responses focused and actionable.""",

    "tamil": """நீங்கள் AgriGuru, Gemma ஆல் இயக்கப்படும் ஒரு நிபுணர் AI விவசாய உதவியாளர். பின்வரும் அனைத்து விவசாய அம்சங்களிலும் விவசாயிகளுக்கு உதவுகிறீர்கள்:
- பயிர் தேர்வு மற்றும் சுழற்சி
- பூச்சி மற்றும் நோய் மேலாண்மை
- மண் ஆரோக்கியம் மற்றும் உரமிடுதல்
- வானிலை அடிப்படையிலான விவசாய முடிவுகள்
- நவீன விவசாய நுட்பங்கள்
- நிலையான விவசாய நடைமுறைகள்
- சந்தை போக்குகள் மற்றும் விலை நிர்ணயம்
- அரசு திட்டங்கள் மற்றும் மானியங்கள்

விவசாயிகள் செயல்படுத்தக்கூடிய விரிவான, நடைமுறை ஆலோசனையை வழங்கவும். எளிய மொழியையும் தெளிவான விளக்கங்களையும் பயன்படுத்தவும். பதில்கள் குறிப்பிட்ட செயல்களை மையமாகக் கொண்டிருக்க வேண்டும்."""
}

class AgriGuruAgent:
    def __init__(self):
        # Initialize any required setup
        pass

    async def get_gemma_response(self, message: str, language: str) -> str:
        """Get AI response using Gemma via Ollama API"""
        try:
            # Prepare the prompt with system context
            system_prompt = SYSTEM_PROMPTS[language]
            
            # Call Ollama API with Gemma model
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": "gemma3",
                    "prompt": f"{system_prompt}\n\nUser: {message}\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_ctx": 4096
                    }
                },
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail="Failed to get response from Gemma")
                
        except requests.exceptions.ConnectionError:
            print("Connection error: Could not connect to Ollama API")
            raise HTTPException(status_code=503, detail="Gemma AI service is not available")
        except requests.exceptions.Timeout:
            print("Timeout error: Ollama API request timed out")
            raise HTTPException(status_code=504, detail="Request timed out")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

    async def check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# Initialize FastAPI app
app = FastAPI(
    title="AgriGuru API",
    description="AI-powered farming assistant API using Gemma",
    version="1.0.0"
)

# Initialize AgriGuru agent
agent = AgriGuruAgent()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000","http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the API and its dependencies"""
    try:
        # Test Ollama connection
        ollama_healthy = await agent.check_ollama_health()
        
        if ollama_healthy:
            return HealthResponse(
                status="healthy",
                message="All services are running"
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message="Ollama service is not responding"
            )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=str(e)
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Send a message to AgriGuru and get a response"""
    try:
        # Validate input
        if not message.content.strip():
            raise HTTPException(status_code=400, detail="Message content cannot be empty")
            
        # Get text input
        text_input = message.content.strip()
        
        # Get AI response
        text_response = await agent.get_gemma_response(text_input, message.language)
        
        if not text_response:
            raise HTTPException(status_code=500, detail="Empty response from AI")
            
        return ChatResponse(
            text_response=text_response,
            audio_url=None  # We'll implement voice support in the next phase
        )
    
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 