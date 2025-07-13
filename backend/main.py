from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator, Any
import requests
import json
import os
import time
import asyncio
from dotenv import load_dotenv
import httpx
from weather_alerts import weather_alerts_handler
from govtschme import get_agricultural_schemes

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
    "english": """You are AgriGuru, an expert AI farming assistant powered by Gemma. You help farmers with all aspects of agriculture.

RESPONSE FORMAT:
1. Keep responses concise - maximum 3 paragraphs
2. Structure your answers clearly with these sections:
   - Main Answer (1-2 sentences overview)
   - Key Points (2-3 bullet points)
   - Practical Tips (1-2 actionable recommendations)
3. Use simple language and clear explanations
4. End your response clearly without trailing off

TOPICS YOU CAN HELP WITH:
- Crop selection and rotation
- Pest and disease management
- Soil health and fertilization
- Weather-based farming decisions
- Modern farming techniques
- Sustainable agriculture practices
- Market trends and pricing
- Government schemes and subsidies""",

    "tamil": """நீங்கள் AgriGuru, Gemma ஆல் இயக்கப்படும் ஒரு நிபுணர் AI விவசாய உதவியாளர்.

பதில் வடிவம்:
1. சுருக்கமான பதில்கள் - அதிகபட்சம் 3 பத்திகள்
2. உங்கள் பதில்களை இந்த பிரிவுகளுடன் தெளிவாக கட்டமைக்கவும்:
   - முக்கிய பதில் (1-2 வாக்கியங்கள் கண்ணோட்டம்)
   - முக்கிய புள்ளிகள் (2-3 புள்ளிகள்)
   - நடைமுறை குறிப்புகள் (1-2 செயல்படுத்தக்கூடிய பரிந்துரைகள்)
3. எளிய மொழியையும் தெளிவான விளக்கங்களையும் பயன்படுத்தவும்
4. உங்கள் பதிலை தெளிவாக முடிக்கவும்

நீங்கள் உதவக்கூடிய தலைப்புகள்:
- பயிர் தேர்வு மற்றும் சுழற்சி
- பூச்சி மற்றும் நோய் மேலாண்மை
- மண் ஆரோக்கியம் மற்றும் உரமிடுதல்
- வானிலை அடிப்படையிலான விவசாய முடிவுகள்
- நவீன விவசாய நுட்பங்கள்
- நிலையான விவசாய நடைமுறைகள்
- சந்தை போக்குகள் மற்றும் விலை நிர்ணயம்
- அரசு திட்டங்கள் மற்றும் மானியங்கள்"""
}

class AgriGuruAgent:
    def __init__(self):
        self.timeout = 120  # Increased timeout for slower machines
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.max_tokens = 500  # Limit response length

    async def generate_response(self, message: str, language: str) -> AsyncGenerator[str, None]:
        """Generate streaming response using Gemma via Ollama API"""
        try:
            system_prompt = SYSTEM_PROMPTS[language]
            print(f"Sending request to Ollama API...")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    'POST',
                    'http://localhost:11434/api/generate',
                    json={
                        "model": "gemma3",
                        "prompt": f"{system_prompt}\n\nUser: {message}\nAssistant: Let me help you with that!\n\n",
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_k": 40,
                            "top_p": 0.9,
                            "num_ctx": 2048,
                            "num_thread": 4,
                            "stop": ["User:", "\n\nHuman:", "\n\nAssistant:"],  # Stop tokens to prevent continuing
                            "num_predict": self.max_tokens  # Limit response length
                        }
                    }
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"Ollama API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=503,
                            detail="Failed to get response from Gemma"
                        )

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if chunk.get('response'):
                                    yield chunk['response']
                                if chunk.get('done', False):
                                    break  # Stop when response is complete
                            except json.JSONDecodeError:
                                print(f"Failed to decode JSON: {line}")
                                continue

        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            print(f"Timeout error: {str(e)}")
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The model might be busy or needs more time."
            )
        except httpx.ConnectError:
            print("Connection error: Could not connect to Ollama API")
            raise HTTPException(
                status_code=503,
                detail="Could not connect to Gemma AI service. Please make sure Ollama is running."
            )
        except Exception as e:
            print(f"Unexpected error in generate_response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )

    async def check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy and Gemma model is available"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    print("Ollama service is not responding properly")
                    return False

                data = response.json()
                if not any(model.get('name', '').startswith('gemma3') for model in data.get('models', [])):
                    print("Gemma model is not available")
                    return False

                print("Ollama service and Gemma model are available")
                return True
        except Exception as e:
            print(f"Error checking Ollama health: {str(e)}")
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
    allow_origins=["http://localhost:8080", "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Custom exception handler to ensure proper error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.get("/health")
@app.get("/api/health")
async def health_check() -> HealthResponse:
    """Check the health status of the API and its dependencies"""
    try:
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

@app.get("/api/schemes")
async def get_schemes():
    """Get latest government schemes"""
    try:
        schemes = get_agricultural_schemes()
        return {"schemes": schemes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weather-alerts")
async def get_weather_alerts(lat: float, lon: float):
    """Get weather alerts for a location"""
    return await weather_alerts_handler(lat, lon)

@app.post("/chat")
@app.post("/api/chat")
async def chat(message: ChatMessage) -> StreamingResponse:
    """Send a message to AgriGuru and get a streaming response"""
    if not message.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    return StreamingResponse(
        agent.generate_response(message.content.strip(), message.language),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 