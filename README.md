# 🌾 AgriGuru: Modular AI-powered Farming Platform

**AgriGuru** is an advanced AI-driven agricultural advisory platform that empowers farmers—especially in Tamil Nadu—with personalized, explainable, and multilingual support through voice and chat interfaces.

---

## 🚀 Key Features

- ✅ **Plant Disease Detection** (with Explainable AI)
- ✅ **Crop Planning Assistant**
- ✅ **Weather-Adaptive Alerts**
- ✅ **Multilingual Voice/Chatbot** (Tamil & English)
- ✅ **Government Schemes Notifier**
- ✅ **Offline-Capable Local LLM (Gemma via Ollama)**

---

## 🎯 Objectives

- Empower farmers using AI-driven, localized advice.
- Support non-literate users via **Tamil voice input**.
- Enable smart planning for crops & timely disease treatment.
- Alert users based on **localized real-time weather data**.
- Help farmers access **government schemes** easily.

---

## 🏗️ Tech Stack & Architecture

| Layer         | Technology                            | Purpose                                              |
|---------------|----------------------------------------|------------------------------------------------------|
| 🖥️ Frontend    | React, TypeScript, Tailwind CSS        | Multilingual UI                                      |
| 🎤 Voice       | Web Speech API                         | Tamil/English speech-to-text                         |
| 🔊 TTS (Planned)| TBD                                   | Text-to-speech (future update)                      |
| 🌐 Backend     | FastAPI (Python)                       | API and routing logic                                |
| 🤖 AI Model    | Gemma 3B via Ollama                    | Local LLM for agri chat                              |
| 🧠 ML/AI       | TensorFlow, Keras, Scikit-learn        | Disease detection & crop recommendation              |
| 📡 APIs        | Weatherbit, RSS Feeds                  | Weather alerts, Govt schemes                         |
| 🔐 Security    | CORS, .env configs                     | Secure API communication                             |

---

## DEMO VIDEO
🎥 **[Watch Demo Video](https://github.com/sweathabalaji/AgriGuru-A-Farming-Ecosystem/blob/56925b68b706b32b453a18785b6d6ed574a4b020/demovid.mp4)**  

## 🌿 Modules

### 1️⃣ Plant Disease Detection

- 📊 **Dataset:** PlantVillage (54,305 images, 38 diseases)  
- 🧬 **Model:** ResNet50  
- 🎯 **Accuracy:** 98%  
- ⚡ **Inference Time:** ~100ms  
- 🖼️ **Explainable AI:** Grad-CAM heatmaps  
- 📄 **Report Output:** PDF (disease name, confidence, symptoms, remedies)  

**Pipeline:**  
_Image ➔ Preprocessing ➔ ResNet ➔ Prediction ➔ Grad-CAM ➔ PDF Report_

---

### 2️⃣ Crop Planning Assistant

- 🎯 **ML Model:** RandomForest (soil, region, investment input)
- 🤖 **AI Planner:** Google Gemini via Flask API
- 📝 **Output:** 
  - Best crops
  - Daily plans (Land prep → Growth → Flowering → Harvest)
- 🈂️ **Tamil Translations** + caching for fast response

**Pipeline:**  
_Excel ➔ Label Encode ➔ RandomForest ➔ Gemini Plan ➔ Tamil Output_

---

### 3️⃣ Weather-Adaptive Alerts

- 🌦️ Uses **Weatherbit API**
- ⚠️ Triggers alerts for:
  - Heat > 35°C
  - Rain > 10mm
  - UV > 8
  - Wind > 40 km/h
- 📍 Personalized using farmer’s **latitude/longitude**
- 🈂️ Bilingual contextual advice: irrigation, pesticide, harvest tips

---

### 4️⃣ Multilingual Voice & Chatbot (Tamil/English)

- 💬 Built on **Gemma 3B via Ollama** with custom system prompts
- 🔁 Real-time **streamed** responses
- 🗣️ **Supports voice & text inputs**

**Flow:**  
_User ➔ React ➔ FastAPI ➔ Ollama ➔ Gemma ➔ Streaming ➔ UI Update_

---

### 5️⃣ Govt Schemes Notifier

- 📡 Aggregates RSS feeds from:
  - The Hindu (Agri)
  - AgriFarming
  - Down To Earth
- 🏷️ Categories:
  - Subsidies, Loans, Insurance, Training
- 🈂️ Tamil & English output for reach and clarity

---

## ⚙️ API Endpoints

| Endpoint             | Method | Purpose                        |
|----------------------|--------|--------------------------------|
| `/health`            | GET    | System health check            |
| `/api/chat`          | POST   | Chat with AI (text/voice)      |
| `/api/weather-alerts`| GET    | Get weather alerts             |
| `/api/schemes`       | GET    | Latest government schemes      |

---


### ⚙️ Local Setup

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
```bash
ollama run gemma3
```
```bash
npm install && npm run dev
```
