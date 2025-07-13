
import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, AlertCircle, Loader2, Mic, MicOff, Square } from 'lucide-react';
import axios from 'axios';

interface ChatbotSectionProps {
  language: 'english' | 'tamil';
}

interface Message {
  type: 'user' | 'bot' | 'error' | 'system';
  content: string;
}

const ChatbotSection = ({ language }: ChatbotSectionProps) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const translations = {
    english: {
      title: "Ask AgriGuru - Your Gemma-Powered Farming Assistant",
      subtitle: "Get instant answers to your farming questions in Tamil or English, powered by Google's Gemma AI",
      placeholder: "Type your farming question here...",
      examples: [
        "What crops are best suited for sandy soil?",
        "How to protect crops during heavy rain?",
        "Best practices for organic pest control?",
        "When is the right time to harvest rice?",
        "How to improve soil fertility naturally?"
      ],
      sendButton: "Send",
      examplesTitle: "Try asking:",
      thinking: "AgriGuru is thinking...",
      errorMessage: "Sorry, there was an error. Please try again.",
      connectionError: "Could not connect to AgriGuru. Please check if the service is running.",
      retryButton: "Retry Connection",
      connecting: "Connecting to AgriGuru...",
      startListening: "Start Voice Input",
      stopListening: "Stop Voice Input",
      listeningMessage: "Listening... Speak now",
      voiceError: "Voice input is not supported in your browser",
      stopGenerating: "Stop Generating",
      responseStopped: "Response generation stopped.",
    },
    tamil: {
      title: "AgriGuru-விடம் கேளுங்கள் - உங்கள் Gemma-ஆல் இயக்கப்படும் விவசாய உதவியாளர்",
      subtitle: "Google's Gemma AI மூலம் தமிழ் அல்லது ஆங்கிலத்தில் உங்கள் விவசாய கேள்விகளுக்கு உடனடி பதில்கள் பெறுங்கள்",
      placeholder: "உங்கள் விவசாய கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
      examples: [
        "மணல் மண்ணுக்கு ஏற்ற பயிர்கள் எவை?",
        "கனமழையின் போது பயிர்களை எவ்வாறு பாதுகாப்பது?",
        "இயற்கை பூச்சி கட்டுப்பாட்டிற்கான சிறந்த முறைகள்?",
        "நெல் அறுவடை செய்ய சரியான நேரம் எப்போது?",
        "மண் வளத்தை இயற்கையாக எப்படி மேம்படுத்துவது?"
      ],
      sendButton: "அனுப்பு",
      examplesTitle: "கேட்க முயற்சிக்கவும்:",
      thinking: "AgriGuru சிந்திக்கிறது...",
      errorMessage: "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.",
      connectionError: "AgriGuru-வுடன் இணைக்க முடியவில்லை. சேவை இயங்குகிறதா என்பதை சரிபார்க்கவும்.",
      retryButton: "மீண்டும் இணைக்க முயற்சிக்கவும்",
      connecting: "AgriGuru-வுடன் இணைக்கிறது...",
      startListening: "குரல் உள்ளீடு தொடங்கவும்",
      stopListening: "குரல் உள்ளீட்டை நிறுத்தவும்",
      listeningMessage: "கேட்கிறது... இப்போது பேசவும்",
      voiceError: "உங்கள் உலாவியில் குரல் உள்ளீடு ஆதரிக்கப்படவில்லை",
      stopGenerating: "பதில் உருவாக்கத்தை நிறுத்து",
      responseStopped: "பதில் உருவாக்கம் நிறுத்தப்பட்டது.",
    }
  };

  const t = translations[language];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    checkConnection();
  }, []);

  useEffect(() => {
    // Initialize speech recognition
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = language === 'tamil' ? 'ta-IN' : 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');
        setMessage(transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setMessages(prev => [...prev, { 
          type: 'error', 
          content: t.voiceError 
        }]);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [language]);

  const toggleVoiceInput = () => {
    if (!recognitionRef.current) {
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: t.voiceError 
      }]);
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      setMessage('');
      recognitionRef.current.start();
      setIsListening(true);
      setMessages(prev => [...prev, { 
        type: 'system', 
        content: t.listeningMessage 
      }]);
    }
  };

  const stopResponseGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      setMessages(prev => [...prev, { 
        type: 'system', 
        content: t.responseStopped 
      }]);
    }
  };

  const API_BASE_URL = 'http://localhost:8000';
  const API_ENDPOINTS = {
    health: ['/health', '/api/health'],
    chat: ['/chat', '/api/chat']
  };

  const FRONTEND_URLS = ['http://localhost:8080', 'http://localhost:5173', 'http://localhost:3000'];

  const checkConnection = async () => {
    try {
      setIsConnected(null); // Set to connecting state
      let connected = false;
      
      // Try both health endpoint formats
      for (const endpoint of API_ENDPOINTS.health) {
        try {
          const response = await axios.get(`${API_BASE_URL}${endpoint}`);
          if (response.data.status === 'healthy') {
            connected = true;
            break;
          }
        } catch (error) {
          console.log(`Failed to connect to ${endpoint}:`, error);
          // Continue trying other endpoints
        }
      }
      
      setIsConnected(connected);
    } catch (error) {
      console.error('Connection check failed:', error);
      setIsConnected(false);
    }
  };

  const handleSendMessage = async () => {
    if (message.trim() && !isLoading) {
      try {
        setIsLoading(true);
        const userMessage = message.trim();
        setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
        setMessage('');

        // Add a placeholder for the bot's response
        setMessages(prev => [...prev, { type: 'bot', content: '' }]);

        // Create new AbortController for this request
        abortControllerRef.current = new AbortController();

        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: userMessage,
            language,
            voice_input: false,
            voice_output: false
          }),
          signal: abortControllerRef.current.signal
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('ReadableStream not supported');
        }

        const decoder = new TextDecoder();
        let accumulatedResponse = '';
        let lastUpdateTime = Date.now();
        const updateInterval = 100; // Update UI every 100ms

        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            accumulatedResponse += chunk;

            // Update UI periodically to avoid too frequent updates
            const now = Date.now();
            if (now - lastUpdateTime >= updateInterval) {
              setMessages(prev => {
                const newMessages = [...prev];
                newMessages[newMessages.length - 1] = {
                  type: 'bot',
                  content: accumulatedResponse
                };
                return newMessages;
              });
              lastUpdateTime = now;
            }
          }
        } catch (readError) {
          if (readError.name === 'AbortError') {
            // Handle aborted request
            reader.cancel();
            return;
          }
          throw readError;
        }

        // Final update
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            type: 'bot',
            content: accumulatedResponse
          };
          return newMessages;
        });

      } catch (error) {
        if (error.name === 'AbortError') {
          return; // Don't show error for aborted requests
        }

        console.error('Error sending message:', error);
        let errorMessage = t.errorMessage;
        
        if (error instanceof Error) {
          if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            setIsConnected(false);
            errorMessage = t.connectionError;
          } else if (error.message.includes('503')) {
            errorMessage = language === 'english' 
              ? "Gemma AI service is not available. Please make sure Ollama is running and the model is installed."
              : "Gemma AI சேவை கிடைக்கவில்லை. Ollama இயங்குகிறதா மற்றும் மாதிரி நிறுவப்பட்டுள்ளதா என்பதை உறுதிப்படுத்தவும்.";
          } else if (error.message.includes('504')) {
            errorMessage = language === 'english'
              ? "The AI model is taking too long to respond. Please try again in a few moments."
              : "AI மாதிரி பதிலளிக்க அதிக நேரம் எடுக்கிறது. சில நிமிடங்களில் மீண்டும் முயற்சிக்கவும்.";
          } else {
            errorMessage = error.message;
          }
        }
        
        // Remove the empty bot message if it exists
        setMessages(prev => {
          if (prev[prev.length - 1].type === 'bot' && prev[prev.length - 1].content === '') {
            return [...prev.slice(0, -1), { type: 'error', content: errorMessage }];
          }
          return [...prev, { type: 'error', content: errorMessage }];
        });
      } finally {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    }
  };

  if (isConnected === null) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-green-500" />
          <p className="text-gray-600">{t.connecting}</p>
        </div>
      </div>
    );
  }

  if (isConnected === false) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-red-600">{t.connectionError}</h3>
          <button
            onClick={checkConnection}
            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
          >
            {t.retryButton}
          </button>
        </div>
      </div>
    );
  }

  return (
    <section className="py-16 px-6 bg-white/50">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12 animate-fade-in">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center shadow-lg animate-bounce-slow">
              <MessageCircle className="w-10 h-10 text-white" />
            </div>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
            {t.title}
          </h2>
          <p className="text-xl text-gray-600">
            {t.subtitle}
          </p>
        </div>

        {/* Chat Interface */}
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden border border-green-100">
          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-6 bg-gradient-to-b from-green-50/30 to-white">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-20">
                <MessageCircle className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p className="text-lg">
                  {language === 'english' ? 'Start a conversation with AgriGuru!' : 'AgriGuru-உடன் உரையாடலைத் தொடங்குங்கள்!'}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                        msg.type === 'user'
                          ? 'bg-green-500 text-white'
                          : msg.type === 'error'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-gray-100 text-gray-800'
                      } shadow-md animate-fade-in`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 text-gray-600 px-4 py-3 rounded-2xl shadow-md animate-pulse flex items-center space-x-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>{t.thinking}</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-6 bg-white border-t border-gray-100">
            <div className="flex items-center gap-2 mt-4">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder={t.placeholder}
                className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                disabled={isLoading}
                />
                <button
                  onClick={toggleVoiceInput}
                className={`p-2 rounded-lg ${isListening ? 'bg-red-500' : 'bg-blue-500'} text-white`}
                disabled={isLoading}
              >
                {isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
              </button>
              {isLoading ? (
                <button
                  onClick={stopResponseGeneration}
                  className="bg-red-500 text-white p-2 rounded-lg hover:bg-red-600"
                  title={t.stopGenerating}
                >
                  <Square className="w-6 h-6" />
                </button>
              ) : (
              <button
                onClick={handleSendMessage}
                  disabled={!message.trim() || isLoading}
                  className="bg-green-500 text-white p-2 rounded-lg disabled:opacity-50"
              >
                  <Send className="w-6 h-6" />
              </button>
              )}
            </div>

            {/* Example Questions */}
            <div className="mt-4">
              <p className="text-sm text-gray-600 mb-2 font-medium">{t.examplesTitle}</p>
              <div className="flex flex-wrap gap-2">
                {t.examples.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setMessage(example)}
                    className="text-xs bg-green-100 text-green-700 px-3 py-1.5 rounded-full hover:bg-green-200 transition-all duration-300 hover:scale-105"
                    disabled={isLoading}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ChatbotSection;
