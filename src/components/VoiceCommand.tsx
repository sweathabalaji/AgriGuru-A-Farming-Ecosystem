import React, { useEffect, useState } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

interface CommandMapping {
  [key: string]: {
    pageId: string;
    description: {
      tamil: string;
      english: string;
    };
  };
}

interface VoiceCommandProps {
  setCurrentPage: (page: string) => void;
  language: string;
}

const COMMANDS: CommandMapping = {
  // Tamil Commands
  'குரு': {
    pageId: 'home',
    description: {
      tamil: 'விவசாய குரு உங்களை வரவேற்கிறது. நான் உங்களுக்கு எப்படி உதவ முடியும்?',
      english: 'Welcome to AgriGuru. How can I help you?'
    }
  },
  'முகப்பு': {
    pageId: 'home',
    description: {
      tamil: 'முகப்பு பக்கத்திற்கு வரவேற்கிறோம்.',
      english: 'Welcome to the home page.'
    }
  },
  'பயிர் உதவி': {
    pageId: 'crop-assistance',
    description: {
      tamil: 'பயிர் உதவி பக்கம். உங்கள் பயிர் வளர்ப்பிற்கான அறிவுரைகளை பெறலாம்.',
      english: 'Crop assistance page. Get advice for your crop cultivation.'
    }
  },
  'சந்தை விலை': {
    pageId: 'market-price',
    description: {
      tamil: 'சந்தை விலை பக்கம். இங்கு பயிர் விலைகள் தெரிந்துகொள்ளலாம்.',
      english: 'Market price page. Here you can check crop prices.'
    }
  },
  'நோய் கண்டறிதல்': {
    pageId: 'disease-detection',
    description: {
      tamil: 'நோய் கண்டறிதல் பக்கம். உங்கள் பயிர்களின் நோய்களை கண்டறியலாம்.',
      english: 'Disease detection page. Here you can identify crop diseases.'
    }
  },
  'வாங்க விற்க': {
    pageId: 'buy-sell',
    description: {
      tamil: 'வாங்க மற்றும் விற்க பக்கம். இங்கு விவசாய பொருட்களை வாங்கலாம் அல்லது விற்கலாம்.',
      english: 'Buy and sell page. Here you can buy or sell agricultural products.'
    }
  },
  'அரசு திட்டங்கள்': {
    pageId: 'government-schemes',
    description: {
      tamil: 'அரசு திட்டங்கள் பக்கம். விவசாயிகளுக்கான அரசு திட்டங்களை பற்றி அறியலாம்.',
      english: 'Government schemes page. Learn about government schemes for farmers.'
    }
  },
  // English Commands
  'agriguru': {
    pageId: 'home',
    description: {
      tamil: 'விவசாய குரு உங்களை வரவேற்கிறது. நான் உங்களுக்கு எப்படி உதவ முடியும்?',
      english: 'Welcome to AgriGuru. How can I help you?'
    }
  },
  'home': {
    pageId: 'home',
    description: {
      tamil: 'முகப்பு பக்கத்திற்கு வரவேற்கிறோம்.',
      english: 'Welcome to the home page.'
    }
  },
  'crop assistance': {
    pageId: 'crop-assistance',
    description: {
      tamil: 'பயிர் உதவி பக்கம். உங்கள் பயிர் வளர்ப்பிற்கான அறிவுரைகளை பெறலாம்.',
      english: 'Crop assistance page. Get advice for your crop cultivation.'
    }
  },
  'market price': {
    pageId: 'market-price',
    description: {
      tamil: 'சந்தை விலை பக்கம். இங்கு பயிர் விலைகள் தெரிந்துகொள்ளலாம்.',
      english: 'Market price page. Here you can check crop prices.'
    }
  },
  'disease detection': {
    pageId: 'disease-detection',
    description: {
      tamil: 'நோய் கண்டறிதல் பக்கம். உங்கள் பயிர்களின் நோய்களை கண்டறியலாம்.',
      english: 'Disease detection page. Here you can identify crop diseases.'
    }
  },
  'buy and sell': {
    pageId: 'buy-sell',
    description: {
      tamil: 'வாங்க மற்றும் விற்க பக்கம். இங்கு விவசாய பொருட்களை வாங்கலாம் அல்லது விற்கலாம்.',
      english: 'Buy and sell page. Here you can buy or sell agricultural products.'
    }
  },
  'government schemes': {
    pageId: 'government-schemes',
    description: {
      tamil: 'அரசு திட்டங்கள் பக்கம். விவசாயிகளுக்கான அரசு திட்டங்களை பற்றி அறியலாம்.',
      english: 'Government schemes page. Learn about government schemes for farmers.'
    }
  }
};

const VoiceCommand: React.FC<VoiceCommandProps> = ({ setCurrentPage, language }) => {
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { transcript, resetTranscript } = useSpeechRecognition();
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [tamilVoice, setTamilVoice] = useState<SpeechSynthesisVoice | null>(null);

  // Initialize speech synthesis and find Tamil voice
  useEffect(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        console.log('Available voices:', voices.map(v => `${v.name} (${v.lang})`));
        
        // Try to find a Tamil voice
        const tamilVoices = voices.filter(voice => 
          voice.lang.toLowerCase().includes('ta') || // ta-IN, ta-LK, etc.
          voice.name.toLowerCase().includes('tamil')
        );
        
        if (tamilVoices.length > 0) {
          console.log('Found Tamil voice:', tamilVoices[0].name);
          setTamilVoice(tamilVoices[0]);
        } else {
          console.log('No Tamil voice found, will use default voice');
        }
      };

      window.speechSynthesis.onvoiceschanged = loadVoices;
      loadVoices();
    }
  }, []);

  const speak = async (text: string) => {
    try {
      setIsLoading(true);
      setIsSpeaking(true);

      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      
      if (language === 'tamil') {
        // Configure for Tamil speech
        utterance.lang = 'ta-IN';
        utterance.rate = 0.9;
        utterance.pitch = 1;
        
        // Use Tamil voice if available
        if (tamilVoice) {
          utterance.voice = tamilVoice;
        }
      } else {
        // Configure for English speech
        utterance.lang = 'en-US';
        utterance.rate = 0.9;
      }

      // Add event listeners
      utterance.onend = () => {
        console.log('Speech finished');
        setIsSpeaking(false);
        setIsLoading(false);
      };

      utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        setIsSpeaking(false);
        setIsLoading(false);
      };

      // Split text into sentences for better speech flow
      const sentences = text.split(/[।.!?]/).filter(Boolean);
      
      for (const sentence of sentences) {
        const sentenceUtterance = new SpeechSynthesisUtterance(sentence.trim());
        Object.assign(sentenceUtterance, {
          lang: utterance.lang,
          rate: utterance.rate,
          pitch: utterance.pitch,
          voice: utterance.voice,
          onend: () => console.log('Sentence finished:', sentence),
          onerror: (e) => console.error('Sentence error:', e)
        });
        
        window.speechSynthesis.speak(sentenceUtterance);
      }

    } catch (error) {
      console.error('Speech synthesis error:', error);
      setIsSpeaking(false);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (transcript) {
      const command = Object.keys(COMMANDS).find(cmd => 
        transcript.toLowerCase().includes(cmd.toLowerCase())
      );

      if (command) {
        const { pageId, description } = COMMANDS[command];
        setCurrentPage(pageId);
        speak(description[language === 'tamil' ? 'tamil' : 'english']);
        resetTranscript();
        setIsListening(false);
      }
    }
  }, [transcript, setCurrentPage, language]);

  const startListening = () => {
    setIsListening(true);
    SpeechRecognition.startListening({ 
      continuous: true,
      language: language === 'tamil' ? 'ta-IN' : 'en-US'
    });
  };

  const stopListening = () => {
    setIsListening(false);
    SpeechRecognition.stopListening();
    resetTranscript();
  };

  if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
    return null;
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={isListening ? stopListening : startListening}
        className="rounded-full p-4 shadow-lg flex items-center gap-2 bg-green-600 text-white hover:bg-green-700 transition-colors"
        style={{ minWidth: '44px', minHeight: '44px' }}
        disabled={isLoading || isSpeaking}
      >
        <span className="material-icons">
          {isListening ? 'mic' : 'mic_none'}
        </span>
        <span className="hidden md:inline">
          {isSpeaking ? (language === 'tamil' ? 'பேசுகிறது...' : 'Speaking...') :
           isListening ? (language === 'tamil' ? 'கேட்டுக்கொண்டிருக்கிறது...' : 'Listening...') :
           (language === 'tamil' ? 'தமிழில் பேச' : 'Speak')}
        </span>
      </button>
      {isListening && transcript && (
        <div className="mt-2 p-2 bg-white rounded shadow text-sm">
          {transcript}
        </div>
      )}
    </div>
  );
};

export default VoiceCommand; 