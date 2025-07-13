
import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface VideoSliderProps {
  language: string;
}

const VideoSlider = ({ language }: VideoSliderProps) => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const translations = {
    english: {
      title: "Empowering Farmers with AI Technology",
      subtitle: "Join thousands of farmers who trust AgriGuru for smarter farming decisions",
      videoTitles: [
        "Smart Crop Planning",
        "Weather-Based Alerts",
        "Disease Detection"
      ]
    },
    tamil: {
      title: "AI தொழில்நுட்பத்துடன் விவசாயிகளை மேம்படுத்துதல்",
      subtitle: "புத்திசாலித்தனமான விவசாய முடிவுகளுக்காக AgriGuru-ஐ நம்பும் ஆயிரக்கணக்கான விவசாயிகளுடன் இணையுங்கள்",
      videoTitles: [
        "புத்திசாலி பயிர் திட்டமிடல்",
        "வானிலை அடிப்படையிலான எச்சரிக்கைகள்",
        "நோய் கண்டறிதல்"
      ]
    }
  };

  const t = translations[language as keyof typeof translations];

  const slides = [
    {
      id: 1,
      title: t.videoTitles[0],
      image: "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      description: language === 'english' ? "AI-powered crop recommendations" : "AI இயங்கும் பயிர் பரிந்துரைகள்"
    },
    {
      id: 2,
      title: t.videoTitles[1],
      image: "https://images.unsplash.com/photo-1464207687429-7505649dae38?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      description: language === 'english' ? "Real-time weather monitoring" : "நிகழ்நேர வானிலை கண்காணிப்பு"
    },
    {
      id: 3,
      title: t.videoTitles[2],
      image: "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      description: language === 'english' ? "Early disease detection" : "முன்கூட்டியே நோய் கண்டறிதல்"
    }
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 4000);
    return () => clearInterval(timer);
  }, [slides.length]);

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  return (
    <section className="py-12 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-800 mb-6 leading-tight">
            {t.title}
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            {t.subtitle}
          </p>
        </div>

        <div className="relative max-w-6xl mx-auto">
          <div className="relative h-96 md:h-[500px] overflow-hidden rounded-3xl shadow-2xl">
            {slides.map((slide, index) => (
              <div
                key={slide.id}
                className={`absolute inset-0 transition-all duration-700 transform ${
                  index === currentSlide
                    ? 'translate-x-0 opacity-100 scale-100'
                    : index < currentSlide
                    ? '-translate-x-full opacity-0 scale-95'
                    : 'translate-x-full opacity-0 scale-95'
                }`}
              >
                <div className="relative h-full">
                  <img
                    src={slide.image}
                    alt={slide.title}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
                  <div className="absolute bottom-8 left-8 right-8 text-white">
                    <h3 className="text-3xl md:text-4xl font-bold mb-2">{slide.title}</h3>
                    <p className="text-lg opacity-90">{slide.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Navigation Buttons */}
          <button
            onClick={prevSlide}
            className="absolute left-4 top-1/2 -translate-y-1/2 bg-white/90 hover:bg-white p-3 rounded-full shadow-lg transition-all duration-300 hover:scale-110"
          >
            <ChevronLeft className="w-6 h-6 text-gray-800" />
          </button>
          <button
            onClick={nextSlide}
            className="absolute right-4 top-1/2 -translate-y-1/2 bg-white/90 hover:bg-white p-3 rounded-full shadow-lg transition-all duration-300 hover:scale-110"
          >
            <ChevronRight className="w-6 h-6 text-gray-800" />
          </button>

          {/* Dots Indicator */}
          <div className="flex justify-center space-x-2 mt-6">
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentSlide(index)}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  index === currentSlide
                    ? 'bg-green-500 scale-125'
                    : 'bg-gray-300 hover:bg-gray-400'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default VideoSlider;
