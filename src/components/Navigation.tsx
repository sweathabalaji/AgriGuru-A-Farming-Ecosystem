
import React from 'react';
import { Home, Sprout, TrendingUp, Bug, ShoppingCart, FileText, Cloud } from 'lucide-react';
import { Switch } from "@/components/ui/switch";

interface NavigationProps {
  currentPage: string;
  setCurrentPage: (page: string) => void;
  language: 'english' | 'tamil';
  setLanguage: (language: 'english' | 'tamil') => void;
}

const Navigation = ({ currentPage, setCurrentPage, language, setLanguage }: NavigationProps) => {
  const translations = {
    english: {
      brand: "AgriGuru",
      home: "Home",
      weatherAlerts: "Weather Alerts",
      cropAssistance: "Crop Assistance",
      diseaseDetection: "Disease Detection",
      governmentSchemes: "Govt Schemes",
      language: "தமிழ்"
    },
    tamil: {
      brand: "விவசாய குரு",
      home: "முகப்பு",
      weatherAlerts: "வானிலை எச்சரிக்கைகள்",
      cropAssistance: "பயிர் உதவி",
      diseaseDetection: "நோய் கண்டறிதல்",
      governmentSchemes: "அரசு திட்டங்கள்",
      language: "English"
    }
  };

  const t = translations[language as keyof typeof translations];

  const menuItems = [
    { id: 'home', label: t.home, icon: Home },
    { id: 'weather-alerts', label: t.weatherAlerts, icon: Cloud },
    { id: 'crop-assistance', label: t.cropAssistance, icon: Sprout },
    { id: 'disease-detection', label: t.diseaseDetection, icon: Bug },
    { id: 'government-schemes', label: t.governmentSchemes, icon: FileText },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md border-b border-green-200 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center">
              <Sprout className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-green-600 to-emerald-700 bg-clip-text text-transparent">
              {t.brand}
            </span>
          </div>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center gap-1.5">
            {menuItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition-all duration-300 hover:bg-green-100 hover:scale-105 ${
                  currentPage === item.id
                    ? 'bg-green-500 text-white shadow-md'
                    : 'text-gray-700 hover:text-green-600'
                }`}
              >
                <item.icon className="w-4.5 h-4.5" />
                <span className="text-[15px] font-medium whitespace-nowrap">{item.label}</span>
              </button>
            ))}
          </div>

          {/* Language Toggle */}
          <div className="flex items-center gap-2">
            <span className="text-[15px] font-medium text-gray-600">
              {language === 'english' ? 'EN' : 'தமிழ்'}
            </span>
            <Switch
              checked={language === 'tamil'}
              onCheckedChange={(checked) => setLanguage(checked ? 'tamil' : 'english')}
              className="data-[state=checked]:bg-green-500"
            />
            <span className="text-[15px] font-medium text-green-600">
              {t.language}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
