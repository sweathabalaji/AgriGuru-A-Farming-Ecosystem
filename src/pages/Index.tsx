
import React, { useState } from 'react';
import Navigation from '../components/Navigation';
import VideoSlider from '../components/VideoSlider';
import ChatbotSection from '../components/ChatbotSection';
import CropAssistance from '../components/CropAssistance';
import DiseaseDetection from '../components/DiseaseDetection';
import GovernmentSchemes from '../components/GovernmentSchemes';
import VoiceCommand from '../components/VoiceCommand';
import WeatherAlerts from '../components/WeatherAlerts';

const Index = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [language, setLanguage] = useState<'english' | 'tamil'>('english');

  const renderContent = () => {
    switch (currentPage) {
      case 'weather-alerts':
        return <WeatherAlerts language={language} />;
      case 'crop-assistance':
        return <CropAssistance language={language} />;
      case 'disease-detection':
        return <DiseaseDetection language={language} />;
      case 'government-schemes':
        return <GovernmentSchemes language={language} />;
      default:
        return (
          <>
            <VideoSlider language={language} />
            <ChatbotSection language={language} />
          </>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-lime-25 to-emerald-50" 
         style={{ backgroundImage: "url('data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23a7f3d0' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')" }}>
      <Navigation 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage} 
        language={language} 
        setLanguage={setLanguage} 
      />
      <main className="pt-20">
        {renderContent()}
      </main>
      <VoiceCommand setCurrentPage={setCurrentPage} language={language} />
    </div>
  );
};

export default Index;
