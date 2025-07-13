
import React from 'react';
import { Sprout, Brain, Globe, Smartphone } from 'lucide-react';

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-green-600 via-green-500 to-emerald-600">
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-20 w-32 h-32 bg-white rounded-full animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-24 h-24 bg-white rounded-full animate-bounce"></div>
        <div className="absolute top-1/2 left-10 w-16 h-16 bg-yellow-300 rounded-full animate-ping"></div>
        <div className="absolute bottom-32 left-1/3 w-20 h-20 bg-blue-300 rounded-full animate-pulse"></div>
      </div>
      
      <div className="container mx-auto px-6 text-center text-white relative z-10">
        <div className="animate-fade-in">
          <div className="flex justify-center mb-8">
            <div className="relative">
              <Sprout className="w-24 h-24 text-yellow-300 animate-bounce" />
              <Brain className="w-12 h-12 text-white absolute -top-2 -right-2 animate-pulse" />
            </div>
          </div>
          
          <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-white via-yellow-200 to-green-200 bg-clip-text text-transparent animate-scale-in">
            AgriGuru
          </h1>
          
          <p className="text-2xl md:text-3xl mb-8 font-light opacity-90 animate-fade-in">
            Your AI-Powered Agricultural Advisor
          </p>
          
          <p className="text-lg md:text-xl mb-12 max-w-3xl mx-auto leading-relaxed opacity-80 animate-fade-in">
            Bridging the gap between cutting-edge technology and grassroots farming needs through intelligent, multilingual agricultural guidance.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center animate-fade-in">
            <button className="bg-yellow-500 text-green-900 px-8 py-4 rounded-full text-lg font-semibold hover:bg-yellow-400 transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl">
              Explore Solutions
            </button>
            <button className="border-2 border-white text-white px-8 py-4 rounded-full text-lg font-semibold hover:bg-white hover:text-green-600 transform hover:scale-105 transition-all duration-300">
              Watch Demo
            </button>
          </div>
          
          <div className="flex justify-center gap-8 mt-16 animate-fade-in">
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full">
              <Globe className="w-5 h-5 text-yellow-300" />
              <span className="text-sm">Multilingual</span>
            </div>
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full">
              <Smartphone className="w-5 h-5 text-yellow-300" />
              <span className="text-sm">Mobile First</span>
            </div>
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full">
              <Brain className="w-5 h-5 text-yellow-300" />
              <span className="text-sm">AI Powered</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
