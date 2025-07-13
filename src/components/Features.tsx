
import React from 'react';
import { Sprout, Cloud, Bug, MessageCircle, TrendingUp, FileText } from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: Sprout,
      title: "Crop Planning Assistant",
      description: "AI-powered crop selection based on soil type, weather, and market conditions",
      benefits: ["Top 3 crop recommendations", "Climate compatibility analysis", "Profitability optimization"],
      color: "from-green-400 to-emerald-500"
    },
    {
      icon: Cloud,
      title: "Weather-Adaptive Alerts",
      description: "Proactive weather-based farming recommendations and alerts",
      benefits: ["7-day weather forecasts", "Risk assessment", "Actionable farming advice"],
      color: "from-blue-400 to-cyan-500"
    },
    {
      icon: Bug,
      title: "Pest & Disease Detector",
      description: "Image-based pest and disease identification with treatment plans",
      benefits: ["CNN-based image recognition", "Instant disease identification", "Treatment recommendations"],
      color: "from-red-400 to-pink-500"
    },
    {
      icon: MessageCircle,
      title: "Voice/Chat AI Bot",
      description: "Multilingual conversational AI for easy farmer interaction",
      benefits: ["Tamil, Hindi, English support", "Voice-to-text processing", "Natural conversation"],
      color: "from-purple-400 to-violet-500"
    },
    {
      icon: TrendingUp,
      title: "Market Price Tracker",
      description: "Real-time mandi prices and optimal selling recommendations",
      benefits: ["Live market data", "Best price locations", "Profit maximization"],
      color: "from-yellow-400 to-orange-500"
    },
    {
      icon: FileText,
      title: "Govt Schemes Notifier",
      description: "Personalized government scheme recommendations and guidance",
      benefits: ["Scheme eligibility check", "Application assistance", "Simplified processes"],
      color: "from-indigo-400 to-blue-500"
    }
  ];

  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            Six Powerful Features
          </h2>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            AgriGuru provides comprehensive agricultural support through six AI-powered modules 
            designed to address every aspect of modern farming.
          </p>
        </div>
        
        <div className="grid lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group bg-gradient-to-br from-gray-50 to-white p-8 rounded-3xl border border-gray-200 hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-500 animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`bg-gradient-to-br ${feature.color} w-20 h-20 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-6 transition-all duration-300`}>
                <feature.icon className="w-10 h-10 text-white" />
              </div>
              
              <h3 className="text-2xl font-bold text-gray-800 mb-4 group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-green-600 group-hover:to-blue-600 group-hover:bg-clip-text transition-all duration-300">
                {feature.title}
              </h3>
              
              <p className="text-gray-600 leading-relaxed mb-6">{feature.description}</p>
              
              <div className="space-y-3">
                {feature.benefits.map((benefit, benefitIndex) => (
                  <div key={benefitIndex} className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">{benefit}</span>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 pt-6 border-t border-gray-200">
                <button className="w-full bg-gradient-to-r from-green-500 to-blue-500 text-white py-3 rounded-xl font-semibold hover:from-green-600 hover:to-blue-600 transform hover:scale-105 transition-all duration-300">
                  Learn More
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
