
import React from 'react';
import { AlertCircle, TrendingDown, Users, MapPin } from 'lucide-react';

const ProblemStatement = () => {
  const challenges = [
    {
      icon: AlertCircle,
      title: "Limited Access to Guidance",
      description: "Farmers lack timely, accurate, and personalized agricultural advice"
    },
    {
      icon: TrendingDown,
      title: "Generic Solutions",
      description: "Existing systems provide one-size-fits-all recommendations"
    },
    {
      icon: Users,
      title: "Language Barriers",
      description: "Technology-heavy solutions are inaccessible to average farmers"
    },
    {
      icon: MapPin,
      title: "Localization Gap",
      description: "Lack of real-time, localized advisory support"
    }
  ];

  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            The Challenge We're Solving
          </h2>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            Agriculture is the backbone of the Indian economy, yet farmers continue to face significant challenges 
            due to limited access to modern agricultural guidance and technology.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {challenges.map((challenge, index) => (
            <div
              key={index}
              className="group bg-gradient-to-br from-red-50 to-orange-50 p-8 rounded-2xl border border-red-100 hover:shadow-xl transform hover:-translate-y-2 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className="bg-red-500 w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <challenge.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-4">{challenge.title}</h3>
              <p className="text-gray-600 leading-relaxed">{challenge.description}</p>
            </div>
          ))}
        </div>
        
        <div className="mt-16 bg-gradient-to-r from-red-500 to-orange-500 rounded-3xl p-12 text-white text-center animate-fade-in">
          <h3 className="text-3xl font-bold mb-6">The Impact</h3>
          <p className="text-xl leading-relaxed max-w-4xl mx-auto">
            These challenges lead to reduced yields, poor pricing decisions, and underutilization of available resources. 
            Farmers struggle with crop selection, weather prediction, pest identification, market pricing, and accessing government schemes.
          </p>
        </div>
      </div>
    </section>
  );
};

export default ProblemStatement;
