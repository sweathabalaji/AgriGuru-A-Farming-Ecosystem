
import React from 'react';
import { Lightbulb, Target, Zap, Shield } from 'lucide-react';

const Solutions = () => {
  const solutions = [
    {
      icon: Lightbulb,
      title: "AI-Powered Intelligence",
      description: "Advanced AI and natural language processing for smart decision making",
      color: "from-yellow-400 to-orange-500"
    },
    {
      icon: Target,
      title: "Personalized Guidance",
      description: "Tailored recommendations based on specific farm conditions and needs",
      color: "from-blue-400 to-purple-500"
    },
    {
      icon: Zap,
      title: "Real-time Insights",
      description: "Live data integration for weather, market prices, and crop conditions",
      color: "from-green-400 to-teal-500"
    },
    {
      icon: Shield,
      title: "Trusted Advisor",
      description: "A reliable, multilingual agricultural guru for every farmer",
      color: "from-purple-400 to-pink-500"
    }
  ];

  return (
    <section className="py-20 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            Our Solution Approach
          </h2>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            AgriGuru is a modular, AI-powered advisory platform that empowers farmers through 
            intelligent technology and accessible interfaces.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {solutions.map((solution, index) => (
            <div
              key={index}
              className="group bg-white p-8 rounded-2xl shadow-lg hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <div className={`bg-gradient-to-br ${solution.color} w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <solution.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-4">{solution.title}</h3>
              <p className="text-gray-600 leading-relaxed">{solution.description}</p>
            </div>
          ))}
        </div>
        
        <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-3xl p-12 text-white text-center animate-fade-in">
          <h3 className="text-3xl font-bold mb-6">Why AgriGuru?</h3>
          <p className="text-xl leading-relaxed max-w-4xl mx-auto mb-8">
            We bridge the gap between cutting-edge technology and grassroots farming needs, 
            making precision agriculture accessible to every farmer, regardless of their technical background.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">No IoT Required</span>
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">Multilingual Support</span>
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">Real-time Data</span>
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">Voice Interface</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Solutions;
