
import React from 'react';
import { Trophy, Users, Lightbulb, Target } from 'lucide-react';

const Team = () => {
  return (
    <section className="py-20 bg-gradient-to-br from-green-50 to-blue-50">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            Team Spammers
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Competing in the Infosys Global Hackathon, we're passionate about revolutionizing 
            agriculture through innovative AI solutions.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          <div className="group bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-300 animate-fade-in">
            <div className="bg-gradient-to-br from-purple-500 to-pink-500 w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Trophy className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-4">Innovation</h3>
            <p className="text-gray-600">Cutting-edge AI solutions for real-world farming challenges</p>
          </div>
          
          <div className="group bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-300 animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <div className="bg-gradient-to-br from-blue-500 to-cyan-500 w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Users className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-4">Collaboration</h3>
            <p className="text-gray-600">Diverse expertise coming together for maximum impact</p>
          </div>
          
          <div className="group bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-300 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="bg-gradient-to-br from-green-500 to-teal-500 w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Lightbulb className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-4">Creativity</h3>
            <p className="text-gray-600">Novel approaches to solve complex agricultural problems</p>
          </div>
          
          <div className="group bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transform hover:-translate-y-3 transition-all duration-300 animate-fade-in" style={{ animationDelay: '0.3s' }}>
            <div className="bg-gradient-to-br from-orange-500 to-red-500 w-16 h-16 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Target className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-4">Impact</h3>
            <p className="text-gray-600">Focused on creating meaningful change for farmers</p>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-3xl p-12 text-white text-center animate-fade-in">
          <h3 className="text-3xl font-bold mb-6">Our Mission</h3>
          <p className="text-xl leading-relaxed max-w-4xl mx-auto mb-8">
            To democratize agricultural technology and empower every farmer with AI-driven insights, 
            breaking down barriers of language, literacy, and technology access.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">Infosys Global Hackathon</span>
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">Team Spammers</span>
            <span className="bg-white/20 backdrop-blur-md px-6 py-3 rounded-full text-lg font-semibold">AI for Agriculture</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Team;
