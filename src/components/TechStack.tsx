
import React from 'react';
import { Code2, Database, Globe, Smartphone, Cpu, Zap } from 'lucide-react';

const TechStack = () => {
  const technologies = [
    {
      category: "AI/ML Frameworks",
      icon: Cpu,
      techs: ["LangChain", "HuggingFace", "OpenAI GPT", "Whisper", "mT5", "BLOOMZ"],
      color: "from-purple-500 to-pink-500"
    },
    {
      category: "Computer Vision",
      icon: Code2,
      techs: ["EfficientNet", "VGG", "CNN Models", "Image Classification"],
      color: "from-blue-500 to-cyan-500"
    },
    {
      category: "APIs & Data",
      icon: Database,
      techs: ["OpenWeatherMap", "Agmarknet", "FAISS Vector Store", "PDF Processing"],
      color: "from-green-500 to-teal-500"
    },
    {
      category: "Voice & Language",
      icon: Globe,
      techs: ["Coqui TTS", "Bark", "Multilingual NLP", "Speech Processing"],
      color: "from-orange-500 to-red-500"
    },
    {
      category: "Frontend & Mobile",
      icon: Smartphone,
      techs: ["React.js", "Tailwind CSS", "Responsive Design", "Mobile-First"],
      color: "from-indigo-500 to-purple-500"
    },
    {
      category: "Automation",
      icon: Zap,
      techs: ["n8n Workflows", "Microservices", "Cloud Hosting", "Auto Triggers"],
      color: "from-yellow-500 to-orange-500"
    }
  ];

  return (
    <section className="py-20 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Cutting-Edge Technology Stack
          </h2>
          <p className="text-xl opacity-90 max-w-4xl mx-auto leading-relaxed">
            Built with the latest AI frameworks and technologies to deliver 
            powerful, scalable, and accessible agricultural solutions.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {technologies.map((tech, index) => (
            <div
              key={index}
              className="group bg-white/10 backdrop-blur-md p-8 rounded-3xl border border-white/20 hover:bg-white/20 hover:border-white/40 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`bg-gradient-to-br ${tech.color} w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <tech.icon className="w-8 h-8 text-white" />
              </div>
              
              <h3 className="text-xl font-bold mb-4">{tech.category}</h3>
              
              <div className="space-y-2">
                {tech.techs.map((techName, techIndex) => (
                  <div key={techIndex} className="flex items-center gap-3">
                    <div className="w-1.5 h-1.5 bg-white rounded-full opacity-70"></div>
                    <span className="text-sm opacity-90">{techName}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        
        <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-3xl p-12 text-center animate-fade-in">
          <h3 className="text-3xl font-bold mb-6">Architecture Highlights</h3>
          <div className="grid md:grid-cols-3 gap-8 text-lg">
            <div>
              <div className="text-2xl font-bold text-yellow-300 mb-2">Modular Design</div>
              <p className="opacity-90">Independently deployable modules for scalability</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-300 mb-2">No IoT Required</div>
              <p className="opacity-90">Software-only solution accessible on any device</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-300 mb-2">AI Agents</div>
              <p className="opacity-90">Autonomous data processing and decision making</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TechStack;
