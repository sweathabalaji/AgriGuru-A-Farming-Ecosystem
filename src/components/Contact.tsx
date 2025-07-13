
import React from 'react';
import { Mail, Github, Linkedin, Globe, Phone, MapPin } from 'lucide-react';

const Contact = () => {
  return (
    <section className="py-20 bg-gradient-to-br from-gray-900 via-green-900 to-blue-900 text-white">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Get in Touch
          </h2>
          <p className="text-xl opacity-90 max-w-3xl mx-auto leading-relaxed">
            Interested in AgriGuru? Want to collaborate or learn more about our solution? 
            We'd love to hear from you!
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-12 mb-16">
          <div className="animate-fade-in">
            <h3 className="text-2xl font-bold mb-6">Let's Collaborate</h3>
            <p className="text-lg opacity-90 mb-8 leading-relaxed">
              Whether you're a farmer, agricultural expert, investor, or technology partner, 
              we're excited to explore how AgriGuru can make a difference together.
            </p>
            
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="bg-green-500 w-12 h-12 rounded-full flex items-center justify-center">
                  <Mail className="w-6 h-6" />
                </div>
                <div>
                  <div className="font-semibold">Email</div>
                  <div className="opacity-80">team.spammers@agriguru.ai</div>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="bg-blue-500 w-12 h-12 rounded-full flex items-center justify-center">
                  <Phone className="w-6 h-6" />
                </div>
                <div>
                  <div className="font-semibold">Phone</div>
                  <div className="opacity-80">+91 98765 43210</div>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="bg-purple-500 w-12 h-12 rounded-full flex items-center justify-center">
                  <MapPin className="w-6 h-6" />
                </div>
                <div>
                  <div className="font-semibold">Location</div>
                  <div className="opacity-80">India</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white/10 backdrop-blur-md p-8 rounded-3xl border border-white/20 animate-fade-in">
            <h3 className="text-2xl font-bold mb-6">Quick Message</h3>
            <form className="space-y-6">
              <div>
                <input
                  type="text"
                  placeholder="Your Name"
                  className="w-full px-4 py-3 bg-white/10 border border-white/30 rounded-xl text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>
              <div>
                <input
                  type="email"
                  placeholder="Your Email"
                  className="w-full px-4 py-3 bg-white/10 border border-white/30 rounded-xl text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>
              <div>
                <textarea
                  rows={4}
                  placeholder="Your Message"
                  className="w-full px-4 py-3 bg-white/10 border border-white/30 rounded-xl text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
                />
              </div>
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-green-500 to-blue-500 py-3 rounded-xl font-semibold hover:from-green-600 hover:to-blue-600 transform hover:scale-105 transition-all duration-300"
              >
                Send Message
              </button>
            </form>
          </div>
        </div>
        
        <div className="text-center animate-fade-in">
          <h3 className="text-2xl font-bold mb-6">Follow Our Journey</h3>
          <div className="flex justify-center gap-6">
            <a href="#" className="bg-white/10 backdrop-blur-md w-14 h-14 rounded-full flex items-center justify-center hover:bg-white/20 transform hover:scale-110 transition-all duration-300">
              <Github className="w-6 h-6" />
            </a>
            <a href="#" className="bg-white/10 backdrop-blur-md w-14 h-14 rounded-full flex items-center justify-center hover:bg-white/20 transform hover:scale-110 transition-all duration-300">
              <Linkedin className="w-6 h-6" />
            </a>
            <a href="#" className="bg-white/10 backdrop-blur-md w-14 h-14 rounded-full flex items-center justify-center hover:bg-white/20 transform hover:scale-110 transition-all duration-300">
              <Globe className="w-6 h-6" />
            </a>
          </div>
          
          <div className="mt-12 pt-8 border-t border-white/20">
            <p className="opacity-70">
              © 2024 AgriGuru by Team Spammers. Built for Infosys Global Hackathon.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
