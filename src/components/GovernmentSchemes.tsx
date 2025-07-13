
import React, { useState, useEffect } from 'react';
import { FileText, Download, CheckCircle, Clock, Loader2 } from 'lucide-react';
import axios from 'axios';

interface Scheme {
  title: string;
  description: string;
  link: string;
  date: string;
  source: string;
  tamil_title?: string;
  tamil_description?: string;
  tamil_date?: string;
  tamil_source?: string;
}

interface GovernmentSchemesProps {
  language: string;
}

const GovernmentSchemes = ({ language }: GovernmentSchemesProps) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [schemes, setSchemes] = useState<Scheme[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Function to determine scheme category based on title and description
  const getSchemeCategory = (scheme: Scheme) => {
    const text = `${scheme.title.toLowerCase()} ${scheme.description.toLowerCase()}`;
    
    if (text.includes('loan') || text.includes('credit') || text.includes('கடன்') || text.includes('card')) {
      return 'loans';
    }
    if (text.includes('insurance') || text.includes('bima') || text.includes('காப்பீடு')) {
      return 'insurance';
    }
    if (text.includes('training') || text.includes('skill') || text.includes('பயிற்சி') || text.includes('education')) {
      return 'training';
    }
    if (text.includes('subsidy') || text.includes('support') || text.includes('மானியம்') || text.includes('fund') || text.includes('assistance')) {
      return 'subsidies';
    }
    return 'all';
  };

  useEffect(() => {
    const fetchSchemes = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get('http://localhost:8000/api/schemes');
        setSchemes(response.data.schemes);
      } catch (err) {
        console.error('Error fetching schemes:', err);
        setError(language === 'tamil' 
          ? 'திட்டங்களை பெறுவதில் பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.'
          : 'Error fetching schemes. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchSchemes();
  }, [language]);

  const translations = {
    english: {
      title: "Government Schemes & Subsidies",
      subtitle: "Discover and apply for government schemes tailored for farmers",
      categories: {
        all: "All Schemes",
        subsidies: "Subsidies",
        loans: "Loans",
        insurance: "Insurance",
        training: "Training"
      },
      source: "Source",
      date: "Published",
      viewDetails: "View Details",
      loading: "Loading schemes...",
      error: "Error loading schemes. Please try again.",
      noSchemes: "No schemes available at the moment.",
      noSchemesCategory: "No {category} schemes available."
    },
    tamil: {
      title: "அரசு திட்டங்கள் மற்றும் மானியங்கள்",
      subtitle: "விவசாயிகளுக்கு ஏற்ற அரசு திட்டங்களைக் கண்டறிந்து விண்ணப்பிக்கவும்",
      categories: {
        all: "அனைத்து திட்டங்கள்",
        subsidies: "மானியங்கள்",
        loans: "கடன்கள்",
        insurance: "காப்பீடு",
        training: "பயிற்சி"
      },
      source: "மூலம்",
      date: "வெளியிடப்பட்டது",
      viewDetails: "விவரங்களைக் காண",
      loading: "திட்டங்கள் ஏற்றப்படுகிறது...",
      error: "திட்டங்களை ஏற்றுவதில் பிழை. மீண்டும் முயற்சிக்கவும்.",
      noSchemes: "தற்போது திட்டங்கள் எதுவும் இல்லை.",
      noSchemesCategory: "{category} திட்டங்கள் எதுவும் இல்லை."
    }
  };

  // Function to get category name in Tamil
  const getCategoryNameInTamil = (category: string) => {
    const categoryTranslations: { [key: string]: string } = {
      'loans': 'கடன்',
      'insurance': 'காப்பீடு',
      'training': 'பயிற்சி',
      'subsidies': 'மானியம்'
    };
    return categoryTranslations[category] || category;
  };

  const t = translations[language as keyof typeof translations];

  // Filter schemes based on selected category
  const filteredSchemes = selectedCategory === 'all' 
    ? schemes
    : schemes.filter(scheme => getSchemeCategory(scheme) === selectedCategory);

  return (
    <section className="py-16 px-6 min-h-screen">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12 animate-fade-in">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg">
              <FileText className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
            {t.title}
          </h1>
          <p className="text-xl text-gray-600">
            {t.subtitle}
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {Object.entries(t.categories).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSelectedCategory(key)}
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                selectedCategory === key
                  ? 'bg-indigo-500 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 border border-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-12">
            <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mb-4" />
            <p className="text-gray-600">{t.loading}</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center py-12">
            <p className="text-red-500">{error}</p>
          </div>
        )}

        {/* No Schemes State */}
        {!loading && !error && filteredSchemes.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-600">
              {selectedCategory === 'all' 
                ? t.noSchemes 
                : t.noSchemesCategory.replace(
                    '{category}', 
                    language === 'tamil' 
                      ? getCategoryNameInTamil(selectedCategory)
                      : selectedCategory
                  )
              }
            </p>
          </div>
        )}

        {/* Schemes Grid */}
        {!loading && !error && filteredSchemes.length > 0 && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSchemes.map((scheme, index) => (
            <div key={index} className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-all duration-300 hover:-translate-y-2">
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-100 to-blue-100 rounded-xl flex items-center justify-center">
                  <FileText className="w-6 h-6 text-indigo-600" />
                </div>
                  <div className="text-sm font-medium px-3 py-1 rounded-full" style={{
                    backgroundColor: getSchemeCategory(scheme) === 'subsidies' ? '#E8F5E9' :
                                   getSchemeCategory(scheme) === 'loans' ? '#E3F2FD' :
                                   getSchemeCategory(scheme) === 'insurance' ? '#FFF3E0' :
                                   getSchemeCategory(scheme) === 'training' ? '#F3E5F5' : '#ECEFF1',
                    color: getSchemeCategory(scheme) === 'subsidies' ? '#2E7D32' :
                           getSchemeCategory(scheme) === 'loans' ? '#1565C0' :
                           getSchemeCategory(scheme) === 'insurance' ? '#E65100' :
                           getSchemeCategory(scheme) === 'training' ? '#7B1FA2' : '#455A64'
                  }}>
                    {language === 'tamil' 
                      ? getCategoryNameInTamil(getSchemeCategory(scheme))
                      : getSchemeCategory(scheme).charAt(0).toUpperCase() + getSchemeCategory(scheme).slice(1)}
                  </div>
                </div>
                
                <h3 className="text-lg font-semibold text-gray-800 mb-3">
                  {language === 'tamil' && scheme.tamil_title ? scheme.tamil_title : scheme.title}
                </h3>
                <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                  {language === 'tamil' && scheme.tamil_description ? scheme.tamil_description : scheme.description}
                </p>
                
                <div className="space-y-2 mb-4">
                  <p className="text-sm text-gray-500">
                    <span className="font-medium">{t.source}:</span> {' '}
                    {language === 'tamil' && scheme.tamil_source ? scheme.tamil_source : scheme.source}
                  </p>
                  <p className="text-sm text-gray-500">
                    <span className="font-medium">{t.date}:</span> {' '}
                    {language === 'tamil' && scheme.tamil_date ? scheme.tamil_date : scheme.date}
                  </p>
              </div>
              
              <div className="flex space-x-3">
                  <a 
                    href={scheme.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 bg-gradient-to-r from-indigo-500 to-blue-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:from-indigo-600 hover:to-blue-700 transition-all duration-300 text-center"
                  >
                    {t.viewDetails}
                  </a>
                </div>
            </div>
          ))}
        </div>
        )}
      </div>
    </section>
  );
};

export default GovernmentSchemes;
