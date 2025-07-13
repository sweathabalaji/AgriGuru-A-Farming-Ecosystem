import React, { useState, useEffect } from 'react';
import { Cloud, Sun, CloudRain, Wind, Thermometer, Droplets, CloudLightning } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Alert } from '@/components/ui/alert';

interface WeatherAlertsProps {
  language: 'english' | 'tamil';
}

interface WeatherData {
  current: {
    temp: number;
    humidity: number;
    windSpeed: number;
    rainfall: number;
    condition: string;
    description: string;
  };
  forecast: Array<{
    day: string;
    high: number;
    low: number;
    condition: string;
    pop: number;
  }>;
  alerts: Array<{
    type: string;
    message: string;
  }>;
  advice: string;
}

const WEATHERBIT_API_KEY = '2def20a210ad4d459fd6c52818694218';
const LOCATION = { lat: 13.0827, lon: 80.2707 }; // Chennai, TN coordinates

const WeatherAlerts = ({ language }: WeatherAlertsProps) => {
  const [currentWeather, setCurrentWeather] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const translations = {
    english: {
      title: "Weather Alerts & Forecasts",
      subtitle: "Real-time weather updates for better farming decisions",
      temperature: "Temperature",
      humidity: "Humidity",
      windSpeed: "Wind Speed",
      rainfall: "Rainfall",
      forecast: "5-Day Forecast",
      alerts: "Weather Alerts",
      loading: "Loading weather data...",
      error: "Error loading weather data. Please try again later.",
      high: "High",
      low: "Low",
      noAlerts: "No severe weather alerts at this time",
      weatherAdvice: "Farming Weather Advice",
      kmh: "km/h",
      mm: "mm",
      precipChance: "Precipitation Chance"
    },
    tamil: {
      title: "வானிலை எச்சரிக்கைகள் & முன்னறிவிப்புகள்",
      subtitle: "சிறந்த விவசாய முடிவுகளுக்கான நேரடி வானிலை புதுப்பிப்புகள்",
      temperature: "வெப்பநிலை",
      humidity: "ஈரப்பதம்",
      windSpeed: "காற்றின் வேகம்",
      rainfall: "மழைப்பொழிவு",
      forecast: "5-நாள் முன்னறிவிப்பு",
      alerts: "வானிலை எச்சரிக்கைகள்",
      loading: "வானிலை தரவு ஏற்றப்படுகிறது...",
      error: "வானிலை தரவை ஏற்றுவதில் பிழை. பிறகு முயற்சிக்கவும்.",
      high: "அதிகம்",
      low: "குறைவு",
      noAlerts: "தற்போது கடுமையான வானிலை எச்சரிக்கைகள் எதுவும் இல்லை",
      weatherAdvice: "விவசாய வானிலை ஆலோசனை",
      kmh: "கி.மீ/மணி",
      mm: "மி.மீ",
      precipChance: "மழை பொழிவு வாய்ப்பு"
    }
  };

  const t = translations[language];

  // Function to generate weather advice based on conditions
  const generateWeatherAdvice = (data: any) => {
    const temp = data.temp;
    const humidity = data.rh;
    const windSpeed = data.wind_spd;
    const precip = data.precip;

    let advice = '';
    if (language === 'english') {
      // Rainfall conditions
      if (precip > 5) {
        advice += 'Heavy rainfall expected. • Postpone spraying of pesticides and fertilizers. • Ensure proper drainage in fields. • Cover harvested produce. • Consider delaying sowing activities. ';
      } else if (precip > 0 && precip <= 5) {
        advice += 'Light rainfall expected. • Good conditions for transplanting. • Ideal for applying fertilizers if rain is expected to stop. • Check field drainage. ';
      }

      // Temperature conditions
      if (temp > 35) {
        advice += 'High temperatures forecasted. • Increase irrigation frequency. • Apply mulching to retain soil moisture. • Provide shade for sensitive crops. • Water plants during early morning or evening. ';
      } else if (temp < 15) {
        advice += 'Cool temperatures expected. • Protect sensitive crops from frost. • Reduce irrigation frequency. • Consider using row covers. ';
      }

      // Wind conditions
      if (windSpeed > 20) {
        advice += 'Strong winds expected. • Postpone spraying activities. • Provide support to tall crops. • Protect young plants. • Delay fertilizer application. ';
      } else if (windSpeed > 10 && windSpeed <= 20) {
        advice += 'Moderate winds forecasted. • Monitor crop support structures. • Exercise caution while spraying. ';
      }

      // Humidity conditions
      if (humidity > 80) {
        advice += 'High humidity conditions. • Monitor for fungal diseases. • Ensure good air circulation in fields. • Consider preventive fungicide application. • Reduce irrigation. ';
      } else if (humidity < 30) {
        advice += 'Low humidity alert. • Increase irrigation frequency. • Consider misting for sensitive crops. • Apply mulching to conserve moisture. ';
      }
    } else {
      // Tamil translations
      if (precip > 5) {
        advice += 'கனமழை எதிர்பார்க்கப்படுகிறது. • பூச்சிக்கொல்லிகள் மற்றும் உரங்களை தெளிப்பதை ஒத்திவைக்கவும். • வயல்களில் சரியான வடிகால் அமைப்பை உறுதிசெய்யவும். • அறுவடை செய்த விளைபொருட்களை மூடி வைக்கவும். • விதைப்பு பணிகளை தாமதப்படுத்த பரிசீலிக்கவும். ';
      } else if (precip > 0 && precip <= 5) {
        advice += 'லேசான மழை எதிர்பார்க்கப்படுகிறது. • நடவு செய்வதற்கு ஏற்ற நேரம். • மழை நின்றவுடன் உரமிடுவதற்கு ஏற்ற நேரம். • வயல் வடிகால் அமைப்பை சரிபார்க்கவும். ';
      }

      if (temp > 35) {
        advice += 'அதிக வெப்பநிலை எதிர்பார்க்கப்படுகிறது. • நீர்ப்பாசன அதிர்வெண்ணை அதிகரிக்கவும். • மண் ஈரப்பதத்தை தக்கவைக்க மல்ச் பயன்படுத்தவும். • உணர்திறன் பயிர்களுக்கு நிழல் வழங்கவும். • காலை அல்லது மாலை நேரத்தில் நீர் பாய்ச்சவும். ';
      } else if (temp < 15) {
        advice += 'குளிர் வெப்பநிலை எதிர்பார்க்கப்படுகிறது. • உணர்திறன் பயிர்களை பனியில் இருந்து பாதுகாக்கவும். • நீர்ப்பாசன அதிர்வெண்ணை குறைக்கவும். • வரிசை உறைகளைப் பயன்படுத்த பரிசீலிக்கவும். ';
      }

      if (windSpeed > 20) {
        advice += 'பலத்த காற்று எதிர்பார்க்கப்படுகிறது. • தெளிப்பு நடவடிக்கைகளை ஒத்திவைக்கவும். • உயரமான பயிர்களுக்கு ஆதரவு வழங்கவும். • இளம் செடிகளை பாதுகாக்கவும். • உரமிடுவதை தாமதப்படுத்தவும். ';
      } else if (windSpeed > 10 && windSpeed <= 20) {
        advice += 'மிதமான காற்று எதிர்பார்க்கப்படுகிறது. • பயிர் ஆதரவு கட்டமைப்புகளை கண்காணிக்கவும். • தெளிக்கும்போது கவனமாக இருக்கவும். ';
      }

      if (humidity > 80) {
        advice += 'அதிக ஈரப்பதம் நிலைமைகள். • பூஞ்சை நோய்களை கண்காணிக்கவும். • வயல்களில் நல்ல காற்றோட்டத்தை உறுதிசெய்யவும். • தடுப்பு பூஞ்சைக்கொல்லி பயன்பாட்டை பரிசீலிக்கவும். • நீர்ப்பாசனத்தை குறைக்கவும். ';
      } else if (humidity < 30) {
        advice += 'குறைந்த ஈரப்பதம் எச்சரிக்கை. • நீர்ப்பாசன அதிர்வெண்ணை அதிகரிக்கவும். • உணர்திறன் பயிர்களுக்கு மூடுபனி பரிசீலிக்கவும். • ஈரப்பதத்தை பாதுகாக்க மல்ச் பயன்படுத்தவும். ';
      }
    }
    
    return advice || (language === 'english' ? 
      'Weather conditions are suitable for normal farming activities. • Continue regular irrigation. • Proceed with planned fertilizer application. • Maintain normal crop monitoring. • Good time for general field maintenance.' :
      'வானிலை நிலைமைகள் சாதாரண விவசாய நடவடிக்கைகளுக்கு ஏற்றவை. • வழக்கமான நீர்ப்பாசனத்தை தொடரவும். • திட்டமிட்ட உர பயன்பாட்டை தொடரவும். • வழக்கமான பயிர் கண்காணிப்பை பராமரிக்கவும். • பொது வயல் பராமரிப்புக்கு நல்ல நேரம்.');
  };

  // Convert Weatherbit icon code to our condition type
  const mapWeatherCode = (code: string): string => {
    const codeNum = parseInt(code);
    if (codeNum < 300) return 'storm'; // Thunderstorm
    if (codeNum < 500) return 'rain'; // Drizzle
    if (codeNum < 600) return 'rain'; // Rain
    if (codeNum < 700) return 'snow'; // Snow
    if (codeNum < 800) return 'cloudy'; // Atmosphere
    if (codeNum === 800) return 'sunny'; // Clear
    return 'cloudy'; // Clouds
  };

  const getWeatherIcon = (condition: string) => {
    switch (condition) {
      case 'sunny':
        return <Sun className="w-6 h-6 text-yellow-500" />;
      case 'cloudy':
        return <Cloud className="w-6 h-6 text-gray-500" />;
      case 'rain':
        return <CloudRain className="w-6 h-6 text-blue-500" />;
      case 'storm':
        return <CloudLightning className="w-6 h-6 text-purple-500" />;
      case 'snow':
        return <CloudRain className="w-6 h-6 text-blue-200" />;
      default:
        return <Cloud className="w-6 h-6 text-gray-500" />;
    }
  };

  useEffect(() => {
    const fetchWeatherData = async () => {
      setLoading(true);
      setError(null);
      try {
        console.log('Fetching current weather data...');
        // Fetch current weather
        const currentUrl = `https://api.weatherbit.io/v2.0/current?lat=${LOCATION.lat}&lon=${LOCATION.lon}&key=${WEATHERBIT_API_KEY}&lang=${language === 'tamil' ? 'ta' : 'en'}`;
        console.log('Current weather URL:', currentUrl);
        
        const currentResponse = await fetch(currentUrl);
        
        if (!currentResponse.ok) {
          const errorText = await currentResponse.text();
          console.error('Current weather error response:', errorText);
          throw new Error(`Failed to fetch current weather: ${currentResponse.status} ${errorText}`);
        }
        
        const currentData = await currentResponse.json();
        console.log('Current weather data:', currentData);
        
        console.log('Fetching forecast data...');
        // Fetch forecast
        const forecastUrl = `https://api.weatherbit.io/v2.0/forecast/daily?lat=${LOCATION.lat}&lon=${LOCATION.lon}&key=${WEATHERBIT_API_KEY}&days=5&lang=${language === 'tamil' ? 'ta' : 'en'}`;
        console.log('Forecast URL:', forecastUrl);
        
        const forecastResponse = await fetch(forecastUrl);
        
        if (!forecastResponse.ok) {
          const errorText = await forecastResponse.text();
          console.error('Forecast error response:', errorText);
          throw new Error(`Failed to fetch forecast: ${forecastResponse.status} ${errorText}`);
        }
        
        const forecastData = await forecastResponse.json();
        console.log('Forecast data:', forecastData);

        if (!currentData.data || !currentData.data[0]) {
          throw new Error('Invalid current weather data format');
        }

        if (!forecastData.data || !Array.isArray(forecastData.data)) {
          throw new Error('Invalid forecast data format');
        }

        // Process current weather data
        const current = currentData.data[0];
        const processedData: WeatherData = {
          current: {
            temp: Math.round(current.temp),
            humidity: current.rh,
            windSpeed: Math.round(current.wind_spd * 3.6), // Convert m/s to km/h
            rainfall: current.precip,
            condition: mapWeatherCode(current.weather.code),
            description: current.weather.description
          },
          forecast: forecastData.data.slice(0, 5).map((day: any) => ({
            day: new Date(day.datetime).toLocaleDateString(language === 'tamil' ? 'ta-IN' : 'en-US', { weekday: 'short' }),
            high: Math.round(day.max_temp),
            low: Math.round(day.min_temp),
            condition: mapWeatherCode(day.weather.code),
            pop: day.pop // Probability of precipitation
          })),
          alerts: [],
          advice: generateWeatherAdvice(current)
        };

        // Add alerts based on conditions
        if (current.precip > 5) {
          processedData.alerts.push({
            type: 'rain',
            message: language === 'english'
              ? `Heavy rainfall of ${current.precip}mm expected. Take necessary precautions.`
              : `${current.precip}மிமீ கனமழை எதிர்பார்க்கப்படுகிறது. தேவையான முன்னெச்சரிக்கை நடவடிக்கைகளை எடுக்கவும்.`
          });
        }
        if (current.wind_spd * 3.6 > 20) {
          processedData.alerts.push({
            type: 'wind',
            message: language === 'english'
              ? `Strong winds of ${Math.round(current.wind_spd * 3.6)}km/h expected.`
              : `${Math.round(current.wind_spd * 3.6)}கிமீ/மணி வேகத்தில் பலத்த காற்று எதிர்பார்க்கப்படுகிறது.`
          });
        }

        console.log('Processed weather data:', processedData);
        setCurrentWeather(processedData);
      } catch (err) {
        console.error('Error fetching weather data:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchWeatherData();
    // Set up auto-refresh every 30 minutes
    const interval = setInterval(fetchWeatherData, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [language]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500" />
        <span className="ml-3 text-gray-600">{t.loading}</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Alert variant="destructive" className="w-full max-w-md">
          <CloudLightning className="h-4 w-4 mr-2" />
          <p className="text-red-800">{error}</p>
        </Alert>
      </div>
    );
  }

  if (!currentWeather) {
    return null;
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">{t.title}</h1>
        <p className="text-gray-600">{t.subtitle}</p>
      </div>

      {/* Current Weather */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">{t.temperature}</p>
              <p className="text-3xl font-bold text-gray-800">{currentWeather.current.temp}°C</p>
            </div>
            <Thermometer className="w-8 h-8 text-red-500" />
          </div>
        </Card>

        <Card className="p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">{t.humidity}</p>
              <p className="text-3xl font-bold text-gray-800">{currentWeather.current.humidity}%</p>
            </div>
            <Droplets className="w-8 h-8 text-blue-500" />
          </div>
        </Card>

        <Card className="p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">{t.windSpeed}</p>
              <p className="text-3xl font-bold text-gray-800">{currentWeather.current.windSpeed} {t.kmh}</p>
            </div>
            <Wind className="w-8 h-8 text-green-500" />
          </div>
        </Card>

        <Card className="p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">{t.rainfall}</p>
              <p className="text-3xl font-bold text-gray-800">{currentWeather.current.rainfall} {t.mm}</p>
            </div>
            <CloudRain className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
      </div>

      {/* Forecast */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">{t.forecast}</h2>
        <div className="grid grid-cols-5 gap-4">
          {currentWeather.forecast.map((day, index) => (
            <div key={index} className="text-center p-4 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
              <p className="font-medium text-gray-600">{day.day}</p>
              {getWeatherIcon(day.condition)}
              <div className="mt-2">
                <p className="text-sm font-medium text-gray-800">{day.high}°<span className="text-xs text-gray-500">/{day.low}°</span></p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Alerts */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">{t.alerts}</h2>
        {currentWeather.alerts.length > 0 ? (
          <div className="space-y-4">
            {currentWeather.alerts.map((alert, index) => (
              <Alert key={index} className="bg-red-50 border-red-200">
                <div className="flex items-center gap-2">
                  <CloudLightning className="w-5 h-5 text-red-500" />
                  <p className="text-red-800">{alert.message}</p>
                </div>
              </Alert>
            ))}
          </div>
        ) : (
          <p className="text-gray-600">{t.noAlerts}</p>
        )}
      </Card>

      {/* Weather Advice */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">{t.weatherAdvice}</h2>
        <p className="text-gray-700">{currentWeather.advice}</p>
      </Card>
    </div>
  );
};

export default WeatherAlerts; 