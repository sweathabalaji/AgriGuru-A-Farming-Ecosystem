import React, { useState } from 'react';
import { Bug, Camera, Upload, Search, Loader, FileText, Image, X } from 'lucide-react';

interface DiseaseDetectionProps {
  language: string;
}

const DiseaseDetection = ({ language }: DiseaseDetectionProps) => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showVisualization, setShowVisualization] = useState(false);

  const translations = {
    english: {
      title: "AI-Powered Disease Detection",
      subtitle: "Upload a photo of your crop to identify diseases and get treatment recommendations",
      uploadImage: "Upload Image",
      takePhoto: "Take Photo",
      analyzeImage: "Analyze Image",
      dragDrop: "Drag and drop an image here, or click to select",
      removeImage: "Remove image",
      loading: "Analyzing...",
      downloadReport: "Download PDF",
      viewVisualization: "View Image",
      closeVisualization: "Close Image",
    },
    tamil: {
      title: "AI இயங்கும் நோய் கண்டறிதல்",
      subtitle: "நோய்களைக் கண்டறிந்து சிகிச்சை பரிந்துரைகளைப் பெற உங்கள் பயிரின் புகைப்படத்தைப் பதிவேற்றவும்",
      uploadImage: "படத்தைப் பதிவேற்றவும்",
      takePhoto: "புகைப்படம் எடுக்கவும்",
      analyzeImage: "படத்தை பகுப்பாய்வு செய்யவும்",
      dragDrop: "இங்கே ஒரு படத்தை இழுத்து விடவும், அல்லது தேர்ந்தெடுக்க கிளிக் செய்யவும்",
      removeImage: "படத்தை அகற்று",
      loading: "பகுப்பாய்வு நடந்து கொண்டிருக்கிறது...",
      downloadReport: "PDF பதிவிறக்கவும்",
      viewVisualization: "படத்தைப் பார்க்கவும்",
      closeVisualization: "படத்தை மூடு",
    }
  };

  const t = translations[language as keyof typeof translations];

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviewImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setResult(null);
    setError(null);
    setShowVisualization(false);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:8080/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to get prediction');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="py-16 px-6 min-h-screen bg-gray-50">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12 animate-fade-in">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-gradient-to-br from-red-500 to-pink-600 rounded-full flex items-center justify-center shadow-lg">
              <Bug className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
            {t.title}
          </h1>
          <p className="text-xl text-gray-600">{t.subtitle}</p>
        </div>

        <div className="bg-white rounded-3xl shadow-2xl p-8 border border-red-100">
          <div className="text-center">
            <div className="border-2 border-dashed border-gray-300 rounded-2xl p-12 mb-6 hover:border-red-400 transition-colors duration-300">
              {previewImage ? (
                <div className="space-y-4">
                  <img
                    src={previewImage}
                    alt="Uploaded crop"
                    className="max-w-full max-h-64 mx-auto rounded-lg shadow-lg"
                  />
                  <button
                    onClick={() => {
                      setSelectedImage(null);
                      setPreviewImage(null);
                      setResult(null);
                      setError(null);
                      setShowVisualization(false);
                    }}
                    className="text-red-500 hover:text-red-600 underline"
                  >
                    {t.removeImage}
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="w-16 h-16 text-gray-400 mx-auto" />
                  <p className="text-gray-600">{t.dragDrop}</p>
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <label className="bg-gradient-to-r from-red-500 to-pink-600 text-white px-6 py-3 rounded-xl font-semibold hover:from-red-600 hover:to-pink-700 transform hover:scale-105 transition-all duration-300 shadow-lg cursor-pointer flex items-center space-x-2">
                <Upload className="w-5 h-5" />
                <span>{t.uploadImage}</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>

              <button className="bg-gray-500 text-white px-6 py-3 rounded-xl font-semibold hover:bg-gray-600 transform hover:scale-105 transition-all duration-300 shadow-lg flex items-center space-x-2">
                <Camera className="w-5 h-5" />
                <span>{t.takePhoto}</span>
              </button>
            </div>

            {previewImage && (
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-green-600 hover:to-emerald-700 transform hover:scale-105 transition-all duration-300 shadow-lg flex items-center space-x-2 mx-auto"
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>{t.loading}</span>
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    <span>{t.analyzeImage}</span>
                  </>
                )}
              </button>
            )}

            {error && (
              <div className="mt-6 text-red-600 font-semibold">{error}</div>
            )}

            {result && (
              <div className="mt-10 bg-gray-100 p-6 rounded-xl text-left shadow-inner border border-gray-200">
                <h3 className="text-xl font-semibold text-green-700 mb-2">
                  🌿 Disease: {result.prediction.disease_name}
                </h3>
                <p className="text-gray-700 mb-4">
                  🔍 Confidence: {result.prediction.confidence}
                </p>

                <div className="flex flex-col sm:flex-row gap-4 mb-4">
                  <a
                    href={`http://127.0.0.1:8080${result.download_urls.report}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold flex items-center space-x-2 shadow-md"
                  >
                    <FileText className="w-5 h-5" />
                    <span>🧾 {t.downloadReport}</span>
                  </a>

                  <button
                    onClick={() => setShowVisualization(!showVisualization)}
                    className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold flex items-center space-x-2 shadow-md"
                  >
                    <Image className="w-5 h-5" />
                    <span>🔥 {showVisualization ? t.closeVisualization : t.viewVisualization}</span>
                  </button>
                </div>

                {showVisualization && (
                  <div className="mt-6">
                    <img
                      src={`http://127.0.0.1:8080${result.download_urls.visualization}`}
                      alt="Disease Visualization"
                      className="w-full rounded-xl shadow-lg border border-gray-300"
                    />
                  </div>
                )}

                <div className="mt-6">
                  <strong className="block text-gray-800 mb-1">🩺 Remedies:</strong>
                  <ul className="list-disc pl-6 text-gray-600">
                    {result.disease_info.remedies?.map((r: string, idx: number) => (
                      <li key={idx}>{r}</li>
                    )) || <li>No information available</li>}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default DiseaseDetection;
