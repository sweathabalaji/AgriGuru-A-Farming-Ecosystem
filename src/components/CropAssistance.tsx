import React, { useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

type DayPlan = {
  day: number;
  date: string;
  phase: string;
  activities: string[];
  water_requirement: string;
  fertilizer_application: string;
  disease_monitoring: string;
  weather_consideration: string;
};

type CropPlanResponse = {
  crop: string;
  overview?: string;
  estimated_yield?: string;
  yield_roi?: string;
  plan: DayPlan[];
};

const CropAssistance: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"predictor" | "planner">("predictor");
  const [formData, setFormData] = useState({
    state: "",
    district: "",
    investment: "50000",
    acres: "1",
    soilType: "Loamy",
    duration: "30"
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictedCrop, setPredictedCrop] = useState<string | null>(null);
  const [cropPlan, setCropPlan] = useState<CropPlanResponse | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const selectDuration = (days: string) => {
    setFormData({ ...formData, duration: days });
  };

  const handlePredict = async () => {
    setError(null);
    setPredictedCrop(null);
    setLoading(true);

    try {
      const payload = {
        state: formData.state,
        district: formData.district,
        investment: formData.investment,
        soil_type: formData.soilType
      };

      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log("🌾 Predict Response:", data);

      if (data.success) {
        setPredictedCrop(data.crop);
      } else {
        setError(data.error || "Failed to predict crop.");
      }
    } catch (err) {
      console.error("Predict Error:", err);
      setError("Something went wrong. Please try again.");
    }
    setLoading(false);
  };

  const handlePlanSubmit = async () => {
    setError(null);
    setCropPlan(null);
    setLoading(true);

    try {
      const payload = {
        state: formData.state,
        district: formData.district,
        investment: Number(formData.investment),
        duration: Number(formData.duration),
        acres: Number(formData.acres),
        soil_type: formData.soilType
      };

      const response = await fetch("http://localhost:5000/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log("🚀 Plan API Response:", data);

      if (data.success) {
        // Defensive parse: fill empty fields with defaults
        setCropPlan({
          crop: data.crop || "Unknown",
          overview: data.overview || "No overview available.",
          estimated_yield: data.estimated_yield || "N/A",
          yield_roi: data.yield_roi || "N/A",
          plan: Array.isArray(data.plan) ? data.plan : []
        });
      } else {
        setError(data.error || "Failed to fetch crop plan.");
      }
    } catch (err) {
      console.error("Plan Error:", err);
      setError("Something went wrong. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-green-500 to-emerald-600 flex justify-center items-center p-6">
      <div className="w-full max-w-3xl bg-white p-8 rounded-3xl shadow-2xl transition-all duration-500 hover:scale-[1.02]">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-green-800 mb-3">🌿 Agri AI Assistant</h1>
          <p className="text-gray-600 font-medium">Predict crops and get AI-powered farming plans in Tamil & English.</p>
        </div>

        {/* Tabs */}
        <div className="flex mb-8 rounded-lg overflow-hidden border-2 border-green-500">
          <button
            className={`flex-1 py-3 font-semibold transition ${activeTab === "predictor" ? "bg-green-500 text-white" : "bg-gray-50 text-green-800 hover:bg-green-100"}`}
            onClick={() => setActiveTab("predictor")}
          >
            🌱 Crop Predictor
          </button>
          <button
            className={`flex-1 py-3 font-semibold transition ${activeTab === "planner" ? "bg-green-500 text-white" : "bg-gray-50 text-green-800 hover:bg-green-100"}`}
            onClick={() => setActiveTab("planner")}
          >
            📝 AI Planner
          </button>
        </div>

        {/* Predictor */}
        {activeTab === "predictor" && (
          <div className="space-y-6">
            <Label>State:</Label>
            <Input name="state" value={formData.state} onChange={handleChange} placeholder="Enter state" />

            <Label>District:</Label>
            <Input name="district" value={formData.district} onChange={handleChange} placeholder="Enter district" />

            <Label>Investment (₹):</Label>
            <Input name="investment" type="number" value={formData.investment} onChange={handleChange} />

            <Button
              onClick={handlePredict}
              disabled={loading}
              className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-xl font-semibold mt-4 transition"
            >
              {loading ? "Predicting..." : "🌾 Predict Crop"}
            </Button>

            {error && <p className="text-center text-red-500">{error}</p>}
            {predictedCrop && (
              <div className="text-center mt-6 p-5 bg-green-100 rounded-xl border-2 border-green-400 shadow animate-pulse">
                <h3 className="text-xl font-bold text-green-800 mb-1">🎯 Recommended Crop</h3>
                <p className="text-lg font-semibold">{predictedCrop}</p>
              </div>
            )}
          </div>
        )}

        {/* Planner */}
        {activeTab === "planner" && !cropPlan && (
          <div className="space-y-6">
            <Label>State:</Label>
            <Input name="state" value={formData.state} onChange={handleChange} placeholder="Enter state" />

            <Label>District:</Label>
            <Input name="district" value={formData.district} onChange={handleChange} placeholder="Enter district" />

            <Label>Investment (₹):</Label>
            <Input name="investment" type="number" value={formData.investment} onChange={handleChange} />

            <Label>Land Size (Acres):</Label>
            <Input name="acres" type="number" value={formData.acres} onChange={handleChange} />

            <Label>Soil Type:</Label>
            <select
              name="soilType"
              value={formData.soilType}
              onChange={handleChange}
              className="w-full p-4 border-2 border-gray-300 rounded-xl focus:border-green-500"
            >
              <option value="">Auto-detect</option>
              <option value="Loamy">Loamy</option>
              <option value="Clayey">Clayey</option>
              <option value="Sandy">Sandy</option>
              <option value="Alluvial">Alluvial</option>
              <option value="Laterite">Laterite</option>
              <option value="Black">Black</option>
              <option value="Red">Red</option>
              <option value="Peaty">Peaty</option>
              <option value="Chalky">Chalky</option>
            </select>

            <Label>Planning Duration:</Label>
            <div className="grid grid-cols-3 gap-4">
              {["30", "60", "90"].map((d) => (
                <div
                  key={d}
                  className={`p-4 text-center border-2 rounded-xl cursor-pointer transition ${
                    formData.duration === d ? "border-green-600 bg-green-50" : "border-gray-300 hover:border-green-400"
                  }`}
                  onClick={() => selectDuration(d)}
                >
                  <h3 className="text-2xl font-bold">{d}</h3>
                  <p>Days</p>
                </div>
              ))}
            </div>

            <Button
              onClick={handlePlanSubmit}
              disabled={loading}
              className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-xl font-semibold mt-4 transition"
            >
              {loading ? "Generating..." : "🚀 Generate AI Crop Plan"}
            </Button>

            {error && <p className="text-center text-red-500">{error}</p>}
          </div>
        )}

        {/* Render the crop plan if available */}
        {cropPlan && (
          <>
            {console.log("🌾 Final Crop Plan:", cropPlan)}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-green-800 mb-2">🌾 {cropPlan.crop}</h1>
              <p className="text-gray-700 font-medium">{cropPlan.overview}</p>
              <p className="mt-2 text-green-700 font-semibold">
                📈 Expected Yield: {cropPlan.estimated_yield} | ROI: {cropPlan.yield_roi}
              </p>
            </div>
            <div className="space-y-6">
              {cropPlan.plan.map((day, i) => (
                <div key={i} className="p-6 bg-green-50 rounded-xl border border-green-300 shadow hover:shadow-lg transition">
                  <h4 className="text-xl font-bold text-green-700 mb-1">Day {day.day} - {day.date}</h4>
                  <p className="font-semibold mb-2">🌱 Phase: {day.phase}</p>
                  <ul className="list-disc list-inside mb-2 space-y-1">
                    {day.activities.map((act, idx) => <li key={idx}>{act}</li>)}
                  </ul>
                  <p>💧 <b>Water:</b> {day.water_requirement}</p>
                  <p>🌿 <b>Fertilizer:</b> {day.fertilizer_application}</p>
                  <p>🔍 <b>Disease:</b> {day.disease_monitoring}</p>
                  <p>🌤️ <b>Weather:</b> {day.weather_consideration}</p>
                </div>
              ))}
            </div>
            <div className="text-center mt-8">
              <Button onClick={() => window.location.reload()} className="bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-xl transition">
                🔄 Create New Plan
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default CropAssistance;
