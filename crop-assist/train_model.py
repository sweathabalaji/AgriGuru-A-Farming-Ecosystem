import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load updated dataset with soil type
df = pd.read_excel("crop_dataset_with_soil.xlsx")

# Label encoding
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()  # New encoder for soil type

df['State_Code'] = state_encoder.fit_transform(df['State_Name'])
df['District_Code'] = district_encoder.fit_transform(df['District_Name'])
df['Crop_Code'] = crop_encoder.fit_transform(df['Crop'])
df['Soil_Code'] = soil_encoder.fit_transform(df['soil_type'])  # Encode soil type

# Features now include Investment_Price and Soil_Code
X = df[['State_Code', 'District_Code', 'Investment_Price', 'Soil_Code']]
y = df['Crop_Code']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'crop_predictor_model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(district_encoder, 'district_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')
joblib.dump(soil_encoder, 'soil_encoder.pkl')  # Save soil encoder

print("Model and encoders saved with investment price and soil type.")
