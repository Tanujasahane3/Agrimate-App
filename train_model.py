import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your training dataset
# Make sure your CSV has columns like: Crop, Area, Seed_Type, Yield_per_Acre, Seed_Cost_per_Acre, Fertilizer_Cost, etc.
data = pd.read_csv("crop_training_data.csv")

# Check columns (adjust if your file has different column names)
# print(data.columns)

# Encode Seed_Type (categorical feature)
seed_type_encoder = LabelEncoder()
data['Seed_Type_Encoded'] = seed_type_encoder.fit_transform(data['Seed_Type'])

# Features and target
# Using 'Area' and 'Seed_Type_Encoded' as features (like your app expects)
X = data[['Area', 'Seed_Type_Encoded']]

# Target variable: Yield per Acre
y = data['Yield_per_Acre']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the model and encoder for your app to load later
joblib.dump(model, "yield_model.pkl")
joblib.dump(seed_type_encoder, "seed_type_encoder.pkl")

print("Training completed and model + encoder saved!")
