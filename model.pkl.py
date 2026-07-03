import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load Dataset
# Replace 'crop_data.csv' with your actual dataset
# Ensure the dataset has features like Nitrogen, Phosphorus, Potassium, etc.
data = pd.read_csv('crop_data.csv')

# Step 2: Define Features and Target
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = data['Crop_Label']  # 'Crop_Label' should be integers representing crops

# Step 3: Preprocess the Data
# Scale the features using MinMaxScaler and StandardScaler
minmax_scaler = MinMaxScaler()
X_minmax_scaled = minmax_scaler.fit_transform(X)

standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X_minmax_scaled)

# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the Model and Scalers
# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the MinMaxScaler
with open('minmaxscaler.pkl', 'wb') as file:
    pickle.dump(minmax_scaler, file)

# Save the StandardScaler
with open('standscaler.pkl', 'wb') as file:
    pickle.dump(standard_scaler, file)

print("Model and scalers saved successfully!")
