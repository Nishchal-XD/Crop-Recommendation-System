import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load Dataset
# Replace 'crop_data.csv' with your actual dataset
data = pd.read_csv('crop_data.csv')

# Step 2: Define Features
# Select the features used for training
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]

# Step 3: Initialize and Fit StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Save the StandardScaler
with open('standscaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("StandardScaler saved as 'standscaler.pkl'")
