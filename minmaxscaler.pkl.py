import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

# Step 1: Load Dataset
# Replace 'crop_data.csv' with your actual dataset
data = pd.read_csv('crop_data.csv')

# Step 2: Define Features
# Select the features used for training
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]

# Step 3: Initialize and Fit MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Step 4: Save the MinMaxScaler
with open('minmaxscaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("MinMaxScaler saved as 'minmaxscaler.pkl'")
