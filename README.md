# Crop Recommendation System using Machine Learning

An intelligent, data-driven web application designed to help farmers and agricultural professionals maximize crop yields. By analyzing environmental and soil parameters, the system leverages machine learning to accurately recommend the most suitable crop to cultivate in a specific field.

---

## 🚀 Features
* **Multi-Factor Analysis:** Recommends crops based on critical soil attributes (Nitrogen, Phosphorus, Potassium) and environmental metrics (Temperature, Humidity, pH, Rainfall).
* **Pre-trained ML Engine:** Utilizes a highly optimized classification model trained on empirical agricultural data.
* **Feature Scaling:** Integrates standardized MinMax and Standard scaling pipelines to ensure seamless data preprocessing before inference.
* **Responsive Web Interface:** Features a clean, user-friendly frontend built to collect metrics and display immediate, actionable crop recommendations.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Serialization:** Pickle (`.pkl`) for saving and loading scaling pipelines and model weights
* **Backend Framework:** Flask / FastAPI
* **Frontend:** HTML5, CSS3 (Bootstrap-based UI components)

---

## 📂 Project Structure
```text
├── app.py                  # Main web application logic and server router
├── crop_data.csv           # Historical agricultural training dataset
├── model.pkl               # Trained Machine Learning classification model
├── minmaxscaler.pkl        # Serialized MinMax Scaler instance
├── standscaler.pkl         # Serialized Standard Scaler instance
├── README.md               # Project documentation
├── screenshots/            # UI application walkthrough images
├── templates/              
│   └── index.html          # Frontend data entry form and UI layout
└── images/                 # Reference assets and UI graphics
