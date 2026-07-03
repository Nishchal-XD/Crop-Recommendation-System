from flask import Flask, request, render_template
import os
import urllib.parse
import numpy as np
import pandas
import pickle

# base directory (script location) so paths work regardless of cwd
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# importing model (use absolute paths)
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'),'rb'))
sc = pickle.load(open(os.path.join(BASE_DIR, 'standscaler.pkl'),'rb'))
ms = pickle.load(open(os.path.join(BASE_DIR, 'minmaxscaler.pkl'),'rb'))

# creating flask app; serve static files from the script folder
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=BASE_DIR)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    feature_df = pandas.DataFrame([feature_list], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'])

    scaled_features = ms.transform(feature_df)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)

        # determine image filename for the recommended crop
        def find_image_for_crop(name):
            base = name.lower().replace(' ', '_')
            for ext in ('.jpg', '.jpeg', '.png'):
                fname = base + ext
                # check inside the script's folder (BASE_DIR)
                if os.path.exists(os.path.join(BASE_DIR, fname)):
                    return fname
                # check inside images/ subfolder
                if os.path.exists(os.path.join(BASE_DIR, 'images', fname)):
                    return os.path.join('images', fname).replace('\\', '/')
            # fallback to local generic image if present
            if os.path.exists(os.path.join(BASE_DIR, 'img.jpg')):
                return 'img.jpg'
            # final fallback: return an external placeholder image URL
            return 'https://via.placeholder.com/600x400?text=' + urllib.parse.quote(name)

        image_filename = find_image_for_crop(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        image_filename = None

    return render_template('index.html', result=result, image=image_filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)




# python main
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)