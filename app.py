from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

from flask_cors import CORS, cross_origin
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import warnings
import tensorflow as tf
import os
from transformers import BertTokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=UserWarning)



app = Flask(__name__, static_folder="./build/static", template_folder="./build")

CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})

# Set up credentials
scope = ["https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("myProject.json", scope)
client = gspread.authorize(creds)

@app.route("/")
def index():
    return render_template("index.html")
# Marriage Age predictor
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data = request.json
    gender = int(data['gender'])
    religion = int(data['religion'])
    caste = int(data['caste'])
    mother_tongue = int(data['mother_tongue'])
    country = int(data['country'])
    height_cms = float(data['height_cms'])

    inputs = [gender, religion, caste, mother_tongue, country, height_cms]
    model = joblib.load("marriage_age_predictor.ml")
    age_pred = model.predict([inputs])
    return str(age_pred[0])

@app.route("/add_data", methods=['POST'])
def add_data():
    print("add_data endpoint was called.")  # Debugging
    try:
        data = request.json
        print(f"Data to be written: {data}")
        sheet = client.open("Myproject").worksheet("Sheet1")  # or whatever your sheet's name is

        # Check for column headers
        headers = ['gender', 'religion', 'caste', 'mother_tongue', 'country', 'height_cms']
        first_row = sheet.row_values(1)  # gets the first row
        if not first_row or first_row != headers:
            sheet.insert_row(headers, 1)  # insert headers at first row

        sheet.append_row([
            data['gender'], 
            data['religion'], 
            data["caste"], 
            data['mother_tongue'],
            data['country'], 
            data['height_cms']
        ])

        return jsonify({"success": True, "message": "Data added to Google Sheet!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Load the cyberbullying model
cyberbullying_model = tf.keras.models.load_model("Stacking_Model.h5")

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
CLASS_NAMES = ["not bully", "troll", "sexual", "religious", "threat"]
# Cyber Bullying detection 
@app.route("/detect_cyberbullying", methods=['GET', 'POST'])
def detect_cyberbullying():
    try:
        data = request.json
        text = data['text']

        # Tokenize the input text
        inputs = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=100, 
            truncation=True,
            padding='max_length',
            return_attention_mask=False,
            return_tensors='tf'
        )

        sequences = inputs["input_ids"]

        # Predict
        predictions = cyberbullying_model.predict(sequences)[0]
        predicted_index = int(tf.argmax(predictions).numpy())
        predicted_class = CLASS_NAMES[predicted_index]

        response = {
            'predicted_index': predicted_index,
            'predicted_class': predicted_class,
            'probabilities': predictions.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


brain_model = joblib.load('brain_model.pkl')

@app.route('/predict_brainstroke', methods=['POST'])
def predict_brain():
    data = request.get_json(force=True)
    # Preprocessing the data
    preprocessed_data = preprocess_input(data)
    # Convert input data to appropriate format
    input_data = [
        preprocessed_data['gender'], 
        preprocessed_data['age'], 
        preprocessed_data['hypertension'], 
        preprocessed_data['heart_disease'],
        preprocessed_data['ever_married'], 
        preprocessed_data['work_type'], 
        preprocessed_data['Residence_type'],
        preprocessed_data['avg_glucose_level'], 
        preprocessed_data['bmi'], 
        preprocessed_data['smoking_status']
    ]
    # Assume model takes data in array format
    prediction = brain_model.predict([input_data])
    # Process the model prediction and return the result
    if isinstance(prediction, (list, np.ndarray)):
        if len(prediction.shape) == 2:
            result = {'prediction': int(prediction[0][0])}
        else:
            result = {'prediction': int(prediction[0])}
    else:
        result = {'prediction': int(prediction)}  # adjust based on your model's output format
    return jsonify(result)


def preprocess_input(data):
    # Encoding gender: Male is 1, Female is 0
    if data['gender'].lower() == 'male':
        data['gender'] = 1
    else:
        data['gender'] = 0

    # Convert age
    data['age'] = float(data['age'])

    # Convert hypertension and heart_disease
    if data['hypertension'].lower() == 'yes':
        data['hypertension'] = 1
    else:
        data['hypertension'] = 0

    if data['heart_disease'].lower() == 'yes':
        data['heart_disease'] = 1
    else:
        data['heart_disease'] = 0
    # Encoding ever_married: Yes is 1, No is 0
    if data['ever_married'].lower() == 'yes':
        data['ever_married'] = 1
    else:
        data['ever_married'] = 0

    # Encoding work_type: (This is just a sample, adapt as needed)
    work_type_map = {
        'private': 1,
        'self-employed': 2,
        'govt_job': 3,
        'children': 4,
        'never_worked': 5
    }
    data['work_type'] = work_type_map.get(data['work_type'].lower(), 0)  # 0 as default if not found

    # Encoding Residence_type: Urban is 1, Rural is 0
    if data['Residence_type'].lower() == 'urban':
        data['Residence_type'] = 1
    else:
        data['Residence_type'] = 0

    # Convert avg_glucose_level and bmi to floats
    data['avg_glucose_level'] = float(data['avg_glucose_level'])
    data['bmi'] = float(data['bmi'])

    # Encoding smoking_status: (This is just a sample, adapt as needed)
    smoking_status_map = {
        'formerly_smoked': 1,
        'never_smoked': 2,
        'smokes': 3,
        'unknown': 4
    }
    data['smoking_status'] = smoking_status_map.get(data['smoking_status'].lower(), 0)  # 0 as default if not found

    return data

if __name__ == "__main__":
    app.run(debug=True)
