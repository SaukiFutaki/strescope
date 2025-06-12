from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)

# Global variables untuk model components
model = None
scaler = None
pca = None
feature_names = None
label_mapping = None
models_loaded = False


def load_models():
    """Load semua model components dengan error handling"""
    global model, scaler, pca, feature_names, label_mapping, models_loaded

    print("🔄 Loading model components...")

    try:
        # Check if all required files exist
        required_files = [
            'stress_model.pkl',
            'scaler.pkl',
            'pca.pkl',
            'feature_names.pkl',
            'label_mapping.pkl'
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            print("🔧 Run 'python create_dummy_models.py' to create dummy models for testing")
            return False

        # Load each component
        print("📦 Loading model...")
        with open('stress_model.pkl', 'rb') as f:
            model = pickle.load(f)

        print("📦 Loading scaler...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        print("📦 Loading PCA...")
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)

        print("📦 Loading feature names...")
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        print("📦 Loading label mapping...")
        with open('label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)

        print("✅ All model components loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Features count: {len(feature_names)}")
        print(f"Labels: {label_mapping}")

        models_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Possible solutions:")
        print("1. Run 'python create_models.py' to create model files")
        print("2. Check if files are corrupted and recreate them")
        print("3. Ensure all dependencies are installed correctly")

        models_loaded = False
        return False


def preprocess_input(data):
    """Preprocess input data"""
    if not models_loaded:
        raise ValueError("Models not loaded")

    try:
        # Buat DataFrame dari input
        df = pd.DataFrame([data])

        # Normalisasi input teks
        df['Gender'] = df['Gender'].str.capitalize()
        df['BMI Category'] = df['BMI Category'].str.capitalize()
        df['Sleep Disorder'] = df['Sleep Disorder'].str.title()
        df['Occupation'] = df['Occupation'].str.lower()

        # Mapping gender
        gender_map = {'Male': 1, 'Female': 0}
        if df['Gender'].iloc[0] not in gender_map:
            raise ValueError(f"Gender tidak valid: {df['Gender'].iloc[0]}")
        df['Gender'] = df['Gender'].map(gender_map)

        # Mapping BMI
        bmi_map = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        if df['BMI Category'].iloc[0] not in bmi_map:
            raise ValueError(f"BMI Category tidak valid: {df['BMI Category'].iloc[0]}")
        df['BMI Category'] = df['BMI Category'].map(bmi_map)

        # Mapping sleep disorder
        sleep_map = {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
        if df['Sleep Disorder'].iloc[0] not in sleep_map:
            raise ValueError(f"Sleep Disorder tidak valid: {df['Sleep Disorder'].iloc[0]}")
        df['Sleep Disorder'] = df['Sleep Disorder'].map(sleep_map)

        # Occupation one-hot encoding
        known_occupations = ['doctor', 'nurse', 'engineer', 'teacher', 'lawyer', 'student', 'manager', 'scientist',
                             'salesman']
        occupation_input = df['Occupation'].iloc[0]
        if occupation_input not in known_occupations:
            raise ValueError(f"Occupation tidak valid: {occupation_input}")

        for occ in known_occupations:
            df[f"Occupation_{occ}"] = 1 if occupation_input == occ else 0
        df.drop(columns='Occupation', inplace=True)

        # Tambahkan kolom yang hilang dari feature_names
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Susun ulang sesuai urutan saat training
        df = df[feature_names]

        return df

    except Exception as e:
        raise ValueError(f"Error preprocessing: {str(e)}")


@app.route('/', methods=['GET'])
def home():
    """Endpoint untuk cek status API"""
    return jsonify({
        'message': 'Stress Level Prediction API',
        'status': 'active' if models_loaded else 'models not loaded',
        'version': '1.0',
        'models_loaded': models_loaded,
        'endpoints': {
            'predict': '/predict (POST)',
            'predict_simple': '/predict/simple (POST)',
            'health': '/health (GET)',
            'reload': '/reload (POST)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'components': {
            'model': model is not None,
            'scaler': scaler is not None,
            'pca': pca is not None,
            'feature_names': feature_names is not None,
            'label_mapping': label_mapping is not None
        }
    })


@app.route('/reload', methods=['POST'])
def reload_models():
    """Reload model components"""
    success = load_models()
    return jsonify({
        'success': success,
        'models_loaded': models_loaded,
        'message': 'Models reloaded successfully' if success else 'Failed to reload models'
    })


@app.route('/predict', methods=['POST'])
def predict_stress():
    """Endpoint untuk prediksi stress level"""

    # Check if models are loaded
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Try /reload endpoint or check model files.',
            'suggestion': 'Run create_dummy_models.py to create model files'
        }), 503

    try:
        # Ambil data dari request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validasi field yang diperlukan
        required_fields = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
            'Physical Activity Level', 'BMI Category', 'Systolic BP', 'Diastolic BP',
            'Heart Rate', 'Daily Steps', 'Sleep Disorder'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400

        # Preprocessing
        processed_data = preprocess_input(data)

        # Scaling
        scaled_data = scaler.transform(processed_data)

        # PCA
        pca_data = pca.transform(scaled_data)

        # Prediksi
        prediction = model.predict(pca_data)[0]
        prediction_proba = model.predict_proba(pca_data)[0]

        # Get prediction label
        stress_level = label_mapping.get(prediction, "Unknown")

        # Confidence score
        confidence = float(max(prediction_proba))

        return jsonify({
            'prediction': {
                'stress_level_numeric': int(prediction),
                'stress_level_label': stress_level,
                'confidence': round(confidence, 3)
            },
            'input_data': data,
            'probabilities': {
                'rendah': round(float(prediction_proba[0]), 3) if len(prediction_proba) > 0 else 0,
                'sedang': round(float(prediction_proba[1]), 3) if len(prediction_proba) > 1 else 0,
                'tinggi': round(float(prediction_proba[2]), 3) if len(prediction_proba) > 2 else 0
            }
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Endpoint sederhana dengan input form-like"""

    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Try /reload endpoint or check model files.'
        }), 503

    try:
        # Ambil data dari form atau JSON
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()

        # Konversi ke format yang sesuai
        try:
            blood_pressure = form_data.get('blood_pressure', '120/80')
            systolic, diastolic = map(int, blood_pressure.split('/'))

            data = {
                'Gender': form_data.get('gender'),
                'Age': int(form_data.get('age')),
                'Occupation': form_data.get('occupation'),
                'Sleep Duration': float(form_data.get('sleep_duration')),
                'Quality of Sleep': int(form_data.get('quality_sleep')),
                'Physical Activity Level': int(form_data.get('physical_activity')),
                'BMI Category': form_data.get('bmi_category'),
                'Systolic BP': systolic,
                'Diastolic BP': diastolic,
                'Heart Rate': int(form_data.get('heart_rate')),
                'Daily Steps': int(form_data.get('daily_steps')),
                'Sleep Disorder': form_data.get('sleep_disorder', 'None')
            }
        except (ValueError, KeyError) as e:
            return jsonify({'error': f'Invalid input format: {str(e)}'}), 400

        # Preprocessing dan prediksi
        processed_data = preprocess_input(data)
        scaled_data = scaler.transform(processed_data)
        pca_data = pca.transform(scaled_data)
        prediction = model.predict(pca_data)[0]
        prediction_proba = model.predict_proba(pca_data)[0]

        stress_level = label_mapping.get(prediction, "Unknown")
        confidence = float(max(prediction_proba))

        return jsonify({
            'result': stress_level,
            'prediction_numeric': int(prediction),
            'confidence': round(confidence, 3),
            'message': f'Tingkat stres Anda diprediksi: {stress_level}'
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# Load models saat startup
load_models()

if __name__ == '__main__':
    print("🚀 Starting Stress Level Prediction API...")

    if not models_loaded:
        print("⚠️ Models not loaded! API will have limited functionality.")
        print("💡 Solutions:")
        print("   1. Run: python create_dummy_models.py")
        print("   2. Or use /reload endpoint after creating model files")

    print("\n📋 Available endpoints:")
    print("  GET  /         - API info")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Full prediction (JSON)")
    print("  POST /predict/simple - Simple prediction")
    print("  POST /reload   - Reload models")

    app.run(debug=True, host='0.0.0.0', port=5000)