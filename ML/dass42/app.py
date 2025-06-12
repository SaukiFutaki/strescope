from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
app = Flask(__name__)


class StressPredictionSystem:
    """Complete stress prediction system"""

    def __init__(self, model_path, config_path, scaler_path, imputer_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load preprocessing objects
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)

        # Load model
        if self.config['model_type'] == 'neural_network':
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

        self.question_columns = self.config['question_columns']
        self.stress_labels = self.config['stress_labels']

    def preprocess_input(self, responses):
        """Preprocess user responses"""
        # Convert responses to DataFrame
        if isinstance(responses, dict):
            # Ensure all required questions are present
            for col in self.question_columns:
                if col not in responses:
                    responses[col] = 0  # Default value for missing responses

            df = pd.DataFrame([responses])
        else:
            df = pd.DataFrame([dict(zip(self.question_columns, responses))])

        # Select only required columns
        df = df[self.question_columns]

        # Handle missing values
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns
        )

        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_imputed),
            columns=df.columns
        )

        return df_scaled

    def predict(self, responses):
        """Make stress prediction"""
        # Preprocess input
        X = self.preprocess_input(responses)

        # Make prediction
        if self.config['model_type'] == 'neural_network':
            pred_proba = self.model.predict(X, verbose=0)
            pred_class = np.argmax(pred_proba, axis=1)[0]
            confidence = float(np.max(pred_proba))
        else:
            pred_class = self.model.predict(X)[0]
            pred_proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(pred_proba))

        # Get stress level label
        stress_level = self.stress_labels[str(pred_class)]

        return {
            'stress_level': stress_level,
            'stress_level_numeric': int(pred_class),
            'confidence': confidence,
            'probabilities': {
                'Low': float(pred_proba[0]),
                'Moderate': float(pred_proba[1]),
                'High': float(pred_proba[2])
            }
        }
# Load model and preprocessing objects
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)

    scaler = joblib.load('stress_scaler.pkl')
    imputer = joblib.load('stress_imputer.pkl')

    if config['model_type'] == 'neural_network':
        model = tf.keras.models.load_model('stress_model.h5')
    else:
        model = joblib.load('stress_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route('/predict', methods=['POST'])
def predict_stress():
    try:
        # Get JSON data
        data = request.get_json()

        # Validate input
        if not data or 'responses' not in data:
            return jsonify({'error': 'Missing responses data'}), 400

        responses = data['responses']

        # Create predictor instance
        predictor = StressPredictionSystem(
            'stress_model.h5' if config['model_type'] == 'neural_network' else 'stress_model.pkl',
            'model_config.json',
            'stress_scaler.pkl',
            'stress_imputer.pkl'
        )

        # Make prediction
        result = predictor.predict(responses)

        return jsonify({
            'status': 'success',
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


@app.route('/questions', methods=['GET'])
def get_questions():
    return jsonify({
        'selected_questions': config['selected_questions'],
        'question_columns': config['question_columns'],
        'description': 'DASS-21 Stress related questions'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)