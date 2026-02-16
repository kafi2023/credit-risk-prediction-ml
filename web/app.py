"""
Flask web application for Credit Risk Prediction
"""
from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from src.prediction.predictor import CreditRiskPredictor

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/')
def index():
    """Home page — renders the prediction form."""
    return render_template('index.html')


@app.route('/schema', methods=['GET'])
def schema():
    """Return the input schema so the frontend can build a dynamic form."""
    return jsonify(CreditRiskPredictor.input_schema())


@app.route('/models', methods=['GET'])
def models():
    """List available trained models."""
    return jsonify({"models": CreditRiskPredictor.available_models()})


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests.

    Expects JSON with 20 feature fields (7 numerical + 13 categorical).
    Returns prediction, probability, risk level, and SHAP explanation.
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"status": "error", "errors": ["Invalid JSON body"]}), 400

        # Optional: allow client to choose model
        model_name = data.pop("model", "random_forest")

        predictor = CreditRiskPredictor.get_instance(model_name)
        result = predictor.predict(data, explain=True)

        status_code = 200 if result["status"] == "success" else 400
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            'status': 'error',
            'errors': [str(e)]
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
