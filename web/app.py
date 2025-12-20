"""
Flask web application for Credit Risk Prediction
"""
from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from request
        data = request.get_json()
        
        # TODO: Implement prediction logic
        # This will be implemented in Milestone 2
        
        response = {
            'status': 'success',
            'message': 'Prediction endpoint - to be implemented',
            'prediction': None,
            'explanation': None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

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
