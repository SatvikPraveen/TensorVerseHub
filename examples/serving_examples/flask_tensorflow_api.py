# Location: /examples/serving_examples/flask_tensorflow_api.py

"""
Flask API for serving TensorFlow models with tf.keras integration.
Provides REST endpoints for model inference, health checks, and metadata.
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import json
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorFlowModelServer:
    """TensorFlow model serving with Flask."""
    
    def __init__(self, model_path: str, model_type: str = 'savedmodel'):
        """
        Initialize model server.
        
        Args:
            model_path: Path to the model
            model_type: Type of model ('savedmodel', 'h5', 'tflite')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.model_metadata = {}
        self.load_model()
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0
        
    def load_model(self) -> None:
        """Load the TensorFlow model."""
        try:
            if self.model_type == 'savedmodel':
                self.model = tf.saved_model.load(self.model_path)
                logger.info(f"Loaded SavedModel from {self.model_path}")
                
            elif self.model_type == 'h5':
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded Keras model from {self.model_path}")
                
            elif self.model_type == 'tflite':
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info(f"Loaded TFLite model from {self.model_path}")
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            # Load metadata if available
            self._load_metadata()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_metadata(self) -> None:
        """Load model metadata."""
        metadata_path = None
        
        if self.model_type == 'savedmodel':
            metadata_path = os.path.join(self.model_path, 'metadata.json')
        elif self.model_type == 'h5':
            metadata_path = os.path.join(os.path.dirname(self.model_path), 'metadata.json')
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
                logger.info("Loaded model metadata")
    
    def preprocess_image(self, image_data: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess base64 encoded image for inference.
        
        Args:
            image_data: Base64 encoded image
            target_size: Target image size
            
        Returns:
            Preprocessed image array
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def preprocess_text(self, text: str, max_length: int = 128) -> np.ndarray:
        """
        Preprocess text for inference (simplified tokenization).
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed text array
        """
        # This is a simplified tokenization - in practice, use the same
        # tokenizer that was used during training
        words = text.lower().split()
        
        # Simple word to integer mapping (would use actual tokenizer in practice)
        word_to_int = {word: i+1 for i, word in enumerate(set(words))}
        
        # Convert words to integers
        token_ids = [word_to_int.get(word, 0) for word in words]
        
        # Pad or truncate to max_length
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))
        else:
            token_ids = token_ids[:max_length]
        
        return np.array([token_ids], dtype=np.int32)
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run model inference.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Model predictions
        """
        start_time = time.time()
        
        try:
            if self.model_type == 'tflite':
                # TFLite inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
                
            else:
                # TensorFlow/Keras inference
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(input_data)
                else:
                    # For SavedModel
                    predictions = self.model(input_data)
                    if isinstance(predictions, dict):
                        predictions = list(predictions.values())[0]
                    predictions = predictions.numpy()
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.inference_count += 1
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        avg_inference_time = (self.total_inference_time / self.inference_count 
                             if self.inference_count > 0 else 0)
        
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'inference_count': self.inference_count,
            'average_inference_time_ms': avg_inference_time * 1000,
            'total_inference_time_s': self.total_inference_time,
            'metadata': self.model_metadata
        }
        
        if self.model_type == 'tflite':
            info['input_shape'] = self.input_details[0]['shape'].tolist()
            info['output_shape'] = self.output_details[0]['shape'].tolist()
            info['input_dtype'] = str(self.input_details[0]['dtype'])
            info['output_dtype'] = str(self.output_details[0]['dtype'])
        
        return info


# Global model server instance
model_server: Optional[TensorFlowModelServer] = None

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Decorators
def require_model(f):
    """Decorator to ensure model is loaded."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if model_server is None:
            return jsonify({'error': 'Model not loaded'}), 503
        return f(*args, **kwargs)
    return decorated_function

def log_request(f):
    """Decorator to log requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.path} - {duration:.3f}s")
        return result
    return decorated_function

# Routes
@app.route('/')
def home():
    """Home page with API documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TensorFlow Model Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #2e7d32; }
            .path { font-family: monospace; background-color: #e8f5e8; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🤖 TensorFlow Model Server</h1>
        <p>RESTful API for TensorFlow model inference with tf.keras integration.</p>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="path">/health</div>
            <p>Health check endpoint - returns server and model status.</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="path">/model/info</div>
            <p>Get detailed model information and performance statistics.</p>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="path">/predict/image</div>
            <p>Image classification inference. Send base64-encoded image in JSON body.</p>
            <pre>{"image": "data:image/jpeg;base64,..."}</pre>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="path">/predict/text</div>
            <p>Text classification inference. Send text in JSON body.</p>
            <pre>{"text": "Your input text here"}</pre>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="path">/predict/batch</div>
            <p>Batch inference for multiple inputs.</p>
            <pre>{"inputs": [input1, input2, ...]}</pre>
        </div>
        
        <h2>Model Status</h2>
        {% if model_loaded %}
        <p>✅ Model loaded: {{ model_type }} from {{ model_path }}</p>
        {% else %}
        <p>❌ No model loaded</p>
        {% endif %}
        
        <h2>Usage Examples</h2>
        <h3>cURL Example</h3>
        <pre>
curl -X POST http://localhost:5000/predict/image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}'
        </pre>
        
        <h3>Python Example</h3>
        <pre>
import requests
import base64

with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/predict/image', 
                        json={'image': f'data:image/jpeg;base64,{image_data}'})
print(response.json())
        </pre>
    </body>
    </html>
    """
    
    return render_template_string(
        html_template,
        model_loaded=model_server is not None,
        model_type=model_server.model_type if model_server else None,
        model_path=model_server.model_path if model_server else None
    )

@app.route('/health')
@log_request
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'tensorflow_version': tf.__version__,
        'model_loaded': model_server is not None,
        'uptime_seconds': time.time() - app.start_time if hasattr(app, 'start_time') else 0
    }
    
    if model_server:
        status.update({
            'model_type': model_server.model_type,
            'inference_count': model_server.inference_count,
            'average_inference_time_ms': (model_server.total_inference_time / model_server.inference_count * 1000) if model_server.inference_count > 0 else 0
        })
    
    return jsonify(status)

@app.route('/model/info')
@require_model
@log_request
def model_info():
    """Get model information."""
    return jsonify(model_server.get_model_info())

@app.route('/predict/image', methods=['POST'])
@require_model
@log_request
def predict_image():
    """Image classification endpoint."""
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        image_data = model_server.preprocess_image(request.json['image'])
        
        # Run inference
        predictions = model_server.predict(image_data)
        
        # Process predictions
        if predictions.ndim > 1 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        # Convert to list for JSON serialization
        predictions_list = predictions.tolist()
        
        # Get top predictions
        if len(predictions_list) > 1:
            top_indices = np.argsort(predictions)[-5:][::-1]  # Top 5
            top_predictions = [
                {
                    'class_id': int(idx),
                    'confidence': float(predictions[idx]),
                    'class_name': model_server.model_metadata.get('class_names', [str(idx)])[idx] if 'class_names' in model_server.model_metadata else str(idx)
                }
                for idx in top_indices
            ]
        else:
            top_predictions = [{'prediction': float(predictions_list[0])}]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions,
            'raw_predictions': predictions_list
        })
        
    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/text', methods=['POST'])
@require_model
@log_request
def predict_text():
    """Text classification endpoint."""
    try:
        if not request.json or 'text' not in request.json:
            return jsonify({'error': 'No text data provided'}), 400
        
        # Preprocess text
        text_data = model_server.preprocess_text(request.json['text'])
        
        # Run inference
        predictions = model_server.predict(text_data)
        
        # Process predictions
        if predictions.ndim > 1 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        predictions_list = predictions.tolist()
        
        # Get top predictions
        if len(predictions_list) > 1:
            top_indices = np.argsort(predictions)[-3:][::-1]  # Top 3
            top_predictions = [
                {
                    'class_id': int(idx),
                    'confidence': float(predictions[idx]),
                    'class_name': model_server.model_metadata.get('class_names', [str(idx)])[idx] if 'class_names' in model_server.model_metadata else str(idx)
                }
                for idx in top_indices
            ]
        else:
            top_predictions = [{'prediction': float(predictions_list[0])}]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions,
            'raw_predictions': predictions_list
        })
        
    except Exception as e:
        logger.error(f"Text prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
@require_model
@log_request
def predict_batch():
    """Batch inference endpoint."""
    try:
        if not request.json or 'inputs' not in request.json:
            return jsonify({'error': 'No input data provided'}), 400
        
        inputs = request.json['inputs']
        if not isinstance(inputs, list):
            return jsonify({'error': 'Inputs must be a list'}), 400
        
        if len(inputs) > 32:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 32)'}), 400
        
        batch_predictions = []
        
        for i, input_item in enumerate(inputs):
            try:
                if 'image' in input_item:
                    processed_input = model_server.preprocess_image(input_item['image'])
                elif 'text' in input_item:
                    processed_input = model_server.preprocess_text(input_item['text'])
                else:
                    batch_predictions.append({'error': f'Invalid input format at index {i}'})
                    continue
                
                predictions = model_server.predict(processed_input)
                if predictions.ndim > 1 and predictions.shape[0] == 1:
                    predictions = predictions[0]
                
                batch_predictions.append({
                    'success': True,
                    'predictions': predictions.tolist()
                })
                
            except Exception as e:
                batch_predictions.append({'error': str(e)})
        
        return jsonify({
            'success': True,
            'batch_size': len(inputs),
            'results': batch_predictions
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/reload', methods=['POST'])
@log_request
def reload_model():
    """Reload the model."""
    global model_server
    
    try:
        if model_server:
            old_path = model_server.model_path
            model_server.load_model()
            logger.info(f"Model reloaded from {old_path}")
            return jsonify({'success': True, 'message': 'Model reloaded successfully'})
        else:
            return jsonify({'error': 'No model to reload'}), 404
            
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({'error': 'Internal server error'}), 500

def initialize_server(model_path: str, model_type: str = 'savedmodel'):
    """Initialize the model server."""
    global model_server
    
    try:
        model_server = TensorFlowModelServer(model_path, model_type)
        app.start_time = time.time()
        logger.info(f"Server initialized with {model_type} model from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize server: {str(e)}")
        return False

def main():
    """Main function to run the Flask server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorFlow Model Server')
    parser.add_argument('--model-path', required=True, help='Path to the model')
    parser.add_argument('--model-type', default='savedmodel', choices=['savedmodel', 'h5', 'tflite'], help='Model type')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize server
    if not initialize_server(args.model_path, args.model_type):
        logger.error("Failed to initialize server")
        exit(1)
    
    # Start Flask app
    logger.info(f"Starting TensorFlow Model Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()