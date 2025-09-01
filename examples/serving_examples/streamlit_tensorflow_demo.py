# Location: /examples/serving_examples/streamlit_tensorflow_demo.py

"""
Streamlit demo application for TensorFlow model inference.
Interactive web interface for testing image and text classification models.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io
import time
from typing import List, Dict, Tuple, Optional
import base64


# Page configuration
st.set_page_config(
    page_title="TensorFlow Model Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str, model_type: str):
    """Load TensorFlow model with caching."""
    try:
        if model_type == "SavedModel":
            model = tf.saved_model.load(model_path)
        elif model_type == "Keras H5":
            model = tf.keras.models.load_model(model_path)
        elif model_type == "TFLite":
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for model inference."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def preprocess_text(text: str, max_length: int = 128) -> np.ndarray:
    """Simple text preprocessing (in practice, use the same tokenizer as training)."""
    # Simple tokenization
    words = text.lower().split()
    
    # Create a simple vocabulary (in practice, use the training vocabulary)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Convert to token IDs
    token_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad or truncate
    if len(token_ids) < max_length:
        token_ids.extend([vocab['<PAD>']] * (max_length - len(token_ids)))
    else:
        token_ids = token_ids[:max_length]
    
    return np.array([token_ids], dtype=np.int32)

def run_inference(model, input_data: np.ndarray, model_type: str) -> np.ndarray:
    """Run model inference."""
    start_time = time.time()
    
    if model_type == "TFLite":
        # TFLite inference
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
    else:
        # TensorFlow/Keras inference
        if hasattr(model, 'predict'):
            predictions = model.predict(input_data)
        else:
            predictions = model(input_data)
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[0]
            predictions = predictions.numpy()
    
    inference_time = time.time() - start_time
    
    return predictions, inference_time

def create_prediction_chart(predictions: np.ndarray, class_names: Optional[List[str]] = None, top_k: int = 5):
    """Create interactive prediction chart."""
    if predictions.ndim > 1:
        predictions = predictions[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_scores = predictions[top_indices]
    
    # Create labels
    if class_names and len(class_names) > max(top_indices):
        labels = [class_names[i] for i in top_indices]
    else:
        labels = [f"Class {i}" for i in top_indices]
    
    # Create horizontal bar chart
    fig = px.bar(
        x=top_scores,
        y=labels,
        orientation='h',
        title=f"Top {top_k} Predictions",
        labels={'x': 'Confidence Score', 'y': 'Class'},
        color=top_scores,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_confidence_gauge(confidence: float):
    """Create confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence %"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 TensorFlow Model Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model selection
    demo_mode = st.sidebar.checkbox("Use Demo Mode (Simulated Predictions)", value=True)
    
    if not demo_mode:
        model_path = st.sidebar.text_input("Model Path", value="")
        model_type = st.sidebar.selectbox("Model Type", ["SavedModel", "Keras H5", "TFLite"])
        
        if model_path:
            model = load_model(model_path, model_type)
        else:
            model = None
            st.sidebar.warning("Please provide a model path")
    else:
        model = "demo"  # Placeholder for demo mode
        model_type = "Demo"
    
    # Application type selection
    app_type = st.sidebar.selectbox("Application Type", ["Image Classification", "Text Classification", "Model Analysis"])
    
    # Class names (optional)
    class_names_input = st.sidebar.text_area("Class Names (one per line, optional)")
    class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()] if class_names_input else None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    if app_type == "Image Classification":
        with col1:
            st.header("📸 Image Classification")
            
            # Image upload
            uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_image is not None:
                # Display image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Prediction button
                if st.button("🔮 Predict", type="primary"):
                    with st.spinner("Running inference..."):
                        if demo_mode:
                            # Simulate predictions for demo
                            time.sleep(1)  # Simulate processing time
                            predictions = np.random.dirichlet(np.ones(10), size=1)[0]
                            inference_time = 0.045
                        else:
                            if model is not None:
                                # Preprocess image
                                processed_image = preprocess_image(image)
                                
                                # Run inference
                                predictions, inference_time = run_inference(model, processed_image, model_type)
                                
                                if predictions.ndim > 1:
                                    predictions = predictions[0]
                            else:
                                st.error("Model not loaded")
                                return
                        
                        # Display results
                        st.success(f"Prediction completed in {inference_time:.3f}s")
                        
                        # Create prediction chart
                        fig = create_prediction_chart(predictions, class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top prediction
                        top_class = np.argmax(predictions)
                        top_confidence = predictions[top_class]
                        
                        class_name = class_names[top_class] if class_names and top_class < len(class_names) else f"Class {top_class}"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Top Prediction</h3>
                            <p><strong>{class_name}</strong></p>
                            <p>Confidence: {top_confidence:.4f} ({top_confidence * 100:.2f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.header("📊 Statistics")
            
            if uploaded_image is not None and 'predictions' in locals():
                # Confidence gauge
                fig_gauge = create_confidence_gauge(np.max(predictions))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Performance metrics
                st.markdown("### ⚡ Performance")
                st.metric("Inference Time", f"{inference_time:.3f}s")
                st.metric("Top Confidence", f"{np.max(predictions):.4f}")
                st.metric("Prediction Entropy", f"{-np.sum(predictions * np.log(predictions + 1e-10)):.3f}")
    
    elif app_type == "Text Classification":
        with col1:
            st.header("📝 Text Classification")
            
            # Text input
            input_text = st.text_area("Enter text for classification", height=100)
            
            # Sample texts
            sample_texts = [
                "This movie is absolutely amazing! Great acting and storyline.",
                "The product quality is poor and arrived damaged.",
                "Neutral statement about weather conditions today.",
                "I love this restaurant, the food is delicious!"
            ]
            
            selected_sample = st.selectbox("Or choose a sample text:", [""] + sample_texts)
            if selected_sample:
                input_text = selected_sample
            
            if input_text and st.button("🔮 Classify Text", type="primary"):
                with st.spinner("Processing text..."):
                    if demo_mode:
                        # Simulate predictions
                        time.sleep(0.5)
                        predictions = np.random.dirichlet(np.ones(5), size=1)[0]
                        inference_time = 0.023
                    else:
                        if model is not None:
                            # Preprocess text
                            processed_text = preprocess_text(input_text)
                            
                            # Run inference
                            predictions, inference_time = run_inference(model, processed_text, model_type)
                            
                            if predictions.ndim > 1:
                                predictions = predictions[0]
                        else:
                            st.error("Model not loaded")
                            return
                    
                    # Display results
                    st.success(f"Classification completed in {inference_time:.3f}s")
                    
                    # Create prediction chart
                    fig = create_prediction_chart(predictions, class_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top prediction
                    top_class = np.argmax(predictions)
                    top_confidence = predictions[top_class]
                    
                    class_name = class_names[top_class] if class_names and top_class < len(class_names) else f"Class {top_class}"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🎯 Classification Result</h3>
                        <p><strong>{class_name}</strong></p>
                        <p>Confidence: {top_confidence:.4f} ({top_confidence * 100:.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.header("📊 Text Analysis")
            
            if input_text:
                # Text statistics
                word_count = len(input_text.split())
                char_count = len(input_text)
                
                st.metric("Word Count", word_count)
                st.metric("Character Count", char_count)
                
                if 'predictions' in locals():
                    st.metric("Inference Time", f"{inference_time:.3f}s")
                    st.metric("Top Confidence", f"{np.max(predictions):.4f}")
                    
                    # Word cloud placeholder
                    st.markdown("### 🔤 Text Preview")
                    st.text_area("Input Text", input_text, height=150, disabled=True)
    
    elif app_type == "Model Analysis":
        st.header("🔍 Model Analysis")
        
        if demo_mode:
            st.info("Demo mode: Showing sample model analysis")
            
            # Sample model info
            model_info = {
                "Model Type": "Demo CNN Classifier",
                "Parameters": "2,456,789",
                "Model Size": "9.4 MB",
                "Input Shape": "(224, 224, 3)",
                "Output Classes": "10",
                "Framework": "TensorFlow 2.15.0"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Model Information")
                for key, value in model_info.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("📈 Performance Metrics")
                
                # Sample performance data
                performance_data = {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                    "Value": [0.947, 0.952, 0.943, 0.947]
                }
                
                df = pd.DataFrame(performance_data)
                
                fig = px.bar(df, x="Metric", y="Value", title="Model Performance")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Layer visualization placeholder
            st.subheader("🏗️ Model Architecture")
            
            layer_data = {
                "Layer": ["Input", "Conv2D_1", "MaxPool2D_1", "Conv2D_2", "MaxPool2D_2", "Dense_1", "Dense_2", "Output"],
                "Output Shape": ["(224, 224, 3)", "(222, 222, 32)", "(111, 111, 32)", "(109, 109, 64)", "(54, 54, 64)", "(512,)", "(128,)", "(10,)"],
                "Parameters": [0, 896, 0, 18496, 0, 1769984, 65664, 1290]
            }
            
            df_layers = pd.DataFrame(layer_data)
            st.dataframe(df_layers, use_container_width=True)
            
        else:
            if model is not None:
                st.info("Model analysis functionality would be implemented here")
                st.write("Model loaded successfully!")
            else:
                st.warning("Please load a model to view analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Built with TensorFlow, Streamlit, and ❤️</p>
        <p><em>TensorVerseHub - Comprehensive TensorFlow Learning Platform</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()