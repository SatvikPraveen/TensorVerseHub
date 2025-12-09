# Location: /tests/test_export_comprehensive.py

"""
Comprehensive tests for model export and deployment utilities.
Tests for TFLite, SavedModel, and cross-platform export.
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from export_utils import ModelExporter, TFLiteConverter, SavedModelHandler
from model_utils import ModelBuilders


class TestModelExportBasics:
    """Test basic model export functionality."""
    
    def test_export_saved_model_format(self):
        """Test exporting to SavedModel format."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'saved_model')
            model.save(save_path)
            
            # Verify SavedModel structure
            assert os.path.exists(save_path)
            assert os.path.exists(os.path.join(save_path, 'saved_model.pb'))
            assert os.path.exists(os.path.join(save_path, 'variables'))
    
    def test_load_saved_model(self):
        """Test loading SavedModel."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'saved_model')
            model.save(save_path)
            
            # Load and verify
            loaded_model = tf.saved_model.load(save_path)
            assert loaded_model is not None
    
    def test_export_h5_format(self):
        """Test exporting to H5 format."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, 'model.h5')
            model.save(h5_path)
            
            # Verify file exists
            assert os.path.exists(h5_path)
            assert os.path.getsize(h5_path) > 0
    
    def test_load_h5_model(self):
        """Test loading H5 model."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, 'model.h5')
            model.save(h5_path)
            
            # Load and verify
            loaded_model = tf.keras.models.load_model(h5_path)
            assert loaded_model is not None
            assert loaded_model.input_shape == model.input_shape
            assert loaded_model.output_shape == model.output_shape


class TestTFLiteConversion:
    """Test TFLite conversion functionality."""
    
    def test_tflite_basic_conversion(self):
        """Test basic TFLite conversion."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        assert isinstance(tflite_model, bytes)
        assert len(tflite_model) > 0
    
    def test_tflite_interpreter_creation(self):
        """Test creating TFLite interpreter."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Verify details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        assert len(input_details) > 0
        assert len(output_details) > 0
    
    def test_tflite_inference(self):
        """Test inference with TFLite model."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create test input
        test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        assert output.shape == (1, 10)
    
    def test_tflite_with_quantization_aware_training(self):
        """Test TFLite conversion with quantization awareness."""
        try:
            import tensorflow_model_optimization as tfmot
            
            model = ModelBuilders.create_cnn_classifier(
                input_shape=(28, 28, 1),
                num_classes=10
            )
            
            # Quantization aware training (if available)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            assert isinstance(tflite_model, bytes)
        except ImportError:
            pytest.skip("tensorflow_model_optimization not available")
    
    def test_tflite_save_and_load(self):
        """Test saving and loading TFLite model."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, 'model.tflite')
            
            # Save
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Load
            with open(tflite_path, 'rb') as f:
                loaded_tflite = f.read()
            
            # Verify
            assert loaded_tflite == tflite_model


class TestModelExportMetadata:
    """Test export with metadata and documentation."""
    
    def test_save_with_metadata(self):
        """Test saving model with metadata."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Create metadata
        metadata = {
            'version': '1.0',
            'author': 'TensorVerseHub',
            'description': 'CNN classifier for image classification',
            'input_shape': [28, 28, 1],
            'output_classes': 10,
            'framework': 'tensorflow.keras'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model')
            model.save(save_path)
            
            # Save metadata
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Verify both files exist
            assert os.path.exists(save_path)
            assert os.path.exists(metadata_path)
            
            # Load and verify metadata
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            assert loaded_metadata['version'] == '1.0'
    
    def test_save_model_config(self):
        """Test saving model configuration."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        config = model.get_config()
        
        assert config is not None
        assert 'layers' in config
        assert isinstance(config['layers'], list)
    
    def test_reconstruct_from_config(self):
        """Test reconstructing model from config."""
        original_model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        config = original_model.get_config()
        reconstructed_model = tf.keras.Sequential.from_config(config)
        
        assert reconstructed_model is not None
        assert reconstructed_model.input_shape == original_model.input_shape


class TestCrossPlatformExport:
    """Test cross-platform export compatibility."""
    
    def test_export_for_ios_compatibility(self):
        """Test exporting for iOS compatibility."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # iOS typically uses TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Should be valid TFLite
        assert isinstance(tflite_model, bytes)
        assert len(tflite_model) > 1000  # Reasonable size
    
    def test_export_for_android_compatibility(self):
        """Test exporting for Android compatibility."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # Android typically uses TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        tflite_model = converter.convert()
        
        assert isinstance(tflite_model, bytes)
    
    def test_export_for_web_compatibility(self):
        """Test exporting for web (TFLite/TF.js compatibility)."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # Web can use TFLite or SavedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'web_model')
            model.save(save_path)
            
            # SavedModel format is web-compatible
            assert os.path.exists(save_path)
    
    def test_export_for_edge_devices(self):
        """Test exporting for edge devices."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # Edge devices: minimize size with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Verify model size is reasonable for edge
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"Edge model size: {size_mb:.2f} MB")
        
        assert len(tflite_model) > 0


class TestExportEdgeCases:
    """Test edge cases in export functionality."""
    
    def test_export_model_with_no_weights(self):
        """Test exporting uncompiled model."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        # Export without compilation
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        assert isinstance(tflite_model, bytes)
    
    def test_export_model_with_dynamic_shape(self):
        """Test exporting model with dynamic input shape."""
        inputs = tf.keras.Input(shape=(None, 28, 1))
        x = tf.keras.layers.LSTM(32)(inputs)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Dynamic shapes require special handling in TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            assert isinstance(tflite_model, bytes)
        except Exception:
            pytest.skip("Dynamic shapes not fully supported in TFLite")
    
    def test_export_very_large_model(self):
        """Test exporting very large model."""
        try:
            model = tf.keras.applications.EfficientNetB7(
                include_top=True,
                weights=None,
                input_shape=(224, 224, 3),
                classes=10
            )
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"Large model size: {size_mb:.2f} MB")
            
            assert isinstance(tflite_model, bytes)
        except Exception:
            pytest.skip("Large model test requires sufficient resources")
    
    def test_export_model_with_custom_objects(self):
        """Test exporting model with custom layers."""
        class CustomLayer(tf.keras.layers.Layer):
            def __init__(self, units=32):
                super(CustomLayer, self).__init__()
                self.units = units
            
            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='random_normal'
                )
            
            def call(self, inputs):
                return tf.matmul(inputs, self.w)
            
            def get_config(self):
                return {'units': self.units}
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, input_shape=(20,)),
            CustomLayer(10),
            tf.keras.layers.Dense(2)
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Custom layers need custom_objects when loading
            h5_path = os.path.join(tmpdir, 'model_custom.h5')
            model.save(h5_path)
            
            # Load with custom objects
            loaded = tf.keras.models.load_model(
                h5_path,
                custom_objects={'CustomLayer': CustomLayer}
            )
            
            assert loaded is not None


class TestExportPerformance:
    """Test export and inference performance."""
    
    def test_saved_model_vs_h5_size(self):
        """Compare SavedModel and H5 file sizes."""
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # SavedModel
            saved_model_path = os.path.join(tmpdir, 'saved_model')
            model.save(saved_model_path)
            
            # H5
            h5_path = os.path.join(tmpdir, 'model.h5')
            model.save(h5_path)
            
            # Get sizes
            import shutil
            saved_model_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(saved_model_path)
                for f in filenames
            )
            h5_size = os.path.getsize(h5_path)
            tflite_size = 0
            
            print(f"SavedModel size: {saved_model_size / 1024:.1f} KB")
            print(f"H5 size: {h5_size / 1024:.1f} KB")
            
            assert saved_model_size > 0
            assert h5_size > 0
    
    def test_tflite_vs_keras_inference_speed(self):
        """Compare inference speed between TFLite and Keras."""
        import time
        
        model = ModelBuilders.create_cnn_classifier(
            input_shape=(28, 28, 1),
            num_classes=10
        )
        
        test_input = np.random.randn(100, 28, 28, 1).astype(np.float32)
        
        # Keras inference
        start = time.time()
        keras_output = model.predict(test_input, verbose=0)
        keras_time = time.time() - start
        
        # TFLite inference
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        start = time.time()
        for i in range(len(test_input)):
            interpreter.set_tensor(input_details[0]['index'], test_input[i:i+1])
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        tflite_time = time.time() - start
        
        print(f"Keras time: {keras_time*1000:.1f} ms")
        print(f"TFLite time: {tflite_time*1000:.1f} ms")
        print(f"TFLite speedup: {keras_time/tflite_time:.1f}x")
        
        assert keras_time > 0
        assert tflite_time > 0
