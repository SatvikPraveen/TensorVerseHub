# Location: /src/data_utils.py

"""
TensorFlow data utilities with tf.keras integration.
Provides comprehensive data pipeline creation, preprocessing, and augmentation utilities.
"""

import tensorflow as tf
import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Callable, Union
import json


class TFRecordHandler:
    """Utilities for creating, writing, and reading TFRecord files."""
    
    @staticmethod
    def _bytes_feature(value: Union[str, bytes]) -> tf.train.Feature:
        """Create a bytes feature."""
        if isinstance(value, str):
            value = value.encode('utf-8')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    @staticmethod
    def _float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
        """Create a float feature."""
        if isinstance(value, (int, float)):
            value = [float(value)]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    @staticmethod
    def _int64_feature(value: Union[int, List[int]]) -> tf.train.Feature:
        """Create an int64 feature."""
        if isinstance(value, int):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    def serialize_image_example(self, image_path: str, label: int, 
                               additional_features: Optional[Dict] = None) -> bytes:
        """
        Serialize an image example for TFRecord storage.
        
        Args:
            image_path: Path to the image file
            label: Integer label for classification
            additional_features: Optional additional features to include
            
        Returns:
            Serialized example as bytes
        """
        # Read and encode image
        image_string = tf.io.read_file(image_path)
        image_shape = tf.io.decode_image(image_string).shape
        
        feature_dict = {
            'image': self._bytes_feature(image_string.numpy()),
            'label': self._int64_feature(label),
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'channels': self._int64_feature(image_shape[2]),
            'filename': self._bytes_feature(os.path.basename(image_path))
        }
        
        if additional_features:
            for key, value in additional_features.items():
                if isinstance(value, str):
                    feature_dict[key] = self._bytes_feature(value)
                elif isinstance(value, (int, list)):
                    feature_dict[key] = self._int64_feature(value)
                elif isinstance(value, (float, list)):
                    feature_dict[key] = self._float_feature(value)
        
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()
    
    def serialize_text_example(self, text: str, label: int,
                              additional_features: Optional[Dict] = None) -> bytes:
        """
        Serialize a text example for TFRecord storage.
        
        Args:
            text: Input text string
            label: Integer label for classification
            additional_features: Optional additional features
            
        Returns:
            Serialized example as bytes
        """
        feature_dict = {
            'text': self._bytes_feature(text),
            'label': self._int64_feature(label),
            'text_length': self._int64_feature(len(text.split()))
        }
        
        if additional_features:
            for key, value in additional_features.items():
                if isinstance(value, str):
                    feature_dict[key] = self._bytes_feature(value)
                elif isinstance(value, (int, list)):
                    feature_dict[key] = self._int64_feature(value)
                elif isinstance(value, (float, list)):
                    feature_dict[key] = self._float_feature(value)
        
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()
    
    def write_tfrecord(self, examples: List[bytes], output_path: str) -> None:
        """
        Write serialized examples to a TFRecord file.
        
        Args:
            examples: List of serialized examples
            output_path: Path for output TFRecord file
        """
        with tf.io.TFRecordWriter(output_path) as writer:
            for example in examples:
                writer.write(example)
        print(f"Written {len(examples)} examples to {output_path}")


class DataPipeline:
    """Comprehensive data pipeline builder with tf.keras integration."""
    
    def __init__(self, batch_size: int = 32, shuffle_buffer: int = 1000):
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.tfrecord_handler = TFRecordHandler()
    
    def create_image_dataset(self, image_paths: List[str], labels: List[int],
                           image_size: Tuple[int, int] = (224, 224),
                           num_channels: int = 3,
                           augment: bool = True,
                           cache: bool = True) -> tf.data.Dataset:
        """
        Create an optimized image dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Corresponding labels
            image_size: Target image size (height, width)
            num_channels: Number of image channels
            augment: Whether to apply data augmentation
            cache: Whether to cache dataset in memory
            
        Returns:
            Configured tf.data.Dataset
        """
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Load and decode images
        def load_and_preprocess_image(path: str, label: int) -> Tuple[tf.Tensor, int]:
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image, channels=num_channels)
            image = tf.image.resize(image, image_size)
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
            return image, label
        
        dataset = dataset.map(
            load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply data augmentation
        if augment:
            dataset = dataset.map(
                self._apply_image_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Cache dataset
        if cache:
            dataset = dataset.cache()
        
        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _apply_image_augmentation(self, image: tf.Tensor, label: int) -> Tuple[tf.Tensor, int]:
        """Apply random image augmentations."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random saturation and hue (for color images)
        if image.shape[-1] == 3:
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)
        
        # Random rotation (small angles)
        angle = tf.random.uniform([], -0.1, 0.1)  # ±5.7 degrees
        image = tf.contrib.image.rotate(image, angle)
        
        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def create_text_dataset(self, texts: List[str], labels: List[int],
                          max_length: int = 128,
                          vocab_size: int = 10000) -> Tuple[tf.data.Dataset, tf.keras.layers.TextVectorization]:
        """
        Create an optimized text dataset with preprocessing.
        
        Args:
            texts: List of text strings
            labels: Corresponding labels
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
            
        Returns:
            Tuple of (dataset, text_vectorizer)
        """
        # Create text vectorization layer
        text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            standardize='lower_and_strip_punctuation',
            split='whitespace'
        )
        
        # Adapt vectorizer to the text data
        text_vectorizer.adapt(texts)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        
        # Apply text vectorization
        def vectorize_text(text: str, label: int) -> Tuple[tf.Tensor, int]:
            return text_vectorizer(text), label
        
        dataset = dataset.map(
            vectorize_text,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset, text_vectorizer
    
    def create_tfrecord_dataset(self, tfrecord_files: List[str],
                               feature_description: Dict,
                               parse_fn: Optional[Callable] = None) -> tf.data.Dataset:
        """
        Create dataset from TFRecord files.
        
        Args:
            tfrecord_files: List of TFRecord file paths
            feature_description: Description of features in TFRecord
            parse_fn: Optional custom parsing function
            
        Returns:
            Parsed tf.data.Dataset
        """
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        
        # Default parsing function
        if parse_fn is None:
            def parse_fn(example_proto):
                return tf.io.parse_single_example(example_proto, feature_description)
        
        # Parse examples
        dataset = dataset.map(
            parse_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_mixed_precision_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Configure dataset for mixed precision training.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset configured for mixed precision
        """
        def cast_to_mixed_precision(features, labels):
            # Cast features to float16 for mixed precision
            if isinstance(features, tf.Tensor):
                features = tf.cast(features, tf.float16)
            elif isinstance(features, dict):
                features = {k: tf.cast(v, tf.float16) if v.dtype == tf.float32 else v 
                           for k, v in features.items()}
            
            # Labels typically stay as int32/int64
            return features, labels
        
        return dataset.map(
            cast_to_mixed_precision,
            num_parallel_calls=tf.data.AUTOTUNE
        )


class DataAugmentation:
    """Advanced data augmentation utilities."""
    
    @staticmethod
    def create_augmentation_layer() -> tf.keras.Sequential:
        """
        Create a tf.keras Sequential model for data augmentation.
        
        Returns:
            Sequential model with augmentation layers
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
    
    @staticmethod
    def mixup(dataset: tf.data.Dataset, alpha: float = 0.2) -> tf.data.Dataset:
        """
        Apply MixUp augmentation to dataset.
        
        Args:
            dataset: Input dataset
            alpha: MixUp interpolation parameter
            
        Returns:
            Dataset with MixUp applied
        """
        def mixup_fn(batch_x, batch_y):
            batch_size = tf.shape(batch_x)[0]
            
            # Generate lambda from Beta distribution
            lam = tf.random.gamma([batch_size, 1, 1, 1], alpha, 1.0)
            lam = lam / (lam + tf.random.gamma([batch_size, 1, 1, 1], alpha, 1.0))
            
            # Shuffle indices
            indices = tf.random.shuffle(tf.range(batch_size))
            x_shuffled = tf.gather(batch_x, indices)
            y_shuffled = tf.gather(batch_y, indices)
            
            # Mix inputs and labels
            mixed_x = lam * batch_x + (1 - lam) * x_shuffled
            mixed_y = tf.squeeze(lam[:, 0, 0, 0:1]) * tf.cast(batch_y, tf.float32) + \
                     (1 - tf.squeeze(lam[:, 0, 0, 0:1])) * tf.cast(y_shuffled, tf.float32)
            
            return mixed_x, mixed_y
        
        return dataset.map(mixup_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    @staticmethod
    def cutmix(dataset: tf.data.Dataset, alpha: float = 1.0) -> tf.data.Dataset:
        """
        Apply CutMix augmentation to dataset.
        
        Args:
            dataset: Input dataset
            alpha: CutMix parameter
            
        Returns:
            Dataset with CutMix applied
        """
        def cutmix_fn(batch_x, batch_y):
            batch_size = tf.shape(batch_x)[0]
            image_h, image_w = tf.shape(batch_x)[1], tf.shape(batch_x)[2]
            
            # Generate lambda
            lam = tf.random.beta([batch_size], alpha, alpha)
            
            # Generate random bounding box
            cut_rat = tf.sqrt(1.0 - lam)
            cut_w = tf.cast(cut_rat * tf.cast(image_w, tf.float32), tf.int32)
            cut_h = tf.cast(cut_rat * tf.cast(image_h, tf.float32), tf.int32)
            
            cx = tf.random.uniform([batch_size], 0, image_w, dtype=tf.int32)
            cy = tf.random.uniform([batch_size], 0, image_h, dtype=tf.int32)
            
            # Calculate box coordinates
            x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_w)
            y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_h)
            x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_w)
            y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_h)
            
            # Shuffle and apply CutMix
            indices = tf.random.shuffle(tf.range(batch_size))
            x_shuffled = tf.gather(batch_x, indices)
            y_shuffled = tf.gather(batch_y, indices)
            
            # Create mask and apply CutMix (simplified version)
            # Note: Full implementation would require more complex indexing
            mixed_x = batch_x  # Placeholder - full implementation needed
            
            # Adjust lambda based on actual cut area
            lam_adjusted = 1.0 - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) / 
                                tf.cast(image_h * image_w, tf.float32))
            
            mixed_y = lam_adjusted * tf.cast(batch_y, tf.float32) + \
                     (1 - lam_adjusted) * tf.cast(y_shuffled, tf.float32)
            
            return mixed_x, mixed_y
        
        return dataset.map(cutmix_fn, num_parallel_calls=tf.data.AUTOTUNE)


def create_feature_description_image() -> Dict:
    """Create feature description for image TFRecords."""
    return {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'filename': tf.io.FixedLenFeature([], tf.string)
    }


def create_feature_description_text() -> Dict:
    """Create feature description for text TFRecords."""
    return {
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'text_length': tf.io.FixedLenFeature([], tf.int64)
    }


def parse_image_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse image TFRecord example.
    
    Args:
        example_proto: Serialized example
        
    Returns:
        Tuple of (image, label)
    """
    feature_description = create_feature_description_image()
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode image
    image = tf.io.decode_image(features['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Get label
    label = tf.cast(features['label'], tf.int32)
    
    return image, label


def parse_text_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse text TFRecord example.
    
    Args:
        example_proto: Serialized example
        
    Returns:
        Tuple of (text, label)
    """
    feature_description = create_feature_description_text()
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Get text and label
    text = features['text']
    label = tf.cast(features['label'], tf.int32)
    
    return text, label


# Convenience functions for common use cases
def create_image_classification_pipeline(
    image_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    validation_split: float = 0.2,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create train/validation datasets from image directory.
    
    Args:
        image_dir: Directory containing class subdirectories
        batch_size: Batch size for datasets
        image_size: Target image size
        validation_split: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    # Use tf.keras.utils.image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    
    # Normalize images to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Optimize performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds


def create_text_classification_pipeline(
    texts: List[str],
    labels: List[int],
    batch_size: int = 32,
    max_tokens: int = 10000,
    sequence_length: int = 128,
    validation_split: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.keras.layers.TextVectorization]:
    """
    Create train/validation datasets for text classification.
    
    Args:
        texts: List of text strings
        labels: Corresponding labels
        batch_size: Batch size for datasets
        max_tokens: Maximum vocabulary size
        sequence_length: Maximum sequence length
        validation_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_dataset, validation_dataset, text_vectorizer)
    """
    # Split data
    split_idx = int(len(texts) * (1 - validation_split))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create text vectorizer
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=sequence_length
    )
    
    # Adapt on all text data
    text_vectorizer.adapt(texts)
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    
    # Apply text vectorization
    def vectorize_text(text, label):
        return text_vectorizer(text), label
    
    train_ds = train_ds.map(vectorize_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(vectorize_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, text_vectorizer