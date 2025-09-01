# Location: /src/visualization.py

"""
Visualization utilities for TensorFlow models and training analysis.
Provides comprehensive plotting, model visualization, and analysis tools.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Dict, Any, Union
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os


class ModelVisualization:
    """Model architecture and component visualization utilities."""
    
    @staticmethod
    def plot_model_architecture(model: tf.keras.Model,
                               save_path: Optional[str] = None,
                               show_shapes: bool = True,
                               show_layer_names: bool = True,
                               rankdir: str = 'TB',
                               dpi: int = 96) -> None:
        """
        Plot model architecture diagram.
        
        Args:
            model: tf.keras model to visualize
            save_path: Path to save the plot
            show_shapes: Whether to show layer shapes
            show_layer_names: Whether to show layer names
            rankdir: Direction of plot ('TB', 'LR')
            dpi: Resolution of the plot
        """
        tf.keras.utils.plot_model(
            model,
            to_file=save_path or 'model_architecture.png',
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            rankdir=rankdir,
            expand_nested=True,
            dpi=dpi
        )
        
        if save_path:
            print(f"Model architecture saved to {save_path}")
        else:
            print("Model architecture saved to model_architecture.png")
    
    @staticmethod
    def visualize_layer_outputs(model: tf.keras.Model,
                               input_image: np.ndarray,
                               layer_names: Optional[List[str]] = None,
                               max_images_per_row: int = 16) -> None:
        """
        Visualize outputs of intermediate layers.
        
        Args:
            model: tf.keras model
            input_image: Input image to process
            layer_names: List of layer names to visualize
            max_images_per_row: Maximum feature maps per row
        """
        if layer_names is None:
            # Get all convolutional layers
            layer_names = [layer.name for layer in model.layers 
                          if isinstance(layer, tf.keras.layers.Conv2D)]
        
        # Create models that output intermediate representations
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(np.expand_dims(input_image, axis=0))
        
        # Plot activations
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            n_cols = min(max_images_per_row, n_features)
            n_rows = n_features // n_cols + (1 if n_features % n_cols else 0)
            
            plt.figure(figsize=(n_cols * 2, n_rows * 2))
            plt.suptitle(f'Layer: {layer_name}', fontsize=16)
            
            for i in range(n_features):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
                plt.axis('off')
                plt.title(f'Feature {i}')
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_filter_weights(model: tf.keras.Model,
                           layer_name: str,
                           max_filters: int = 64) -> None:
        """
        Visualize convolutional layer filter weights.
        
        Args:
            model: tf.keras model
            layer_name: Name of convolutional layer
            max_filters: Maximum number of filters to show
        """
        layer = model.get_layer(layer_name)
        
        if not isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Layer {layer_name} is not a Conv2D layer")
            return
        
        weights = layer.get_weights()[0]  # Get kernel weights
        
        # Normalize weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        n_filters = min(weights.shape[-1], max_filters)
        n_cols = 8
        n_rows = n_filters // n_cols + (1 if n_filters % n_cols else 0)
        
        plt.figure(figsize=(n_cols * 2, n_rows * 2))
        plt.suptitle(f'Filters from layer: {layer_name}', fontsize=16)
        
        for i in range(n_filters):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Handle different kernel sizes and channels
            if weights.shape[2] == 1:  # Grayscale
                plt.imshow(weights[:, :, 0, i], cmap='viridis')
            else:  # RGB - show first 3 channels
                filter_img = weights[:, :, :min(3, weights.shape[2]), i]
                if filter_img.shape[2] == 1:
                    plt.imshow(filter_img[:, :, 0], cmap='viridis')
                else:
                    plt.imshow(filter_img)
            
            plt.axis('off')
            plt.title(f'Filter {i}')
        
        plt.tight_layout()
        plt.show()


class TrainingVisualization:
    """Training process visualization utilities."""
    
    @staticmethod
    def plot_training_history(history: tf.keras.callbacks.History,
                             metrics: Optional[List[str]] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot training history with multiple metrics.
        
        Args:
            history: Keras training history
            metrics: List of metrics to plot
            save_path: Path to save the plot
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in history.history]
        
        if not available_metrics:
            print("No valid metrics found in history")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Plot training metric
            epochs = range(1, len(history.history[metric]) + 1)
            ax.plot(epochs, history.history[metric], 'b-', label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(epochs, history.history[val_metric], 'r-', label=f'Validation {metric}')
            
            ax.set_title(f'{metric.capitalize()} Over Epochs')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_sizes: np.ndarray,
                           train_scores: np.ndarray,
                           val_scores: np.ndarray,
                           metric_name: str = 'Accuracy',
                           save_path: Optional[str] = None) -> None:
        """
        Plot learning curves showing performance vs training set size.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            metric_name: Name of the metric being plotted
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        color='blue', alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        color='red', alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel(metric_name)
        plt.title(f'Learning Curves - {metric_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = False,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
            class_names: List of class names
            save_path: Path to save the plot
        """
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        n_classes = y_true.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve for each class
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-Class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()


class DataVisualization:
    """Data exploration and visualization utilities."""
    
    @staticmethod
    def plot_image_samples(images: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          class_names: Optional[List[str]] = None,
                          num_samples: int = 25,
                          figsize: Tuple[int, int] = (10, 10)) -> None:
        """
        Plot sample images from dataset.
        
        Args:
            images: Array of images
            labels: Corresponding labels
            class_names: List of class names
            num_samples: Number of samples to show
            figsize: Figure size
        """
        n_cols = int(np.sqrt(num_samples))
        n_rows = num_samples // n_cols + (1 if num_samples % n_cols else 0)
        
        plt.figure(figsize=figsize)
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Handle different image formats
            img = images[i]
            if len(img.shape) == 3:
                if img.shape[-1] == 1:  # Grayscale
                    plt.imshow(img.squeeze(), cmap='gray')
                else:  # RGB
                    plt.imshow(img)
            else:  # 2D grayscale
                plt.imshow(img, cmap='gray')
            
            plt.axis('off')
            
            # Add label if available
            if labels is not None:
                label = labels[i]
                if class_names and label < len(class_names):
                    title = class_names[int(label)]
                else:
                    title = f'Label: {label}'
                plt.title(title, fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_class_distribution(labels: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot distribution of classes in dataset.
        
        Args:
            labels: Array of labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(range(len(unique)), counts)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        
        # Set x-tick labels
        if class_names:
            plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
        else:
            plt.xticks(range(len(unique)), unique)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_data_augmentation_samples(original_image: np.ndarray,
                                     augmentation_layer: tf.keras.layers.Layer,
                                     num_samples: int = 9) -> None:
        """
        Show samples of data augmentation applied to an image.
        
        Args:
            original_image: Original image
            augmentation_layer: tf.keras augmentation layer
            num_samples: Number of augmented samples to show
        """
        n_cols = 3
        n_rows = (num_samples + 2) // n_cols  # +2 for original
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        # Show original
        plt.subplot(n_rows, n_cols, 1)
        if len(original_image.shape) == 3 and original_image.shape[-1] == 1:
            plt.imshow(original_image.squeeze(), cmap='gray')
        else:
            plt.imshow(original_image)
        plt.title('Original')
        plt.axis('off')
        
        # Show augmented samples
        batch_image = np.expand_dims(original_image, axis=0)
        
        for i in range(num_samples):
            plt.subplot(n_rows, n_cols, i + 2)
            
            # Apply augmentation
            augmented = augmentation_layer(batch_image, training=True)
            augmented_img = augmented[0].numpy()
            
            if len(augmented_img.shape) == 3 and augmented_img.shape[-1] == 1:
                plt.imshow(augmented_img.squeeze(), cmap='gray')
            else:
                plt.imshow(augmented_img)
            
            plt.title(f'Augmented {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


class AdvancedVisualization:
    """Advanced visualization techniques for model interpretation."""
    
    @staticmethod
    def plot_gradient_flow(model: tf.keras.Model,
                          named_parameters: List[Tuple[str, tf.Variable]]) -> None:
        """
        Plot gradient flow through the network layers.
        
        Args:
            model: tf.keras model
            named_parameters: List of (name, parameter) tuples with gradients
        """
        ave_grads = []
        layers = []
        
        for name, param in named_parameters:
            if param.gradient is not None:
                layers.append(name)
                ave_grads.append(param.gradient.numpy().mean())
        
        plt.figure(figsize=(12, 6))
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_training_dashboard(history: tf.keras.callbacks.History,
                                model: tf.keras.Model,
                                validation_data: tf.data.Dataset,
                                save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive training dashboard.
        
        Args:
            history: Training history
            model: Trained model
            validation_data: Validation dataset
            save_path: Path to save the dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Training history plots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Loss plot
        ax1 = fig.add_subplot(gs[0, 0:2])
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = fig.add_subplot(gs[0, 2:4])
        if 'accuracy' in history.history:
            ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in history.history:
                ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Model architecture summary (text)
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.axis('off')
        
        # Get model summary as string
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        summary_text = '\n'.join(summary_str[:15])  # First 15 lines
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax3.set_title('Model Architecture Summary')
        
        # Sample predictions (if validation data available)
        ax4 = fig.add_subplot(gs[1, 2:4])
        try:
            # Get a batch of validation data
            for x_val, y_val in validation_data.take(1):
                predictions = model.predict(x_val[:9])  # First 9 samples
                
                # Simple visualization of first few samples
                ax4.set_title('Sample Predictions vs True Labels')
                ax4.axis('off')
                
                pred_text = "Sample predictions:\n"
                for i in range(min(9, len(predictions))):
                    pred_class = np.argmax(predictions[i])
                    true_class = y_val[i].numpy() if hasattr(y_val[i], 'numpy') else y_val[i]
                    pred_text += f"Sample {i+1}: Pred={pred_class}, True={true_class}\n"
                
                ax4.text(0.05, 0.95, pred_text, transform=ax4.transAxes,
                        verticalalignment='top', fontsize=10)
                break
        except Exception as e:
            ax4.text(0.5, 0.5, f"Could not generate predictions:\n{str(e)[:50]}...",
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Sample Predictions (Error)')
        
        # Training metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create metrics summary
        final_metrics = {}
        for key, values in history.history.items():
            if values:
                final_metrics[key] = values[-1]
        
        metrics_text = "Final Training Metrics:\n"
        for key, value in final_metrics.items():
            metrics_text += f"{key}: {value:.4f}  "
        
        ax5.text(0.5, 0.5, metrics_text, transform=ax5.transAxes,
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training dashboard saved to {save_path}")
        
        plt.show()


# Convenience function for quick visualization
def quick_model_analysis(model: tf.keras.Model,
                        history: tf.keras.callbacks.History,
                        test_data: Optional[tf.data.Dataset] = None,
                        save_dir: Optional[str] = None) -> None:
    """
    Perform quick analysis and visualization of trained model.
    
    Args:
        model: Trained tf.keras model
        history: Training history
        test_data: Test dataset for evaluation
        save_dir: Directory to save visualizations
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot training history
    TrainingVisualization.plot_training_history(
        history,
        save_path=os.path.join(save_dir, 'training_history.png') if save_dir else None
    )
    
    # Plot model architecture
    ModelVisualization.plot_model_architecture(
        model,
        save_path=os.path.join(save_dir, 'model_architecture.png') if save_dir else None
    )
    
    # If test data is available, create confusion matrix
    if test_data:
        try:
            y_true = []
            y_pred = []
            
            for x, y in test_data:
                predictions = model.predict(x, verbose=0)
                y_pred.extend(np.argmax(predictions, axis=1))
                y_true.extend(y.numpy())
            
            TrainingVisualization.plot_confusion_matrix(
                np.array(y_true),
                np.array(y_pred),
                save_path=os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
            )
            
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
    
    print("Model analysis complete!")


def setup_plotting_style():
    """Setup consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set default figure parameters
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.size': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })
    
    print("Plotting style configured for TensorVerseHub visualizations")