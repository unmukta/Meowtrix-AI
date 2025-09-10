#!/usr/bin/env python3
"""
MeowTrix-AI Training Script
Complete training pipeline for deepfake detection system
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import logging
from typing import Tuple, Dict, List, Any, Optional
import warnings
import time
from datetime import datetime

# Import MeowTrix modules
from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from classifier import MeowTrixClassifier
from evaluation import ModelEvaluator

# Suppress warnings
warnings.filterwarnings('ignore')

class MeowTrixTrainer:
    """
    Complete training pipeline for MeowTrix-AI deepfake detection system
    """

    def __init__(self, config_path: str = "meowtrix_config.json"):
        """
        Initialize trainer with configuration

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.setup_logging()

        # Initialize components
        self.image_processor = ImageProcessor(
            target_size=tuple(self.config['model_settings']['image_size'])
        )

        self.feature_extractor = FeatureExtractor(
            lbp_radius=self.config['feature_extraction']['lbp_radius'],
            lbp_n_points=self.config['feature_extraction']['lbp_n_points'],
            hog_orientations=self.config['feature_extraction']['hog_orientations'],
            hog_pixels_per_cell=tuple(self.config['feature_extraction']['hog_pixels_per_cell']),
            hog_cells_per_block=tuple(self.config['feature_extraction']['hog_cells_per_block'])
        )

        self.classifier = MeowTrixClassifier(
            random_state=self.config['model_settings']['random_state']
        )

        self.evaluator = ModelEvaluator(save_plots=True)

        self.logger.info("MeowTrix-AI Trainer initialized")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default config
            return {
                "model_settings": {
                    "image_size": [128, 128],
                    "batch_size": 32,
                    "validation_split": 0.15,
                    "test_split": 0.15,
                    "random_state": 42
                },
                "feature_extraction": {
                    "lbp_radius": 1,
                    "lbp_n_points": 8,
                    "hog_orientations": 9,
                    "hog_pixels_per_cell": [16, 16],
                    "hog_cells_per_block": [2, 2]
                },
                "dataset": {
                    "real_faces_count": 5000,
                    "fake_faces_count": 5000,
                    "target_accuracy": 0.94,
                    "target_precision": 0.93,
                    "target_recall": 0.95,
                    "target_f1": 0.94
                }
            }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('meowtrix_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MeowTrix.Trainer')

    def load_images_from_directory(self, directory: str, label: int, 
                                 max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load images from directory and assign labels

        Args:
            directory: Directory containing images
            label: Label to assign (0 for fake, 1 for real)
            max_images: Maximum number of images to load

        Returns:
            Tuple of (images, labels)
        """
        try:
            images = []
            labels = []

            # Supported image extensions
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

            # Get all image files
            image_files = []
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(os.path.join(directory, file))

            # Shuffle and limit if needed
            image_files = shuffle(image_files, random_state=self.config['model_settings']['random_state'])
            if max_images:
                image_files = image_files[:max_images]

            self.logger.info(f"Loading {len(image_files)} images from {directory}")

            # Load and process images
            for i, image_path in enumerate(image_files):
                try:
                    # Load and preprocess image
                    image = self.image_processor.load_image(image_path)
                    if image is None:
                        continue

                    processed_image = self.image_processor.preprocess_image(image)

                    images.append(processed_image)
                    labels.append(label)

                    if (i + 1) % 100 == 0:
                        self.logger.info(f"Loaded {i + 1}/{len(image_files)} images")

                except Exception as e:
                    self.logger.warning(f"Failed to load {image_path}: {str(e)}")
                    continue

            self.logger.info(f"Successfully loaded {len(images)} images with label {label}")
            return images, labels

        except Exception as e:
            self.logger.error(f"Failed to load images from {directory}: {str(e)}")
            raise

    def prepare_dataset(self, real_dir: str, fake_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare complete dataset from real and fake directories

        Args:
            real_dir: Directory containing real face images
            fake_dir: Directory containing fake face images

        Returns:
            Tuple of (features, labels)
        """
        try:
            self.logger.info("Preparing dataset...")

            # Load images
            real_images, real_labels = self.load_images_from_directory(
                real_dir, label=1, max_images=self.config['dataset']['real_faces_count']
            )

            fake_images, fake_labels = self.load_images_from_directory(
                fake_dir, label=0, max_images=self.config['dataset']['fake_faces_count']
            )

            # Combine datasets
            all_images = real_images + fake_images
            all_labels = real_labels + fake_labels

            self.logger.info(f"Total dataset size: {len(all_images)} images")
            self.logger.info(f"Real images: {sum(all_labels)} ({sum(all_labels)/len(all_labels):.1%})")
            self.logger.info(f"Fake images: {len(all_labels) - sum(all_labels)} ({(len(all_labels) - sum(all_labels))/len(all_labels):.1%})")

            # Extract features
            self.logger.info("Extracting features from all images...")
            features = []

            for i, image in enumerate(all_images):
                try:
                    # Extract combined LBP + HOG features
                    image_features = self.feature_extractor.extract_combined_features(image)
                    features.append(image_features)

                    if (i + 1) % 100 == 0:
                        self.logger.info(f"Extracted features from {i + 1}/{len(all_images)} images")

                except Exception as e:
                    self.logger.warning(f"Failed to extract features from image {i}: {str(e)}")
                    continue

            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(all_labels[:len(features)])  # Match length in case some failed

            # Shuffle the dataset
            X, y = shuffle(X, y, random_state=self.config['model_settings']['random_state'])

            self.logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
            self.logger.info(f"Feature vector dimension: {X.shape[1]}")

            return X, y

        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            raise

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split dataset into train, validation, and test sets

        Args:
            X: Feature matrix
            y: Label array

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            test_size = self.config['model_settings']['test_split']
            val_size = self.config['model_settings']['validation_split']
            random_state = self.config['model_settings']['random_state']

            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Second split: separate validation from training
            # Adjust val_size for remaining data
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_temp
            )

            self.logger.info("Dataset split completed:")
            self.logger.info(f"  Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
            self.logger.info(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
            self.logger.info(f"  Testing:    {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            self.logger.error(f"Dataset splitting failed: {str(e)}")
            raise

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    models_to_train: List[str] = None) -> Dict[str, Dict]:
        """
        Train all models with cross-validation

        Args:
            X_train: Training features
            y_train: Training labels  
            models_to_train: List of model names to train

        Returns:
            Training results for all models
        """
        try:
            self.logger.info("Starting model training...")
            start_time = time.time()

            if models_to_train is None:
                models_to_train = ['svm_rbf', 'svm_linear', 'random_forest']

            # Train all models
            training_results = self.classifier.train_all_models(
                X_train, y_train, models_to_train=models_to_train, cv_folds=5
            )

            training_time = time.time() - start_time
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")

            # Log results
            self.logger.info("Training Results Summary:")
            for model_name, results in training_results.items():
                self.logger.info(f"  {model_name}: CV Score = {results['best_score']:.4f} ± {results['cv_std']:.4f}")

            self.logger.info(f"Best model: {self.classifier.best_model_name}")

            return training_results

        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       training_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test set

        Args:
            X_test: Test features
            y_test: Test labels
            training_results: Results from training phase

        Returns:
            Evaluation results for all models
        """
        try:
            self.logger.info("Starting model evaluation...")

            evaluation_results = {}

            for model_name in training_results.keys():
                try:
                    self.logger.info(f"Evaluating {model_name}...")

                    # Make predictions
                    y_pred = self.classifier.predict(X_test, model_name=model_name)
                    y_proba = self.classifier.predict_proba(X_test, model_name=model_name)[:, 1]

                    # Comprehensive evaluation
                    eval_results = self.evaluator.evaluate_model_comprehensive(
                        y_test, y_pred, y_proba, model_name
                    )

                    evaluation_results[model_name] = eval_results

                    # Print summary
                    self.evaluator.print_evaluation_summary(eval_results)

                    # Save individual report
                    self.evaluator.save_evaluation_report(
                        eval_results, f"{model_name}_evaluation_report.json"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                    continue

            # Create comparison plot
            if len(evaluation_results) > 1:
                metrics_dict = {}
                for model_name, results in evaluation_results.items():
                    metrics_dict[model_name] = results['metrics']

                self.evaluator.create_metrics_comparison_plot(
                    metrics_dict, "MeowTrix-AI Model Comparison"
                )

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def save_final_model(self, output_path: str) -> None:
        """
        Save the best trained model

        Args:
            output_path: Path to save the model
        """
        try:
            self.classifier.save_model(output_path)
            self.logger.info(f"Best model saved to {output_path}")

            # Save model info
            model_info = {
                'model_name': self.classifier.best_model_name,
                'training_timestamp': datetime.now().isoformat(),
                'config': self.config,
                'feature_info': self.feature_extractor.get_feature_info()
            }

            info_path = output_path.replace('.joblib', '_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)

            self.logger.info(f"Model info saved to {info_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def run_complete_training(self, real_dir: str, fake_dir: str, 
                             output_model: str, models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Run complete training pipeline

        Args:
            real_dir: Directory with real face images
            fake_dir: Directory with fake face images
            output_model: Path to save final model
            models_to_train: List of models to train

        Returns:
            Complete training and evaluation results
        """
        try:
            self.logger.info("🚀 Starting MeowTrix-AI complete training pipeline")
            start_time = time.time()

            # Step 1: Prepare dataset
            self.logger.info("📊 Step 1: Preparing dataset...")
            X, y = self.prepare_dataset(real_dir, fake_dir)

            # Step 2: Split dataset
            self.logger.info("✂️  Step 2: Splitting dataset...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X, y)

            # Step 3: Train models
            self.logger.info("🏋️  Step 3: Training models...")
            training_results = self.train_models(X_train, y_train, models_to_train)

            # Step 4: Evaluate models
            self.logger.info("📈 Step 4: Evaluating models...")
            evaluation_results = self.evaluate_models(X_test, y_test, training_results)

            # Step 5: Save best model
            self.logger.info("💾 Step 5: Saving best model...")
            os.makedirs(os.path.dirname(output_model), exist_ok=True)
            self.save_final_model(output_model)

            total_time = time.time() - start_time

            # Compile final results
            final_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'best_model': self.classifier.best_model_name,
                'dataset_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'feature_dimension': X.shape[1]
                },
                'training_time_seconds': total_time,
                'timestamp': datetime.now().isoformat()
            }

            # Save complete results
            results_path = output_model.replace('.joblib', '_complete_results.json')
            with open(results_path, 'w') as f:
                # Create JSON-serializable version
                json_results = final_results.copy()
                if 'evaluation_results' in json_results:
                    for model_name in json_results['evaluation_results']:
                        if 'plots' in json_results['evaluation_results'][model_name]:
                            json_results['evaluation_results'][model_name]['plots'] = list(
                                json_results['evaluation_results'][model_name]['plots'].keys()
                            )
                json.dump(json_results, f, indent=4)

            self.logger.info(f"\n🎉 Training pipeline completed successfully!")
            self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            self.logger.info(f"🏆 Best model: {self.classifier.best_model_name}")
            self.logger.info(f"📁 Model saved to: {output_model}")
            self.logger.info(f"📊 Results saved to: {results_path}")

            return final_results

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main training script entry point"""
    parser = argparse.ArgumentParser(description="MeowTrix-AI Training Pipeline")
    parser.add_argument('real_dir', help='Directory containing real face images')
    parser.add_argument('fake_dir', help='Directory containing fake/generated face images') 
    parser.add_argument('output_model', help='Path to save the trained model')
    parser.add_argument('--config', default='meowtrix_config.json', 
                       help='Configuration file path')
    parser.add_argument('--models', nargs='+', 
                       choices=['svm_rbf', 'svm_linear', 'svm_poly', 'random_forest'],
                       default=['svm_rbf', 'svm_linear', 'random_forest'],
                       help='Models to train')

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.real_dir):
        print(f"Error: Real images directory not found: {args.real_dir}")
        sys.exit(1)

    if not os.path.exists(args.fake_dir):
        print(f"Error: Fake images directory not found: {args.fake_dir}")
        sys.exit(1)

    try:
        # Initialize trainer
        trainer = MeowTrixTrainer(args.config)

        # Run complete training pipeline
        results = trainer.run_complete_training(
            args.real_dir, args.fake_dir, args.output_model, args.models
        )

        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"Best Model: {results['best_model']}")
        print(f"Training Time: {results['training_time_seconds']:.1f} seconds")
        print(f"Model saved to: {args.output_model}")
        print("="*60)

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
