"""
MeowTrix-AI Classifier Module
Implements SVM and Random Forest classifiers for deepfake detection
with hyperparameter tuning and model evaluation capabilities
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

class MeowTrixClassifier:
    """
    Machine Learning classifier for deepfake detection using SVM and Random Forest
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.logger = self._setup_logger()

        # Define model configurations
        self.model_configs = {
            'svm_rbf': {
                'model': SVC(probability=True, random_state=random_state),
                'params': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'classifier__kernel': ['rbf']
                }
            },
            'svm_linear': {
                'model': SVC(probability=True, random_state=random_state),
                'params': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__kernel': ['linear']
                }
            },
            'svm_poly': {
                'model': SVC(probability=True, random_state=random_state),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__degree': [2, 3, 4],
                    'classifier__kernel': ['poly']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'classifier__n_estimators': [50, 100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__bootstrap': [True, False]
                }
            }
        }

        self.logger.info("MeowTrix Classifier initialized")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the classifier"""
        logger = logging.getLogger('MeowTrix.Classifier')
        logger.setLevel(logging.INFO)
        return logger

    def create_pipeline(self, model) -> Pipeline:
        """
        Create a preprocessing and classification pipeline

        Args:
            model: The classifier model

        Returns:
            Complete pipeline with scaler and classifier
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

    def train_single_model(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          model_name: str,
                          cv_folds: int = 5,
                          scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model to train
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for hyperparameter tuning

        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Training {model_name} model...")

            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")

            config = self.model_configs[model_name]
            pipeline = self.create_pipeline(config['model'])

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                config['params'],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )

            # Fit the model
            grid_search.fit(X_train, y_train)

            # Store the best model
            self.models[model_name] = grid_search.best_estimator_

            # Calculate cross-validation scores
            cv_scores = cross_val_score(
                grid_search.best_estimator_, 
                X_train, 
                y_train, 
                cv=cv_folds,
                scoring=scoring
            )

            results = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            self.logger.info(f"{model_name} training completed:")
            self.logger.info(f"  Best CV Score: {grid_search.best_score_:.4f}")
            self.logger.info(f"  CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Training failed for {model_name}: {str(e)}")
            raise

    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        models_to_train: Optional[list] = None,
                        cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train all configured models and select the best one

        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train (None for all)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with all training results
        """
        try:
            if models_to_train is None:
                models_to_train = list(self.model_configs.keys())

            results = {}
            best_score = 0

            self.logger.info(f"Training {len(models_to_train)} models...")

            for model_name in models_to_train:
                try:
                    model_results = self.train_single_model(X_train, y_train, model_name, cv_folds)
                    results[model_name] = model_results

                    # Update best model
                    if model_results['best_score'] > best_score:
                        best_score = model_results['best_score']
                        self.best_model = model_results['model']
                        self.best_model_name = model_name

                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {str(e)}")
                    continue

            if self.best_model is not None:
                self.is_fitted = True
                self.logger.info(f"Best model: {self.best_model_name} (score: {best_score:.4f})")
            else:
                self.logger.error("No models were successfully trained")

            return results

        except Exception as e:
            self.logger.error(f"Training all models failed: {str(e)}")
            raise

    def predict(self, X: np.ndarray, use_best_model: bool = True, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using trained model

        Args:
            X: Feature matrix
            use_best_model: Whether to use the best model
            model_name: Specific model name to use

        Returns:
            Prediction array
        """
        try:
            if use_best_model and self.best_model is not None:
                model = self.best_model
                used_model = self.best_model_name
            elif model_name and model_name in self.models:
                model = self.models[model_name]
                used_model = model_name
            else:
                raise ValueError("No valid model available for prediction")

            predictions = model.predict(X)
            self.logger.info(f"Predictions made using {used_model}")
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_proba(self, X: np.ndarray, use_best_model: bool = True, model_name: Optional[str] = None) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            X: Feature matrix
            use_best_model: Whether to use the best model
            model_name: Specific model name to use

        Returns:
            Probability matrix
        """
        try:
            if use_best_model and self.best_model is not None:
                model = self.best_model
                used_model = self.best_model_name
            elif model_name and model_name in self.models:
                model = self.models[model_name]
                used_model = model_name
            else:
                raise ValueError("No valid model available for prediction")

            probabilities = model.predict_proba(X)
            self.logger.info(f"Probabilities computed using {used_model}")
            return probabilities

        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation

        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Specific model to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Make predictions
            y_pred = self.predict(X_test, model_name=model_name)
            y_proba = self.predict_proba(X_test, model_name=model_name)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_proba[:, 1])
            }

            self.logger.info("Model evaluation completed:")
            for metric, value in metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def save_model(self, filepath: str, model_name: Optional[str] = None) -> None:
        """
        Save trained model to file

        Args:
            filepath: Path to save the model
            model_name: Specific model to save (None for best model)
        """
        try:
            if model_name and model_name in self.models:
                model_to_save = self.models[model_name]
                save_name = model_name
            elif self.best_model is not None:
                model_to_save = self.best_model
                save_name = self.best_model_name
            else:
                raise ValueError("No model available to save")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save model and metadata
            save_data = {
                'model': model_to_save,
                'model_name': save_name,
                'is_fitted': self.is_fitted
            }

            joblib.dump(save_data, filepath)
            self.logger.info(f"Model {save_name} saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file

        Args:
            filepath: Path to load the model from
        """
        try:
            save_data = joblib.load(filepath)

            self.best_model = save_data['model']
            self.best_model_name = save_data['model_name']
            self.is_fitted = save_data['is_fitted']

            # Also store in models dict
            self.models[self.best_model_name] = self.best_model

            self.logger.info(f"Model {self.best_model_name} loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get feature importance if available

        Args:
            model_name: Specific model name

        Returns:
            Feature importance array or None
        """
        try:
            if model_name and model_name in self.models:
                model = self.models[model_name]
            elif self.best_model is not None:
                model = self.best_model
            else:
                return None

            # Check if model has feature importance
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                return classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                return np.abs(classifier.coef_[0])
            else:
                return None

        except Exception as e:
            self.logger.error(f"Feature importance extraction failed: {str(e)}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about trained models

        Returns:
            Dictionary with model information
        """
        return {
            'available_models': list(self.models.keys()),
            'best_model': self.best_model_name,
            'is_fitted': self.is_fitted,
            'model_configs': list(self.model_configs.keys())
        }
