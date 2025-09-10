"""
MeowTrix-AI Evaluation Module
Comprehensive model evaluation with metrics, visualization, and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

class ModelEvaluator:
    """
    Comprehensive model evaluation class for deepfake detection models
    """

    def __init__(self, save_plots: bool = True, plots_dir: str = "evaluation_plots"):
        """
        Initialize the model evaluator

        Args:
            save_plots: Whether to save generated plots
            plots_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.logger = self._setup_logger()

        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)

        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")

        self.logger.info("Model Evaluator initialized")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the evaluator"""
        logger = logging.getLogger('MeowTrix.Evaluator')
        logger.setLevel(logging.INFO)
        return logger

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary with basic metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred)
            }

            # Add AUC metrics if probabilities provided
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                except ValueError as e:
                    self.logger.warning(f"Could not calculate AUC metrics: {str(e)}")
                    metrics['roc_auc'] = 0.0
                    metrics['pr_auc'] = 0.0

            self.logger.info("Basic metrics calculated successfully")
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate basic metrics: {str(e)}")
            raise

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return 0.0
        except:
            return 0.0

    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   title: str = "Confusion Matrix") -> plt.Figure:
        """
        Create confusion matrix visualization

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                       ax=ax)

            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)

            # Add percentage annotations
            total = cm.sum()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    percentage = cm[i, j] / total * 100
                    ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                           ha='center', va='center', fontsize=10, color='gray')

            plt.tight_layout()

            if self.save_plots:
                filename = f"{title.lower().replace(' ', '_')}.png"
                filepath = os.path.join(self.plots_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Confusion matrix saved to {filepath}")

            return fig

        except Exception as e:
            self.logger.error(f"Failed to create confusion matrix plot: {str(e)}")
            raise

    def create_roc_curve_plot(self, y_true: np.ndarray, y_proba: np.ndarray, 
                             title: str = "ROC Curve") -> plt.Figure:
        """
        Create ROC curve visualization

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot ROC curve
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if self.save_plots:
                filename = f"{title.lower().replace(' ', '_')}.png"
                filepath = os.path.join(self.plots_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"ROC curve saved to {filepath}")

            return fig

        except Exception as e:
            self.logger.error(f"Failed to create ROC curve plot: {str(e)}")
            raise

    def create_precision_recall_curve_plot(self, y_true: np.ndarray, y_proba: np.ndarray,
                                         title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Create Precision-Recall curve visualization

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        try:
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            ap_score = average_precision_score(y_true, y_proba)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot PR curve
            ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap_score:.3f})')

            # Add baseline
            baseline = np.sum(y_true) / len(y_true)
            ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                      label=f'Baseline (AP = {baseline:.3f})')

            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if self.save_plots:
                filename = f"{title.lower().replace(' ', '_')}.png"
                filepath = os.path.join(self.plots_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"PR curve saved to {filepath}")

            return fig

        except Exception as e:
            self.logger.error(f"Failed to create PR curve plot: {str(e)}")
            raise

    def create_metrics_comparison_plot(self, results_dict: Dict[str, Dict[str, float]],
                                     title: str = "Model Comparison") -> plt.Figure:
        """
        Create comparison plot of different models' metrics

        Args:
            results_dict: Dictionary with model results
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        try:
            # Prepare data for plotting
            metrics_df = pd.DataFrame(results_dict).T

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')

            # Plot each metric
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']

            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i // 2, i % 2]

                if metric in metrics_df.columns:
                    values = metrics_df[metric].values
                    models = metrics_df.index.tolist()

                    bars = ax.bar(models, values, alpha=0.7)
                    ax.set_title(metric.title().replace('_', ' '), fontsize=12, fontweight='bold')
                    ax.set_ylabel('Score', fontsize=10)
                    ax.set_ylim(0, 1)

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                    # Rotate x-axis labels if needed
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if self.save_plots:
                filename = f"{title.lower().replace(' ', '_')}.png"
                filepath = os.path.join(self.plots_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Comparison plot saved to {filepath}")

            return fig

        except Exception as e:
            self.logger.error(f"Failed to create comparison plot: {str(e)}")
            raise

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     target_names: List[str] = ['Fake', 'Real']) -> str:
        """
        Generate detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Class names

        Returns:
            Classification report as string
        """
        try:
            report = classification_report(y_true, y_pred, target_names=target_names)
            self.logger.info("Classification report generated")
            return report

        except Exception as e:
            self.logger.error(f"Failed to generate classification report: {str(e)}")
            return f"Error generating report: {str(e)}"

    def evaluate_model_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None,
                                   model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation with all metrics and visualizations

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            model_name: Name of the model being evaluated

        Returns:
            Comprehensive evaluation results
        """
        try:
            self.logger.info(f"Starting comprehensive evaluation for {model_name}")

            # Calculate basic metrics
            metrics = self.calculate_basic_metrics(y_true, y_pred, y_proba)

            # Generate classification report
            classification_rep = self.generate_classification_report(y_true, y_pred)

            # Create visualizations
            plots = {}

            # Confusion matrix
            plots['confusion_matrix'] = self.create_confusion_matrix_plot(
                y_true, y_pred, f"{model_name} - Confusion Matrix"
            )

            # ROC and PR curves if probabilities available
            if y_proba is not None:
                plots['roc_curve'] = self.create_roc_curve_plot(
                    y_true, y_proba, f"{model_name} - ROC Curve"
                )
                plots['pr_curve'] = self.create_precision_recall_curve_plot(
                    y_true, y_proba, f"{model_name} - Precision-Recall Curve"
                )

            # Compile results
            results = {
                'model_name': model_name,
                'metrics': metrics,
                'classification_report': classification_rep,
                'plots': plots,
                'evaluation_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Comprehensive evaluation completed for {model_name}")
            return results

        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {str(e)}")
            raise

    def save_evaluation_report(self, results: Dict[str, Any], 
                             filename: str = "evaluation_report.json") -> None:
        """
        Save evaluation results to JSON file

        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        try:
            # Prepare results for JSON serialization (remove plot objects)
            json_results = results.copy()
            if 'plots' in json_results:
                json_results['plots'] = list(json_results['plots'].keys())

            # Save to file
            filepath = os.path.join(self.plots_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=4)

            self.logger.info(f"Evaluation report saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save evaluation report: {str(e)}")
            raise

    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary to console

        Args:
            results: Evaluation results dictionary
        """
        try:
            print(f"\n{'='*60}")
            print(f"EVALUATION SUMMARY - {results['model_name']}")
            print(f"{'='*60}")

            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"-" * 30)
            metrics = results['metrics']
            print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
            print(f"Precision:    {metrics['precision']:.4f} ({metrics['precision']:.1%})")
            print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']:.1%})")
            print(f"F1-Score:     {metrics['f1_score']:.4f} ({metrics['f1_score']:.1%})")
            print(f"Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']:.1%})")

            if 'roc_auc' in metrics:
                print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
                print(f"PR AUC:       {metrics['pr_auc']:.4f}")

            print(f"\n📋 CLASSIFICATION REPORT:")
            print(f"-" * 50)
            print(results['classification_report'])

            print(f"\n🎯 TARGET PERFORMANCE (MeowTrix-AI Goals):")
            print(f"-" * 50)
            targets = {'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1_score': 0.94}

            for metric, target in targets.items():
                actual = metrics[metric]
                status = "✅ PASS" if actual >= target else "❌ BELOW TARGET"
                print(f"{metric.title():12}: {actual:.4f} (Target: {target:.4f}) {status}")

            print(f"\n⏰ Evaluation completed at: {results['evaluation_timestamp']}")
            print(f"{'='*60}\n")

        except Exception as e:
            self.logger.error(f"Failed to print evaluation summary: {str(e)}")
