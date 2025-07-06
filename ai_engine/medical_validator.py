"""
Medical Validation Framework for Pneumonia Detection
Provides scientific validation with ROC analysis, sensitivity/specificity calculations,
and optimal threshold determination for medical AI models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

class MedicalValidator:
    """
    Comprehensive medical validation framework for pneumonia detection models.
    Provides scientifically rigorous evaluation metrics and threshold optimization.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "validation_plots"):
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Medical validation parameters
        self.medical_thresholds = np.arange(0.01, 0.96, 0.01)  # Fine-grained threshold search
        self.results_cache = {}
        
        logger.info("Medical Validator initialized")
    
    def validate_model_performance(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        model_name: str = "model",
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive medical validation of model performance.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=pneumonia)
            y_scores: Predicted probabilities for pneumonia class
            model_name: Name of the model being validated
            generate_plots: Whether to generate validation plots
            
        Returns:
            Comprehensive validation results dictionary
        """
        logger.info(f"Starting medical validation for {model_name}")
        
        # Basic dataset statistics
        n_total = len(y_true)
        n_pneumonia = np.sum(y_true)
        n_normal = n_total - n_pneumonia
        prevalence = n_pneumonia / n_total
        
        logger.info(f"Dataset: {n_total} images ({n_normal} normal, {n_pneumonia} pneumonia)")
        logger.info(f"Prevalence: {prevalence:.3f}")
        
        # Generate ROC analysis
        roc_results = self._generate_roc_analysis(y_true, y_scores, model_name)
        
        # Generate Precision-Recall analysis
        pr_results = self._generate_precision_recall_analysis(y_true, y_scores, model_name)
        
        # Find optimal thresholds using multiple criteria
        optimal_thresholds = self._find_optimal_thresholds(y_true, y_scores)
        
        # Generate detailed threshold analysis
        threshold_analysis = self._analyze_thresholds(y_true, y_scores)
        
        # Calculate medical performance metrics at optimal threshold
        optimal_threshold = optimal_thresholds['youden_j']['threshold']
        medical_metrics = self._calculate_medical_metrics(
            y_true, y_scores, optimal_threshold, prevalence
        )
        
        # Generate confusion matrix analysis
        confusion_analysis = self._generate_confusion_analysis(
            y_true, y_scores, optimal_threshold, model_name
        )
        
        # Create comprehensive results
        validation_results = {
            'model_name': model_name,
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_statistics': {
                'total_samples': n_total,
                'normal_samples': n_normal,
                'pneumonia_samples': n_pneumonia,
                'prevalence': prevalence
            },
            'roc_analysis': roc_results,
            'precision_recall_analysis': pr_results,
            'optimal_thresholds': optimal_thresholds,
            'threshold_analysis': threshold_analysis,
            'medical_metrics': medical_metrics,
            'confusion_analysis': confusion_analysis,
            'recommendations': self._generate_medical_recommendations(
                medical_metrics, optimal_thresholds, prevalence
            )
        }
        
        # Generate plots if requested
        if generate_plots:
            self._generate_validation_plots(validation_results, y_true, y_scores, model_name)
        
        # Cache results
        self.results_cache[model_name] = validation_results
        
        logger.info(f"Medical validation completed for {model_name}")
        return validation_results
    
    def _generate_roc_analysis(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        model_name: str
    ) -> Dict[str, Any]:
        """Generate comprehensive ROC analysis."""
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal operating point using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        
        return {
            'auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_point': {
                'threshold': float(optimal_threshold),
                'sensitivity': float(optimal_sensitivity),
                'specificity': float(optimal_specificity),
                'youden_j': float(j_scores[optimal_idx])
            },
            'interpretation': self._interpret_auc(roc_auc)
        }
    
    def _generate_precision_recall_analysis(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        model_name: str
    ) -> Dict[str, Any]:
        """Generate Precision-Recall analysis."""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Find optimal F1 threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        
        return {
            'average_precision': float(avg_precision),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_f1_point': {
                'threshold': float(optimal_f1_threshold),
                'precision': float(precision[optimal_f1_idx]),
                'recall': float(recall[optimal_f1_idx]),
                'f1_score': float(f1_scores[optimal_f1_idx])
            }
        }
    
    def _find_optimal_thresholds(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Find optimal thresholds using multiple medical criteria."""
        
        optimal_thresholds = {}
        
        # 1. Youden's J statistic (balanced sensitivity/specificity)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        youden_idx = np.argmax(j_scores)
        
        optimal_thresholds['youden_j'] = {
            'threshold': float(thresholds[youden_idx]),
            'sensitivity': float(tpr[youden_idx]),
            'specificity': float(1 - fpr[youden_idx]),
            'youden_j': float(j_scores[youden_idx]),
            'rationale': 'Balanced sensitivity and specificity for general medical use'
        }
        
        # 2. High sensitivity threshold (minimize false negatives)
        high_sens_target = 0.95
        high_sens_indices = np.where(tpr >= high_sens_target)[0]
        if len(high_sens_indices) > 0:
            high_sens_idx = high_sens_indices[0]
            optimal_thresholds['high_sensitivity'] = {
                'threshold': float(thresholds[high_sens_idx]),
                'sensitivity': float(tpr[high_sens_idx]),
                'specificity': float(1 - fpr[high_sens_idx]),
                'rationale': 'High sensitivity for screening scenarios'
            }
        
        # 3. High specificity threshold (minimize false positives)
        high_spec_target = 0.95
        high_spec_indices = np.where((1 - fpr) >= high_spec_target)[0]
        if len(high_spec_indices) > 0:
            high_spec_idx = high_spec_indices[-1]  # Last index maintains specificity
            optimal_thresholds['high_specificity'] = {
                'threshold': float(thresholds[high_spec_idx]),
                'sensitivity': float(tpr[high_spec_idx]),
                'specificity': float(1 - fpr[high_spec_idx]),
                'rationale': 'High specificity for confirmatory scenarios'
            }
        
        # 4. Cost-sensitive threshold (assuming 10:1 cost ratio FN:FP)
        cost_ratios = 10 * fpr + (1 - tpr)  # 10x weight on false positives
        cost_optimal_idx = np.argmin(cost_ratios)
        
        optimal_thresholds['cost_sensitive'] = {
            'threshold': float(thresholds[cost_optimal_idx]),
            'sensitivity': float(tpr[cost_optimal_idx]),
            'specificity': float(1 - fpr[cost_optimal_idx]),
            'cost_ratio': float(cost_ratios[cost_optimal_idx]),
            'rationale': 'Minimizes medical costs assuming 10:1 FN:FP cost ratio'
        }
        
        return optimal_thresholds
    
    def _analyze_thresholds(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, List[float]]:
        """Analyze performance across all thresholds."""
        
        threshold_results = {
            'thresholds': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'f1_score': [],
            'accuracy': [],
            'youden_j': []
        }
        
        for threshold in self.medical_thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            youden_j = sensitivity + specificity - 1
            
            threshold_results['thresholds'].append(threshold)
            threshold_results['sensitivity'].append(sensitivity)
            threshold_results['specificity'].append(specificity)
            threshold_results['precision'].append(precision)
            threshold_results['f1_score'].append(f1)
            threshold_results['accuracy'].append(accuracy)
            threshold_results['youden_j'].append(youden_j)
        
        return threshold_results
    
    def _calculate_medical_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        threshold: float,
        prevalence: float
    ) -> Dict[str, float]:
        """Calculate comprehensive medical performance metrics."""
        
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Medical-specific metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        ppv = precision  # Positive Predictive Value (same as precision)
        
        # Likelihood ratios
        lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
        lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # F1 and balanced accuracy
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Youden's J statistic
        youden_j = sensitivity + specificity - 1
        
        return {
            'threshold': threshold,
            'sensitivity_recall': sensitivity,
            'specificity': specificity,
            'precision_ppv': ppv,
            'negative_predictive_value': npv,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1_score,
            'youden_j_statistic': youden_j,
            'likelihood_ratio_positive': lr_positive,
            'likelihood_ratio_negative': lr_negative,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'prevalence': prevalence
        }
    
    def _generate_confusion_analysis(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        threshold: float,
        model_name: str
    ) -> Dict[str, Any]:
        """Generate detailed confusion matrix analysis."""
        
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Get detailed classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Normal', 'Pneumonia'],
            output_dict=True
        )
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'normalized_confusion_matrix': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }
    
    def _interpret_auc(self, auc_score: float) -> str:
        """Interpret AUC score in medical context."""
        if auc_score >= 0.95:
            return "Excellent discrimination (AUC ≥ 0.95)"
        elif auc_score >= 0.90:
            return "Very good discrimination (0.90 ≤ AUC < 0.95)"
        elif auc_score >= 0.80:
            return "Good discrimination (0.80 ≤ AUC < 0.90)"
        elif auc_score >= 0.70:
            return "Fair discrimination (0.70 ≤ AUC < 0.80)"
        elif auc_score >= 0.60:
            return "Poor discrimination (0.60 ≤ AUC < 0.70)"
        else:
            return "No discrimination (AUC < 0.60)"
    
    def _generate_medical_recommendations(
        self, 
        medical_metrics: Dict[str, float],
        optimal_thresholds: Dict[str, Dict[str, float]],
        prevalence: float
    ) -> Dict[str, Any]:
        """Generate medical recommendations based on validation results."""
        
        recommendations = {
            'primary_threshold': optimal_thresholds['youden_j']['threshold'],
            'clinical_interpretation': [],
            'deployment_recommendations': [],
            'monitoring_recommendations': []
        }
        
        # Clinical interpretation
        sensitivity = medical_metrics['sensitivity_recall']
        specificity = medical_metrics['specificity']
        
        if sensitivity >= 0.95:
            recommendations['clinical_interpretation'].append(
                "High sensitivity - suitable for screening applications"
            )
        elif sensitivity >= 0.85:
            recommendations['clinical_interpretation'].append(
                "Good sensitivity - appropriate for general diagnostic use"
            )
        else:
            recommendations['clinical_interpretation'].append(
                "Lower sensitivity - requires careful consideration of false negatives"
            )
        
        if specificity >= 0.95:
            recommendations['clinical_interpretation'].append(
                "High specificity - suitable for confirmatory applications"
            )
        elif specificity >= 0.85:
            recommendations['clinical_interpretation'].append(
                "Good specificity - appropriate for general diagnostic use"
            )
        else:
            recommendations['clinical_interpretation'].append(
                "Lower specificity - may generate excessive false positives"
            )
        
        # Deployment recommendations
        if medical_metrics['youden_j_statistic'] >= 0.8:
            recommendations['deployment_recommendations'].append(
                "Model shows excellent balanced performance for clinical deployment"
            )
        elif medical_metrics['youden_j_statistic'] >= 0.6:
            recommendations['deployment_recommendations'].append(
                "Model shows good performance - suitable with clinical oversight"
            )
        else:
            recommendations['deployment_recommendations'].append(
                "Model requires improvement before clinical deployment"
            )
        
        # Monitoring recommendations
        recommendations['monitoring_recommendations'] = [
            "Continuous monitoring of sensitivity and specificity in clinical use",
            "Regular recalibration with new data",
            "Monitor for dataset drift and demographic bias",
            "Validate performance on external datasets"
        ]
        
        return recommendations
    
    def _generate_validation_plots(
        self, 
        results: Dict[str, Any],
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str
    ):
        """Generate comprehensive validation plots."""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Medical Validation Results: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        ax1 = axes[0, 0]
        roc_data = results['roc_analysis']
        ax1.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax1.scatter([1 - roc_data['optimal_point']['specificity']], 
                   [roc_data['optimal_point']['sensitivity']], 
                   color='red', s=100, label='Optimal Point')
        ax1.set_xlabel('False Positive Rate (1-Specificity)')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax2 = axes[0, 1]
        pr_data = results['precision_recall_analysis']
        ax2.plot(pr_data['recall'], pr_data['precision'], 'g-', linewidth=2,
                label=f'PR Curve (AP = {pr_data["average_precision"]:.3f})')
        ax2.axhline(y=results['dataset_statistics']['prevalence'], 
                   color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Recall (Sensitivity)')
        ax2.set_ylabel('Precision (PPV)')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Threshold Analysis
        ax3 = axes[0, 2]
        thresh_data = results['threshold_analysis']
        ax3.plot(thresh_data['thresholds'], thresh_data['sensitivity'], 
                'b-', label='Sensitivity', linewidth=2)
        ax3.plot(thresh_data['thresholds'], thresh_data['specificity'], 
                'r-', label='Specificity', linewidth=2)
        ax3.plot(thresh_data['thresholds'], thresh_data['youden_j'], 
                'g-', label="Youden's J", linewidth=2)
        ax3.axvline(x=results['optimal_thresholds']['youden_j']['threshold'], 
                   color='orange', linestyle='--', alpha=0.7, label='Optimal Threshold')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Threshold Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        ax4 = axes[1, 0]
        cm = np.array(results['confusion_analysis']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax4, cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        ax4.set_title('Confusion Matrix')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        
        # 5. Performance Metrics Bar Chart
        ax5 = axes[1, 1]
        metrics = results['medical_metrics']
        metric_names = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Accuracy']
        metric_values = [
            metrics['sensitivity_recall'],
            metrics['specificity'],
            metrics['precision_ppv'],
            metrics['f1_score'],
            metrics['accuracy']
        ]
        bars = ax5.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        ax5.set_ylim(0, 1)
        ax5.set_title('Performance Metrics')
        ax5.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Score Distribution
        ax6 = axes[1, 2]
        normal_scores = y_scores[y_true == 0]
        pneumonia_scores = y_scores[y_true == 1]
        ax6.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        ax6.hist(pneumonia_scores, bins=30, alpha=0.7, label='Pneumonia', color='red', density=True)
        ax6.axvline(x=results['optimal_thresholds']['youden_j']['threshold'], 
                   color='orange', linestyle='--', linewidth=2, label='Optimal Threshold')
        ax6.set_xlabel('Predicted Probability')
        ax6.set_ylabel('Density')
        ax6.set_title('Score Distribution')
        ax6.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = self.plot_dir / f'{model_name}_validation_report.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to {plot_path}")
        
        plt.show()
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model validation results."""
        
        comparison = {
            'model_comparison': {},
            'best_model_by_metric': {},
            'recommendations': []
        }
        
        metrics_to_compare = [
            'auc', 'sensitivity_recall', 'specificity', 
            'precision_ppv', 'f1_score', 'youden_j_statistic'
        ]
        
        for metric in metrics_to_compare:
            best_model = None
            best_score = -1
            
            for model_name, results in results_dict.items():
                if metric == 'auc':
                    score = results['roc_analysis']['auc']
                else:
                    score = results['medical_metrics'][metric]
                
                comparison['model_comparison'][f'{model_name}_{metric}'] = score
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            comparison['best_model_by_metric'][metric] = {
                'model': best_model,
                'score': best_score
            }
        
        # Generate overall recommendations
        if len(results_dict) > 1:
            # Find model with best balanced performance (Youden's J)
            best_balanced = comparison['best_model_by_metric']['youden_j_statistic']
            comparison['recommendations'].append(
                f"Best balanced performance: {best_balanced['model']} "
                f"(Youden's J = {best_balanced['score']:.3f})"
            )
            
            # Find model with best AUC
            best_auc = comparison['best_model_by_metric']['auc']
            comparison['recommendations'].append(
                f"Best discrimination: {best_auc['model']} "
                f"(AUC = {best_auc['score']:.3f})"
            )
        
        return comparison
    
    def save_validation_report(
        self, 
        results: Dict[str, Any], 
        filepath: str
    ):
        """Save comprehensive validation report to JSON."""
        
        report_path = Path(filepath)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")

# Global validator instance
medical_validator = MedicalValidator()