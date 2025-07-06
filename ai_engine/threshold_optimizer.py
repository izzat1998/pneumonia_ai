"""
Threshold Optimizer for Medical AI Models
Provides scientific threshold optimization using multiple medical criteria
including Youden's J statistic, cost-sensitive analysis, and clinical scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.optimize import minimize_scalar
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

class MedicalThresholdOptimizer:
    """
    Optimize decision thresholds for medical AI models using multiple criteria.
    Provides scientifically rigorous threshold selection for clinical deployment.
    """
    
    def __init__(self, output_dir: str = "threshold_optimization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Medical cost ratios for different scenarios
        self.cost_scenarios = {
            'screening': {'fn_cost': 10, 'fp_cost': 1, 'description': 'Screening scenario - high cost for missed cases'},
            'diagnostic': {'fn_cost': 5, 'fp_cost': 2, 'description': 'General diagnostic - balanced costs'},
            'confirmatory': {'fn_cost': 3, 'fp_cost': 5, 'description': 'Confirmatory scenario - high cost for false positives'},
            'emergency': {'fn_cost': 20, 'fp_cost': 1, 'description': 'Emergency setting - critical to catch all cases'}
        }
        
        # Clinical performance targets
        self.clinical_targets = {
            'high_sensitivity': {'min_sensitivity': 0.95, 'min_specificity': 0.70},
            'high_specificity': {'min_sensitivity': 0.70, 'min_specificity': 0.95},
            'balanced': {'min_sensitivity': 0.85, 'min_specificity': 0.85},
            'screening_optimal': {'min_sensitivity': 0.90, 'min_specificity': 0.80}
        }
        
        logger.info("Medical Threshold Optimizer initialized")
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "model",
        prevalence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive threshold optimization for medical applications.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=pneumonia)
            y_scores: Predicted probabilities for pneumonia class
            model_name: Name of the model
            prevalence: Disease prevalence (if different from dataset)
            
        Returns:
            Comprehensive optimization results
        """
        logger.info(f"Starting threshold optimization for {model_name}")
        
        if prevalence is None:
            prevalence = np.mean(y_true)
        
        # Calculate ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Comprehensive threshold analysis
        optimization_results = {
            'model_name': model_name,
            'optimization_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(y_true),
                'n_positive': int(np.sum(y_true)),
                'n_negative': int(len(y_true) - np.sum(y_true)),
                'prevalence': float(prevalence)
            },
            'roc_analysis': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': float(auc(fpr, tpr))
            },
            'optimal_thresholds': {}
        }
        
        # 1. Youden's J statistic (balanced sensitivity/specificity)
        optimization_results['optimal_thresholds']['youden_j'] = self._optimize_youden_j(
            fpr, tpr, roc_thresholds
        )
        
        # 2. Cost-sensitive optimization for different scenarios
        for scenario, costs in self.cost_scenarios.items():
            optimization_results['optimal_thresholds'][f'cost_{scenario}'] = self._optimize_cost_sensitive(
                y_true, y_scores, costs['fn_cost'], costs['fp_cost'], scenario, prevalence
            )
        
        # 3. Clinical target optimization
        for target_name, targets in self.clinical_targets.items():
            optimization_results['optimal_thresholds'][f'clinical_{target_name}'] = self._optimize_clinical_targets(
                fpr, tpr, roc_thresholds, targets['min_sensitivity'], targets['min_specificity'], target_name
            )
        
        # 4. F1-score optimization
        optimization_results['optimal_thresholds']['f1_optimal'] = self._optimize_f1_score(
            y_true, y_scores
        )
        
        # 5. Precision-Recall optimization
        optimization_results['optimal_thresholds']['pr_optimal'] = self._optimize_precision_recall(
            precision, recall, pr_thresholds
        )
        
        # 6. Medical utility optimization
        optimization_results['optimal_thresholds']['medical_utility'] = self._optimize_medical_utility(
            y_true, y_scores, prevalence
        )
        
        # Comprehensive comparison and recommendations
        optimization_results['threshold_comparison'] = self._compare_thresholds(
            optimization_results['optimal_thresholds'], y_true, y_scores
        )
        
        optimization_results['recommendations'] = self._generate_threshold_recommendations(
            optimization_results, prevalence
        )
        
        # Generate visualization
        self._plot_threshold_optimization(optimization_results, y_true, y_scores, model_name)
        
        # Save results
        self._save_optimization_results(optimization_results, model_name)
        
        logger.info(f"Threshold optimization completed for {model_name}")
        return optimization_results
    
    def _optimize_youden_j(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray
    ) -> Dict[str, float]:
        """Optimize using Youden's J statistic (sensitivity + specificity - 1)."""
        
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        return {
            'threshold': float(thresholds[optimal_idx]),
            'sensitivity': float(tpr[optimal_idx]),
            'specificity': float(1 - fpr[optimal_idx]),
            'youden_j': float(j_scores[optimal_idx]),
            'rationale': 'Maximizes sum of sensitivity and specificity',
            'clinical_use': 'General diagnostic use with balanced performance'
        }
    
    def _optimize_cost_sensitive(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fn_cost: float,
        fp_cost: float,
        scenario: str,
        prevalence: float
    ) -> Dict[str, Any]:
        """Optimize threshold based on cost-sensitive analysis."""
        
        def cost_function(threshold):
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate expected cost
            total_cost = (fn * fn_cost + fp * fp_cost) / len(y_true)
            return total_cost
        
        # Find optimal threshold
        result = minimize_scalar(cost_function, bounds=(0.01, 0.99), method='bounded')
        optimal_threshold = result.x
        
        # Calculate performance at optimal threshold
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_optimal == 1))
        tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
        fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
        fn = np.sum((y_true == 1) & (y_pred_optimal == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': float(optimal_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'expected_cost': float(result.fun),
            'fn_cost_ratio': fn_cost,
            'fp_cost_ratio': fp_cost,
            'scenario': scenario,
            'rationale': f'Minimizes expected cost for {scenario} scenario',
            'clinical_use': self.cost_scenarios[scenario]['description']
        }
    
    def _optimize_clinical_targets(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray,
        min_sensitivity: float,
        min_specificity: float,
        target_name: str
    ) -> Dict[str, Any]:
        """Optimize threshold to meet clinical performance targets."""
        
        specificity = 1 - fpr
        
        # Find thresholds that meet minimum requirements
        valid_indices = (tpr >= min_sensitivity) & (specificity >= min_specificity)
        
        if not np.any(valid_indices):
            # No threshold meets both requirements, find best compromise
            sensitivity_deficit = np.maximum(0, min_sensitivity - tpr)
            specificity_deficit = np.maximum(0, min_specificity - specificity)
            total_deficit = sensitivity_deficit + specificity_deficit
            
            optimal_idx = np.argmin(total_deficit)
            meets_targets = False
        else:
            # Among valid thresholds, maximize Youden's J
            valid_j_scores = (tpr - fpr)[valid_indices]
            relative_optimal_idx = np.argmax(valid_j_scores)
            optimal_idx = np.where(valid_indices)[0][relative_optimal_idx]
            meets_targets = True
        
        return {
            'threshold': float(thresholds[optimal_idx]),
            'sensitivity': float(tpr[optimal_idx]),
            'specificity': float(specificity[optimal_idx]),
            'meets_targets': meets_targets,
            'target_sensitivity': min_sensitivity,
            'target_specificity': min_specificity,
            'youden_j': float(tpr[optimal_idx] - fpr[optimal_idx]),
            'rationale': f'Optimized for {target_name} clinical targets',
            'clinical_use': f'Clinical scenario requiring sens≥{min_sensitivity:.2f}, spec≥{min_specificity:.2f}'
        }
    
    def _optimize_f1_score(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """Optimize threshold for maximum F1-score."""
        
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate final metrics
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_optimal == 1))
        tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
        fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
        fn = np.sum((y_true == 1) & (y_pred_optimal == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': float(optimal_threshold),
            'f1_score': float(f1_scores[optimal_idx]),
            'precision': float(precision),
            'sensitivity_recall': float(recall),
            'specificity': float(specificity),
            'rationale': 'Maximizes F1-score (harmonic mean of precision and recall)',
            'clinical_use': 'Balanced optimization for precision-recall trade-off'
        }
    
    def _optimize_precision_recall(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        thresholds: np.ndarray
    ) -> Dict[str, float]:
        """Optimize threshold for precision-recall balance."""
        
        # Find threshold that maximizes precision-recall product
        pr_product = precision[:-1] * recall[:-1]  # Exclude last element (precision has n+1 elements)
        optimal_idx = np.argmax(pr_product)
        
        return {
            'threshold': float(thresholds[optimal_idx]),
            'precision': float(precision[optimal_idx]),
            'recall': float(recall[optimal_idx]),
            'pr_product': float(pr_product[optimal_idx]),
            'rationale': 'Maximizes precision-recall product',
            'clinical_use': 'Balances positive predictive value and sensitivity'
        }
    
    def _optimize_medical_utility(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        prevalence: float
    ) -> Dict[str, Any]:
        """Optimize threshold for medical utility considering prevalence."""
        
        def medical_utility_function(threshold):
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate confusion matrix
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate rates
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Medical utility considering prevalence and clinical impact
            # Weighted by prevalence and clinical importance
            utility = (
                sensitivity * prevalence * 2.0 +  # 2x weight for catching pneumonia
                specificity * (1 - prevalence) * 1.0  # 1x weight for correctly identifying normal
            )
            
            return -utility  # Minimize negative utility (maximize utility)
        
        # Find optimal threshold
        result = minimize_scalar(medical_utility_function, bounds=(0.01, 0.99), method='bounded')
        optimal_threshold = result.x
        
        # Calculate performance at optimal threshold
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_optimal == 1))
        tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
        fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
        fn = np.sum((y_true == 1) & (y_pred_optimal == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': float(optimal_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'medical_utility': float(-result.fun),
            'prevalence_used': float(prevalence),
            'rationale': 'Maximizes medical utility considering disease prevalence and clinical impact',
            'clinical_use': 'Prevalence-adjusted optimization for real-world deployment'
        }
    
    def _compare_thresholds(
        self,
        optimal_thresholds: Dict[str, Dict[str, Any]],
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Compare all optimal thresholds across different metrics."""
        
        comparison_data = []
        
        for method, threshold_data in optimal_thresholds.items():
            threshold = threshold_data['threshold']
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate comprehensive metrics
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            youden_j = sensitivity + specificity - 1
            
            comparison_data.append({
                'method': method,
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'accuracy': accuracy,
                'f1_score': f1,
                'youden_j': youden_j,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best performing threshold for each metric
        best_by_metric = {}
        for metric in ['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score', 'youden_j']:
            best_idx = comparison_df[metric].idxmax()
            best_method = comparison_df.iloc[best_idx]['method']
            best_value = comparison_df.iloc[best_idx][metric]
            best_by_metric[metric] = {
                'method': best_method,
                'value': float(best_value),
                'threshold': float(comparison_df.iloc[best_idx]['threshold'])
            }
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_by_metric': best_by_metric,
            'threshold_range': {
                'min': float(comparison_df['threshold'].min()),
                'max': float(comparison_df['threshold'].max()),
                'mean': float(comparison_df['threshold'].mean()),
                'std': float(comparison_df['threshold'].std())
            }
        }
    
    def _generate_threshold_recommendations(
        self,
        optimization_results: Dict[str, Any],
        prevalence: float
    ) -> Dict[str, Any]:
        """Generate recommendations for threshold selection."""
        
        recommendations = {
            'primary_recommendation': {},
            'scenario_recommendations': {},
            'clinical_considerations': [],
            'validation_requirements': []
        }
        
        # Primary recommendation based on balanced performance
        youden_j_result = optimization_results['optimal_thresholds']['youden_j']
        recommendations['primary_recommendation'] = {
            'method': 'youden_j',
            'threshold': youden_j_result['threshold'],
            'rationale': 'Provides optimal balance of sensitivity and specificity for general medical use',
            'sensitivity': youden_j_result['sensitivity'],
            'specificity': youden_j_result['specificity']
        }
        
        # Scenario-specific recommendations
        if prevalence < 0.1:  # Low prevalence
            recommendations['scenario_recommendations']['low_prevalence'] = {
                'recommended_method': 'cost_screening',
                'rationale': 'High sensitivity needed for rare disease screening',
                'considerations': ['Higher false positive rate acceptable', 'Focus on not missing cases']
            }
        elif prevalence > 0.3:  # High prevalence
            recommendations['scenario_recommendations']['high_prevalence'] = {
                'recommended_method': 'clinical_balanced',
                'rationale': 'Balanced approach suitable for common conditions',
                'considerations': ['Both sensitivity and specificity important', 'Cost-effectiveness balanced']
            }
        else:  # Moderate prevalence
            recommendations['scenario_recommendations']['moderate_prevalence'] = {
                'recommended_method': 'youden_j',
                'rationale': 'Youden\'s J optimal for moderate prevalence',
                'considerations': ['Balanced performance preferred', 'Consider clinical context']
            }
        
        # Clinical considerations
        recommendations['clinical_considerations'] = [
            'Consider patient population and clinical setting',
            'Evaluate cost of false positives vs false negatives',
            'Validate threshold on external datasets before deployment',
            'Monitor performance continuously in clinical use',
            'Consider ensemble methods for improved reliability'
        ]
        
        # Validation requirements
        recommendations['validation_requirements'] = [
            'Prospective validation on independent dataset',
            'Cross-validation across different patient populations',
            'Comparison with expert radiologist performance',
            'Clinical utility assessment in real-world setting',
            'Regular recalibration with new data'
        ]
        
        return recommendations
    
    def _plot_threshold_optimization(
        self,
        optimization_results: Dict[str, Any],
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str
    ):
        """Generate comprehensive threshold optimization plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Threshold Optimization Results: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve with optimal points
        ax1 = axes[0, 0]
        roc_data = optimization_results['roc_analysis']
        ax1.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # Plot key optimal points
        key_methods = ['youden_j', 'cost_screening', 'clinical_balanced']
        colors = ['red', 'green', 'orange']
        
        for method, color in zip(key_methods, colors):
            if method in optimization_results['optimal_thresholds']:
                result = optimization_results['optimal_thresholds'][method]
                fpr_point = 1 - result['specificity']
                tpr_point = result['sensitivity']
                ax1.scatter([fpr_point], [tpr_point], color=color, s=100, 
                           label=f'{method}: {result["threshold"]:.3f}')
        
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve with Optimal Thresholds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Threshold vs Performance Metrics
        ax2 = axes[0, 1]
        thresholds = np.linspace(0.01, 0.99, 99)
        sensitivities = []
        specificities = []
        youden_js = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            youden_js.append(sensitivity + specificity - 1)
        
        ax2.plot(thresholds, sensitivities, 'b-', label='Sensitivity', linewidth=2)
        ax2.plot(thresholds, specificities, 'r-', label='Specificity', linewidth=2)
        ax2.plot(thresholds, youden_js, 'g-', label="Youden's J", linewidth=2)
        
        # Mark optimal Youden's J threshold
        youden_result = optimization_results['optimal_thresholds']['youden_j']
        ax2.axvline(x=youden_result['threshold'], color='orange', linestyle='--', 
                   alpha=0.7, label=f'Optimal: {youden_result["threshold"]:.3f}')
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Performance vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cost Analysis
        ax3 = axes[0, 2]
        cost_methods = [k for k in optimization_results['optimal_thresholds'].keys() if k.startswith('cost_')]
        if cost_methods:
            cost_data = []
            for method in cost_methods:
                result = optimization_results['optimal_thresholds'][method]
                cost_data.append({
                    'Scenario': method.replace('cost_', ''),
                    'Threshold': result['threshold'],
                    'Expected Cost': result['expected_cost']
                })
            
            cost_df = pd.DataFrame(cost_data)
            bars = ax3.bar(cost_df['Scenario'], cost_df['Expected Cost'], 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax3.set_xlabel('Cost Scenario')
            ax3.set_ylabel('Expected Cost')
            ax3.set_title('Cost-Sensitive Optimization')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add threshold labels on bars
            for bar, threshold in zip(bars, cost_df['Threshold']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{threshold:.3f}', ha='center', va='bottom')
        
        # 4. Threshold Comparison Table
        ax4 = axes[1, 0]
        comparison_data = optimization_results['threshold_comparison']['comparison_table']
        comparison_df = pd.DataFrame(comparison_data)
        
        # Select key methods for display
        display_methods = ['youden_j', 'cost_screening', 'clinical_balanced', 'f1_optimal']
        display_df = comparison_df[comparison_df['method'].isin(display_methods)]
        
        # Create heatmap of key metrics
        metrics_for_heatmap = ['sensitivity', 'specificity', 'precision', 'f1_score']
        heatmap_data = display_df[metrics_for_heatmap].values
        
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax4.set_xticks(range(len(metrics_for_heatmap)))
        ax4.set_xticklabels(metrics_for_heatmap, rotation=45)
        ax4.set_yticks(range(len(display_df)))
        ax4.set_yticklabels(display_df['method'].tolist())
        ax4.set_title('Performance Comparison Heatmap')
        
        # Add text annotations
        for i in range(len(display_df)):
            for j in range(len(metrics_for_heatmap)):
                ax4.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                        ha='center', va='center', color='black', fontsize=8)
        
        # 5. Score Distribution with Thresholds
        ax5 = axes[1, 1]
        normal_scores = y_scores[y_true == 0]
        pneumonia_scores = y_scores[y_true == 1]
        
        ax5.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        ax5.hist(pneumonia_scores, bins=30, alpha=0.7, label='Pneumonia', color='red', density=True)
        
        # Mark key thresholds
        youden_threshold = optimization_results['optimal_thresholds']['youden_j']['threshold']
        ax5.axvline(x=youden_threshold, color='orange', linestyle='--', linewidth=2, 
                   label=f'Youden: {youden_threshold:.3f}')
        
        if 'cost_screening' in optimization_results['optimal_thresholds']:
            screening_threshold = optimization_results['optimal_thresholds']['cost_screening']['threshold']
            ax5.axvline(x=screening_threshold, color='green', linestyle='--', linewidth=2,
                       label=f'Screening: {screening_threshold:.3f}')
        
        ax5.set_xlabel('Predicted Probability')
        ax5.set_ylabel('Density')
        ax5.set_title('Score Distribution with Optimal Thresholds')
        ax5.legend()
        
        # 6. Clinical Performance Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        youden_result = optimization_results['optimal_thresholds']['youden_j']
        summary_text = f"""
RECOMMENDED THRESHOLD SUMMARY

Primary Recommendation (Youden's J):
Threshold: {youden_result['threshold']:.3f}
Sensitivity: {youden_result['sensitivity']:.3f}
Specificity: {youden_result['specificity']:.3f}
Youden's J: {youden_result['youden_j']:.3f}

Clinical Interpretation:
- Balanced sensitivity/specificity
- Suitable for general diagnostic use
- Optimal point on ROC curve

Validation Required:
- External dataset testing
- Clinical utility assessment
- Radiologist comparison study
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'{model_name}_threshold_optimization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold optimization plots saved to {plot_path}")
    
    def _save_optimization_results(
        self,
        results: Dict[str, Any],
        model_name: str
    ):
        """Save optimization results to file."""
        
        results_path = self.output_dir / f"{model_name}_threshold_optimization.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Threshold optimization results saved to {results_path}")
    
    def compare_threshold_methods(
        self,
        results_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare threshold optimization results across multiple models."""
        
        comparison = {
            'method_performance': {},
            'consistency_analysis': {},
            'recommendations': []
        }
        
        # Extract performance for each method across models
        all_methods = set()
        for model_results in results_dict.values():
            all_methods.update(model_results['optimal_thresholds'].keys())
        
        for method in all_methods:
            method_performance = []
            for model_name, results in results_dict.items():
                if method in results['optimal_thresholds']:
                    method_data = results['optimal_thresholds'][method]
                    method_performance.append({
                        'model': model_name,
                        'threshold': method_data['threshold'],
                        'sensitivity': method_data.get('sensitivity', 0),
                        'specificity': method_data.get('specificity', 0)
                    })
            
            if method_performance:
                comparison['method_performance'][method] = method_performance
        
        return comparison

# Global threshold optimizer instance
threshold_optimizer = MedicalThresholdOptimizer()