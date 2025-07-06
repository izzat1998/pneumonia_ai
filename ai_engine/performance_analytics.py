"""
Performance Analytics for Medical AI Models
Provides comprehensive performance analysis, comparison, and monitoring
for pneumonia detection models with medical-grade metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings

from .medical_validator import medical_validator
from .dataset_processor import dataset_processor

logger = logging.getLogger(__name__)

class MedicalPerformanceAnalytics:
    """
    Comprehensive performance analytics for medical AI models.
    Provides detailed analysis, comparison, and monitoring capabilities.
    """
    
    def __init__(self, output_dir: str = "performance_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analytics configuration
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.medical_thresholds = {
            'high_sensitivity': 0.95,
            'high_specificity': 0.95,
            'balanced': 0.85
        }
        
        # Results storage
        self.analysis_results = {}
        self.comparison_results = {}
        
        logger.info("Performance Analytics initialized")
    
    def comprehensive_model_analysis(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        detailed_predictions: Optional[List[Dict[str, Any]]] = None,
        image_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single model's performance.
        
        Args:
            model_name: Name of the model
            y_true: Ground truth labels
            y_scores: Predicted probabilities
            detailed_predictions: Detailed prediction results
            image_paths: Paths to the images (optional)
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive analysis for {model_name}")
        
        # Basic validation
        validation_results = medical_validator.validate_model_performance(
            y_true, y_scores, model_name, generate_plots=True
        )
        
        # Extended performance metrics
        extended_metrics = self._calculate_extended_metrics(y_true, y_scores)
        
        # Calibration analysis
        calibration_analysis = self._analyze_model_calibration(y_true, y_scores, model_name)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(y_true, y_scores)
        
        # Clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            validation_results, extended_metrics, statistical_analysis
        )
        
        # Performance by subgroups (if detailed predictions available)
        subgroup_analysis = {}
        if detailed_predictions:
            subgroup_analysis = self._analyze_performance_by_subgroups(
                y_true, y_scores, detailed_predictions
            )
        
        # Error analysis
        error_analysis = self._perform_error_analysis(
            y_true, y_scores, validation_results['optimal_thresholds']['youden_j']['threshold'],
            image_paths
        )
        
        # Combine all results
        comprehensive_results = {
            'model_name': model_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_validation': validation_results,
            'extended_metrics': extended_metrics,
            'calibration_analysis': calibration_analysis,
            'statistical_analysis': statistical_analysis,
            'clinical_interpretation': clinical_interpretation,
            'subgroup_analysis': subgroup_analysis,
            'error_analysis': error_analysis,
            'recommendations': self._generate_comprehensive_recommendations(
                validation_results, extended_metrics, clinical_interpretation
            )
        }
        
        # Store results
        self.analysis_results[model_name] = comprehensive_results
        
        # Save detailed report
        self._save_analysis_report(comprehensive_results, model_name)
        
        logger.info(f"Comprehensive analysis completed for {model_name}")
        return comprehensive_results
    
    def _calculate_extended_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate extended performance metrics beyond basic validation."""
        
        # Find optimal threshold using Youden's J
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        # Extended metrics
        extended_metrics = {
            'optimal_threshold': float(optimal_threshold),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            'matthews_correlation': float(matthews_corrcoef(y_true, y_pred)),
            'brier_score': float(brier_score_loss(y_true, y_scores)),
            'log_loss': float(-np.mean(y_true * np.log(y_scores + 1e-15) + 
                                     (1 - y_true) * np.log(1 - y_scores + 1e-15)))
        }
        
        # Calculate metrics at different confidence levels
        confidence_metrics = {}
        for conf_level in self.confidence_levels:
            high_conf_mask = np.maximum(y_scores, 1 - y_scores) >= conf_level
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask])
                confidence_metrics[f'accuracy_at_{conf_level:.0%}'] = {
                    'accuracy': float(high_conf_accuracy),
                    'coverage': float(np.mean(high_conf_mask)),
                    'n_samples': int(np.sum(high_conf_mask))
                }
        
        extended_metrics['confidence_metrics'] = confidence_metrics
        
        # Discrimination metrics
        extended_metrics['discrimination'] = {
            'gini_coefficient': float(2 * auc(fpr, tpr) - 1),
            'ks_statistic': float(np.max(tpr - fpr)),
            'auc_precision_recall': float(average_precision_score(y_true, y_scores))
        }
        
        return extended_metrics
    
    def _analyze_model_calibration(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Analyze model calibration (reliability)."""
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=10, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        # Generate calibration plot
        self._plot_calibration_curve(
            fraction_of_positives, mean_predicted_value, model_name, ece, mce
        )
        
        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'interpretation': self._interpret_calibration(ece, mce)
        }
    
    def _perform_statistical_analysis(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical analysis of model performance."""
        
        # Bootstrap confidence intervals for AUC
        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        
        auc_scores = []
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            
            fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
            auc_scores.append(auc(fpr, tpr))
        
        auc_scores = np.array(auc_scores)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(auc_scores, 2.5)
        ci_upper = np.percentile(auc_scores, 97.5)
        
        # Distribution analysis
        normal_scores = y_scores[y_true == 0]
        pneumonia_scores = y_scores[y_true == 1]
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(normal_scores, pneumonia_scores)
        
        # Mann-Whitney U test
        u_statistic, u_p_value = stats.mannwhitneyu(
            pneumonia_scores, normal_scores, alternative='greater'
        )
        
        return {
            'auc_bootstrap': {
                'mean': float(np.mean(auc_scores)),
                'std': float(np.std(auc_scores)),
                'ci_lower_95': float(ci_lower),
                'ci_upper_95': float(ci_upper),
                'n_bootstraps': n_bootstraps
            },
            'distribution_tests': {
                'kolmogorov_smirnov': {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_p_value),
                    'significant': ks_p_value < 0.05
                },
                'mann_whitney_u': {
                    'statistic': float(u_statistic),
                    'p_value': float(u_p_value),
                    'significant': u_p_value < 0.05
                }
            },
            'score_distributions': {
                'normal_mean': float(np.mean(normal_scores)),
                'normal_std': float(np.std(normal_scores)),
                'pneumonia_mean': float(np.mean(pneumonia_scores)),
                'pneumonia_std': float(np.std(pneumonia_scores)),
                'effect_size_cohens_d': float(
                    (np.mean(pneumonia_scores) - np.mean(normal_scores)) /
                    np.sqrt((np.std(pneumonia_scores)**2 + np.std(normal_scores)**2) / 2)
                )
            }
        }
    
    def _analyze_performance_by_subgroups(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        detailed_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance by different subgroups."""
        
        subgroup_analysis = {}
        
        # Analyze by image quality if available
        if detailed_predictions and 'image_quality' in detailed_predictions[0]:
            quality_groups = self._group_by_image_quality(detailed_predictions)
            
            for quality_level, indices in quality_groups.items():
                if len(indices) > 10:  # Minimum group size
                    group_results = medical_validator.validate_model_performance(
                        y_true[indices], y_scores[indices], 
                        f"quality_{quality_level}", generate_plots=False
                    )
                    subgroup_analysis[f'image_quality_{quality_level}'] = {
                        'n_samples': len(indices),
                        'auc': group_results['roc_analysis']['auc'],
                        'sensitivity': group_results['medical_metrics']['sensitivity_recall'],
                        'specificity': group_results['medical_metrics']['specificity']
                    }
        
        # Analyze by processing time (proxy for image complexity)
        if detailed_predictions and 'processing_time' in detailed_predictions[0]:
            processing_times = np.array([p.get('processing_time', 0) for p in detailed_predictions])
            time_groups = self._group_by_processing_time(processing_times)
            
            for time_group, indices in time_groups.items():
                if len(indices) > 10:
                    group_results = medical_validator.validate_model_performance(
                        y_true[indices], y_scores[indices],
                        f"processing_{time_group}", generate_plots=False
                    )
                    subgroup_analysis[f'processing_time_{time_group}'] = {
                        'n_samples': len(indices),
                        'auc': group_results['roc_analysis']['auc'],
                        'sensitivity': group_results['medical_metrics']['sensitivity_recall'],
                        'specificity': group_results['medical_metrics']['specificity']
                    }
        
        return subgroup_analysis
    
    def _perform_error_analysis(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
        image_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        
        y_pred = (y_scores >= threshold).astype(int)
        
        # Identify error types
        true_positives = (y_true == 1) & (y_pred == 1)
        true_negatives = (y_true == 0) & (y_pred == 0)
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        error_analysis = {
            'error_counts': {
                'false_positives': int(np.sum(false_positives)),
                'false_negatives': int(np.sum(false_negatives)),
                'true_positives': int(np.sum(true_positives)),
                'true_negatives': int(np.sum(true_negatives))
            },
            'error_rates': {
                'false_positive_rate': float(np.sum(false_positives) / np.sum(y_true == 0)),
                'false_negative_rate': float(np.sum(false_negatives) / np.sum(y_true == 1))
            }
        }
        
        # Analyze confidence of errors
        if np.sum(false_positives) > 0:
            fp_confidences = y_scores[false_positives]
            error_analysis['false_positive_analysis'] = {
                'mean_confidence': float(np.mean(fp_confidences)),
                'std_confidence': float(np.std(fp_confidences)),
                'high_confidence_fps': int(np.sum(fp_confidences > 0.8)),
                'top_fp_indices': np.where(false_positives)[0][np.argsort(fp_confidences)[-5:]].tolist()
            }
        
        if np.sum(false_negatives) > 0:
            fn_confidences = y_scores[false_negatives]
            error_analysis['false_negative_analysis'] = {
                'mean_confidence': float(np.mean(fn_confidences)),
                'std_confidence': float(np.std(fn_confidences)),
                'low_confidence_fns': int(np.sum(fn_confidences < 0.2)),
                'top_fn_indices': np.where(false_negatives)[0][np.argsort(fn_confidences)[:5]].tolist()
            }
        
        # Add image paths for error cases if available
        if image_paths:
            error_analysis['error_examples'] = {
                'false_positives': [image_paths[i] for i in np.where(false_positives)[0][:10]],
                'false_negatives': [image_paths[i] for i in np.where(false_negatives)[0][:10]]
            }
        
        return error_analysis
    
    def _generate_clinical_interpretation(
        self,
        validation_results: Dict[str, Any],
        extended_metrics: Dict[str, Any],
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate clinical interpretation of results."""
        
        auc = validation_results['roc_analysis']['auc']
        sensitivity = validation_results['medical_metrics']['sensitivity_recall']
        specificity = validation_results['medical_metrics']['specificity']
        
        clinical_interpretation = {
            'overall_performance': self._interpret_overall_performance(auc, sensitivity, specificity),
            'clinical_utility': self._assess_clinical_utility(sensitivity, specificity),
            'deployment_readiness': self._assess_deployment_readiness(
                validation_results, extended_metrics
            ),
            'risk_assessment': self._assess_clinical_risks(sensitivity, specificity),
            'comparison_to_benchmarks': self._compare_to_clinical_benchmarks(
                auc, sensitivity, specificity
            )
        }
        
        return clinical_interpretation
    
    def _interpret_overall_performance(
        self, 
        auc: float, 
        sensitivity: float, 
        specificity: float
    ) -> Dict[str, str]:
        """Interpret overall model performance."""
        
        performance_level = "poor"
        if auc >= 0.9 and sensitivity >= 0.85 and specificity >= 0.85:
            performance_level = "excellent"
        elif auc >= 0.8 and sensitivity >= 0.8 and specificity >= 0.8:
            performance_level = "good"
        elif auc >= 0.7 and sensitivity >= 0.7 and specificity >= 0.7:
            performance_level = "fair"
        
        return {
            'level': performance_level,
            'summary': f"Model shows {performance_level} performance with AUC={auc:.3f}, "
                      f"sensitivity={sensitivity:.3f}, specificity={specificity:.3f}"
        }
    
    def _assess_clinical_utility(
        self, 
        sensitivity: float, 
        specificity: float
    ) -> Dict[str, Any]:
        """Assess clinical utility based on sensitivity and specificity."""
        
        utility_assessment = {}
        
        # Screening utility (requires high sensitivity)
        if sensitivity >= 0.95:
            utility_assessment['screening'] = {
                'suitable': True,
                'reason': 'High sensitivity minimizes missed pneumonia cases'
            }
        else:
            utility_assessment['screening'] = {
                'suitable': False,
                'reason': f'Sensitivity ({sensitivity:.3f}) too low for screening'
            }
        
        # Confirmatory utility (requires high specificity)
        if specificity >= 0.95:
            utility_assessment['confirmatory'] = {
                'suitable': True,
                'reason': 'High specificity minimizes false positive diagnoses'
            }
        else:
            utility_assessment['confirmatory'] = {
                'suitable': False,
                'reason': f'Specificity ({specificity:.3f}) too low for confirmation'
            }
        
        # General diagnostic utility (balanced performance)
        if sensitivity >= 0.85 and specificity >= 0.85:
            utility_assessment['general_diagnostic'] = {
                'suitable': True,
                'reason': 'Balanced performance suitable for general diagnostic use'
            }
        else:
            utility_assessment['general_diagnostic'] = {
                'suitable': False,
                'reason': 'Insufficient balance for general diagnostic use'
            }
        
        return utility_assessment
    
    def _assess_deployment_readiness(
        self,
        validation_results: Dict[str, Any],
        extended_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess readiness for clinical deployment."""
        
        readiness_factors = {
            'performance_threshold': validation_results['medical_metrics']['youden_j_statistic'] >= 0.7,
            'calibration_quality': extended_metrics.get('expected_calibration_error', 1.0) < 0.1,
            'statistical_significance': validation_results.get('statistical_analysis', {}).get(
                'distribution_tests', {}
            ).get('mann_whitney_u', {}).get('significant', False),
            'sufficient_sensitivity': validation_results['medical_metrics']['sensitivity_recall'] >= 0.8,
            'sufficient_specificity': validation_results['medical_metrics']['specificity'] >= 0.8
        }
        
        readiness_score = sum(readiness_factors.values()) / len(readiness_factors)
        
        if readiness_score >= 0.8:
            readiness_level = 'ready'
        elif readiness_score >= 0.6:
            readiness_level = 'needs_improvement'
        else:
            readiness_level = 'not_ready'
        
        return {
            'level': readiness_level,
            'score': readiness_score,
            'factors': readiness_factors,
            'recommendations': self._generate_deployment_recommendations(readiness_factors)
        }
    
    def _assess_clinical_risks(
        self, 
        sensitivity: float, 
        specificity: float
    ) -> Dict[str, Any]:
        """Assess clinical risks associated with model performance."""
        
        risks = {}
        
        # False negative risk (missed pneumonia)
        fn_rate = 1 - sensitivity
        if fn_rate > 0.2:
            risks['false_negative'] = {
                'level': 'high',
                'rate': fn_rate,
                'implication': 'High risk of missing pneumonia cases'
            }
        elif fn_rate > 0.1:
            risks['false_negative'] = {
                'level': 'medium',
                'rate': fn_rate,
                'implication': 'Moderate risk of missing pneumonia cases'
            }
        else:
            risks['false_negative'] = {
                'level': 'low',
                'rate': fn_rate,
                'implication': 'Low risk of missing pneumonia cases'
            }
        
        # False positive risk (unnecessary treatment)
        fp_rate = 1 - specificity
        if fp_rate > 0.2:
            risks['false_positive'] = {
                'level': 'high',
                'rate': fp_rate,
                'implication': 'High risk of unnecessary treatment and anxiety'
            }
        elif fp_rate > 0.1:
            risks['false_positive'] = {
                'level': 'medium',
                'rate': fp_rate,
                'implication': 'Moderate risk of unnecessary treatment'
            }
        else:
            risks['false_positive'] = {
                'level': 'low',
                'rate': fp_rate,
                'implication': 'Low risk of unnecessary treatment'
            }
        
        return risks
    
    def _compare_to_clinical_benchmarks(
        self, 
        auc: float, 
        sensitivity: float, 
        specificity: float
    ) -> Dict[str, Any]:
        """Compare performance to clinical benchmarks."""
        
        # Typical radiologist performance benchmarks for pneumonia detection
        radiologist_benchmarks = {
            'expert_radiologist': {'sensitivity': 0.85, 'specificity': 0.90, 'auc': 0.88},
            'general_radiologist': {'sensitivity': 0.80, 'specificity': 0.85, 'auc': 0.83},
            'resident_radiologist': {'sensitivity': 0.75, 'specificity': 0.80, 'auc': 0.78}
        }
        
        comparisons = {}
        for benchmark_name, benchmark_values in radiologist_benchmarks.items():
            comparison = {
                'auc_difference': auc - benchmark_values['auc'],
                'sensitivity_difference': sensitivity - benchmark_values['sensitivity'],
                'specificity_difference': specificity - benchmark_values['specificity']
            }
            
            if (comparison['auc_difference'] >= 0 and 
                comparison['sensitivity_difference'] >= 0 and 
                comparison['specificity_difference'] >= 0):
                comparison['overall'] = 'superior'
            elif (comparison['auc_difference'] >= -0.05 and 
                  comparison['sensitivity_difference'] >= -0.05 and 
                  comparison['specificity_difference'] >= -0.05):
                comparison['overall'] = 'comparable'
            else:
                comparison['overall'] = 'inferior'
            
            comparisons[benchmark_name] = comparison
        
        return comparisons
    
    def _generate_deployment_recommendations(
        self, 
        readiness_factors: Dict[str, bool]
    ) -> List[str]:
        """Generate deployment recommendations based on readiness factors."""
        
        recommendations = []
        
        if not readiness_factors['performance_threshold']:
            recommendations.append("Improve overall model performance (Youden's J < 0.7)")
        
        if not readiness_factors['calibration_quality']:
            recommendations.append("Improve model calibration to reduce overconfidence")
        
        if not readiness_factors['statistical_significance']:
            recommendations.append("Ensure statistical significance of discrimination")
        
        if not readiness_factors['sufficient_sensitivity']:
            recommendations.append("Improve sensitivity to reduce false negatives")
        
        if not readiness_factors['sufficient_specificity']:
            recommendations.append("Improve specificity to reduce false positives")
        
        if not recommendations:
            recommendations.append("Model meets deployment criteria - proceed with clinical validation")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(
        self,
        validation_results: Dict[str, Any],
        extended_metrics: Dict[str, Any],
        clinical_interpretation: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations."""
        
        recommendations = {
            'immediate_actions': [],
            'medium_term_improvements': [],
            'long_term_goals': []
        }
        
        # Immediate actions based on performance
        auc = validation_results['roc_analysis']['auc']
        if auc < 0.8:
            recommendations['immediate_actions'].append(
                "Critical: Improve model discrimination (AUC < 0.8)"
            )
        
        sensitivity = validation_results['medical_metrics']['sensitivity_recall']
        specificity = validation_results['medical_metrics']['specificity']
        
        if sensitivity < 0.8:
            recommendations['immediate_actions'].append(
                "Address low sensitivity to reduce missed pneumonia cases"
            )
        
        if specificity < 0.8:
            recommendations['immediate_actions'].append(
                "Address low specificity to reduce false positives"
            )
        
        # Medium-term improvements
        ece = extended_metrics.get('expected_calibration_error', 0)
        if ece > 0.1:
            recommendations['medium_term_improvements'].append(
                "Implement calibration techniques to improve reliability"
            )
        
        recommendations['medium_term_improvements'].extend([
            "Validate on external datasets",
            "Implement uncertainty quantification",
            "Add explainability features for clinical use"
        ])
        
        # Long-term goals
        recommendations['long_term_goals'].extend([
            "Conduct prospective clinical trial",
            "Integrate with clinical decision support systems",
            "Implement continuous learning from clinical feedback",
            "Validate across diverse patient populations"
        ])
        
        return recommendations
    
    def _plot_calibration_curve(
        self,
        fraction_of_positives: np.ndarray,
        mean_predicted_value: np.ndarray,
        model_name: str,
        ece: float,
        mce: float
    ):
        """Plot calibration curve."""
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve - {model_name}\nECE: {ece:.3f}, MCE: {mce:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_calibration_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {plot_path}")
    
    def _interpret_calibration(self, ece: float, mce: float) -> str:
        """Interpret calibration metrics."""
        
        if ece < 0.05:
            return "Excellent calibration"
        elif ece < 0.1:
            return "Good calibration"
        elif ece < 0.15:
            return "Fair calibration - some overconfidence"
        else:
            return "Poor calibration - significant overconfidence"
    
    def _group_by_image_quality(
        self, 
        detailed_predictions: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Group predictions by image quality."""
        
        quality_groups = {'high': [], 'medium': [], 'low': []}
        
        for i, pred in enumerate(detailed_predictions):
            quality = pred.get('image_quality', {})
            overall_quality = quality.get('overall_medical_quality', 0.5)
            
            if overall_quality >= 0.8:
                quality_groups['high'].append(i)
            elif overall_quality >= 0.6:
                quality_groups['medium'].append(i)
            else:
                quality_groups['low'].append(i)
        
        return {k: np.array(v) for k, v in quality_groups.items()}
    
    def _group_by_processing_time(
        self, 
        processing_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Group by processing time quartiles."""
        
        quartiles = np.percentile(processing_times, [25, 50, 75])
        
        groups = {
            'fast': np.where(processing_times <= quartiles[0])[0],
            'medium': np.where((processing_times > quartiles[0]) & 
                              (processing_times <= quartiles[2]))[0],
            'slow': np.where(processing_times > quartiles[2])[0]
        }
        
        return groups
    
    def _save_analysis_report(
        self, 
        results: Dict[str, Any], 
        model_name: str
    ):
        """Save comprehensive analysis report."""
        
        report_path = self.output_dir / f"{model_name}_performance_analysis.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Performance analysis report saved to {report_path}")

# Global performance analytics instance
performance_analytics = MedicalPerformanceAnalytics()