"""
CNB Model Monitoring System
Advanced monitoring for banking ML models with regulatory compliance
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from azure_storage import CNBAzureStorage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CNBModelMonitor:
    def __init__(self, model=None, window_size=100, drift_threshold=0.1):
        """
        Initialize banking model monitoring system
        
        Args:
            model: Trained ML model to monitor
            window_size (int): Size of sliding window for drift detection
            drift_threshold (float): Threshold for detecting model drift
        """
        self.model = model
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Monitoring data storage
        self.predictions_log = deque(maxlen=1000)  # Last 1000 predictions
        self.performance_history = []
        self.feature_distributions = defaultdict(deque)
        
        # Model performance tracking
        self.baseline_accuracy = None
        self.current_accuracy = None
        self.drift_detected = False
        
        # Initialize Azure Storage for logging
        self.azure_storage = CNBAzureStorage()
        
        logger.info("CNB Model Monitor initialized")
    
    def log_prediction(self, features, prediction, actual=None, metadata=None):
        """
        Log a single prediction for monitoring
        
        Args:
            features (dict): Input features used for prediction
            prediction (dict): Model prediction result
            actual (dict): Actual outcome (if available)
            metadata (dict): Additional metadata (customer_id, timestamp, etc.)
        """
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }
        
        self.predictions_log.append(log_entry)
        
        # Update feature distributions for drift detection
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                self.feature_distributions[feature_name].append(value)
        
        # Log to Azure Storage for audit trail
        if self.azure_storage.is_connected():
            try:
                self._store_prediction_log(log_entry)
            except Exception as e:
                logger.warning(f"Failed to store prediction log: {e}")
        
        logger.debug(f"Logged prediction: {prediction.get('risk_level', 'unknown')} risk")
    
    def _store_prediction_log(self, log_entry):
        """Store prediction log in Azure Storage"""
        filename = f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        
        blob_client = self.azure_storage.blob_service_client.get_blob_client(
            container=self.azure_storage.container_name,
            blob=f"monitoring/{filename}"
        )
        
        blob_client.upload_blob(
            json.dumps(log_entry, indent=2),
            overwrite=True,
            metadata={
                'content_type': 'application/json',
                'data_class': 'monitoring',
                'risk_level': log_entry['prediction'].get('risk_level', 'unknown')
            }
        )
    
    def calculate_performance_metrics(self, actual_outcomes=None):
        """
        Calculate current model performance metrics
        
        Args:
            actual_outcomes (list): List of actual outcomes for recent predictions
        
        Returns:
            dict: Performance metrics
        """
        if len(self.predictions_log) < 10:
            return {'error': 'Insufficient data for performance calculation'}
        
        # Get recent predictions
        recent_predictions = list(self.predictions_log)[-50:]  # Last 50 predictions
        
        if actual_outcomes:
            # Calculate metrics with actual outcomes
            predicted_risks = [p['prediction']['is_risky'] for p in recent_predictions[:len(actual_outcomes)]]
            
            try:
                accuracy = accuracy_score(actual_outcomes, predicted_risks)
                precision = precision_score(actual_outcomes, predicted_risks)
                recall = recall_score(actual_outcomes, predicted_risks)
                f1 = f1_score(actual_outcomes, predicted_risks)
                
                self.current_accuracy = accuracy
                
                performance = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'sample_size': len(actual_outcomes),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.performance_history.append(performance)
                return performance
                
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
                return {'error': f'Metric calculation failed: {str(e)}'}
        
        # Calculate distribution-based metrics
        risk_scores = [p['prediction']['risk_score'] for p in recent_predictions]
        risk_levels = [p['prediction']['risk_level'] for p in recent_predictions]
        
        performance = {
            'avg_risk_score': float(np.mean(risk_scores)),
            'risk_score_std': float(np.std(risk_scores)),
            'high_risk_percentage': len([r for r in risk_levels if r in ['HIGH', 'CRITICAL']]) / len(risk_levels),
            'prediction_count': len(recent_predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        return performance
    
    def detect_model_drift(self):
        """
        Detect if model performance has drifted significantly
        
        Returns:
            dict: Drift detection results
        """
        if len(self.predictions_log) < self.window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Feature distribution drift detection
        feature_drifts = {}
        recent_window = list(self.predictions_log)[-self.window_size:]
        baseline_window = list(self.predictions_log)[-2*self.window_size:-self.window_size] if len(self.predictions_log) >= 2*self.window_size else None
        
        if baseline_window:
            for feature_name in self.feature_distributions.keys():
                recent_values = [p['features'].get(feature_name) for p in recent_window if feature_name in p['features']]
                baseline_values = [p['features'].get(feature_name) for p in baseline_window if feature_name in p['features']]
                
                if recent_values and baseline_values:
                    recent_values = [v for v in recent_values if isinstance(v, (int, float))]
                    baseline_values = [v for v in baseline_values if isinstance(v, (int, float))]
                    
                    if recent_values and baseline_values:
                        recent_mean = np.mean(recent_values)
                        baseline_mean = np.mean(baseline_values)
                        
                        # Calculate relative drift
                        if baseline_mean != 0:
                            drift_ratio = abs(recent_mean - baseline_mean) / abs(baseline_mean)
                            feature_drifts[feature_name] = drift_ratio
        
        # Overall drift assessment
        significant_drifts = {k: v for k, v in feature_drifts.items() if v > self.drift_threshold}
        drift_detected = len(significant_drifts) > 0
        
        # Performance drift (if we have accuracy history)
        performance_drift = False
        if self.baseline_accuracy and self.current_accuracy:
            accuracy_drift = abs(self.current_accuracy - self.baseline_accuracy)
            if accuracy_drift > self.drift_threshold:
                performance_drift = True
        
        drift_result = {
            'drift_detected': drift_detected or performance_drift,
            'feature_drifts': significant_drifts,
            'performance_drift': performance_drift,
            'drift_threshold': self.drift_threshold,
            'recommendation': 'Model retraining recommended' if (drift_detected or performance_drift) else 'Model performance stable',
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_detected = drift_result['drift_detected']
        
        if self.drift_detected:
            logger.warning(f"Model drift detected: {significant_drifts}")
        
        return drift_result
    
    def generate_monitoring_report(self):
        """
        Generate comprehensive monitoring report for regulatory compliance
        
        Returns:
            dict: Comprehensive monitoring report
        """
        current_time = datetime.now()
        
        # Basic statistics
        total_predictions = len(self.predictions_log)
        recent_predictions = list(self.predictions_log)[-24:] if total_predictions >= 24 else list(self.predictions_log)  # Last 24 hours or all
        
        if not recent_predictions:
            return {'error': 'No predictions available for reporting'}
        
        # Risk distribution
        risk_levels = [p['prediction']['risk_level'] for p in recent_predictions]
        risk_distribution = {
            'CRITICAL': risk_levels.count('CRITICAL'),
            'HIGH': risk_levels.count('HIGH'),
            'MEDIUM': risk_levels.count('MEDIUM'),
            'LOW': risk_levels.count('LOW'),
            'MINIMAL': risk_levels.count('MINIMAL')
        }
        
        # Feature analysis
        feature_stats = {}
        for feature_name in ['amount', 'account_balance', 'monthly_income']:
            values = [p['features'].get(feature_name) for p in recent_predictions if feature_name in p['features']]
            values = [v for v in values if isinstance(v, (int, float))]
            
            if values:
                feature_stats[feature_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Performance metrics
        performance = self.calculate_performance_metrics()
        
        # Drift detection
        drift_analysis = self.detect_model_drift()
        
        # Compliance summary
        high_risk_count = risk_distribution['HIGH'] + risk_distribution['CRITICAL']
        alert_rate = high_risk_count / len(recent_predictions) if recent_predictions else 0
        
        report = {
            'report_metadata': {
                'generated_at': current_time.isoformat(),
                'report_period': '24_hours',
                'total_predictions': total_predictions,
                'recent_predictions': len(recent_predictions),
                'model_version': 'CNB_Advanced_Risk_v2.0'
            },
            'risk_distribution': risk_distribution,
            'performance_metrics': performance,
            'feature_statistics': feature_stats,
            'drift_analysis': drift_analysis,
            'compliance_summary': {
                'alert_rate': float(alert_rate),
                'high_risk_transactions': high_risk_count,
                'model_stability': 'STABLE' if not drift_analysis['drift_detected'] else 'UNSTABLE',
                'regulatory_status': 'COMPLIANT' if alert_rate < 0.2 else 'REVIEW_REQUIRED'
            },
            'recommendations': self._generate_recommendations(drift_analysis, alert_rate)
        }
        
        # Store report in Azure
        if self.azure_storage.is_connected():
            try:
                self._store_monitoring_report(report)
            except Exception as e:
                logger.warning(f"Failed to store monitoring report: {e}")
        
        return report
    
    def _generate_recommendations(self, drift_analysis, alert_rate):
        """Generate actionable recommendations based on monitoring data"""
        recommendations = []
        
        if drift_analysis['drift_detected']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Model Retraining',
                'reason': 'Significant model drift detected',
                'timeline': 'Within 48 hours'
            })
        
        if alert_rate > 0.25:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Review Risk Thresholds',
                'reason': f'High alert rate: {alert_rate:.1%}',
                'timeline': 'Within 1 week'
            })
        
        if len(self.predictions_log) > 500:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Performance Review',
                'reason': 'Sufficient data for comprehensive model evaluation',
                'timeline': 'Next monthly review'
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'INFO',
                'action': 'Continue Monitoring',
                'reason': 'Model performance within acceptable parameters',
                'timeline': 'Ongoing'
            })
        
        return recommendations
    
    def _store_monitoring_report(self, report):
        """Store monitoring report in Azure Storage"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"monitoring_report_{timestamp}.json"
        
        blob_client = self.azure_storage.blob_service_client.get_blob_client(
            container=self.azure_storage.container_name,
            blob=f"reports/{filename}"
        )
        
        blob_client.upload_blob(
            json.dumps(report, indent=2),
            overwrite=True,
            metadata={
                'content_type': 'application/json',
                'data_class': 'monitoring_report',
                'alert_rate': str(report['compliance_summary']['alert_rate']),
                'model_status': report['compliance_summary']['model_stability']
            }
        )
        
        logger.info(f"Monitoring report stored: {filename}")
    
    def set_baseline_performance(self, accuracy):
        """Set baseline accuracy for drift detection"""
        self.baseline_accuracy = accuracy
        logger.info(f"Baseline accuracy set to: {accuracy:.3f}")
    
    def get_monitoring_dashboard_data(self):
        """Get data formatted for monitoring dashboard"""
        if not self.predictions_log:
            return {'error': 'No monitoring data available'}
        
        recent_predictions = list(self.predictions_log)[-50:]
        
        # Time series data for charts
        hourly_stats = defaultdict(lambda: {'count': 0, 'high_risk': 0})
        
        for prediction in recent_predictions:
            timestamp = datetime.fromisoformat(prediction['timestamp'])
            hour_key = timestamp.strftime('%H:00')
            
            hourly_stats[hour_key]['count'] += 1
            if prediction['prediction']['risk_level'] in ['HIGH', 'CRITICAL']:
                hourly_stats[hour_key]['high_risk'] += 1
        
        # Convert to chart format
        chart_data = []
        for hour, stats in sorted(hourly_stats.items()):
            chart_data.append({
                'hour': hour,
                'total_predictions': stats['count'],
                'high_risk_count': stats['high_risk'],
                'risk_rate': stats['high_risk'] / stats['count'] if stats['count'] > 0 else 0
            })
        
        dashboard_data = {
            'chart_data': chart_data,
            'summary_stats': self.calculate_performance_metrics(),
            'drift_status': self.detect_model_drift(),
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard_data


# Test function
def test_model_monitoring():
    """Test the monitoring system"""
    print("🧪 Testing CNB Model Monitoring System...")
    
    monitor = CNBModelMonitor()
    
    # Simulate some predictions
    test_features = [
        {'amount': 50000, 'account_balance': 25000, 'monthly_income': 40000},
        {'amount': 100000, 'account_balance': 5000, 'monthly_income': 30000},
        {'amount': 5000, 'account_balance': 50000, 'monthly_income': 60000}
    ]
    
    test_predictions = [
        {'risk_score': 0.3, 'risk_level': 'LOW', 'is_risky': False},
        {'risk_score': 0.8, 'risk_level': 'HIGH', 'is_risky': True},
        {'risk_score': 0.1, 'risk_level': 'MINIMAL', 'is_risky': False}
    ]
    
    # Log test predictions
    for i, (features, prediction) in enumerate(zip(test_features, test_predictions)):
        monitor.log_prediction(
            features=features,
            prediction=prediction,
            metadata={'test_id': i}
        )
    
    # Generate test report
    report = monitor.generate_monitoring_report()
    
    print("✅ Monitoring system test completed")
    print(f"📊 Logged {len(monitor.predictions_log)} predictions")
    print(f"📋 Generated report with {len(report['risk_distribution'])} risk categories")
    
    return monitor, report


if __name__ == "__main__":
    test_model_monitoring()