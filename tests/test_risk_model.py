"""
CNB Risk Model - Automated Tests
Banking-grade testing for CI/CD pipeline
"""

import pytest
import json
import sys
import os

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, predict_advanced_transaction_risk


class TestCNBRiskModel:
    """Test suite pro CNB Risk Management Platform"""
    
    @pytest.fixture
    def client(self):
        """Flask test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
        assert 'timestamp' in data
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint"""
        response = client.get('/api/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        required_fields = [
            'total_transactions', 'risky_transactions', 
            'safe_transactions', 'model_accuracy'
        ]
        for field in required_fields:
            assert field in data
    
    def test_predict_endpoint_valid_data(self, client):
        """Test prediction endpoint with valid data"""
        test_transaction = {
            "amount": 50000,
            "account_balance": 25000,
            "transaction_type": "withdrawal",
            "monthly_income": 40000,
            "customer_age": 35,
            "country": "CZ"
        }
        
        response = client.post('/api/predict', 
                              data=json.dumps(test_transaction),
                              content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'risk_score' in data
        assert 'is_risky' in data
        assert 'risk_level' in data
        assert 0 <= data['risk_score'] <= 1
        assert data['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data"""
        invalid_transactions = [
            {"amount": -1000},  # Negative amount
            {"amount": 50000, "transaction_type": "invalid_type"},  # Invalid type
            {"amount": 50000, "account_balance": -100},  # Negative balance
            {}  # Empty data
        ]
        
        for invalid_data in invalid_transactions:
            response = client.post('/api/predict',
                                  data=json.dumps(invalid_data),
                                  content_type='application/json')
            assert response.status_code == 400
    
    def test_risk_prediction_function(self):
        """Test core risk prediction function"""
        # Test low risk scenario
        result = predict_advanced_transaction_risk(
            amount=5000,
            account_balance=50000,
            transaction_type="deposit",
            monthly_income=60000,
            customer_age=45,
            country="CZ"
        )
        
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert isinstance(result['risk_score'], (int, float))
        assert result['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_high_risk_scenario(self):
        """Test high risk scenario detection"""
        result = predict_advanced_transaction_risk(
            amount=100000,  # High amount
            account_balance=5000,  # Low balance
            transaction_type="withdrawal",
            monthly_income=30000,
            customer_age=22,  # Young customer
            account_age_days=30,  # New account
            country="OFFSHORE",  # High risk country
            transaction_hour=2,  # Night time
            is_cash_business=1  # Cash business
        )
        
        # Should be high risk due to multiple factors
        assert result['risk_score'] > 0.5
        assert result['risk_level'] in ['HIGH', 'CRITICAL']
    
    def test_banking_business_rules(self):
        """Test banking-specific business rules"""
        # Test AML structuring detection (just under reporting limit)
        structuring_amount = 15000 * 24 * 0.95  # Just under limit
        
        result = predict_advanced_transaction_risk(
            amount=structuring_amount,
            account_balance=20000,
            transaction_type="transfer",
            country="OFFSHORE"
        )
        
        # Should flag potential structuring
        assert result['risk_score'] > 0.3
    
    def test_model_consistency(self):
        """Test model prediction consistency"""
        # Same input should give same output
        test_params = {
            "amount": 50000,
            "account_balance": 25000,
            "transaction_type": "transfer",
            "monthly_income": 45000
        }
        
        result1 = predict_advanced_transaction_risk(**test_params)
        result2 = predict_advanced_transaction_risk(**test_params)
        
        # Results should be identical
        assert result1['risk_score'] == result2['risk_score']
        assert result1['risk_level'] == result2['risk_level']
    
    def test_adaptive_learning_flag(self):
        """Test that adaptive learning is properly flagged"""
        result = predict_advanced_transaction_risk(
            amount=30000,
            account_balance=15000,
            transaction_type="withdrawal"
        )
        
        # Should indicate adaptive learning capability
        assert result.get('adaptive_learning') == True
    
    def test_risk_factors_explanation(self):
        """Test risk factors are properly explained"""
        # High risk scenario should provide explanations
        result = predict_advanced_transaction_risk(
            amount=200000,  # Very high amount
            account_balance=5000,  # Low balance
            transaction_type="withdrawal",
            monthly_income=25000,  # Low income vs amount
            transaction_hour=3,  # Night time
            country="HIGH_RISK"
        )
        
        assert 'risk_factors' in result
        assert len(result['risk_factors']) > 0
        assert 'explanation' in result


class TestCNBSecurityCompliance:
    """Security and compliance tests for banking regulations"""
    
    def test_no_sensitive_data_exposure(self, client):
        """Ensure no sensitive data is exposed in responses"""
        response = client.get('/api/stats')
        data = json.loads(response.data)
        
        # Should not contain any customer-specific data
        sensitive_fields = ['customer_id', 'account_number', 'ssn', 'personal_id']
        for field in sensitive_fields:
            assert field not in str(data)
    
    def test_input_validation_security(self, client):
        """Test security input validation"""
        # Test SQL injection attempt
        malicious_input = {
            "amount": "50000; DROP TABLE users;",
            "transaction_type": "<script>alert('xss')</script>"
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(malicious_input),
                              content_type='application/json')
        
        # Should either reject or sanitize
        assert response.status_code in [400, 500]  # Should not succeed
    
    def test_rate_limiting_preparation(self, client):
        """Test that app can handle multiple requests (DoS protection)"""
        # Send multiple rapid requests
        for i in range(10):
            response = client.get('/health')
            assert response.status_code == 200


class TestCNBPerformance:
    """Performance tests for production readiness"""
    
    def test_prediction_response_time(self):
        """Test prediction response time is acceptable"""
        import time
        
        start_time = time.time()
        
        result = predict_advanced_transaction_risk(
            amount=50000,
            account_balance=25000,
            transaction_type="transfer"
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should respond within 2 seconds (banking requirement)
        assert response_time < 2.0
        assert 'risk_score' in result
    
    def test_concurrent_predictions(self):
        """Test model can handle concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                result = predict_advanced_transaction_risk(
                    amount=50000,
                    account_balance=25000,
                    transaction_type="transfer"
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create 5 concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All predictions should succeed
        assert len(errors) == 0
        assert len(results) == 5


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])