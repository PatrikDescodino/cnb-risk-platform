"""
CNB Risk Model - Automated Tests (Fixed Version)
Banking-grade testing for CI/CD pipeline
"""

import pytest
import json
import sys
import os
import time

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class TestCNBRiskModel:
    """Test suite pro CNB Risk Management Platform"""
    
    @pytest.fixture
    def client(self):
        """Flask test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            # Wait for model to initialize
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = client.get('/health')
                    data = json.loads(response.data)
                    if data.get('model_loaded'):
                        break
                    time.sleep(2)
                except:
                    time.sleep(2)
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
        
        # Check for error message in case model is still training
        if 'error' in data:
            if 'training in progress' in data['error'].lower():
                pytest.skip("Model training in progress - skipping test")
            else:
                pytest.fail(f"Unexpected error: {data['error']}")
        
        # Normal validation
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
    
    def test_basic_prediction_logic(self):
        """Test basic prediction without full model"""
        # Import the function directly
        try:
            from app import predict_advanced_transaction_risk
        except ImportError:
            pytest.skip("Cannot import prediction function")
        
        # Test with basic parameters
        try:
            result = predict_advanced_transaction_risk(
                amount=5000,
                account_balance=50000,
                transaction_type="deposit",
                monthly_income=60000,
                customer_age=45,
                country="CZ"
            )
            
            # Check if we get an error or valid result
            if isinstance(result, dict) and 'error' in result:
                if 'training in progress' in result['error'].lower():
                    pytest.skip("Model training in progress")
                else:
                    pytest.fail(f"Prediction error: {result['error']}")
            
            # Validate successful prediction
            assert 'risk_score' in result
            assert 'risk_level' in result
            assert isinstance(result['risk_score'], (int, float))
            assert result['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            
        except Exception as e:
            if "training in progress" in str(e).lower():
                pytest.skip("Model training in progress")
            else:
                raise


class TestCNBSecurityCompliance:
    """Security and compliance tests for banking regulations"""
    
    @pytest.fixture
    def client(self):
        """Flask test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
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
        # Test malicious input
        malicious_input = {
            "amount": "50000; DROP TABLE users;",
            "transaction_type": "<script>alert('xss')</script>"
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(malicious_input),
                              content_type='application/json')
        
        # Should either reject or sanitize
        assert response.status_code in [400, 500]  # Should not succeed
    
    def test_basic_endpoints_availability(self, client):
        """Test that basic endpoints are available"""
        endpoints = ['/health', '/api/stats']
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestCNBPerformance:
    """Performance tests for production readiness"""
    
    def test_health_endpoint_response_time(self):
        """Test health endpoint response time"""
        import time
        
        app.config['TESTING'] = True
        with app.test_client() as client:
            start_time = time.time()
            response = client.get('/health')
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Health check should be very fast
            assert response_time < 5.0
            assert response.status_code == 200
    
    def test_stats_endpoint_response_time(self):
        """Test stats endpoint response time"""
        import time
        
        app.config['TESTING'] = True
        with app.test_client() as client:
            start_time = time.time()
            response = client.get('/api/stats')
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Stats should respond quickly
            assert response_time < 10.0
            assert response.status_code == 200


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])