"""
CNB Risk Platform - Azure Deployment Fixed
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime, timedelta
import os
import logging
import threading

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and data
model = None
model_accuracy = 0
data_stats = {}
feature_columns = []
model_training_lock = threading.Lock()
model_trained = False

# Constants
VALID_TRANSACTION_TYPES = ['deposit', 'withdrawal', 'transfer', 'payment', 'card_payment']
VALID_COUNTRIES = ['CZ', 'SK', 'DE', 'AT', 'PL', 'OFFSHORE', 'HIGH_RISK', 'OTHER']

def ensure_model_trained():
    """Ensure model is trained with thread safety"""
    global model_trained
    
    if model_trained and model is not None:
        return True
    
    with model_training_lock:
        if model_trained and model is not None:
            return True
        
        try:
            logger.info("Training model on demand...")
            success = train_quick_risk_model()
            if success:
                model_trained = True
                logger.info("Model trained successfully")
                return True
            else:
                logger.error("Model training failed")
                return False
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

def generate_quick_banking_data():
    """Generate smaller dataset for faster Azure startup"""
    n_rows = 5000  # Menší dataset pro rychlejší startup
    
    logger.info(f"Generating {n_rows} synthetic transactions...")
    
    # Basic data generation (simplified)
    customer_ids = np.random.randint(1000, 9999, size=n_rows)
    
    # Simple time features
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]
    hours = np.random.randint(0, 24, size=n_rows)
    
    # Transaction types
    transaction_types = np.random.choice(
        VALID_TRANSACTION_TYPES, 
        size=n_rows,
        p=[0.15, 0.20, 0.25, 0.25, 0.15]
    )
    
    # Customer profiles
    customer_profiles = np.random.choice(['low_income', 'middle_income', 'high_income', 'business'], 
                                       size=n_rows, p=[0.3, 0.5, 0.15, 0.05])
    
    # Demographics
    ages = np.random.normal(45, 15, n_rows).astype(int)
    ages = np.clip(ages, 18, 85)
    
    account_ages = np.random.exponential(500, n_rows).astype(int)
    account_ages = np.clip(account_ages, 1, 3650)
    
    # Income-based amounts (simplified)
    amounts = np.random.lognormal(8, 1.5, n_rows)
    monthly_incomes = np.random.normal(50000, 20000, n_rows)
    monthly_incomes = np.clip(monthly_incomes, 10000, 300000)
    
    # Account balances
    account_balances = monthly_incomes * np.random.uniform(0.5, 2.0, n_rows)
    account_balances = np.clip(account_balances, 0, None)
    
    # Countries
    countries = np.random.choice(['CZ', 'SK', 'DE', 'AT', 'PL', 'OFFSHORE', 'HIGH_RISK'], 
                                size=n_rows, 
                                p=[0.70, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01])
    
    country_risk_scores = {
        'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
        'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
    }
    
    # Frequency features
    transactions_7d = np.random.poisson(3, n_rows)
    transactions_30d = np.random.poisson(12, n_rows)
    
    # Cash business
    is_cash_business = np.random.choice([0, 1], size=n_rows, p=[0.85, 0.15])
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'transaction_date': dates,
        'transaction_hour': hours,
        'amount': amounts,
        'transaction_type': transaction_types,
        'account_balance': account_balances,
        'monthly_income': monthly_incomes,
        'customer_age': ages,
        'account_age_days': account_ages,
        'country': countries,
        'country_risk_score': [country_risk_scores[c] for c in countries],
        'transactions_last_7d': transactions_7d,
        'transactions_last_30d': transactions_30d,
        'customer_profile': customer_profiles,
        'is_cash_business': is_cash_business
    })
    
    logger.info(f"Generated {len(df)} transactions")
    return df

def create_quick_risk_labels(df):
    """Simplified risk labeling"""
    risk_score = np.zeros(len(df))
    
    # Simplified risk factors
    reporting_limit = 360000  # 15K EUR in CZK
    
    # High amount transactions
    high_amount = df['amount'] > (df['monthly_income'] * 0.5)
    risk_score += high_amount * 2
    
    # Country risk
    high_country_risk = df['country_risk_score'] >= 7
    risk_score += high_country_risk * 3
    
    # Night transactions
    night_hours = (df['transaction_hour'] <= 6) | (df['transaction_hour'] >= 23)
    risk_score += night_hours * 2
    
    # New accounts with large transactions
    new_account_risk = (df['account_age_days'] < 90) & (df['amount'] > df['monthly_income'])
    risk_score += new_account_risk * 2
    
    # Cash business
    cash_risk = (df['is_cash_business'] == 1) & (df['amount'] > 100000)
    risk_score += cash_risk * 2
    
    # Define risky transactions
    risk_threshold = np.percentile(risk_score, 85)
    is_risky = (risk_score >= max(2, risk_threshold)).astype(int)
    
    return is_risky, risk_score

def train_quick_risk_model():
    """Train simplified model for Azure deployment"""
    global model, model_accuracy, data_stats, feature_columns
    
    try:
        logger.info("🔄 Training quick risk model for Azure...")
        
        # Generate data
        data = generate_quick_banking_data()
        data['is_risky'], data['risk_score_raw'] = create_quick_risk_labels(data)
        
        # Simple feature engineering
        data['is_night_time'] = (data['transaction_hour'] <= 6) | (data['transaction_hour'] >= 22)
        data['is_business_hours'] = (data['transaction_hour'] >= 9) & (data['transaction_hour'] <= 17)
        data['amount_to_income_ratio'] = data['amount'] / (data['monthly_income'] + 1)
        data['amount_to_balance_ratio'] = data['amount'] / (data['account_balance'] + 1)
        
        # Select features (simplified)
        feature_cols = [
            'amount', 'account_balance', 'monthly_income', 'customer_age', 'account_age_days',
            'country_risk_score', 'transactions_last_7d', 'transactions_last_30d',
            'amount_to_income_ratio', 'amount_to_balance_ratio', 'is_cash_business',
            'is_night_time', 'is_business_hours', 'transaction_hour'
        ]
        
        # Add dummy variables (simplified)
        for trans_type in VALID_TRANSACTION_TYPES:
            data[f'trans_{trans_type}'] = (data['transaction_type'] == trans_type).astype(int)
            feature_cols.append(f'trans_{trans_type}')
        
        for profile in ['low_income', 'middle_income', 'high_income', 'business']:
            data[f'profile_{profile}'] = (data['customer_profile'] == profile).astype(int)
            feature_cols.append(f'profile_{profile}')
        
        # Prepare training data
        X = data[feature_cols]
        y = data['is_risky']
        
        # Store feature columns
        feature_columns = feature_cols
        
        # Fill NaN values
        X = X.fillna(0)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train simplified model
        model = RandomForestClassifier(
            n_estimators=50,  # Menší počet pro rychlejší training
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            class_weight='balanced',
            n_jobs=1  # Single thread pro Azure
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, predictions)
        
        # Calculate statistics
        risky_count = int(data['is_risky'].sum())
        safe_count = len(data) - risky_count
        
        data_stats = {
            'total_transactions': len(data),
            'risky_transactions': risky_count,
            'safe_transactions': safe_count,
            'risk_rate': float(risky_count / len(data)),
            'model_accuracy': float(model_accuracy),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns),
            'avg_transaction_amount': float(data['amount'].mean()),
            'avg_account_balance': float(data['account_balance'].mean()),
            'avg_monthly_income': float(data['monthly_income'].mean()),
            'high_risk_countries': int((data['country_risk_score'] >= 7).sum()),
            'night_transactions': int(data['is_night_time'].sum()),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"✅ Quick model trained: {model_accuracy:.1%} accuracy, {len(feature_columns)} features")
        return True
        
    except Exception as e:
        logger.error(f"Error training quick model: {e}")
        return False

def predict_transaction_risk(amount, account_balance, transaction_type, 
                           monthly_income=50000, customer_age=35, 
                           account_age_days=365, country='CZ',
                           transaction_hour=12, is_cash_business=0):
    """Simplified prediction function"""
    
    if not ensure_model_trained():
        return {'error': 'Model not available'}
    
    try:
        # Country risk mapping
        country_risk_map = {
            'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
            'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
        }
        country_risk_score = country_risk_map.get(country, 5)
        
        # Derived features
        amount_to_income_ratio = amount / max(monthly_income, 1)
        amount_to_balance_ratio = amount / max(account_balance + 1, 1)
        
        # Time features
        is_night_time = 1 if (transaction_hour <= 6 or transaction_hour >= 22) else 0
        is_business_hours = 1 if (9 <= transaction_hour <= 17) else 0
        
        # Default frequency
        transactions_last_7d = 3
        transactions_last_30d = 12
        
        # Create feature vector
        features = {
            'amount': amount,
            'account_balance': account_balance,
            'monthly_income': monthly_income,
            'customer_age': customer_age,
            'account_age_days': account_age_days,
            'country_risk_score': country_risk_score,
            'transactions_last_7d': transactions_last_7d,
            'transactions_last_30d': transactions_last_30d,
            'amount_to_income_ratio': amount_to_income_ratio,
            'amount_to_balance_ratio': amount_to_balance_ratio,
            'is_cash_business': is_cash_business,
            'is_night_time': is_night_time,
            'is_business_hours': is_business_hours,
            'transaction_hour': transaction_hour
        }
        
        # Add transaction type dummies
        for trans_type in VALID_TRANSACTION_TYPES:
            features[f'trans_{trans_type}'] = 1 if transaction_type == trans_type else 0
        
        # Add profile dummies (estimated)
        if monthly_income < 30000:
            profile = 'low_income'
        elif monthly_income < 70000:
            profile = 'middle_income'
        elif monthly_income < 150000:
            profile = 'high_income'
        else:
            profile = 'business'
        
        for prof in ['low_income', 'middle_income', 'high_income', 'business']:
            features[f'profile_{prof}'] = 1 if profile == prof else 0
        
        # Create feature array in correct order
        feature_array = np.array([[features[col] for col in feature_columns]])
        
        # Predict
        risk_probability = model.predict_proba(feature_array)[0][1]
        risk_prediction = model.predict(feature_array)[0]
        
        # Risk level
        if risk_probability >= 0.8:
            risk_level = 'CRITICAL'
        elif risk_probability >= 0.6:
            risk_level = 'HIGH'
        elif risk_probability >= 0.4:
            risk_level = 'MEDIUM'
        elif risk_probability >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        # Risk factors
        risk_factors = []
        if amount_to_income_ratio > 0.5:
            risk_factors.append(f"Vysoká částka vs. příjem ({amount_to_income_ratio:.1%})")
        if country_risk_score >= 7:
            risk_factors.append(f"Riziková země ({country})")
        if is_night_time:
            risk_factors.append("Noční transakce")
        if account_age_days < 90:
            risk_factors.append("Nový účet")
        if amount_to_balance_ratio > 0.8:
            risk_factors.append("Vysoký poměr k zůstatku")
        
        return {
            'risk_score': float(risk_probability),
            'is_risky': bool(risk_prediction),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'explanation': f"Model identifikoval {len(risk_factors)} rizikových faktorů" if risk_factors else "Transakce vyhodnocena jako standardní"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {'error': f'Prediction error: {str(e)}'}

# Flask routes
@app.route('/')
def dashboard():
    try:
        # Ensure model is trained before rendering
        if not ensure_model_trained():
            # Return simple error page if model training fails
            return '''
            <h1>ČNB Risk Platform</h1>
            <p>Model se právě načítá... Zkuste obnovit stránku za chvíli.</p>
            <script>setTimeout(function(){ window.location.reload(); }, 5000);</script>
            ''', 503
        
        return render_template('dashboard.html', stats=data_stats)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return f'<h1>Error</h1><p>{str(e)}</p>', 500

@app.route('/api/stats')
def get_stats():
    try:
        if not ensure_model_trained():
            return jsonify({'error': 'Model not ready'}), 503
        return jsonify(data_stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    try:
        if not ensure_model_trained():
            return jsonify({'error': 'Model not ready, please try again in a moment'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get and validate inputs
        amount = float(data.get('amount', 0))
        account_balance = float(data.get('account_balance', 0))
        transaction_type = data.get('transaction_type', 'transfer')
        
        # Optional fields
        monthly_income = float(data.get('monthly_income', 50000))
        customer_age = int(data.get('customer_age', 35))
        account_age_days = int(data.get('account_age_days', 365))
        country = data.get('country', 'CZ')
        transaction_hour = int(data.get('transaction_hour', 12))
        is_cash_business = int(data.get('is_cash_business', 0))
        
        # Validation
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        if amount > 10000000:
            return jsonify({'error': 'Amount too large'}), 400
        if account_balance < 0:
            return jsonify({'error': 'Account balance cannot be negative'}), 400
        if transaction_type not in VALID_TRANSACTION_TYPES:
            return jsonify({'error': f'Invalid transaction type'}), 400
        if country not in VALID_COUNTRIES:
            return jsonify({'error': f'Invalid country'}), 400
        
        result = predict_transaction_risk(
            amount=amount,
            account_balance=account_balance,
            transaction_type=transaction_type,
            monthly_income=monthly_income,
            customer_age=customer_age,
            account_age_days=account_age_days,
            country=country,
            transaction_hour=transaction_hour,
            is_cash_business=is_cash_business
        )
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        global model_trained
        model_trained = False  # Force retrain
        
        if ensure_model_trained():
            return jsonify({
                'message': 'Model retrained successfully',
                'accuracy': model_accuracy,
                'features': len(feature_columns),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Retraining failed'}), 500
    except Exception as e:
        logger.error(f"Error retraining: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        model_ready = model is not None and model_trained
        return jsonify({
            'status': 'healthy' if model_ready else 'initializing',
            'model_loaded': model_ready,
            'model_type': 'Quick Banking Risk Model',
            'features': len(feature_columns) if feature_columns else 0,
            'accuracy': float(model_accuracy) if model_accuracy else None,
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0-azure'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize on first request (Azure-friendly)
@app.before_first_request
def initialize_model():
    """Initialize model on first request"""
    try:
        logger.info("🚀 Initializing model on first request...")
        # Don't block startup, train in background
        threading.Thread(target=ensure_model_trained, daemon=True).start()
    except Exception as e:
        logger.error(f"Error in before_first_request: {e}")

if __name__ == '__main__':
    # For local development
    ensure_model_trained()
    
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting CNB Risk Platform on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)