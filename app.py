"""
CNB Risk Platform - Advanced Banking Risk Model
Enhanced with realistic banking risk factors for AML, Fraud Detection, and Credit Risk
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime, timedelta
import os
import logging
from azure_storage import CNBAzureStorage
from dotenv import load_dotenv
from model_monitoring import CNBModelMonitor

# Load environment variables
load_dotenv()

# Initialize Azure Storage
azure_storage = CNBAzureStorage()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and data
model = None
model_monitor = None
model_accuracy = 0
data_stats = {}
feature_columns = []

# Constants
VALID_TRANSACTION_TYPES = ['deposit', 'withdrawal', 'transfer', 'payment', 'card_payment']
VALID_COUNTRIES = ['CZ', 'SK', 'DE', 'AT', 'PL', 'OFFSHORE', 'HIGH_RISK', 'OTHER']

def generate_advanced_banking_data():
    """
    Generate synthetic banking transaction data with realistic risk factors
    Enhanced with AML, Fraud Detection, and Credit Risk features
    """
    n_rows = 15000  # Více dat pro lepší training
    
    logger.info(f"Generating {n_rows} synthetic banking transactions...")
    
    # Basic customer data
    customer_ids = np.random.randint(1000, 9999, size=n_rows)
    
    # Advanced time features
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]
    hours = np.random.choice(range(24), size=n_rows)
    
    # Transaction types with realistic distribution
    transaction_types = np.random.choice(VALID_TRANSACTION_TYPES, size=n_rows)
    
    # Advanced customer profiles
    customer_profiles = np.random.choice(['low_income', 'middle_income', 'high_income', 'business'], size=n_rows)
    
    # Customer demographics
    ages = np.random.normal(45, 15, n_rows).astype(int)
    ages = np.clip(ages, 18, 85)
    
    # Account age (days since account opening)
    account_ages = np.random.exponential(500, n_rows).astype(int)
    account_ages = np.clip(account_ages, 1, 3650)  # 1 den až 10 let
    
    # Income-based transaction amounts
    amounts = []
    monthly_incomes = []
    
    for profile in customer_profiles:
        if profile == 'low_income':
            income = np.random.normal(25000, 5000)
            amount = np.random.lognormal(7, 1.5)  # Menší transakce
        elif profile == 'middle_income':
            income = np.random.normal(50000, 10000)
            amount = np.random.lognormal(8, 1.8)
        elif profile == 'high_income':
            income = np.random.normal(100000, 20000)
            amount = np.random.lognormal(9, 2.0)  # Větší transakce
        else:  # business
            income = np.random.normal(200000, 50000)
            amount = np.random.lognormal(10, 2.5)  # Největší transakce
            
        amounts.append(max(1, amount))
        monthly_incomes.append(max(10000, income / 12))  # Měsíční příjem
    
    amounts = np.array(amounts)
    monthly_incomes = np.array(monthly_incomes)
    
    # Account balances (souvisí s příjmem)
    account_balances = []
    for income in monthly_incomes:
        # Zůstatek = 0.5 až 3x měsíční příjem + noise
        balance = income * np.random.uniform(0.5, 3.0) + np.random.normal(0, income * 0.2)
        account_balances.append(max(0, balance))
    
    account_balances = np.array(account_balances)
    
    # Geographical risk (simulace rizikových zemí)
    countries = np.random.choice(VALID_COUNTRIES[:-1],  # Exclude 'OTHER' from generation
                                size=n_rows, 
                                p=[0.70, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01])
    
    country_risk_scores = {
        'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
        'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
    }
    
    # Frequency features (kolik transakcí za posledních 7/30 dní)
    transactions_7d = np.random.poisson(3, n_rows)  # Průměr 3 transakce/týden
    transactions_30d = np.random.poisson(12, n_rows)  # Průměr 12 transakcí/měsíc
    
    # Cash-intensive business indicators
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
    
    logger.info(f"Generated {len(df)} transactions successfully")
    return df

def create_advanced_risk_labels(df):
    """
    Create sophisticated risk labels based on multiple banking risk factors
    """
    risk_score = np.zeros(len(df))
    
    # 1. AML RISK FACTORS
    
    # Strukturované transakce (těsně pod reporting limity)
    reporting_limit = 15000 * 24  # EUR to CZK conversion
    structured_transactions = (df['amount'] > reporting_limit * 0.9) & (df['amount'] < reporting_limit)
    risk_score += structured_transactions * 3
    
    # Neobvyklé částky vs příjem (více než 50% měsíčního příjmu)
    unusual_amount = df['amount'] > (df['monthly_income'] * 0.5)
    risk_score += unusual_amount * 2
    
    # Vysoký country risk
    high_country_risk = df['country_risk_score'] >= 7
    risk_score += high_country_risk * 4
    
    # Častá aktivita (velocity risk)
    high_frequency = df['transactions_last_7d'] > 10
    risk_score += high_frequency * 2
    
    # 2. FRAUD DETECTION FACTORS
    
    # Podezřelé hodiny (0:00 - 6:00)
    suspicious_hours = (df['transaction_hour'] <= 6) | (df['transaction_hour'] >= 23)
    risk_score += suspicious_hours * 2
    
    # Mladý účet s velkými transakcemi
    new_account_risk = (df['account_age_days'] < 30) & (df['amount'] > df['monthly_income'])
    risk_score += new_account_risk * 3
    
    # Výběry převyšující zůstatek (overdraft risk)
    overdraft_risk = (df['transaction_type'] == 'withdrawal') & (df['amount'] > df['account_balance'] * 0.8)
    risk_score += overdraft_risk * 2
    
    # 3. BUSINESS LOGIC RISK
    
    # Cash-intensive business s velkými hotovostními transakcemi
    cash_business_risk = (df['is_cash_business'] == 1) & (df['amount'] > 100000)
    risk_score += cash_business_risk * 2
    
    # Nízký zůstatek s velkými transakcemi
    low_balance_risk = (df['account_balance'] < df['monthly_income'] * 0.2) & (df['amount'] > df['monthly_income'] * 0.3)
    risk_score += low_balance_risk * 2
    
    # Stáří zákazníka vs. riziková aktivita
    young_high_risk = (df['customer_age'] < 25) & (df['amount'] > 50000)
    risk_score += young_high_risk * 1
    
    # 4. KOMBINOVANÉ FAKTORY
    
    # Offshore + vysoká částka + cash business
    offshore_combo = (df['country_risk_score'] >= 7) & (df['amount'] > 200000) & (df['is_cash_business'] == 1)
    risk_score += offshore_combo * 5
    
    # Definice rizikových transakcí (prahová hodnota)
    # Místo pevného prahu použijeme percentile
    risk_threshold = np.percentile(risk_score, 85)  # Top 15% nejrizikovějších
    is_risky = (risk_score >= max(3, risk_threshold)).astype(int)
    
    return is_risky, risk_score

def train_advanced_risk_model():
    """
    Train enhanced risk assessment model with banking-grade features
    """
    global model, model_accuracy, data_stats, feature_columns
    
    logger.info("Training advanced banking risk model...")
    
    try:
        # Generate enhanced dataset
        data = generate_advanced_banking_data()
        data['is_risky'], data['risk_score_raw'] = create_advanced_risk_labels(data)
        
        # Feature Engineering
        
        # Časové features
        data['is_weekend'] = pd.to_datetime(data['transaction_date']).dt.weekday >= 5
        data['is_night_time'] = (data['transaction_hour'] <= 6) | (data['transaction_hour'] >= 22)
        data['is_business_hours'] = (data['transaction_hour'] >= 9) & (data['transaction_hour'] <= 17)
        
        # Poměrové features
        data['amount_to_income_ratio'] = data['amount'] / data['monthly_income']
        data['amount_to_balance_ratio'] = data['amount'] / (data['account_balance'] + 1)
        data['balance_to_income_ratio'] = data['account_balance'] / data['monthly_income']
        data['frequency_velocity'] = data['transactions_last_7d'] / 7  # Denní průměr
        
        # Kategorizace částek
        data['amount_category'] = pd.cut(data['amount'], 
                                       bins=[0, 1000, 10000, 50000, 200000, float('inf')],
                                       labels=['micro', 'small', 'medium', 'large', 'jumbo'])
        
        # Dummy variables with error handling
        try:
            transaction_dummies = pd.get_dummies(data['transaction_type'], prefix='trans')
            profile_dummies = pd.get_dummies(data['customer_profile'], prefix='profile')
            amount_cat_dummies = pd.get_dummies(data['amount_category'], prefix='amount')
        except Exception as e:
            logger.error(f"Error creating dummy variables: {e}")
            raise
        
        # Numerical features
        numerical_features = [
            'amount', 'account_balance', 'monthly_income', 'customer_age', 'account_age_days',
            'country_risk_score', 'transactions_last_7d', 'transactions_last_30d',
            'amount_to_income_ratio', 'amount_to_balance_ratio', 'balance_to_income_ratio',
            'frequency_velocity', 'is_cash_business', 'is_weekend', 'is_night_time', 
            'is_business_hours', 'transaction_hour'
        ]
        
        # Combine all features
        X = pd.concat([
            data[numerical_features],
            transaction_dummies,
            profile_dummies,
            amount_cat_dummies
        ], axis=1)
        
        # Store feature columns for prediction
        feature_columns = X.columns.tolist()
        
        y = data['is_risky']
        
        # Validate data quality
        if X.isnull().any().any():
            logger.warning("NaN values detected in features, filling with 0")
            X = X.fillna(0)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model with better parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',  # Handle imbalanced data
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, predictions)
        
        # Detailed evaluation
        print("\n" + "="*60)
        print("🏦 ADVANCED CNB RISK ASSESSMENT MODEL")
        print("="*60)
        print(f"✅ Model accuracy: {model_accuracy:.1%}")
        print(f"✅ Training samples: {len(X_train):,}")
        print(f"✅ Test samples: {len(X_test):,}")
        print(f"✅ Features used: {len(feature_columns)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 10 RISK INDICATORS:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.3f}")
        
        # Calculate enhanced statistics
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
        
        print(f"📈 Risk rate: {data_stats['risk_rate']:.1%}")
        print(f"💰 Avg transaction: {data_stats['avg_transaction_amount']:,.0f} Kč")
        print("🚀 Advanced risk model ready!")
        
        logger.info("Model training completed successfully")

        global model_monitor
        model_monitor = CNBModelMonitor(model=model)
        if model_accuracy:
            model_monitor.set_baseline_performance(model_accuracy)
        
        logger.info("Model monitor initialized successfully")
        return model, model_accuracy, data_stats
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def predict_advanced_transaction_risk(amount, account_balance, transaction_type, 
                                    monthly_income=50000, customer_age=35, 
                                    account_age_days=365, country='CZ',
                                    transaction_hour=12, is_cash_business=0):
    """
    Advanced risk prediction with realistic banking factors
    """
    global model, feature_columns
    
    if model is None:
        logger.error("Model not trained - cannot make prediction")
        return {'error': 'Model not trained'}
    
    try:
        # Country risk mapping with validation
        country_risk_map = {
            'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
            'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
        }
        country_risk_score = country_risk_map.get(country, 5)
        
        # Derived features with safe division
        amount_to_income_ratio = amount / max(monthly_income, 1)
        amount_to_balance_ratio = amount / max(account_balance + 1, 1)
        balance_to_income_ratio = account_balance / max(monthly_income, 1)
        
        # Time features
        is_weekend = 0  # Simplified for demo
        is_night_time = 1 if (transaction_hour <= 6 or transaction_hour >= 22) else 0
        is_business_hours = 1 if (9 <= transaction_hour <= 17) else 0
        
        # Estimate frequency (simplified)
        transactions_last_7d = 3  # Default assumption
        transactions_last_30d = 12
        frequency_velocity = transactions_last_7d / 7
        
        # Amount category
        if amount <= 1000:
            amount_cat = 'micro'
        elif amount <= 10000:
            amount_cat = 'small'
        elif amount <= 50000:
            amount_cat = 'medium'
        elif amount <= 200000:
            amount_cat = 'large'
        else:
            amount_cat = 'jumbo'
        
        # Customer profile estimation based on income
        if monthly_income < 30000:
            customer_profile = 'low_income'
        elif monthly_income < 70000:
            customer_profile = 'middle_income'
        elif monthly_income < 150000:
            customer_profile = 'high_income'
        else:
            customer_profile = 'business'
        
        # Create feature vector matching training data
        feature_dict = {}
        
        # Numerical features
        numerical_values = {
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
            'balance_to_income_ratio': balance_to_income_ratio,
            'frequency_velocity': frequency_velocity,
            'is_cash_business': is_cash_business,
            'is_weekend': is_weekend,
            'is_night_time': is_night_time,
            'is_business_hours': is_business_hours,
            'transaction_hour': transaction_hour
        }
        
        # Initialize all features to 0
        for col in feature_columns:
            feature_dict[col] = 0
        
        # Set numerical features
        for key, value in numerical_values.items():
            if key in feature_dict:
                feature_dict[key] = value
        
        # Set dummy variables with validation
        trans_col = f'trans_{transaction_type}'
        if trans_col in feature_dict:
            feature_dict[trans_col] = 1
        
        profile_col = f'profile_{customer_profile}'
        if profile_col in feature_dict:
            feature_dict[profile_col] = 1
            
        amount_col = f'amount_{amount_cat}'
        if amount_col in feature_dict:
            feature_dict[amount_col] = 1
        
        # Create feature array
        features = np.array([[feature_dict[col] for col in feature_columns]])
        
        # Predict
        risk_probability = model.predict_proba(features)[0][1]
        risk_prediction = model.predict(features)[0]
        
        # Enhanced risk level classification
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
        
        # Risk factors explanation
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
        
        result = {
            'risk_score': float(risk_probability),
            'is_risky': bool(risk_prediction),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'explanation': f"Model identifikoval {len(risk_factors)} rizikových faktorů" if risk_factors else "Transakce vyhodnocena jako standardní"
        }
        
        logger.info(f"Prediction completed: Risk score {risk_probability:.3f}, Level: {risk_level}")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {'error': f'Prediction error: {str(e)}'}

# Flask routes with improved error handling
@app.route('/')
def dashboard():
    try:
        return render_template('dashboard.html', stats=data_stats)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return jsonify({'error': 'Dashboard unavailable'}), 500

@app.route('/api/stats')
def get_stats():
    try:
        return jsonify(data_stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Stats unavailable'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Basic required fields with validation
        amount = float(data.get('amount', 0))
        account_balance = float(data.get('account_balance', 0))
        transaction_type = data.get('transaction_type', 'transfer')
        
        # Advanced optional fields with defaults
        monthly_income = float(data.get('monthly_income', 50000))
        customer_age = int(data.get('customer_age', 35))
        account_age_days = int(data.get('account_age_days', 365))
        country = data.get('country', 'CZ')
        transaction_hour = int(data.get('transaction_hour', 12))
        is_cash_business = int(data.get('is_cash_business', 0))
        
        # Enhanced validation
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        if amount > 10000000:  # 10M limit
            return jsonify({'error': 'Amount too large'}), 400
        if account_balance < 0:
            return jsonify({'error': 'Account balance cannot be negative'}), 400
        if transaction_type not in VALID_TRANSACTION_TYPES:
            return jsonify({'error': f'Invalid transaction type. Valid types: {VALID_TRANSACTION_TYPES}'}), 400
        if country not in VALID_COUNTRIES:
            return jsonify({'error': f'Invalid country. Valid countries: {VALID_COUNTRIES}'}), 400
        if not (0 <= transaction_hour <= 23):
            return jsonify({'error': 'Transaction hour must be between 0 and 23'}), 400
        if not (18 <= customer_age <= 120):
            return jsonify({'error': 'Customer age must be between 18 and 120'}), 400
        
        result = predict_advanced_transaction_risk(
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
        
        if azure_storage.is_connected():
            upload_result = azure_storage.upload_risk_analysis(
                transaction_data=data,
                risk_result=result,
                customer_id=f"customer_{hash(str(data.get('amount', 0)))}"
            )

        model_monitor.log_prediction(
            features=data,
            prediction=result,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'source': 'web_api'
            }
        )
    
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        logger.info("Retraining model requested")
        train_advanced_risk_model()
        return jsonify({
            'message': 'Advanced model retrained successfully',
            'accuracy': model_accuracy,
            'features': len(feature_columns),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'model_type': 'Advanced Banking Risk Model',
            'features': len(feature_columns) if feature_columns else 0,
            'accuracy': float(model_accuracy) if model_accuracy else None,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
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

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

def initialize_app():
    """Initialize the application with proper error handling"""
    try:
        logger.info("🏦 Initializing Advanced CNB Risk Platform...")
        train_advanced_risk_model()
        logger.info("✅ Advanced banking risk model ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False
    
@app.route('/monitoring')
def monitoring_page():
    """Monitoring dashboard page"""
    try:
        return render_template('monitoring.html')
    except Exception as e:
        logger.error(f"Error rendering monitoring page: {e}")
        return jsonify({'error': 'Monitoring page unavailable'}), 500
    
@app.route('/api/monitoring/dashboard')
def monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        dashboard_data = model_monitor.get_monitoring_dashboard_data()
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/report')
def monitoring_report():
    """Generate comprehensive monitoring report"""
    try:
        report = model_monitor.generate_monitoring_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/drift')
def drift_detection():
    """Check for model drift"""
    try:
        drift_analysis = model_monitor.detect_model_drift()
        return jsonify(drift_analysis)
    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        return jsonify({'error': str(e)}), 500

initialize_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=debug)