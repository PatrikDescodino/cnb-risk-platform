"""
CNB Risk Platform - Advanced Banking Risk Model
WITH ADAPTIVE LEARNING from Azure Storage
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
import sys
import os
import logging
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import monitoring and storage modules
try:
    from model_monitoring import CNBModelMonitor
    from azure_storage import CNBAzureStorage
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Monitoring modules not available: {e}")
    MONITORING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and data
model = None
model_accuracy = 0
data_stats = {}
feature_columns = []
model_training_lock = threading.Lock()
model_training_in_progress = False

# Adaptive learning variables
real_data_cache = []  # Cache for real transaction data
last_retrain_time = None
auto_retrain_threshold = 50  # Retrain after 50 new predictions

# Initialize monitoring and storage
model_monitor = None
azure_storage = None

if MONITORING_AVAILABLE:
    try:
        model_monitor = CNBModelMonitor(window_size=100, drift_threshold=0.1)
        azure_storage = CNBAzureStorage()
        logger.info("✅ Monitoring and storage initialized")
    except Exception as e:
        logger.warning(f"⚠️ Monitoring initialization failed: {e}")
        model_monitor = None
        azure_storage = None

# Constants
VALID_TRANSACTION_TYPES = ['deposit', 'withdrawal', 'transfer', 'payment', 'card_payment']
VALID_COUNTRIES = ['CZ', 'SK', 'DE', 'AT', 'PL', 'OFFSHORE', 'HIGH_RISK', 'OTHER']

def ensure_model_trained():
    """Ensure model is trained before use (lazy loading)"""
    global model, model_training_in_progress
    
    if model is not None:
        return True
    
    with model_training_lock:
        # Double-check pattern
        if model is not None:
            return True
        
        if model_training_in_progress:
            return False
        
        try:
            model_training_in_progress = True
            logger.info("Training model on first use...")
            train_adaptive_risk_model()
            return True
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return False
        finally:
            model_training_in_progress = False

def load_real_data_from_storage():
    """Load real transaction data from Azure Storage"""
    global real_data_cache
    
    if not azure_storage or not azure_storage.is_connected():
        logger.info("No storage connection - using synthetic data only")
        return pd.DataFrame()
    
    try:
        logger.info("📦 Loading real transaction data from Azure Storage...")
        
        # Get list of stored analyses
        analyses_result = azure_storage.list_risk_analyses(limit=1000)
        
        if not analyses_result.get('success') or not analyses_result.get('analyses'):
            logger.info("No real data found in storage yet")
            return pd.DataFrame()
        
        real_transactions = []
        
        # Download and parse each analysis
        for analysis_info in analyses_result['analyses']:
            try:
                filename = analysis_info['filename']
                download_result = azure_storage.download_analysis(filename)
                
                if download_result.get('success'):
                    data = download_result['data']
                    
                    # Extract transaction data and risk result
                    transaction_data = data.get('transaction_data', {})
                    risk_analysis = data.get('risk_analysis', {})
                    
                    # Convert to training format
                    if transaction_data and risk_analysis:
                        real_transaction = {
                            'amount': transaction_data.get('amount', 0),
                            'transaction_type': transaction_data.get('transaction_type', 'transfer'),
                            'country': transaction_data.get('country', 'CZ'),
                            'transaction_hour': transaction_data.get('transaction_hour', 12),
                            # Risk label from actual prediction
                            'is_risky': risk_analysis.get('is_risky', False),
                            'risk_score': risk_analysis.get('risk_score', 0.0),
                            'risk_level': risk_analysis.get('risk_level', 'LOW'),
                            # Timestamp for tracking
                            'timestamp': data.get('timestamp'),
                            'source': 'real_data'
                        }
                        real_transactions.append(real_transaction)
                        
            except Exception as e:
                logger.warning(f"Failed to parse analysis {filename}: {e}")
                continue
        
        if real_transactions:
            real_df = pd.DataFrame(real_transactions)
            real_data_cache = real_transactions  # Update cache
            logger.info(f"✅ Loaded {len(real_df)} real transactions from storage")
            return real_df
        else:
            logger.info("No valid real transaction data found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading real data from storage: {e}")
        return pd.DataFrame()

def generate_advanced_banking_data(include_real_data=True):
    """
    Generate synthetic banking transaction data + include real data from storage
    """
    # Load real data from storage
    real_df = pd.DataFrame()
    if include_real_data:
        real_df = load_real_data_from_storage()
    
    # Determine synthetic data size based on real data availability
    if len(real_df) > 0:
        # If we have real data, generate fewer synthetic samples
        synthetic_ratio = max(2, int(5000 / len(real_df)))  # At least 2x real data
        n_synthetic = min(5000, len(real_df) * synthetic_ratio)
        logger.info(f"📊 Using {len(real_df)} real + {n_synthetic} synthetic transactions")
    else:
        n_synthetic = 5000
        logger.info(f"📊 Using {n_synthetic} synthetic transactions (no real data yet)")
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_banking_data(n_synthetic)
    
    # Combine real and synthetic data
    if len(real_df) > 0:
        # Ensure real data has all required columns
        real_df_enhanced = enhance_real_data_with_features(real_df)
        
        # Combine datasets
        combined_df = pd.concat([synthetic_df, real_df_enhanced], ignore_index=True)
        logger.info(f"✅ Combined dataset: {len(combined_df)} total transactions ({len(real_df)} real, {len(synthetic_df)} synthetic)")
    else:
        combined_df = synthetic_df
        logger.info(f"✅ Using synthetic dataset: {len(combined_df)} transactions")
    
    return combined_df

def enhance_real_data_with_features(real_df):
    """Enhance real data with missing features using reasonable defaults"""
    enhanced_df = real_df.copy()
    
    # Get the number of rows
    n_rows = len(enhanced_df)
    
    # Add missing features with realistic defaults
    if 'customer_id' not in enhanced_df.columns:
        enhanced_df['customer_id'] = np.random.randint(1000, 9999, size=n_rows)
    
    if 'transaction_date' not in enhanced_df.columns:
        enhanced_df['transaction_date'] = [
            datetime.now() - timedelta(days=np.random.randint(0, 30)) 
            for _ in range(n_rows)
        ]
    
    if 'account_balance' not in enhanced_df.columns:
        # Use amount from real data if available, otherwise default
        account_balances = []
        for _, row in enhanced_df.iterrows():
            amount = row.get('amount', 50000)
            balance = max(amount * np.random.uniform(0.5, 3.0), 1000)
            account_balances.append(balance)
        enhanced_df['account_balance'] = account_balances
    
    if 'monthly_income' not in enhanced_df.columns:
        # Estimate income based on transaction amount
        monthly_incomes = []
        for _, row in enhanced_df.iterrows():
            amount = row.get('amount', 50000)
            income = estimate_income_from_amount(amount)
            monthly_incomes.append(income)
        enhanced_df['monthly_income'] = monthly_incomes
    
    if 'customer_age' not in enhanced_df.columns:
        enhanced_df['customer_age'] = np.random.randint(25, 65, size=n_rows)
    
    if 'account_age_days' not in enhanced_df.columns:
        enhanced_df['account_age_days'] = np.random.randint(90, 1825, size=n_rows)
    
    if 'country_risk_score' not in enhanced_df.columns:
        # Get country risk score from real data if available
        country_risk_scores = []
        for _, row in enhanced_df.iterrows():
            country = row.get('country', 'CZ')
            risk_score = get_country_risk_score(country)
            country_risk_scores.append(risk_score)
        enhanced_df['country_risk_score'] = country_risk_scores
    
    if 'transactions_last_7d' not in enhanced_df.columns:
        enhanced_df['transactions_last_7d'] = np.random.randint(1, 8, size=n_rows)
    
    if 'transactions_last_30d' not in enhanced_df.columns:
        enhanced_df['transactions_last_30d'] = np.random.randint(5, 25, size=n_rows)
    
    if 'customer_profile' not in enhanced_df.columns:
        # Estimate profile based on income
        customer_profiles = []
        for _, row in enhanced_df.iterrows():
            income = row.get('monthly_income', 50000)
            profile = estimate_profile_from_income(income)
            customer_profiles.append(profile)
        enhanced_df['customer_profile'] = customer_profiles
    
    if 'is_cash_business' not in enhanced_df.columns:
        enhanced_df['is_cash_business'] = np.random.choice([0, 1], size=n_rows, p=[0.85, 0.15])
    
    # Ensure source column exists
    if 'source' not in enhanced_df.columns:
        enhanced_df['source'] = 'real_data'
    
    return enhanced_df

def estimate_income_from_amount(amount):
    """Estimate monthly income based on transaction amount"""
    if amount < 5000:
        return np.random.normal(35000, 10000)
    elif amount < 20000:
        return np.random.normal(50000, 15000)
    elif amount < 100000:
        return np.random.normal(80000, 20000)
    else:
        return np.random.normal(120000, 30000)

def estimate_profile_from_income(income):
    """Estimate customer profile from income"""
    if income < 30000:
        return 'low_income'
    elif income < 70000:
        return 'middle_income'
    elif income < 120000:
        return 'high_income'
    else:
        return 'business'

def get_country_risk_score(country):
    """Get risk score for country"""
    risk_map = {
        'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
        'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
    }
    return risk_map.get(country, 5)

def generate_synthetic_banking_data(n_rows):
    """Generate synthetic banking transaction data"""
    logger.info(f"Generating {n_rows} synthetic banking transactions...")
    
    # Basic customer data
    customer_ids = np.random.randint(1000, 9999, size=n_rows)
    
    # Advanced time features - FIXED PROBABILITIES
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]
    
    # FIXED: Properly normalized probabilities that sum to 1.0
    hour_probs = np.array([
        0.01, 0.01, 0.005, 0.005, 0.005, 0.01,  # 0-5: Velmi nízká aktivita
        0.02, 0.04, 0.06, 0.08, 0.09, 0.10,     # 6-11: Rostoucí aktivita
        0.08, 0.07, 0.08, 0.09, 0.07, 0.06,     # 12-17: Nejvyšší aktivita
        0.05, 0.04, 0.03, 0.02, 0.015, 0.01     # 18-23: Klesající aktivita
    ])
    
    # Normalize to ensure sum = 1.0
    hour_probs = hour_probs / hour_probs.sum()
    
    hours = np.random.choice(range(24), size=n_rows, p=hour_probs)
    
    # Transaction types
    transaction_types = np.random.choice(
        VALID_TRANSACTION_TYPES, 
        size=n_rows,
        p=[0.15, 0.20, 0.25, 0.25, 0.15]
    )
    
    # Customer profiles
    customer_profiles = np.random.choice(['low_income', 'middle_income', 'high_income', 'business'], 
                                       size=n_rows, p=[0.3, 0.5, 0.15, 0.05])
    
    # Customer demographics
    ages = np.random.normal(45, 15, n_rows).astype(int)
    ages = np.clip(ages, 18, 85)
    
    # Account age
    account_ages = np.random.exponential(500, n_rows).astype(int)
    account_ages = np.clip(account_ages, 1, 3650)
    
    # Income-based amounts
    amounts = []
    monthly_incomes = []
    
    for profile in customer_profiles:
        if profile == 'low_income':
            income = np.random.normal(25000, 5000)
            amount = np.random.lognormal(7, 1.5)
        elif profile == 'middle_income':
            income = np.random.normal(50000, 10000)
            amount = np.random.lognormal(8, 1.8)
        elif profile == 'high_income':
            income = np.random.normal(100000, 20000)
            amount = np.random.lognormal(9, 2.0)
        else:  # business
            income = np.random.normal(200000, 50000)
            amount = np.random.lognormal(10, 2.5)
            
        amounts.append(max(1, amount))
        monthly_incomes.append(max(10000, income / 12))
    
    amounts = np.array(amounts)
    monthly_incomes = np.array(monthly_incomes)
    
    # Account balances
    account_balances = []
    for income in monthly_incomes:
        balance = income * np.random.uniform(0.5, 3.0) + np.random.normal(0, income * 0.2)
        account_balances.append(max(0, balance))
    
    account_balances = np.array(account_balances)
    
    # Geographical risk - FIXED PROBABILITIES
    country_probs = np.array([0.70, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01])
    country_probs = country_probs / country_probs.sum()  # Ensure normalization
    
    countries = np.random.choice(VALID_COUNTRIES[:-1], 
                                size=n_rows, 
                                p=country_probs)
    
    country_risk_scores = {
        'CZ': 1, 'SK': 2, 'DE': 1, 'AT': 1, 'PL': 2, 
        'OFFSHORE': 8, 'HIGH_RISK': 10, 'OTHER': 5
    }
    
    # Frequency features
    transactions_7d = np.random.poisson(3, n_rows)
    transactions_30d = np.random.poisson(12, n_rows)
    
    # Cash-intensive business
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
        'is_cash_business': is_cash_business,
        'source': 'synthetic'  # Mark as synthetic
    })
    
    return df

def create_advanced_risk_labels(df):
    """Create sophisticated risk labels - enhanced for real data"""
    risk_score = np.zeros(len(df))
    
    # For real data, use existing labels if available
    if 'is_risky' in df.columns and 'source' in df.columns:
        real_data_mask = df['source'] == 'real_data'
        if real_data_mask.any():
            logger.info(f"Using existing labels for {real_data_mask.sum()} real transactions")
            # For real data, keep existing labels and create risk scores based on risk level
            for idx, row in df[real_data_mask].iterrows():
                if row['risk_level'] == 'CRITICAL':
                    risk_score[idx] = 10
                elif row['risk_level'] == 'HIGH':
                    risk_score[idx] = 8
                elif row['risk_level'] == 'MEDIUM':
                    risk_score[idx] = 5
                elif row['risk_level'] == 'LOW':
                    risk_score[idx] = 2
                else:  # MINIMAL
                    risk_score[idx] = 1
    
    # For synthetic data, calculate risk scores
    synthetic_mask = df.get('source', 'synthetic') == 'synthetic'
    if synthetic_mask.any():
        # AML factors
        reporting_limit = 15000 * 24
        structured_transactions = (df['amount'] > reporting_limit * 0.9) & (df['amount'] < reporting_limit)
        risk_score += structured_transactions * 3
        
        unusual_amount = df['amount'] > (df['monthly_income'] * 0.5)
        risk_score += unusual_amount * 2
        
        high_country_risk = df['country_risk_score'] >= 7
        risk_score += high_country_risk * 4
        
        high_frequency = df['transactions_last_7d'] > 10
        risk_score += high_frequency * 2
        
        # Fraud factors
        suspicious_hours = (df['transaction_hour'] <= 6) | (df['transaction_hour'] >= 23)
        risk_score += suspicious_hours * 2
        
        new_account_risk = (df['account_age_days'] < 30) & (df['amount'] > df['monthly_income'])
        risk_score += new_account_risk * 3
        
        overdraft_risk = (df['transaction_type'] == 'withdrawal') & (df['amount'] > df['account_balance'] * 0.8)
        risk_score += overdraft_risk * 2
        
        # Business logic
        cash_business_risk = (df['is_cash_business'] == 1) & (df['amount'] > 100000)
        risk_score += cash_business_risk * 2
        
        low_balance_risk = (df['account_balance'] < df['monthly_income'] * 0.2) & (df['amount'] > df['monthly_income'] * 0.3)
        risk_score += low_balance_risk * 2
        
        young_high_risk = (df['customer_age'] < 25) & (df['amount'] > 50000)
        risk_score += young_high_risk * 1
        
        # Combined factors
        offshore_combo = (df['country_risk_score'] >= 7) & (df['amount'] > 200000) & (df['is_cash_business'] == 1)
        risk_score += offshore_combo * 5
    
    # Create binary labels
    if 'is_risky' in df.columns and 'source' in df.columns:
        # Preserve real data labels, calculate for synthetic
        is_risky = df['is_risky'].copy()
        synthetic_indices = df[df['source'] == 'synthetic'].index
        if len(synthetic_indices) > 0:
            synthetic_risk_scores = risk_score[synthetic_indices]
            risk_threshold = np.percentile(synthetic_risk_scores, 85) if len(synthetic_risk_scores) > 0 else 3
            is_risky.loc[synthetic_indices] = (synthetic_risk_scores >= max(3, risk_threshold)).astype(int)
    else:
        # All synthetic data
        risk_threshold = np.percentile(risk_score, 85)
        is_risky = (risk_score >= max(3, risk_threshold)).astype(int)
    
    return is_risky, risk_score

def train_adaptive_risk_model():
    """Train enhanced risk assessment model with adaptive learning"""
    global model, model_accuracy, data_stats, feature_columns, last_retrain_time
    
    logger.info("🧠 Training ADAPTIVE banking risk model...")
    
    try:
        # Generate dataset with real + synthetic data
        data = generate_advanced_banking_data(include_real_data=True)
        data['is_risky'], data['risk_score_raw'] = create_advanced_risk_labels(data)
        
        # Feature Engineering
        data['is_weekend'] = pd.to_datetime(data['transaction_date']).dt.weekday >= 5
        data['is_night_time'] = (data['transaction_hour'] <= 6) | (data['transaction_hour'] >= 22)
        data['is_business_hours'] = (data['transaction_hour'] >= 9) & (data['transaction_hour'] <= 17)
        
        # Ratios with safe division
        data['amount_to_income_ratio'] = data['amount'] / np.maximum(data['monthly_income'], 1)
        data['amount_to_balance_ratio'] = data['amount'] / np.maximum(data['account_balance'] + 1, 1)
        data['balance_to_income_ratio'] = data['account_balance'] / np.maximum(data['monthly_income'], 1)
        data['frequency_velocity'] = data['transactions_last_7d'] / 7
        
        # Categories
        data['amount_category'] = pd.cut(data['amount'], 
                                       bins=[0, 1000, 10000, 50000, 200000, float('inf')],
                                       labels=['micro', 'small', 'medium', 'large', 'jumbo'])
        
        # Dummy variables
        transaction_dummies = pd.get_dummies(data['transaction_type'], prefix='trans')
        profile_dummies = pd.get_dummies(data['customer_profile'], prefix='profile')
        amount_cat_dummies = pd.get_dummies(data['amount_category'], prefix='amount')
        
        # Numerical features
        numerical_features = [
            'amount', 'account_balance', 'monthly_income', 'customer_age', 'account_age_days',
            'country_risk_score', 'transactions_last_7d', 'transactions_last_30d',
            'amount_to_income_ratio', 'amount_to_balance_ratio', 'balance_to_income_ratio',
            'frequency_velocity', 'is_cash_business', 'is_weekend', 'is_night_time', 
            'is_business_hours', 'transaction_hour'
        ]
        
        # Combine features
        X = pd.concat([
            data[numerical_features],
            transaction_dummies,
            profile_dummies,
            amount_cat_dummies
        ], axis=1)
        
        feature_columns = X.columns.tolist()
        y = data['is_risky']
        
        # Validate data
        if X.isnull().any().any():
            logger.warning("NaN values detected, filling with 0")
            X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Train-test split - stratify to ensure both real and synthetic in both sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Enhanced model for mixed data
        model = RandomForestClassifier(
            n_estimators=150,  # Increased for better real data handling
            max_depth=12,      # Slightly deeper
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            class_weight='balanced',
            n_jobs=1  # Single core pro Azure
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, predictions)
        
        # MONITORING INTEGRATION - Set baseline performance
        if model_monitor:
            model_monitor.set_baseline_performance(model_accuracy)
            model_monitor.model = model  # Attach model to monitor
            
            # Store training data in Azure if available
            if azure_storage and azure_storage.is_connected():
                try:
                    model_metadata = {
                        'version': 'v2.2.0_adaptive',
                        'accuracy': float(model_accuracy),
                        'features': len(feature_columns),
                        'training_date': datetime.now().isoformat(),
                        'total_samples': len(X_train),
                        'real_data_samples': len(data[data.get('source', '') == 'real_data']),
                        'synthetic_samples': len(data[data.get('source', '') == 'synthetic']),
                        'adaptive_learning': True
                    }
                    azure_storage.upload_training_data(data, model_metadata)
                    logger.info("📦 Training data uploaded to Azure Storage")
                except Exception as e:
                    logger.warning(f"Failed to upload training data: {e}")
        
        # Enhanced statistics
        risky_count = int(data['is_risky'].sum())
        safe_count = len(data) - risky_count
        real_data_count = len(data[data.get('source', '') == 'real_data'])
        
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
            'real_data_count': real_data_count,
            'synthetic_data_count': len(data) - real_data_count,
            'adaptive_learning': True,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        last_retrain_time = datetime.now()
        
        logger.info(f"✅ ADAPTIVE model trained successfully - Accuracy: {model_accuracy:.1%}")
        logger.info(f"📊 Data composition: {real_data_count} real + {len(data) - real_data_count} synthetic")
        logger.info(f"✅ Features: {len(feature_columns)}, Risk rate: {data_stats['risk_rate']:.1%}")
        return model, model_accuracy, data_stats
        
    except Exception as e:
        logger.error(f"Error training adaptive model: {e}")
        raise

def check_auto_retrain():
    """Check if automatic retraining should be triggered"""
    global last_retrain_time
    
    if not model_monitor or not azure_storage:
        return False
    
    # Check if enough new predictions have been made
    if len(model_monitor.predictions_log) < auto_retrain_threshold:
        return False
    
    # Check if enough time has passed (at least 1 hour)
    if last_retrain_time:
        time_since_retrain = datetime.now() - last_retrain_time
        if time_since_retrain.total_seconds() < 3600:  # 1 hour
            return False
    
    # Check if there's new real data to learn from
    try:
        current_real_data = load_real_data_from_storage()
        if len(current_real_data) > len(real_data_cache):
            logger.info(f"🔄 Auto-retrain triggered: {len(current_real_data)} vs {len(real_data_cache)} real transactions")
            return True
    except Exception as e:
        logger.warning(f"Auto-retrain check failed: {e}")
    
    return False

def predict_advanced_transaction_risk(amount, account_balance, transaction_type, 
                                    monthly_income=50000, customer_age=35, 
                                    account_age_days=365, country='CZ',
                                    transaction_hour=12, is_cash_business=0):
    """Advanced risk prediction with adaptive learning and monitoring integration"""
    global model, feature_columns
    
    if not ensure_model_trained():
        return {'error': 'Model training in progress, please try again in a moment'}
    
    if model is None:
        return {'error': 'Model not available'}
    
    try:
        # Country risk mapping
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
        is_weekend = 0
        is_night_time = 1 if (transaction_hour <= 6 or transaction_hour >= 22) else 0
        is_business_hours = 1 if (9 <= transaction_hour <= 17) else 0
        
        # Frequency
        transactions_last_7d = 3
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
        
        # Customer profile
        if monthly_income < 30000:
            customer_profile = 'low_income'
        elif monthly_income < 70000:
            customer_profile = 'middle_income'
        elif monthly_income < 150000:
            customer_profile = 'high_income'
        else:
            customer_profile = 'business'
        
        # Create feature vector
        feature_dict = {}
        
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
        
        # Initialize features
        for col in feature_columns:
            feature_dict[col] = 0
        
        # Set numerical features
        for key, value in numerical_values.items():
            if key in feature_dict:
                feature_dict[key] = value
        
        # Set dummy variables
        trans_col = f'trans_{transaction_type}'
        if trans_col in feature_dict:
            feature_dict[trans_col] = 1
        
        profile_col = f'profile_{customer_profile}'
        if profile_col in feature_dict:
            feature_dict[profile_col] = 1
            
        amount_col = f'amount_{amount_cat}'
        if amount_col in feature_dict:
            feature_dict[amount_col] = 1
        
        # Create array
        features = np.array([[feature_dict[col] for col in feature_columns]])
        
        # Handle any remaining NaN/inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict
        risk_probability = model.predict_proba(features)[0][1]
        risk_prediction = model.predict(features)[0]
        
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
        
        result = {
            'risk_score': float(risk_probability),
            'is_risky': bool(risk_prediction),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'explanation': f"Model identifikoval {len(risk_factors)} rizikových faktorů" if risk_factors else "Transakce vyhodnocena jako standardní",
            'adaptive_learning': True  # Flag indicating adaptive model
        }
        
        # MONITORING INTEGRATION - Log prediction
        if model_monitor:
            try:
                # Prepare features for monitoring (only relevant ones)
                monitoring_features = {
                    'amount': amount,
                    'account_balance': account_balance,
                    'monthly_income': monthly_income,
                    'transaction_type': transaction_type,
                    'country': country,
                    'transaction_hour': transaction_hour
                }
                
                # Generate customer hash for privacy
                customer_hash = hash(f"{amount}_{account_balance}_{datetime.now().timestamp()}") % 1000000
                
                # Log to monitoring system
                model_monitor.log_prediction(
                    features=monitoring_features,
                    prediction=result,
                    metadata={'customer_hash': customer_hash, 'adaptive_model': True}
                )
                
                # Store in Azure ALWAYS (for learning) - not just high risk
                if azure_storage and azure_storage.is_connected():
                    azure_storage.upload_risk_analysis(
                        transaction_data=monitoring_features,
                        risk_result=result,
                        customer_id=str(customer_hash)
                    )
                
            except Exception as e:
                logger.warning(f"Monitoring logging failed: {e}")
        
        # Check if auto-retrain should be triggered
        try:
            if check_auto_retrain():
                # Trigger background retraining
                threading.Thread(target=background_retrain, daemon=True).start()
        except Exception as e:
            logger.warning(f"Auto-retrain check failed: {e}")
        
        logger.info(f"Prediction completed: Risk score {risk_probability:.3f}, Level: {risk_level}")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {'error': f'Prediction error: {str(e)}'}

def background_retrain():
    """Background retraining function"""
    try:
        logger.info("🔄 Starting background adaptive retraining...")
        with model_training_lock:
            if not model_training_in_progress:
                train_adaptive_risk_model()
                logger.info("✅ Background retraining completed")
            else:
                logger.info("⏸️ Training already in progress, skipping")
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

# Flask routes
@app.route('/')
def dashboard():
    try:
        return render_template('dashboard.html', stats=data_stats)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return jsonify({'error': 'Dashboard unavailable'}), 500

@app.route('/monitoring')
def monitoring_dashboard():
    """Monitoring dashboard page"""
    try:
        return render_template('monitoring.html')
    except Exception as e:
        logger.error(f"Error rendering monitoring dashboard: {e}")
        return jsonify({'error': 'Monitoring dashboard unavailable'}), 500

@app.route('/api/stats')
def get_stats():
    try:
        if not data_stats:
            # Return default stats if model not trained yet
            return jsonify({
                'total_transactions': 0,
                'risky_transactions': 0,
                'safe_transactions': 0,
                'risk_rate': 0.0,
                'model_accuracy': 0.0,
                'test_samples': 0,
                'feature_count': 0,
                'avg_transaction_amount': 0.0,
                'avg_account_balance': 0.0,
                'avg_monthly_income': 0.0,
                'high_risk_countries': 0,
                'night_transactions': 0,
                'real_data_count': 0,
                'synthetic_data_count': 0,
                'adaptive_learning': False,
                'last_updated': 'Model not trained yet'
            })
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
        
        # Basic validation
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
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# MONITORING API ENDPOINTS
@app.route('/api/monitoring/dashboard')
def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        if not model_monitor:
            return jsonify({'error': 'Monitoring not available'})
        
        dashboard_data = model_monitor.get_monitoring_dashboard_data()
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        return jsonify({'error': 'Monitoring dashboard unavailable'}), 500

@app.route('/api/monitoring/report')
def get_monitoring_report():
    """Get comprehensive monitoring report"""
    try:
        if not model_monitor:
            return jsonify({'error': 'Monitoring not available'})
        
        report = model_monitor.generate_monitoring_report()
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        return jsonify({'error': 'Monitoring report unavailable'}), 500

@app.route('/api/monitoring/drift')
def get_drift_analysis():
    """Get model drift analysis"""
    try:
        if not model_monitor:
            return jsonify({'error': 'Monitoring not available'})
        
        drift_analysis = model_monitor.detect_model_drift()
        return jsonify(drift_analysis)
        
    except Exception as e:
        logger.error(f"Error getting drift analysis: {e}")
        return jsonify({'error': 'Drift analysis unavailable'}), 500

@app.route('/api/storage/stats')
def get_storage_stats():
    """Get Azure storage statistics"""
    try:
        if not azure_storage:
            return jsonify({'error': 'Azure storage not available'})
        
        stats = azure_storage.get_storage_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return jsonify({'error': 'Storage stats unavailable'}), 500

@app.route('/api/storage/analyses')
def list_risk_analyses():
    """List recent risk analyses from storage"""
    try:
        if not azure_storage:
            return jsonify({'error': 'Azure storage not available'})
        
        limit = request.args.get('limit', 10, type=int)
        analyses = azure_storage.list_risk_analyses(limit=limit)
        return jsonify(analyses)
        
    except Exception as e:
        logger.error(f"Error listing analyses: {e}")
        return jsonify({'error': 'Cannot list analyses'}), 500

@app.route('/api/adaptive/info')
def get_adaptive_info():
    """Get adaptive learning information"""
    try:
        info = {
            'adaptive_learning_enabled': True,
            'real_data_count': len(real_data_cache),
            'auto_retrain_threshold': auto_retrain_threshold,
            'last_retrain_time': last_retrain_time.isoformat() if last_retrain_time else None,
            'storage_connected': azure_storage is not None and azure_storage.is_connected(),
            'predictions_since_retrain': len(model_monitor.predictions_log) if model_monitor else 0,
            'next_retrain_predictions': auto_retrain_threshold - (len(model_monitor.predictions_log) if model_monitor else 0)
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting adaptive info: {e}")
        return jsonify({'error': 'Adaptive info unavailable'}), 500

@app.route('/debug/storage')
def debug_storage():
    """Debug storage connection"""
    try:
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        
        debug_info = {
            'connection_string_exists': connection_string is not None,
            'connection_string_length': len(connection_string) if connection_string else 0,
            'connection_string_preview': connection_string[:50] + '...' if connection_string else None,
            'azure_storage_object': azure_storage is not None,
            'is_connected': azure_storage.is_connected() if azure_storage else False,
            'monitoring_available': MONITORING_AVAILABLE,
            'environment_variables': {
                'PORT': os.environ.get('PORT'),
                'FLASK_ENV': os.environ.get('FLASK_ENV'),
                'HOST': os.environ.get('HOST')
            }
        }
        
        # Try to initialize storage if connection string exists but object is None
        if connection_string and not azure_storage and MONITORING_AVAILABLE:
            try:
                from azure_storage import CNBAzureStorage
                test_storage = CNBAzureStorage(connection_string)
                debug_info['test_storage_connected'] = test_storage.is_connected()
            except Exception as e:
                debug_info['test_storage_error'] = str(e)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'}), 500

@app.route('/debug/env')
def debug_environment():
    """Debug environment variables"""
    try:
        # Show all environment variables that start with AZURE or are relevant
        relevant_env = {}
        for key, value in os.environ.items():
            if any(keyword in key.upper() for keyword in ['AZURE', 'STORAGE', 'CNB', 'FLASK', 'PORT']):
                # Hide sensitive parts of connection strings
                if 'CONNECTION_STRING' in key or 'KEY' in key:
                    relevant_env[key] = value[:20] + '...' if value else None
                else:
                    relevant_env[key] = value
        
        return jsonify({
            'relevant_environment_variables': relevant_env,
            'total_env_vars': len(os.environ),
            'python_path': os.environ.get('PYTHONPATH'),
            'working_directory': os.getcwd()
        })
        
    except Exception as e:
        return jsonify({'error': f'Environment debug failed: {str(e)}'}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        logger.info("Manual retraining requested")
        with model_training_lock:
            train_adaptive_risk_model()
        return jsonify({
            'message': 'Adaptive model retrained successfully',
            'accuracy': model_accuracy,
            'features': len(feature_columns),
            'real_data_count': data_stats.get('real_data_count', 0),
            'synthetic_data_count': data_stats.get('synthetic_data_count', 0),
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
            'model_training': model_training_in_progress,
            'model_type': 'Adaptive Banking Risk Model',
            'features': len(feature_columns) if feature_columns else 0,
            'accuracy': float(model_accuracy) if model_accuracy else None,
            'monitoring_available': model_monitor is not None,
            'storage_available': azure_storage is not None and azure_storage.is_connected(),
            'adaptive_learning': True,
            'real_data_count': len(real_data_cache),
            'auto_retrain_threshold': auto_retrain_threshold,
            'timestamp': datetime.now().isoformat(),
            'version': '2.2.0_adaptive'
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

if __name__ == '__main__':
    # Quick startup for Azure
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"🏦 CNB Risk Platform v2.2.0 (ADAPTIVE) starting on {host}:{port}")
    logger.info("⚡ Using lazy loading with ADAPTIVE LEARNING")
    logger.info("🧠 Model will learn from real transaction data")
    logger.info("🔄 Auto-retraining enabled")
    
    if MONITORING_AVAILABLE:
        logger.info("📊 Monitoring and storage integration enabled")
    else:
        logger.warning("⚠️ Monitoring modules not available - running in basic mode")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)# Force redeploy
