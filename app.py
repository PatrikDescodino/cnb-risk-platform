"""
CNB Risk Platform - Flask Web Application
Banking-grade risk assessment platform for České národní banka

"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and data
# PROČ GLOBAL: Trénování modelu je pomalé (5-10 sekund)
# Místo trénování při každém requestu, natrénujeme jednou při startu
model = None              # Zde uložíme natrénovaný model
model_accuracy = 0        # Přesnost modelu (pro zobrazení uživateli)
data_stats = {}          # Statistiky o datech (pro dashboard)

def generate_sample_data():
    """
    Generate synthetic banking transaction data
    
    
    Returns:
        pandas.DataFrame: 10,000 synthetic banking transactions
    """
    n_rows = 10000  # Počet transakcí k vygenerování
    
    # Vygeneruj náhodné customer ID (1 až 10000)
    customer_ids = list(range(1, n_rows + 1))
    
    # Náhodné typy transakcí (vklad, výběr, převod)
    transaction_types = np.random.choice(['deposit', 'withdrawal', 'transfer'], size=n_rows)
    
    # Náhodné částky (1 až 50,000 Kč)
    amounts = np.random.randint(1, 50000, size=n_rows)
    
    # Náhodná data transakcí (2023-2024)
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    random_indices = np.random.randint(0, len(date_range), size=n_rows)
    transaction_dates = date_range[random_indices]
    
    # Náhodné zůstatky na účtech (0 až 100,000 Kč)
    account_balances = np.random.randint(0, 100000, size=n_rows)
    
    # Vytvoř DataFrame (tabulku)
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'transaction_date': transaction_dates,
        'amount': amounts,
        'transaction_type': transaction_types,
        'account_balance': account_balances
    })
    
    return df

def create_risk_label(df):
    """
    Create risk labels based on banking criteria
    
    Rizikové transakce = splňují 2+ z těchto kritérií:
    - Vysoká částka (>30,000 Kč)
    - Nízký zůstatek (<10,000 Kč)  
    - Typ: výběr
    
    Args:
        df (pandas.DataFrame): Data s transakcemi
        
    Returns:
        pandas.Series: 1 = riziková, 0 = bezpečná transakce
    """
    # Kritérium 1: Vysoká částka
    high_amount = df['amount'] > 30000
    
    # Kritérium 2: Nízký zůstatek
    low_balance = df['account_balance'] < 10000
    
    # Kritérium 3: Typ transakce je výběr
    withdrawal = df['transaction_type'] == 'withdrawal'
    
    # Sečti kritéria (každé = 1 bod)
    risk_score = high_amount.astype(int) + low_balance.astype(int) + withdrawal.astype(int)
    
    # Riziková = 2+ body
    is_risky = risk_score >= 2
    
    return is_risky.astype(int)  # Převeď na 0/1

def train_risk_model():
    """
    Train the risk assessment model
    
    ROZŠÍŘENÁ VERZE TVÉHO KÓDU:
    - Přidány dodatečné features pro lepší predikci
    - Uložení do global variables pro Flask
    - Výpočet statistik pro dashboard
    
    Returns:
        tuple: (model, accuracy, statistics)
    """
    global model, model_accuracy, data_stats
    
    print("🔄 Trénuji model...")
    
    # 1. STEJNÉ JAKO U TEBE: Vygeneruj data
    data = generate_sample_data()
    data['is_risky'] = create_risk_label(data)
    
    # 2. Feature engineering (vytvoření dodatečných příznaků)
    # Proč? ML modely fungují lépe s více příznaky
    
    # 2a. Dummy variables pro typ transakce
    # Převede 'deposit'/'withdrawal'/'transfer' na 3 sloupce s 0/1
    transaction_dummies = pd.get_dummies(data['transaction_type'], prefix='trans')
    # Výsledek: trans_deposit, trans_transfer, trans_withdrawal (každý 0 nebo 1)
    
    # 2b. Poměr částky k zůstatku (důležitý indikátor rizika)
    data['amount_to_balance_ratio'] = data['amount'] / (data['account_balance'] + 1)
    # +1 proti dělení nulou
    # Příklad: 50000 Kč výběr z účtu s 10000 Kč = poměr 5.0 = velmi rizikové
    
    # 2c. Binární příznaky (0/1)
    data['is_large_transaction'] = (data['amount'] > data['amount'].quantile(0.8)).astype(int)
    # quantile(0.8) = 80% největších transakcí
    
    data['is_low_balance'] = (data['account_balance'] < data['account_balance'].quantile(0.2)).astype(int)
    # quantile(0.2) = 20% nejnižších zůstatků
    
    # 3. PŘÍPRAVA DAT PRO ML MODEL
    # Spojí všechny příznaky do jedné tabulky
    X = pd.concat([
        # Původní číselné sloupce
        data[['amount', 'account_balance', 'amount_to_balance_ratio', 'is_large_transaction', 'is_low_balance']],
        # Dummy variables
        transaction_dummies
    ], axis=1)
    
    # Target variable (co chceme predikovat)
    y = data['is_risky']
    
    # 4. STEJNÉ JAKO U TEBE: Rozdělení dat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. STEJNÉ JAKO U TEBE: Trénování modelu
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. STEJNÉ JAKO U TEBE: Evaluace
    predictions = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, predictions)
    
    # 7. NOVÉ: Výpočet statistik pro web dashboard
    data_stats = {
        'total_transactions': len(data),
        'risky_transactions': int(data['is_risky'].sum()),
        'safe_transactions': int(len(data) - data['is_risky'].sum()),
        'model_accuracy': float(model_accuracy),
        'test_samples': len(y_test),
        'avg_transaction_amount': float(data['amount'].mean()),
        'avg_account_balance': float(data['account_balance'].mean()),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"✅ Model natrénován! Přesnost: {model_accuracy:.1%}")
    
    return model, model_accuracy, data_stats

def predict_transaction_risk(amount, account_balance, transaction_type):
    """
    Predict risk for a single transaction
    
    NOVÁ FUNKCE pro jednotlivé predikce z webu
    
    Args:
        amount (float): Částka transakce
        account_balance (float): Zůstatek na účtu
        transaction_type (str): Typ transakce ('deposit'/'withdrawal'/'transfer')
        
    Returns:
        dict: Výsledek predikce s rizikovým skóre a úrovní
    """
    global model
    
    # Kontrola, že model je natrénovaný
    if model is None:
        return {'error': 'Model not trained'}
    
    # PŘÍPRAVA FEATURES PRO PREDIKCI
    # Musí být stejné jako při trénování!
    
    # 1. Základní features
    amount_to_balance_ratio = amount / (account_balance + 1)
    
    # 2. Binární features (používám fixed threshold místo quantile)
    # PROČ: při predikci nemáme celý dataset pro výpočet quantile
    is_large_transaction = 1 if amount > 40000 else 0      # 80th percentile aproximace
    is_low_balance = 1 if account_balance < 20000 else 0   # 20th percentile aproximace
    
    # 3. Dummy variables pro typ transakce
    trans_deposit = 1 if transaction_type == 'deposit' else 0
    trans_transfer = 1 if transaction_type == 'transfer' else 0
    trans_withdrawal = 1 if transaction_type == 'withdrawal' else 0
    
    # 4. Vytvoř feature array (musí mít stejné pořadí jako při trénování!)
    features = np.array([[
        amount,                    # sloupec 0
        account_balance,           # sloupec 1
        amount_to_balance_ratio,   # sloupec 2
        is_large_transaction,      # sloupec 3
        is_low_balance,           # sloupec 4
        trans_deposit,            # sloupec 5
        trans_transfer,           # sloupec 6
        trans_withdrawal          # sloupec 7
    ]])
    
    # 5. PREDIKCE
    risk_probability = model.predict_proba(features)[0][1]  # Pravděpodobnost rizika (0-1)
    risk_prediction = model.predict(features)[0]            # Binární predikce (0/1)
    
    # 6. Převod na čitelnou formu
    return {
        'risk_score': float(risk_probability),    # 0.0 - 1.0
        'is_risky': bool(risk_prediction),        # True/False
        'risk_level': 'HIGH' if risk_probability > 0.7 else 'MEDIUM' if risk_probability > 0.3 else 'LOW'
    }

# ==========================================
# FLASK ROUTES (webové endpoint)
# ==========================================

@app.route('/')
def dashboard():
    """
    Main dashboard page
    
    WEBOVÁ STRÁNKA - GET request
    """
    return render_template('dashboard.html', stats=data_stats)

@app.route('/api/stats')
def get_stats():
    """
    API endpoint for statistics
    
    GET request - vrátí statistiky jako JSON
    Použití: AJAX calls z frontend JavaScript
    """
    return jsonify(data_stats)

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """
    API endpoint for risk prediction
    
    POST request - příjme data transakce, vrátí rizikové skóre
    
    Expected JSON input:
    {
        "amount": 50000,
        "account_balance": 10000,
        "transaction_type": "withdrawal"
    }
    
    Returns JSON:
    {
        "risk_score": 0.85,
        "is_risky": true,
        "risk_level": "HIGH"
    }
    """
    try:
        # 1. Získej data z POST requestu
        data = request.get_json()
        
        # 2. Extrahuj jednotlivé hodnoty
        amount = float(data.get('amount', 0))
        account_balance = float(data.get('account_balance', 0))
        transaction_type = data.get('transaction_type', 'transfer')
        
        # 3. VALIDACE VSTUPŮ
        # Proč? Uživatel může poslat špatná data
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        if account_balance < 0:
            return jsonify({'error': 'Account balance cannot be negative'}), 400
        if transaction_type not in ['deposit', 'withdrawal', 'transfer']:
            return jsonify({'error': 'Invalid transaction type'}), 400
        
        # 4. SPOČÍTEJ RIZIKO
        result = predict_transaction_risk(amount, account_balance, transaction_type)
        
        # 5. VRAŤ VÝSLEDEK
        return jsonify(result)
        
    except Exception as e:
        # POKUD SE NĚCO POKAZÍ, NESPADNI
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """
    API endpoint to retrain the model
    
    POST request - přetrénuje model s novými daty
    Použití: admin rozhraní
    """
    try:
        train_risk_model()
        return jsonify({
            'message': 'Model retrained successfully',
            'accuracy': model_accuracy,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint
    
    GET request - zkontroluje, že aplikace běží
    Použití: monitoring systémy
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# ==========================================
# INITIALIZATION
# ==========================================

def initialize_app():
    """
    Initialize the application
    
    Spustí se při startu aplikace:
    1. Natrénuje model
    2. Připraví statistiky
    3. Vypíše status
    """
    print("🏦 Initializing CNB Risk Platform...")
    train_risk_model()
    print(f"✅ Model trained with accuracy: {model_accuracy:.1%}")
    print(f"✅ Processing {data_stats['total_transactions']} transactions")
    print("🚀 CNB Risk Platform ready!")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    # Spustí se při startu aplikace
    initialize_app()
    
    # Spustí Flask web server
    port = int(os.environ.get('PORT', 8000))  # Azure nastaví PORT, lokálně 8000
    app.run(host='0.0.0.0', port=port, debug=False)

