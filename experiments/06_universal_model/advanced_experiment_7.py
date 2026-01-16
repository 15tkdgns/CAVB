"""
Advanced Universal Model Experiments
Experiment 7: Time Series Models - ARIMA, GARCH, LSTM, GRU

Specialized time series models for volatility prediction
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    from arch import arch_model
    HAS_GARCH = True
except ImportError:
    HAS_GARCH = False

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

# LSTM Model Definition
if HAS_PYTORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return out

def load_data():
    all_data = {}
    for ticker in ASSETS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data['Return'] = data['Close'].pct_change()
        data['RV_1d'] = data['Return'].abs() * np.sqrt(252) * 100
        data['RV_5d'] = data['Return'].rolling(5).std() * np.sqrt(252) * 100
        data['RV_22d'] = data['Return'].rolling(22).std() * np.sqrt(252) * 100
        all_data[ticker] = data
    
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data[('Close', '^VIX')]
    else:
        vix = vix_data['Close']
    return all_data, vix

def create_features(df, vix):
    df = df.copy()
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    return df.dropna()

def create_sequences(X, y, seq_length=22):
    """Create sequences for LSTM/GRU"""
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i-seq_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_arima(train_target, test_len):
    """Train ARIMA model"""
    if not HAS_ARIMA:
        return None
    try:
        model = ARIMA(train_target, order=(5, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_len)
        return forecast.values
    except:
        return None

def train_garch(train_returns, test_len):
    """Train GARCH model for volatility"""
    if not HAS_GARCH:
        return None
    try:
        model = arch_model(train_returns * 100, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=test_len)
        # Return conditional volatility forecasts
        return np.sqrt(forecast.variance.values[-1])
    except:
        return None

def train_lstm_gru(X_train, y_train, X_test, y_test, model_type='lstm'):
    """Train LSTM or GRU model"""
    if not HAS_PYTORCH:
        return None
    
    try:
        device = torch.device('cpu')
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # Create sequences
        seq_len = 22
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled.flatten(), seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, seq_len)
        
        if len(X_train_seq) < 50 or len(X_test_seq) < 10:
            return None
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_seq).to(device)
        y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test_seq).to(device)
        
        # Model
        input_size = X_train.shape[1]
        if model_type == 'lstm':
            model = LSTMModel(input_size, hidden_size=32, num_layers=2).to(device)
        else:
            model = GRUModel(input_size, hidden_size=32, num_layers=2).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Train
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_t).cpu().numpy()
        
        predictions = scaler_y.inverse_transform(predictions).flatten()
        
        return r2_score(y_test_seq, predictions)
    except Exception as e:
        print(f"    Error in {model_type}: {e}")
        return None

def run_experiment():
    print("=" * 65)
    print("Advanced Universal Model Experiment 7")
    print("Time Series Models: ARIMA, GARCH, LSTM, GRU")
    print("=" * 65)
    
    print(f"\nPackage availability:")
    print(f"  ARIMA (statsmodels): {HAS_ARIMA}")
    print(f"  GARCH (arch): {HAS_GARCH}")
    print(f"  LSTM/GRU (pytorch): {HAS_PYTORCH}")
    
    print("\n[1/2] Loading data...")
    all_data, vix = load_data()
    
    all_results = {}
    
    for ticker in ASSETS:
        print(f"\n[2/2] Processing {ticker}...")
        df = create_features(all_data[ticker], vix)
        
        feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX', 'VIX_lag1', 
                       'VIX_lag5', 'VIX_change', 'CAVB', 'CAVB_lag1', 'CAVB_lag5']
        X = df[feature_cols].values
        y = df['Target'].values
        target_series = df['Target']
        returns = df['Return'].values
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        test_len = n - val_end
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        results = {}
        
        # 1. ElasticNet baseline
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        elastic = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        elastic.fit(X_train_s, y_train)
        results['elasticnet'] = r2_score(y_test, elastic.predict(X_test_s))
        print(f"    ElasticNet: {results['elasticnet']:.4f}")
        
        # 2. ARIMA
        if HAS_ARIMA:
            arima_pred = train_arima(target_series.iloc[:val_end], test_len)
            if arima_pred is not None and len(arima_pred) == test_len:
                results['arima'] = r2_score(y_test, arima_pred)
            else:
                results['arima'] = None
            arima_r2 = results['arima']
            print(f"    ARIMA: {arima_r2:.4f}" if arima_r2 is not None else "    ARIMA: Failed")
        else:
            results['arima'] = None
            print("    ARIMA: Not available")
        
        # 3. GARCH (for volatility)
        if HAS_GARCH:
            garch_pred = train_garch(returns[:val_end], test_len)
            if garch_pred is not None:
                results['garch'] = r2_score(y_test, garch_pred[:test_len])
            else:
                results['garch'] = None
            garch_r2 = results['garch']
            print(f"    GARCH: {garch_r2:.4f}" if garch_r2 is not None else "    GARCH: Failed")
        else:
            results['garch'] = None
            print("    GARCH: Not available")
        
        # 4. LSTM
        if HAS_PYTORCH:
            results['lstm'] = train_lstm_gru(X[:val_end], y[:val_end], 
                                             X[val_end:], df['Target'].iloc[val_end:], 'lstm')
            lstm_r2 = results['lstm']
            print(f"    LSTM: {lstm_r2:.4f}" if lstm_r2 is not None else "    LSTM: Failed")
        else:
            results['lstm'] = None
            print("    LSTM: Not available")
        
        # 5. GRU
        if HAS_PYTORCH:
            results['gru'] = train_lstm_gru(X[:val_end], y[:val_end],
                                           X[val_end:], df['Target'].iloc[val_end:], 'gru')
            gru_r2 = results['gru']
            print(f"    GRU: {gru_r2:.4f}" if gru_r2 is not None else "    GRU: Failed")
        else:
            results['gru'] = None
            print("    GRU: Not available")
        
        all_results[ticker] = results
    
    # Summary
    print("\n" + "=" * 65)
    print("AVERAGE R² BY MODEL")
    print("=" * 65)
    
    models = ['elasticnet', 'arima', 'garch', 'lstm', 'gru']
    baseline = np.mean([all_results[t]['elasticnet'] for t in ASSETS])
    
    for m in models:
        valid_results = [all_results[t][m] for t in ASSETS if all_results[t][m] is not None]
        if valid_results:
            avg = np.mean(valid_results)
            imp = (avg - baseline) / abs(baseline) * 100
            print(f"{m:<15}: {avg:.4f} ({imp:+.1f}%) [{len(valid_results)}/5 assets]")
        else:
            print(f"{m:<15}: Not available")
    
    # Save
    output = {
        'experiment': 'Exp7_TimeSeries_Models',
        'packages': {
            'arima': HAS_ARIMA,
            'garch': HAS_GARCH,
            'pytorch': HAS_PYTORCH
        },
        'results': all_results,
        'summary': {}
    }
    
    for m in models:
        valid = [all_results[t][m] for t in ASSETS if all_results[t][m] is not None]
        if valid:
            output['summary'][m] = {
                'avg_r2': float(np.mean(valid)),
                'n_assets': len(valid)
            }
    
    with open('results/advanced_experiment_7.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_7.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
