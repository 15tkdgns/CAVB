"""
Universal VRP Prediction Model (Production Ready)
- 5-Day and 22-Day horizons
- Feature selection + optimal models
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UniversalVRPModel:
    """Universal VRP Prediction Model for multiple assets and horizons"""
    
    ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
    
    # Selected features based on Exp 17
    FEATURES_5D = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22', 'VIX_change',
                   'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
                   'RV_5d', 'RV_5d_lag', 'RV_22d', 'RV_22d_lag',
                   'SKEW_zscore', 'US10Y', 'DXY_momentum', 'VIX_term']
    
    FEATURES_22D = ['VIX', 'VIX_ma22', 'VIX_term', 'VIX3M',
                    'CAVB', 'CAVB_ma5',
                    'RV_5d', 'RV_22d',
                    'SKEW', 'SKEW_zscore',
                    'US10Y', 'DXY_momentum', 'Credit_spread', 'RSI']
    
    def __init__(self, horizon=5):
        """Initialize model for specified horizon (5 or 22 days)"""
        self.horizon = horizon
        self.models = {}
        self.scalers = {}
        self.data = {}
        
        # Model selection based on horizon
        if horizon == 5:
            self.model_class = SVR
            self.model_params = {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1}
            self.features = self.FEATURES_5D
        else:
            self.model_class = RandomForestRegressor
            self.model_params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 20, 'random_state': 42}
            self.features = self.FEATURES_22D
        
        print(f"UniversalVRPModel initialized for {horizon}-day prediction")
        print(f"Model: {self.model_class.__name__}")
        print(f"Features: {len(self.features)}")
    
    def load_data(self, start='2015-01-01', end='2025-01-01'):
        """Load all required data"""
        print("\nLoading data...")
        
        # Asset prices
        for ticker in self.ASSETS:
            d = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d['Ret'] = d['Close'].pct_change()
            d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
            d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
            self.data[ticker] = d
        
        # VIX
        vix = yf.download('^VIX', start=start, end=end, progress=False)
        self.data['VIX'] = vix[('Close', '^VIX')] if isinstance(vix.columns, pd.MultiIndex) else vix['Close']
        
        # Extended data
        extended = {
            'VIX3M': '^VIX3M', 'SKEW': '^SKEW', 'US10Y': '^TNX',
            'DXY': 'DX-Y.NYB', 'HYG': 'HYG', 'LQD': 'LQD'
        }
        
        for name, ticker in extended.items():
            try:
                d = yf.download(ticker, start=start, end=end, progress=False)
                self.data[name] = d[('Close', ticker)] if isinstance(d.columns, pd.MultiIndex) else d['Close']
            except:
                self.data[name] = None
        
        print("Data loaded successfully")
        return self
    
    def create_features(self, df, ticker):
        """Create features for a single asset"""
        df = df.copy()
        vix = self.data['VIX']
        
        # VIX features
        df['VIX'] = vix
        df['VIX_lag1'] = df['VIX'].shift(1)
        df['VIX_lag5'] = df['VIX'].shift(5)
        df['VIX_ma22'] = df['VIX'].rolling(22).mean()
        df['VIX_change'] = df['VIX'].pct_change()
        
        # CAVB features
        df['CAVB'] = df['VIX'] - df['RV_22d']
        df['CAVB_lag1'] = df['CAVB'].shift(1)
        df['CAVB_lag5'] = df['CAVB'].shift(5)
        df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
        
        # RV features
        df['RV_5d_lag'] = df['RV_5d'].shift(1)
        df['RV_22d_lag'] = df['RV_22d'].shift(1)
        
        # Extended features
        if self.data.get('VIX3M') is not None:
            df['VIX3M'] = self.data['VIX3M']
            df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)
        
        if self.data.get('SKEW') is not None:
            df['SKEW'] = self.data['SKEW']
            df['SKEW_zscore'] = (df['SKEW'] - df['SKEW'].rolling(22).mean()) / (df['SKEW'].rolling(22).std() + 0.01)
        
        if self.data.get('US10Y') is not None:
            df['US10Y'] = self.data['US10Y']
        
        if self.data.get('DXY') is not None:
            df['DXY_momentum'] = self.data['DXY'].pct_change(22)
        
        if self.data.get('HYG') is not None and self.data.get('LQD') is not None:
            df['Credit_spread'] = (self.data['LQD'] / self.data['HYG']) - 1
        
        # Technical
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 0.001)))
        
        # Target
        df['Target'] = df['VIX'] - df['RV_22d'].shift(-self.horizon)
        
        return df
    
    def train(self, test_ratio=0.2):
        """Train models for all assets"""
        print(f"\nTraining {self.horizon}-day models...")
        
        results = {}
        
        for ticker in self.ASSETS:
            print(f"\n{ticker}:")
            
            df = self.create_features(self.data[ticker], ticker)
            
            # Get available features
            available_features = [f for f in self.features if f in df.columns]
            df = df.dropna(subset=available_features + ['Target'])
            
            X = df[available_features]
            y = df['Target']
            
            n = len(X)
            tr_end = int(n * (1 - test_ratio))
            
            Xtr, Xte = X.iloc[:tr_end], X.iloc[tr_end:]
            ytr, yte = y.iloc[:tr_end], y.iloc[tr_end:]
            
            # Scale
            scaler = StandardScaler()
            Xts = scaler.fit_transform(Xtr)
            Xtes = scaler.transform(Xte)
            
            # Train model
            model = self.model_class(**self.model_params)
            model.fit(Xts, ytr)
            
            # Evaluate
            y_pred = model.predict(Xtes)
            r2 = r2_score(yte, y_pred)
            mae = mean_absolute_error(yte, y_pred)
            
            print(f"  R² = {r2:.4f}, MAE = {mae:.4f}")
            
            # Store
            self.models[ticker] = model
            self.scalers[ticker] = scaler
            results[ticker] = {'r2': r2, 'mae': mae, 'n_features': len(available_features)}
        
        # Summary
        avg_r2 = np.mean([r['r2'] for r in results.values()])
        print(f"\nAverage R²: {avg_r2:.4f}")
        
        return results
    
    def predict(self, ticker, latest_features):
        """Make prediction for a single asset"""
        if ticker not in self.models:
            raise ValueError(f"Model not trained for {ticker}")
        
        X = np.array(latest_features).reshape(1, -1)
        X_scaled = self.scalers[ticker].transform(X)
        
        return self.models[ticker].predict(X_scaled)[0]
    
    def save(self, filepath='universal_vrp_model.pkl'):
        """Save trained models"""
        state = {
            'horizon': self.horizon,
            'models': self.models,
            'scalers': self.scalers,
            'features': self.features,
            'model_class': self.model_class.__name__,
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='universal_vrp_model.pkl'):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(horizon=state['horizon'])
        model.models = state['models']
        model.scalers = state['scalers']
        model.features = state['features']
        
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    print("="*70)
    print("Universal VRP Prediction Model - Training")
    print("="*70)
    
    # Train 5-day model
    print("\n" + "="*50)
    print("5-DAY MODEL")
    print("="*50)
    
    model_5d = UniversalVRPModel(horizon=5)
    model_5d.load_data()
    results_5d = model_5d.train()
    model_5d.save('models/universal_vrp_5d.pkl')
    
    # Train 22-day model
    print("\n" + "="*50)
    print("22-DAY MODEL")
    print("="*50)
    
    model_22d = UniversalVRPModel(horizon=22)
    model_22d.load_data()
    results_22d = model_22d.train()
    model_22d.save('models/universal_vrp_22d.pkl')
    
    # Save summary
    summary = {
        'model_type': 'UniversalVRPModel',
        'created': datetime.now().isoformat(),
        '5d': {
            'model': 'SVR_Linear',
            'results': {k: float(v['r2']) for k, v in results_5d.items()},
            'avg_r2': float(np.mean([r['r2'] for r in results_5d.values()]))
        },
        '22d': {
            'model': 'RandomForest',
            'results': {k: float(v['r2']) for k, v in results_22d.items()},
            'avg_r2': float(np.mean([r['r2'] for r in results_22d.values()]))
        }
    }
    
    with open('models/universal_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n5-Day Average R²: {summary['5d']['avg_r2']:.4f}")
    print(f"22-Day Average R²: {summary['22d']['avg_r2']:.4f}")
    print("\nModels saved to models/ directory")
