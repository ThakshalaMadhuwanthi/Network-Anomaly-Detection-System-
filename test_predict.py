import pandas as pd
import joblib
import numpy as np

selected_features = [
    'same_srv_rate',
    'dst_host_srv_count',
    'logged_in',
    'dst_host_same_srv_rate',
    'serror_rate',
    'srv_serror_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'count',
    'dst_host_count',
    'dst_host_rerror_rate',
    'rerror_rate',
    'srv_rerror_rate'
]

normal = {
  "same_srv_rate": 0.90,
  "dst_host_srv_count": 50,
  "logged_in": 1,
  "dst_host_same_srv_rate": 0.85,
  "serror_rate": 0.00,
  "srv_serror_rate": 0.00,
  "dst_host_serror_rate": 0.00,
  "dst_host_srv_serror_rate": 0.00,
  "count": 5,
  "dst_host_count": 100,
  "dst_host_rerror_rate": 0.0,
  "rerror_rate": 0.0,
  "srv_rerror_rate": 0.0
}

anomalous = {
  "same_srv_rate": 0.05,
  "dst_host_srv_count": 200,
  "logged_in": 0,
  "dst_host_same_srv_rate": 0.02,
  "serror_rate": 0.85,
  "srv_serror_rate": 0.80,
  "dst_host_serror_rate": 0.70,
  "dst_host_srv_serror_rate": 0.65,
  "count": 300,
  "dst_host_count": 250,
  "dst_host_rerror_rate": 0.90,
  "rerror_rate": 0.88,
  "srv_rerror_rate": 0.87
}

def load(name):
    try:
        return joblib.load(name)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

def debug_sample(sample, label):
    df = pd.DataFrame([sample])
    # enforce column order
    df = df[selected_features]
    print(f"\n--- {label} input (ordered columns) ---")
    print(df.to_string(index=False))

    scaler = load('scaler.pkl')
    rf = load('random_forest.pkl')
    xgb = load('xgboost.pkl')

    if scaler is None or rf is None:
        print('Missing scaler or random_forest model. Aborting.')
        return

    Xs = scaler.transform(df)
    np.set_printoptions(precision=4, suppress=True)
    print('\nScaled features:')
    print(Xs)

    print('\nScaler mean_ (first 5):', getattr(scaler, 'mean_', None)[:5] if hasattr(scaler, 'mean_') else None)

    # Random Forest
    try:
        proba = rf.predict_proba(Xs)[0]
        pred = rf.predict(Xs)[0]
        print(f"\nRandomForest -> pred={pred}, proba={proba}")
    except Exception as e:
        print('RF prediction error:', e)

    # XGBoost
    if xgb is not None:
        try:
            proba2 = xgb.predict_proba(Xs)[0]
            pred2 = xgb.predict(Xs)[0]
            print(f"XGBoost -> pred={pred2}, proba={proba2}")
        except Exception as e:
            print('XGBoost prediction error:', e)

if __name__ == '__main__':
    print('Running debug for normal sample')
    debug_sample(normal, 'NORMAL')
    print('\nRunning debug for anomalous sample')
    debug_sample(anomalous, 'ANOMALOUS')
