import pandas as pd
import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import layers, models, regularizers, optimizers, callbacks

# Load datasets
train_df = pd.read_csv('Sf_training_gem.csv')
val_df = pd.read_csv('Sf_validation_gem.csv')
test_df = pd.read_csv('Sf_testing_gem.csv')

# Clean columns and sort by date
for df in [train_df, val_df, test_df]:
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.sort_values('Date', inplace=True)

# Define feature sets
base_features = [col for col in train_df.columns if col not in ['Date']]
features_with = base_features.copy()
features_without = [col for col in base_features if col not in ['Vol', 'OpenInt']]

def evaluate_forecasts(train, val, test, feature_cols):
    mae_days = {}
    
    for day in range(1, 11):
        tr = train.copy()
        va = val.copy()
        te = test.copy()
        
        tr['Target'] = tr['Close'].shift(-day)
        va['Target'] = va['Close'].shift(-day)
        te['Target'] = te['Close'].shift(-day)
        
        tr.dropna(subset=feature_cols + ['Target'], inplace=True)
        va.dropna(subset=feature_cols + ['Target'], inplace=True)
        te.dropna(subset=feature_cols + ['Target'], inplace=True)
        
        X_tr = tr[feature_cols].astype(float).values
        y_tr = tr['Target'].astype(float).values
        X_v = va[feature_cols].astype(float).values
        y_v = va['Target'].astype(float).values
        X_te = te[feature_cols].astype(float).values
        y_te = te['Target'].values
        
        # Normalize
        mu_x = np.nanmean(X_tr, axis=0)
        sigma_x = np.nanstd(X_tr, axis=0) + 1e-12
        X_tr_n = (X_tr - mu_x) / sigma_x
        X_v_n = (X_v - mu_x) / sigma_x
        X_te_n = (X_te - mu_x) / sigma_x
        
        mu_y = float(np.nanmean(y_tr))
        sigma_y = float(np.nanstd(y_tr)) + 1e-12
        y_tr_n = (y_tr - mu_y) / sigma_y
        y_v_n = (y_v - mu_y) / sigma_y
        
        inp = layers.Input(shape=(len(feature_cols),))
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
        out = layers.Dense(1, activation="linear")(x)
        
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.Adam(1e-4, clipnorm=0.5), loss="mae")
        
        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_tr_n, y_tr_n, validation_data=(X_v_n, y_v_n), epochs=40, batch_size=64, callbacks=[es], verbose=0)
        
        preds_n = model.predict(X_te_n, verbose=0).flatten()
        preds = preds_n * sigma_y + mu_y
        
        mae_days[f'Day_{day}'] = float(np.mean(np.abs(preds - y_te)))
        
    return mae_days

# Run comparison
mae_with = evaluate_forecasts(train_df, val_df, test_df, features_with)
mae_without = evaluate_forecasts(train_df, val_df, test_df, features_without)

comparison_df = pd.DataFrame({
    'With Vol & OpenInt': mae_with,
    'Without Vol & OpenInt': mae_without
})
comparison_df['Delta (Without - With)'] = comparison_df['Without Vol & OpenInt'] - comparison_df['With Vol & OpenInt']

print(comparison_df.to_string())