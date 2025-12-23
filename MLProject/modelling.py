import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- SETUP PATH BARU ---
# Karena data ada di dalam folder yang sama dengan script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data ada di folder 'namadataset_preprocessing' di sebelah script
DATA_PATH = os.path.join(BASE_DIR, 'world-data-2023_preprocessing', 'processed_data.csv')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data tidak ditemukan di {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# --- SPLIT DATA ---
target_col = 'Life expectancy'
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLFLOW AUTOLOG ---
mlflow.set_experiment("Eksperimen_Basic_Benedictus")
mlflow.autolog()

with mlflow.start_run(run_name="Basic_Run"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds)}")
