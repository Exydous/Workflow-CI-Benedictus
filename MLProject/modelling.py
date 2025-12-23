import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. SETUP PATH DINAMIS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Cek lokasi data (sesuaikan jika nama folder berbeda)
DATA_PATH = os.path.join(BASE_DIR, 'world-data-2023_preprocessing', 'processed_data.csv')

print(f"Mencari data di: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    # Fallback: Cari di folder parent jika struktur folder berbeda di runner
    DATA_PATH = os.path.join(BASE_DIR, '..', 'world-data-2023_preprocessing', 'processed_data.csv')
    if not os.path.exists(DATA_PATH):
         # Fallback terakhir: Cek langsung di folder MLProject jika file di-copy ke sana
        DATA_PATH = os.path.join(BASE_DIR, 'processed_data.csv')
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"FATAL: Data tidak ditemukan. Pastikan file CSV ada.")

# --- 2. LOAD & SPLIT DATA ---
df = pd.read_csv(DATA_PATH)
target_col = 'Life expectancy' 

X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. KONFIGURASI MLFLOW (REVISI LOGIKA) ---
mlflow.autolog()

# PERBAIKAN DI SINI:
# Cek apakah 'mlflow run' telah menetapkan ID di environment variable
# Ini adalah cara paling akurat untuk membedakan Mode CI/CD vs Manual
if os.environ.get("MLFLOW_RUN_ID"):
    print("🚀 Mode: CI/CD Pipeline (Dijalankan via mlflow run)")
    # PENTING: start_run() TANPA argumen akan otomatis mengambil ID dari environment
    run_context = mlflow.start_run()
else:
    print("💻 Mode: Manual / Local Run")
    mlflow.set_experiment("Eksperimen_Basic_Benedictus")
    run_context = mlflow.start_run(run_name="Manual_Run")

# --- 4. TRAINING ---
with run_context:
    print("Memulai Training...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Training Selesai. MAE: {mae}")
    
    # Log metrik manual (opsional, autolog sudah menangkap banyak hal)
    mlflow.log_metric("mae_manual", mae)
    print("Run selesai.")
