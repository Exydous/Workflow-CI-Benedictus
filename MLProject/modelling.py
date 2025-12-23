import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. SETUP PATH DINAMIS ---
# Mendapatkan lokasi folder tempat script ini berada (MLProject)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Menentukan lokasi data (Asumsi: ada di folder 'world-data-2023_preprocessing' di dalam MLProject)
# Jika nama folder data Anda berbeda, sesuaikan string 'world-data-2023_preprocessing' di bawah ini
DATA_PATH = os.path.join(BASE_DIR, 'world-data-2023_preprocessing', 'processed_data.csv')

print(f"Mencari data di: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    # Fallback: Coba cari di root jika tidak ketemu di folder
    print("Data tidak ditemukan di path utama, mencoba mencari di root...")
    DATA_PATH = os.path.join(BASE_DIR, '..', 'world-data-2023_preprocessing', 'processed_data.csv')
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"FATAL: Data tidak ditemukan di {DATA_PATH}")

# --- 2. LOAD & SPLIT DATA ---
df = pd.read_csv(DATA_PATH)

# Pastikan nama kolom target sesuai dengan dataset Anda
target_col = 'Life expectancy' 

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. KONFIGURASI MLFLOW (LOGIKA BARU) ---
# Aktifkan Autologging untuk menangkap parameter & metrik otomatis
mlflow.autolog()

# Logika Penentuan Environment:
# Jika script dijalankan oleh 'mlflow run' (GitHub Actions), active_run() akan bernilai True.
# Jika dijalankan manual (python modelling.py), active_run() akan None.

if mlflow.active_run():
    print("🚀 Mode: CI/CD Pipeline (Dijalankan via mlflow run)")
    # Gunakan run yang sudah dibuatkan oleh GitHub/MLflow Project
    run_context = mlflow.start_run()
else:
    print("💻 Mode: Manual / Local Run")
    # Set eksperimen hanya jika jalan manual
    mlflow.set_experiment("Eksperimen_Basic_Benedictus")
    # Buat run baru
    run_context = mlflow.start_run(run_name="Manual_Run")

# --- 4. TRAINING & EVALUASI ---
with run_context:
    print("Memulai Training...")
    
    # Inisialisasi Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Training (Autolog akan mencatat ini otomatis)
    model.fit(X_train, y_train)
    
    # Prediksi
    preds = model.predict(X_test)
    
    # Evaluasi Manual (Opsional, karena autolog biasanya sudah mencatat ini)
    mae = mean_absolute_error(y_test, preds)
    print(f"Training Selesai. MAE: {mae}")
    
    # Log metrik tambahan jika perlu
    mlflow.log_metric("mae_manual", mae)
    
    print("Run selesai & tersimpan di MLflow.")
