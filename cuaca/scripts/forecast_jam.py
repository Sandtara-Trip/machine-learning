import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== KONFIGURASI ==================
LATITUDE = -8.65
LONGITUDE = 115.22

# Features yang akan digunakan untuk prediksi (sesuai dengan training)
NUM_FEATS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "cloudcover",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
]

# API URL untuk Open-Meteo
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_API_URL = (
    f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&current=temperature_2m,relative_humidity_2m,precipitation,cloudcover,"
    f"windspeed_10m,winddirection_10m,surface_pressure,weathercode"
)

# File paths - flexible path detection
HISTORY_CSV = "weather_history.csv"
PAST_HOURS = 24
FUTURE_HOURS = 12

# Function to find model files
def find_model_files():
    """Find model files in possible locations"""
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "models"),           # Same dir/models
        os.path.join(script_dir, "..", "models"),     # Parent dir/models
        os.path.join(script_dir, "."),                # Current directory
        os.path.join(script_dir, ".."),               # Parent directory
        "models",                                     # Relative models
        "../models",                                  # Relative parent/models
    ]
    
    required_files = {
        "scaler": ["scaler2.pkl"],
        "model": ["best_hourly_model2.h5"],
        "encoder": ["weather_label_encoder2.pkl"]
    }
    
    found_files = {}
    
    for base_path in possible_paths:
        abs_path = os.path.abspath(base_path)
        if os.path.exists(abs_path):
            for file_type, filenames in required_files.items():
                if file_type in found_files:
                    continue
                for filename in filenames:
                    full_path = os.path.join(abs_path, filename)
                    if os.path.exists(full_path):
                        found_files[file_type] = full_path
                        logger.info(f"Found {file_type}: {full_path}")
                        break
    
    return found_files

# ================== LOAD MODEL & PREPROCESSORS ==================
def load_model_and_preprocessors():
    """Load trained model, scaler, and label encoder"""
    try:
        # Find model files
        model_files = find_model_files()
        
        # Check if all required files are found
        required_types = ["scaler", "model", "encoder"]
        missing_files = [f for f in required_types if f not in model_files]
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            logger.error("Please ensure the following files exist in one of these locations:")
            logger.error("- ./models/scaler.pkl")
            logger.error("- ./models/best_hourly_model.h5") 
            logger.error("- ./models/weather_label_encoder.pkl")
            logger.error("Or in current directory or parent directory")
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load the files
        logger.info("Loading scaler...")
        scaler = joblib.load(model_files["scaler"])
        
        logger.info("Loading model...")
        # Try loading model with different approaches
        try:
            model = load_model(model_files["model"], compile=False)
        except Exception as model_error:
            logger.warning(f"Failed to load model with compile=False: {model_error}")
            logger.info("Trying to load model with custom objects...")
            import tensorflow.keras.metrics as metrics
            custom_objects = {
                'mse': metrics.MeanSquaredError,
                'mae': metrics.MeanAbsoluteError,
                'accuracy': metrics.CategoricalAccuracy
            }
            model = load_model(model_files["model"], custom_objects=custom_objects, compile=False)
        
        logger.info("Loading encoder...")
        encoder = joblib.load(model_files["encoder"])
        
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        logger.info(f"Encoder classes: {len(encoder.classes_)} classes")
        
        logger.info("Model dan preprocessors berhasil dimuat")
        return model, scaler, encoder
    
    except Exception as e:
        logger.error(f"Error loading model/preprocessors: {e}")
        raise

# ================== DATA FETCHING ==================
def fetch_current_weather():
    """Fetch current weather data from Open-Meteo API"""
    try:
        response = requests.get(CURRENT_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        current = data["current"]
        
        # Extract semua fitur yang diperlukan
        weather_data = {
            "time": current["time"],
            "temperature_2m": current["temperature_2m"],
            "relative_humidity_2m": current["relative_humidity_2m"], 
            "precipitation": current["precipitation"],
            "cloudcover": current["cloudcover"],
            "windspeed_10m": current["windspeed_10m"],
            "winddirection_10m": current["winddirection_10m"],
            "surface_pressure": current["surface_pressure"],
            "weathercode": current["weathercode"]
        }
        
        logger.info(f"Data cuaca saat ini berhasil diambil: {current['time']}")
        return weather_data
        
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise

def fetch_historical_data():
    """Fetch historical hourly data untuk bootstrap history.csv - FIXED VERSION"""
    try:
        now = datetime.datetime.utcnow()
        start_time = (now - datetime.timedelta(hours=PAST_HOURS))
        end_time = now
        
        # FORMAT DATE YANG BENAR untuk Open-Meteo API
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        
        # URL yang benar - gunakan start_date dan end_date, bukan start_time/end_time
        historical_url = (
            f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation,cloudcover,"
            f"windspeed_10m,winddirection_10m,surface_pressure,weathercode"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone=auto"
        )
        
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        logger.info(f"API URL: {historical_url}")
        
        response = requests.get(historical_url, timeout=30)
        
        # Debug response
        if response.status_code != 200:
            logger.error(f"API Error {response.status_code}: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        # Convert ke DataFrame
        hourly_data = data["hourly"]
        df = pd.DataFrame({
            "time": hourly_data["time"],
            "temperature_2m": hourly_data["temperature_2m"],
            "relative_humidity_2m": hourly_data["relative_humidity_2m"],
            "precipitation": hourly_data["precipitation"], 
            "cloudcover": hourly_data["cloudcover"],
            "windspeed_10m": hourly_data["windspeed_10m"],
            "winddirection_10m": hourly_data["winddirection_10m"],
            "surface_pressure": hourly_data["surface_pressure"],
            "weathercode": hourly_data["weathercode"]
        })
        
        # Filter untuk 24 jam terakhir saja
        df['time'] = pd.to_datetime(df['time'])
        cutoff_time = now - datetime.timedelta(hours=PAST_HOURS)
        df = df[df['time'] >= cutoff_time].copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Keep only the last 24 records
        df = df.tail(PAST_HOURS).reset_index(drop=True)
        
        # Convert time back to string format
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M')
        
        df.to_csv(HISTORY_CSV, index=False)
        logger.info(f"Historical data berhasil disimpan: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        
        # FALLBACK: Buat dummy data jika API gagal
        logger.warning("Creating dummy historical data as fallback...")
        create_dummy_historical_data()

def create_dummy_historical_data():
    """Create dummy historical data as fallback"""
    try:
        # Buat data dummy berdasarkan current weather
        current_weather = fetch_current_weather()
        
        dummy_data = []
        base_time = datetime.datetime.now() - datetime.timedelta(hours=PAST_HOURS)
        
        for i in range(PAST_HOURS):
            time_stamp = (base_time + datetime.timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M')
            
            # Add some variation to make it realistic
            variation = np.random.normal(0, 0.1)  # Small random variation
            
            dummy_record = {
                "time": time_stamp,
                "temperature_2m": current_weather["temperature_2m"] + variation * 5,
                "relative_humidity_2m": max(0, min(100, current_weather["relative_humidity_2m"] + variation * 10)),
                "precipitation": max(0, current_weather["precipitation"] + variation * 0.5),
                "cloudcover": max(0, min(100, current_weather["cloudcover"] + variation * 20)),
                "windspeed_10m": max(0, current_weather["windspeed_10m"] + variation * 2),
                "winddirection_10m": (current_weather["winddirection_10m"] + variation * 30) % 360,
                "surface_pressure": current_weather["surface_pressure"] + variation * 5,
                "weathercode": current_weather["weathercode"]
            }
            dummy_data.append(dummy_record)
        
        df = pd.DataFrame(dummy_data)
        df.to_csv(HISTORY_CSV, index=False)
        logger.info(f"Dummy historical data created: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error creating dummy data: {e}")
        raise

# ================== PREDICTION ==================
def update_history_and_predict(model, scaler, encoder):
    """Update history dengan data terbaru dan buat prediksi"""
    try:
        # 1. Fetch current weather
        current_weather = fetch_current_weather()
        
        # 2. Load history atau buat baru jika tidak ada
        if not os.path.exists(HISTORY_CSV):
            logger.info("History file tidak ditemukan, membuat data historis...")
            fetch_historical_data()
        
        # 3. Load history dan append current data
        history_df = pd.read_csv(HISTORY_CSV)
        
        # Buat dataframe untuk current weather (hanya fitur numerik untuk prediksi)
        current_df = pd.DataFrame([{
            "time": current_weather["time"],
            **{feat: current_weather[feat] for feat in NUM_FEATS},
            "weathercode": current_weather["weathercode"]
        }])
        
        # Gabungkan dengan history
        updated_df = pd.concat([history_df, current_df], ignore_index=True)
        
        # Keep hanya 24 jam terakhir + current
        updated_df = updated_df.tail(PAST_HOURS + 1).reset_index(drop=True)
        
        # Simpan kembali history yang sudah diupdate
        updated_df.to_csv(HISTORY_CSV, index=False)
        
        # 4. Prepare data untuk prediksi
        # Ambil 24 jam terakhir sebelum current time untuk input
        input_df = updated_df.tail(PAST_HOURS)[NUM_FEATS]
        
        logger.info(f"Input data shape: {input_df.shape}")
        logger.info(f"Input data columns: {list(input_df.columns)}")
        
        # Handle missing values
        input_df = input_df.fillna(input_df.mean())
        
        # Check for any remaining NaN atau infinite values
        if input_df.isnull().any().any():
            logger.warning("Found NaN values in input data after fillna")
            input_df = input_df.fillna(0)
        
        if np.isinf(input_df.values).any():
            logger.warning("Found infinite values in input data")
            input_df = input_df.replace([np.inf, -np.inf], 0)
        
        # Scale input features
        logger.info("Scaling input features...")
        X_scaled = scaler.transform(input_df)
        X_input = X_scaled.reshape(1, PAST_HOURS, len(NUM_FEATS))
        
        logger.info(f"Scaled input shape: {X_input.shape}")
        logger.info(f"Input range: min={X_input.min():.4f}, max={X_input.max():.4f}")
        
        # 5. Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(X_input, verbose=0)
        
        logger.info(f"Number of prediction outputs: {len(predictions)}")
        for i, pred in enumerate(predictions):
            logger.info(f"Prediction output {i} shape: {pred.shape}")
        
        # 6. Process predictions
        # predictions[0] = classification output (weather), predictions[1] = regression output (numeric features)
        y_class_pred = predictions[0]  # shape: (1, future_hours, num_classes)
        y_reg_pred = predictions[1]    # shape: (1, future_hours, num_features)
        
        logger.info(f"Classification prediction shape: {y_class_pred.shape}")
        logger.info(f"Regression prediction shape: {y_reg_pred.shape}")
        
        # Ambil indeks kelas (weathercode) langsung tanpa decode ke deskripsi
        class_indices = np.argmax(y_class_pred[0], axis=-1)  # array of shape (future_hours,)
        logger.info(f"Predicted weathercode indices: {class_indices}")
        
        # Inverse transform regression predictions (numeric features)
        logger.info("Inverse transforming regression predictions...")
        numeric_predictions = scaler.inverse_transform(y_reg_pred[0])
        
        logger.info(f"Numeric predictions shape: {numeric_predictions.shape}")
        logger.info(f"Numeric predictions range: min={numeric_predictions.min():.4f}, max={numeric_predictions.max():.4f}")
        
        # 7. Combine predictions => hanya weathercode + angka hasil prediksi fitur numerik
        prediction_results = []
        base_time = pd.to_datetime(current_weather["time"])
        
        for i in range(FUTURE_HOURS):
            pred_time = base_time + pd.Timedelta(hours=i+1)
            
            pred_data = {
                "time": pred_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "weathercode": int(class_indices[i]),  # Ganti dari weather_description menjadi angka weathercode
                **{f"predicted_{NUM_FEATS[j]}": float(numeric_predictions[i, j]) 
                   for j in range(len(NUM_FEATS))}
            }
            prediction_results.append(pred_data)
        
        logger.info(f"Prediksi berhasil dibuat untuk {FUTURE_HOURS} jam ke depan")
        return current_weather, prediction_results
        
    except Exception as e:
        logger.error(f"Error dalam prediksi: {e}")
        raise

# ================== SAVE RESULTS ==================
def save_predictions_to_json(current_weather, predictions):
    """Simpan hasil prediksi ke file JSON dengan enhanced debugging"""
    try:
        print(f"🔍 DEBUG: Mulai menyimpan JSON...")
        print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
        
        result = {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "hourly_predictions": predictions,
            "metadata": {
                "model_type": "LSTM Multi-output",
                "prediction_horizon_hours": FUTURE_HOURS,
                "features_used": NUM_FEATS
            }
        }
        
        print(f"🔍 DEBUG: Result dictionary created, keys: {list(result.keys())}")
        print(f"🔍 DEBUG: Predictions count: {len(predictions)}")
        
        # Simpan dengan timestamp di nama file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_predictions_{timestamp}.json"
        latest_filename = "latest_weather_predictions.json"
        
        print(f"🔍 DEBUG: Attempting to save to: {os.path.abspath(filename)}")
        
        # Save timestamped file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Timestamped file saved: {filename}")
        
        # Save latest file
        with open(latest_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Latest file saved: {latest_filename}")
        
        # Verify files exist
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ File {filename} exists, size: {size} bytes")
        
        if os.path.exists(latest_filename):
            size = os.path.getsize(latest_filename)
            print(f"✅ File {latest_filename} exists, size: {size} bytes")
        
        # List JSON files in current directory
        print(f"📁 JSON files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"   - {file}")
        
        logger.info(f"Predictions saved to {filename}")
        
    except Exception as e:
        print(f"❌ ERROR in save_predictions_to_json: {e}")
        logger.error(f"Error saving predictions: {e}")
        raise

# ================== MAIN PREDICTION JOB ==================
def run_prediction_job():
    """Main job yang dijalankan setiap jam - Enhanced Debug Version"""
    try:
        print("🚀 Starting prediction job...")
        logger.info("========== Starting hourly prediction job ==========")
        
        # Check if model files exist first
        print("🔍 Checking for model files...")
        model_files = find_model_files()
        print(f"🔍 Found model files: {model_files}")
        
        if len(model_files) < 3:
            print("❌ Missing model files! Cannot proceed.")
            print("Required files:")
            print("  - scaler.pkl")
            print("  - best_hourly_model.h5")  
            print("  - weather_label_encoder.pkl")
            return
        
        # Load model dan preprocessors
        print("📚 Loading model and preprocessors...")
        model, scaler, encoder = load_model_and_preprocessors()
        print("✅ Model loaded successfully")
        
        # Update history dan buat prediksi
        print("🌤️ Fetching weather data and making predictions...")
        current_weather, predictions = update_history_and_predict(model, scaler, encoder)
        print(f"✅ Predictions generated: {len(predictions)} hourly forecasts")
        
        # Debug prediction results
        print(f"🔍 Current weather keys: {list(current_weather.keys())}")
        print(f"🔍 First prediction keys: {list(predictions[0].keys()) if predictions else 'No predictions'}")
        
        # Simpan hasil ke JSON
        print("💾 Saving predictions to JSON...")
        save_predictions_to_json(current_weather, predictions)
        print("✅ JSON files saved successfully")
        
        logger.info("========== Prediction job completed successfully ==========")
        
    except Exception as e:
        print(f"❌ PREDICTION JOB FAILED: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        logger.error(f"Prediction job failed: {e}")

# ================== SCHEDULER ==================
def start_scheduler():
    """Start APScheduler untuk menjalankan prediksi setiap jam"""
    scheduler = BlockingScheduler()
    
    # Jalankan setiap jam pada menit ke-5 (untuk memastikan data API sudah tersedia)
    scheduler.add_job(
        run_prediction_job,
        CronTrigger(minute=5),  # Jalankan setiap jam pada menit ke-5
        id='hourly_weather_prediction',
        name='Hourly Weather Prediction',
        misfire_grace_time=300  # Allow 5 minutes grace time
    )
    
    logger.info("Scheduler started. Weather predictions will run every hour at minute 5.")
    logger.info("Press Ctrl+C to stop the scheduler.")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
        scheduler.shutdown()

# Quick test function untuk debugging
def test_api_and_files():
    """Test function untuk debugging API dan file access"""
    print("=== DEBUGGING TEST ===")
    
    # Test 1: Current directory dan file permissions
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Directory contents: {os.listdir('.')}")
    
    # Test 2: Model files
    print("\n🔍 Looking for model files...")
    model_files = find_model_files()
    for file_type, path in model_files.items():
        print(f"  {file_type}: {path}")
    
    # Test 3: API access - Current weather
    print("\n🌐 Testing Current Weather API...")
    try:
        response = requests.get(CURRENT_API_URL, timeout=10)
        print(f"  Current API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Current API Data keys: {list(data.keys())}")
            if 'current' in data:
                print(f"  Current weather time: {data['current'].get('time', 'N/A')}")
        else:
            print(f"  Current API Error: {response.text}")
    except Exception as api_e:
        print(f"  Current API Exception: {api_e}")
    
    # Test 4: API access - Historical data
    print("\n🌐 Testing Historical Data API...")
    try:
        now = datetime.datetime.utcnow()
        start_date = (now - datetime.timedelta(hours=2)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        
        test_historical_url = (
            f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&hourly=temperature_2m,cloudcover"
            f"&start_date={start_date}&end_date={end_date}"
        )
        
        print(f"  Test URL: {test_historical_url}")
        response = requests.get(test_historical_url, timeout=10)
        print(f"  Historical API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Historical API Data keys: {list(data.keys())}")
        else:
            print(f"  Historical API Error: {response.text}")
    except Exception as api_e:
        print(f"  Historical API Exception: {api_e}")
    
    # Test 5: Write permissions
    print("\n📝 Testing write permissions...")
    try:
        with open("test_write.json", 'w') as f:
            json.dump({"test": "data"}, f)
        print("  ✅ JSON write OK")
        os.remove("test_write.json")
    except Exception as write_e:
        print(f"  ❌ Write error: {write_e}")

# ================== MAIN ==================
if __name__ == "__main__":
    print("=== Weather Prediction System for Denpasar ===")
    print("1. Run prediction once")
    print("2. Start continuous hourly predictions")
    print("3. Debug test (recommended first)")
    
    choice = input("Choose option (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Run prediction once
        run_prediction_job()
    elif choice == "2":
        # Start scheduler
        start_scheduler()
    elif choice == "3":
        # Debug test
        test_api_and_files()
    else:
        print("Invalid choice. Exiting...")