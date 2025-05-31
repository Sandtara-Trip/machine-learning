import os
import json
import time
import datetime
import calendar
import requests
import pandas as pd
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_weather_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== KONFIGURASI ==================
LATITUDE = -8.65
LONGITUDE = 115.22

# Features yang akan digunakan untuk prediksi (sesuai dengan training)
DAILY_FEATS = [
    "temperature_2m_max",
    "temperature_2m_min", 
    "precipitation_sum",
    "day_length",
    "day_of_month"
]

# API URL untuk Open-Meteo (daily data)
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_API_URL = (
    f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset"
    f"&timezone=auto"
)

# File paths
DAILY_HISTORY_CSV = "daily_weather_history.csv"
PAST_DAYS = 30
FUTURE_DAYS = 1

# Function to find model files
def find_model_files():
    """Find daily model files in possible locations"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "models"),
        os.path.join(script_dir, "..", "models"),
        os.path.join(script_dir, "."),
        os.path.join(script_dir, ".."),
        "models",
        "../models",
    ]
    
    required_files = {
        "scaler": ["daily_scaler_rule4.pkl"],
        "model": ["best_model_rule4.h5"]
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
    """Load trained daily model, scaler, and label encoder"""
    try:
        model_files = find_model_files()
        
        required_types = ["scaler", "model"]
        missing_files = [f for f in required_types if f not in model_files]
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            logger.error("Please ensure the following files exist:")
            logger.error("- daily_scaler_rule4.pkl")
            logger.error("- best_model_rule4.h5")
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load scaler dan label encoder
        logger.info("Loading scaler and label encoder...")
        with open(model_files["scaler"], 'rb') as f:
            dump = pickle.load(f)
            scaler = dump['scaler']
            label_encoder = dump['label_encoder']
        
        logger.info("Loading model...")
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
        
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        logger.info(f"Label encoder classes: {label_encoder.classes_}")
        
        logger.info("Daily model dan preprocessors berhasil dimuat")
        return model, scaler, label_encoder
    
    except Exception as e:
        logger.error(f"Error loading daily model/preprocessors: {e}")
        raise

# ================== DATA FETCHING ==================
def fetch_daily_weather_data(days_back=35):
    """Fetch daily weather data from Open-Meteo API"""
    try:
        # Calculate date range
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        # Build API URL
        api_url = (
            f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset"
            f"&start_date={start_date.strftime('%Y-%m-%d')}"
            f"&end_date={end_date.strftime('%Y-%m-%d')}"
            f"&timezone=auto"
        )
        
        logger.info(f"Fetching daily data from {start_date} to {end_date}")
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        daily_data = data["daily"]
        df = pd.DataFrame({
            "time": daily_data["time"],
            "temperature_2m_max": daily_data["temperature_2m_max"],
            "temperature_2m_min": daily_data["temperature_2m_min"],
            "precipitation_sum": daily_data["precipitation_sum"],
            "sunrise": daily_data["sunrise"],
            "sunset": daily_data["sunset"]
        })
        
        # Convert time columns to datetime
        df['time'] = pd.to_datetime(df['time'])
        df['sunrise'] = pd.to_datetime(df['sunrise'])
        df['sunset'] = pd.to_datetime(df['sunset'])
        
        # Handle missing values
        for col in ['temperature_2m_max', 'temperature_2m_min']:
            df[col] = df[col].fillna(df[col].mean())
        if df['precipitation_sum'].isna().any():
            df['precipitation_sum'] = df['precipitation_sum'].fillna(0.0)
        
        # Feature engineering (sama seperti training)
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_length'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600.0
        df['day_of_month'] = df['time'].dt.day
        
        # Rule-based weather description (sama seperti training)
        def rule_weather_desc(row):
            precip = row['precipitation_sum']
            if precip > 2.0:
                return 'Hujan'
            elif 0 < precip <= 2.0:
                return 'Hujan Ringan'
            else:
                if row['delta_temp'] >= 6.0:
                    return 'Cerah'
                else:
                    return 'Berawan'
        
        df['weather_desc'] = df.apply(rule_weather_desc, axis=1)
        
        logger.info(f"Daily weather data berhasil diambil: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching daily weather data: {e}")
        raise

def create_dummy_daily_data():
    """Create dummy daily data as fallback"""
    try:
        dummy_data = []
        base_date = datetime.date.today() - datetime.timedelta(days=PAST_DAYS)
        
        for i in range(PAST_DAYS):
            date = base_date + datetime.timedelta(days=i)
            
            # Create reasonable dummy values for Denpasar
            dummy_record = {
                "time": date.strftime('%Y-%m-%d'),
                "temperature_2m_max": 30 + np.random.normal(0, 2),
                "temperature_2m_min": 24 + np.random.normal(0, 1.5),
                "precipitation_sum": max(0, np.random.exponential(1)),
                "sunrise": f"{date}T06:00:00",
                "sunset": f"{date}T18:30:00"
            }
            dummy_data.append(dummy_record)
        
        df = pd.DataFrame(dummy_data)
        df['time'] = pd.to_datetime(df['time'])
        df['sunrise'] = pd.to_datetime(df['sunrise'])
        df['sunset'] = pd.to_datetime(df['sunset'])
        
        # Add engineered features
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_length'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600.0
        df['day_of_month'] = df['time'].dt.day
        
        def rule_weather_desc(row):
            precip = row['precipitation_sum']
            if precip > 2.0:
                return 'Hujan'
            elif 0 < precip <= 2.0:
                return 'Hujan Ringan'
            else:
                if row['delta_temp'] >= 6.0:
                    return 'Cerah'
                else:
                    return 'Berawan'
        
        df['weather_desc'] = df.apply(rule_weather_desc, axis=1)
        
        df.to_csv(DAILY_HISTORY_CSV, index=False)
        logger.info(f"Dummy daily data created: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error creating dummy daily data: {e}")
        raise

# ================== PREDICTION ==================
def make_daily_prediction(model, scaler, label_encoder):
    """Make daily weather prediction"""
    try:
        # 1. Load atau fetch daily data
        if not os.path.exists(DAILY_HISTORY_CSV):
            logger.info("Daily history file tidak ditemukan, fetching data...")
            df_daily = fetch_daily_weather_data()
            df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        else:
            df_daily = pd.read_csv(DAILY_HISTORY_CSV)
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            df_daily['sunrise'] = pd.to_datetime(df_daily['sunrise'])
            df_daily['sunset'] = pd.to_datetime(df_daily['sunset'])
            
            # Update dengan data terbaru jika perlu
            last_date = df_daily['time'].max().date()
            today = datetime.date.today()
            
            if (today - last_date).days > 1:
                logger.info("Updating daily data...")
                df_updated = fetch_daily_weather_data()
                df_daily = df_updated
                df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        
        # 2. Prepare features (sama seperti training)
        if 'delta_temp' not in df_daily.columns:
            df_daily['delta_temp'] = df_daily['temperature_2m_max'] - df_daily['temperature_2m_min']
        if 'day_length' not in df_daily.columns:
            df_daily['day_length'] = (df_daily['sunset'] - df_daily['sunrise']).dt.total_seconds() / 3600.0
        if 'day_of_month' not in df_daily.columns:
            df_daily['day_of_month'] = df_daily['time'].dt.day
        
        # 3. Scale features
        df_daily_scaled = df_daily.copy()
        df_daily_scaled[DAILY_FEATS] = scaler.transform(df_daily_scaled[DAILY_FEATS])
        
        # 4. Prepare input sequence (last 30 days)
        last_30_days = df_daily_scaled.tail(PAST_DAYS)[DAILY_FEATS].values
        X_input = last_30_days.reshape(1, PAST_DAYS, len(DAILY_FEATS))
        
        logger.info(f"Input shape: {X_input.shape}")
        
        # 5. Make prediction
        logger.info("Making daily prediction...")
        predictions = model.predict(X_input, verbose=0)
        
        # 6. Process predictions
        y_class_pred = predictions[0]  # (1, 1, num_classes) 
        y_reg_pred = predictions[1]    # (1, 1, 3) - temp_max, temp_min, precip
        
        # Get weather class
        class_probs = y_class_pred[0, 0, :]
        class_idx = np.argmax(class_probs)
        weather_desc_pred = label_encoder.inverse_transform([class_idx])[0]
        confidence = float(np.max(class_probs))
        
        # Inverse transform regression predictions
        reg_scaled = y_reg_pred[0, 0, :]
        
        # Create dummy array for inverse transform
        dummy = np.zeros((1, len(DAILY_FEATS)))
        dummy[0, 0] = reg_scaled[0]  # temp_max
        dummy[0, 1] = reg_scaled[1]  # temp_min  
        dummy[0, 2] = reg_scaled[2]  # precipitation
        
        # Get current month for day_length average
        current_month = datetime.date.today().month
        avg_day_length = df_daily.groupby(df_daily['time'].dt.month)['day_length'].mean().get(current_month, 12.0)
        dummy[0, 3] = scaler.transform([[0, 0, 0, avg_day_length, 1]])[0, 3]  # normalized day_length
        dummy[0, 4] = scaler.transform([[0, 0, 0, 0, datetime.date.today().day]])[0, 4]  # normalized day_of_month
        
        inv = scaler.inverse_transform(dummy)
        pred_temp_max = float(inv[0, 0])
        pred_temp_min = float(inv[0, 1])
        pred_precip_sum = float(max(0, inv[0, 2]))  # Ensure non-negative precipitation
        
        # 7. Create prediction result
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        
        prediction_result = {
            "date": tomorrow.strftime("%Y-%m-%d"),
            "weather_description": weather_desc_pred,
            "confidence": confidence,
            "temperature_2m_max": pred_temp_max,
            "temperature_2m_min": pred_temp_min,
            "precipitation_sum": pred_precip_sum,
            "temperature_range": f"{pred_temp_min:.1f}°C - {pred_temp_max:.1f}°C"
        }
        
        # Get current weather for context
        current_weather = {
            "date": df_daily['time'].max().strftime("%Y-%m-%d"),
            "temperature_2m_max": float(df_daily['temperature_2m_max'].iloc[-1]),
            "temperature_2m_min": float(df_daily['temperature_2m_min'].iloc[-1]),
            "precipitation_sum": float(df_daily['precipitation_sum'].iloc[-1]),
            "weather_description": df_daily['weather_desc'].iloc[-1] if 'weather_desc' in df_daily.columns else "Unknown"
        }
        
        logger.info(f"Daily prediction completed for {tomorrow}")
        return current_weather, prediction_result
        
    except Exception as e:
        logger.error(f"Error in daily prediction: {e}")
        raise

# ================== EXTENDED PREDICTION (hingga akhir bulan) ==================
def make_extended_daily_prediction(model, scaler, label_encoder):
    """Make extended daily prediction until end of month (like original code)"""
    try:
        # Load daily data
        if not os.path.exists(DAILY_HISTORY_CSV):
            df_daily = fetch_daily_weather_data()
            df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        else:
            df_daily = pd.read_csv(DAILY_HISTORY_CSV)
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            df_daily['sunrise'] = pd.to_datetime(df_daily['sunrise'])
            df_daily['sunset'] = pd.to_datetime(df_daily['sunset'])
        
        # Prepare features
        if 'delta_temp' not in df_daily.columns:
            df_daily['delta_temp'] = df_daily['temperature_2m_max'] - df_daily['temperature_2m_min']
        if 'day_length' not in df_daily.columns:
            df_daily['day_length'] = (df_daily['sunset'] - df_daily['sunrise']).dt.total_seconds() / 3600.0
        if 'day_of_month' not in df_daily.columns:
            df_daily['day_of_month'] = df_daily['time'].dt.day
        
        # Scale features
        df_daily_scaled = df_daily.copy()
        df_daily_scaled[DAILY_FEATS] = scaler.transform(df_daily_scaled[DAILY_FEATS])
        
        # Setup prediction loop (sama seperti kode asli)
        last_date = df_daily['time'].max().date()
        start_date = last_date + datetime.timedelta(days=1)
        
        forecast_year = last_date.year
        forecast_month = last_date.month
        last_day = calendar.monthrange(forecast_year, forecast_month)[1]
        end_of_month = datetime.date(forecast_year, forecast_month, last_day)
        
        results = []
        last_window = df_daily_scaled.tail(PAST_DAYS)[DAILY_FEATS].reset_index(drop=True).copy()
        avg_day_length_per_month = df_daily.groupby(df_daily['time'].dt.month)['day_length'].mean().to_dict()
        
        current_date = start_date
        while current_date <= end_of_month:
            X_input = last_window.values.reshape(1, PAST_DAYS, len(DAILY_FEATS))
            predictions = model.predict(X_input, verbose=0)
            
            # Process classification
            class_probs = predictions[0][0, 0, :]
            class_idx = np.argmax(class_probs)
            desc_pred = label_encoder.inverse_transform([class_idx])[0]
            confidence = float(np.max(class_probs))
            
            # Process regression
            reg_scaled = predictions[1][0, 0, :]
            dummy = np.zeros((1, len(DAILY_FEATS)))
            dummy[0, 0] = reg_scaled[0]
            dummy[0, 1] = reg_scaled[1] 
            dummy[0, 2] = reg_scaled[2]
            inv = scaler.inverse_transform(dummy)
            pred_temp_max = float(inv[0, 0])
            pred_temp_min = float(inv[0, 1])
            pred_precip_sum = float(max(0, inv[0, 2]))
            
            results.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'weather_description': desc_pred,
                'confidence': confidence,
                'temperature_2m_max': pred_temp_max,
                'temperature_2m_min': pred_temp_min,
                'precipitation_sum': pred_precip_sum,
                'temperature_range': f"{pred_temp_min:.1f}°C - {pred_temp_max:.1f}°C"
            })
            
            # Update sliding window
            dom = current_date.day
            month_idx = current_date.month
            avg_day_len = avg_day_length_per_month.get(month_idx, 12.0)
            new_feature = np.array([[pred_temp_max, pred_temp_min, pred_precip_sum, avg_day_len, dom]])
            new_feature_scaled = scaler.transform(new_feature)[0]
            
            last_window = last_window.drop(index=0).reset_index(drop=True).copy()
            last_window.loc[PAST_DAYS - 1] = new_feature_scaled
            
            current_date += datetime.timedelta(days=1)
        
        logger.info(f"Extended prediction completed: {len(results)} days")
        return results
        
    except Exception as e:
        logger.error(f"Error in extended daily prediction: {e}")
        raise

# ================== SAVE RESULTS ==================
def save_daily_predictions_to_json(current_weather, tomorrow_prediction, extended_predictions=None):
    """Save daily prediction results to JSON"""
    try:
        result = {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "tomorrow_prediction": tomorrow_prediction,
            "metadata": {
                "model_type": "LSTM Multi-output Daily Weather Prediction",
                "prediction_horizon": "Daily",
                "features_used": DAILY_FEATS,
                "weather_categories": ["Berawan", "Cerah", "Hujan", "Hujan Ringan"]
            }
        }
        
        if extended_predictions:
            result["extended_predictions"] = extended_predictions
            result["metadata"]["extended_days"] = len(extended_predictions)
        
        # Save files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"daily_weather_predictions_{timestamp}.json"
        latest_filename = "latest_daily_weather_predictions.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(latest_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Daily predictions saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving daily predictions: {e}")
        raise

# ================== MAIN PREDICTION JOB ==================
def run_daily_prediction_job():
    """Main job untuk daily prediction"""
    try:
        logger.info("========== Starting daily prediction job ==========")
        
        # Load model
        model, scaler, label_encoder = load_model_and_preprocessors()
        
        # Make predictions
        current_weather, tomorrow_prediction = make_daily_prediction(model, scaler, label_encoder)
        
        # Optionally make extended predictions
        try:
            extended_predictions = make_extended_daily_prediction(model, scaler, label_encoder)
        except Exception as e:
            logger.warning(f"Extended prediction failed: {e}")
            extended_predictions = None
        
        # Save results
        save_daily_predictions_to_json(current_weather, tomorrow_prediction, extended_predictions)
        
        logger.info("========== Daily prediction job completed successfully ==========")
        
    except Exception as e:
        logger.error(f"Daily prediction job failed: {e}")
        raise

# ================== SCHEDULER ==================
def start_daily_scheduler():
    """Start scheduler untuk daily predictions"""
    scheduler = BlockingScheduler()
    
    # Run daily at 6:00 AM
    scheduler.add_job(
        run_daily_prediction_job,
        CronTrigger(hour=6, minute=0),
        id='daily_weather_prediction',
        name='Daily Weather Prediction',
        misfire_grace_time=1800  # 30 minutes grace time
    )
    
    logger.info("Daily scheduler started. Predictions will run every day at 6:00 AM.")
    logger.info("Press Ctrl+C to stop the scheduler.")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Daily scheduler stopped by user.")
        scheduler.shutdown()

# ================== DEBUGGING ==================
def test_daily_api_and_files():
    """Test function untuk debugging daily prediction"""
    print("=== DAILY PREDICTION DEBUGGING TEST ===")
    
    # Test 1: File access
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Test 2: Model files
    print("\n🔍 Looking for daily model files...")
    model_files = find_model_files()
    for file_type, path in model_files.items():
        print(f"  {file_type}: {path}")
    
    # Test 3: API access
    print("\n🌐 Testing Daily Weather API...")
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=7)
        
        test_url = (
            f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&start_date={start_date.strftime('%Y-%m-%d')}"
            f"&end_date={end_date.strftime('%Y-%m-%d')}"
        )
        
        response = requests.get(test_url, timeout=10)
        print(f"  API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  API Data keys: {list(data.keys())}")
            if 'daily' in data:
                daily_data = data['daily']
                print(f"  Daily data keys: {list(daily_data.keys())}")
                print(f"  Records count: {len(daily_data['time'])}")
        else:
            print(f"  API Error: {response.text}")
    except Exception as api_e:
        print(f"  API Exception: {api_e}")

# ================== MAIN ==================
if __name__ == "__main__":
    print("=== Daily Weather Prediction System for Denpasar ===")
    print("1. Run daily prediction once")
    print("2. Start daily scheduler (runs at 6:00 AM)")
    print("3. Debug test")
    
    choice = input("Choose option (1, 2, or 3): ").strip()
    
    if choice == "1":
        run_daily_prediction_job()
    elif choice == "2":
        start_daily_scheduler()
    elif choice == "3":
        test_daily_api_and_files()
    else:
        print("Invalid choice. Exiting...")