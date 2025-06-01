import os
import json
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
from flask import Flask, jsonify, request
import sys
from io import StringIO
import pytz

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

# Inisialisasi Flask app
app = Flask(__name__)

# Zona waktu WITA (UTC+8)
WITA = pytz.timezone('Asia/Makassar')

# ================== KONFIGURASI ==================
LATITUDE = -8.65
LONGITUDE = 115.22

# Features yang akan digunakan untuk prediksi 
DAILY_FEATS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "delta_temp",
    "day_of_month",
    "month"
]

# Regression targets 
REG_TARGETS = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']

# API URL untuk Open-Meteo (base URL)
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"

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
        "scaler": ["daily_scaler.pkl", "daily_scaler_rule4.pkl"],
        "model": ["best_model_daily.h5", "best_model_rule4.h5"]
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
            logger.error("- daily_scaler.pkl or daily_scaler_rule4.pkl")
            logger.error("- best_model_daily.h5 or best_model_rule4.h5")
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

# ================== HELPER FUNCTION FOR API URL ==================
def build_daily_api_url(start_date, end_date, additional_params=None):
    """Build API URL for daily weather data with date range"""
    base_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "timezone": "Asia/Makassar"
    }
    
    # Add date range
    base_params["start_date"] = start_date.strftime('%Y-%m-%d')
    base_params["end_date"] = end_date.strftime('%Y-%m-%d')
    
    # Add additional parameters if provided
    if additional_params:
        base_params.update(additional_params)
    
    # Build URL
    params_str = "&".join([f"{k}={v}" for k, v in base_params.items()])
    return f"{BASE_API_URL}?{params_str}"

# ================== WEATHER CODE MAPPING  ==================
def map_weather_group(code):
    """Map weather code to weather group"""
    if code in [0, 1, 2, 3]:
        return 'Cerah'
    elif code in [51]:
        return 'Berawan'
    elif code in [53, 55]:
        return 'Gerimis'
    else:
        return 'Hujan'

# ================== DATA FETCHING ==================
def fetch_daily_weather_data(days_back=35):
    """Fetch daily weather data from Open-Meteo API"""
    try:
        # Calculate date range berdasarkan WITA
        end_date = datetime.datetime.now(WITA).date()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        # Build API URL using helper function
        api_url = build_daily_api_url(start_date, end_date)
        
        logger.info(f"Fetching daily data from {start_date} to {end_date}")
        logger.info(f"API URL: {api_url}")
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check if daily data exists
        if "daily" not in data:
            logger.error("No 'daily' key in API response")
            raise ValueError("Invalid API response format")
        
        # Convert to DataFrame
        daily_data = data["daily"]
        
        # Check required fields
        required_fields = ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code"]
        missing_fields = [field for field in required_fields if field not in daily_data]
        if missing_fields:
            logger.error(f"Missing fields in API response: {missing_fields}")
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        df = pd.DataFrame({
            "time": daily_data["time"],
            "temperature_2m_max": daily_data["temperature_2m_max"],
            "temperature_2m_min": daily_data["temperature_2m_min"],
            "precipitation_sum": daily_data["precipitation_sum"],
            "weather_code": daily_data["weather_code"],
        })
        
        logger.info(f"Raw DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"Sample data:\n{df.head()}")
        
        # Convert time columns to datetime dengan WITA
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(WITA)
        
        # Handle missing values 
        for col in ['temperature_2m_max', 'temperature_2m_min', 'weather_code']:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col}, filling with mean")
                df[col] = df[col].fillna(df[col].mean())
        
        if df['precipitation_sum'].isna().any():
            logger.warning("Found NaN values in precipitation_sum, filling with 0")
            df['precipitation_sum'] = df['precipitation_sum'].fillna(0.0)
        
        # Feature engineering 
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        
        # Map weather code to weather group 
        # Ensure weather_code is numeric and handle NaN
        df['weather_code'] = pd.to_numeric(df['weather_code'], errors='coerce')
        df['weather_code'] = df['weather_code'].fillna(0)  # Fill NaN with clear weather code
        df['weather_group'] = df['weather_code'].astype(int).apply(map_weather_group)
        
        logger.info(f"Daily weather data berhasil diambil: {len(df)} records")
        logger.info(f"Weather groups distribution: {df['weather_group'].value_counts().to_dict()}")
        logger.info(f"Final DataFrame columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching daily weather data: {e}")
        # Try to create dummy data as fallback
        logger.info("Attempting to create dummy data as fallback...")
        return create_dummy_daily_data()

def create_dummy_daily_data():
    """Create dummy daily data as fallback"""
    try:
        dummy_data = []
        base_date = datetime.datetime.now(WITA).date() - datetime.timedelta(days=PAST_DAYS)
        
        for i in range(PAST_DAYS):
            date = base_date + datetime.timedelta(days=i)
            
            # Create reasonable dummy values for Denpasar
            temp_max = 30 + np.random.normal(0, 2)
            temp_min = 24 + np.random.normal(0, 1.5)
            precip = max(0, np.random.exponential(1))
            
            # Generate realistic weather code
            if precip > 5:
                weather_code = np.random.choice([61, 63, 65])  # Rain codes
            elif precip > 0:
                weather_code = np.random.choice([53, 55])      # Drizzle codes
            elif (temp_max - temp_min) < 5:
                weather_code = 51                              # Fog/mist
            else:
                weather_code = np.random.choice([0, 1, 2, 3])  # Clear codes
            
            dummy_record = {
                "time": date.strftime('%Y-%m-%d'),
                "temperature_2m_max": temp_max,
                "temperature_2m_min": temp_min,
                "precipitation_sum": precip,
                "weather_code": weather_code
            }
            dummy_data.append(dummy_record)
        
        df = pd.DataFrame(dummy_data)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(WITA)
        
        # Add engineered features 
        df['delta_temp'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['weather_group'] = df['weather_code'].astype(int).apply(map_weather_group)
        
        logger.info(f"Dummy daily data created: {len(df)} records")
        logger.info(f"Dummy data columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating dummy daily data: {e}")
        raise

# ================== EXTENDED PREDICTION (MAIN FUNCTION) ==================
def make_extended_daily_prediction(model, scaler, label_encoder):
    """Make extended daily prediction (DEFAULT OUTPUT)"""
    try:
        # Load or fetch daily data
        if not os.path.exists(DAILY_HISTORY_CSV):
            logger.info("CSV file not found, fetching new data...")
            df_daily = fetch_daily_weather_data()
            df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        else:
            logger.info("Loading existing CSV file...")
            try:
                df_daily = pd.read_csv(DAILY_HISTORY_CSV)
                df_daily['time'] = pd.to_datetime(df_daily['time'])
                
                # Check if time column has timezone info
                if df_daily['time'].dt.tz is None:
                    df_daily['time'] = df_daily['time'].dt.tz_localize(WITA)
                
                # Check if we have the required columns
                required_columns_from_api = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'weather_code']
                missing_api_columns = [col for col in required_columns_from_api if col not in df_daily.columns]
                
                # If missing essential columns, fetch fresh data
                if missing_api_columns:
                    logger.warning(f"CSV file missing required columns: {missing_api_columns}")
                    logger.info("Fetching fresh data from API...")
                    df_daily = fetch_daily_weather_data()
                    df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
                else:
                    # Update data if needed
                    last_date = df_daily['time'].max().date()
                    today = datetime.datetime.now(WITA).date()
                    
                    if (today - last_date).days > 1:
                        logger.info("Updating daily data...")
                        df_updated = fetch_daily_weather_data()
                        df_daily = df_updated
                        df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
                        
            except Exception as csv_error:
                logger.warning(f"Error reading CSV: {csv_error}")
                logger.info("Fetching fresh data...")
                df_daily = fetch_daily_weather_data()
                df_daily.to_csv(DAILY_HISTORY_CSV, index=False)
        
        logger.info(f"Working with DataFrame shape: {df_daily.shape}")
        logger.info(f"DataFrame columns: {df_daily.columns.tolist()}")
        
        # Ensure required columns exist
        required_base_columns = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'weather_code', 'time']
        missing_columns = [col for col in required_base_columns if col not in df_daily.columns]
        if missing_columns:
            logger.error(f"Missing required base columns: {missing_columns}")
            raise ValueError(f"Missing required base columns: {missing_columns}")
        
        # Prepare features 
        if 'delta_temp' not in df_daily.columns:
            df_daily['delta_temp'] = df_daily['temperature_2m_max'] - df_daily['temperature_2m_min']
        if 'day_of_month' not in df_daily.columns:
            df_daily['day_of_month'] = df_daily['time'].dt.day
        if 'month' not in df_daily.columns:
            df_daily['month'] = df_daily['time'].dt.month
        if 'weather_group' not in df_daily.columns:
            # Ensure weather_code is numeric
            df_daily['weather_code'] = pd.to_numeric(df_daily['weather_code'], errors='coerce')
            df_daily['weather_code'] = df_daily['weather_code'].fillna(0)
            df_daily['weather_group'] = df_daily['weather_code'].astype(int).apply(map_weather_group)
        
        # Verify all required features exist
        missing_features = [feat for feat in DAILY_FEATS if feat not in df_daily.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Scale features
        df_daily_scaled = df_daily.copy()
        logger.info(f"Scaling features: {DAILY_FEATS}")
        df_daily_scaled[DAILY_FEATS] = scaler.transform(df_daily_scaled[DAILY_FEATS])
        
        # Determine prediction range based on last_date
        last_date = df_daily['time'].max().date()
        year_ld = last_date.year
        month_ld = last_date.month
        last_day_of_month = calendar.monthrange(year_ld, month_ld)[1]
        
        logger.info(f"Last date in data: {last_date}")
        logger.info(f"Last day of month {month_ld}: {last_day_of_month}")
        
        # Get current weather info for context
        current_weather = {
            "date": last_date.strftime("%Y-%m-%d"),
            "temperature_2m_max": float(df_daily['temperature_2m_max'].iloc[-1]),
            "temperature_2m_min": float(df_daily['temperature_2m_min'].iloc[-1]),
            "precipitation_sum": float(df_daily['precipitation_sum'].iloc[-1]),
            "weather_code": int(df_daily['weather_code'].iloc[-1]),
            "weather_description": df_daily['weather_group'].iloc[-1] if 'weather_group' in df_daily.columns else "Unknown"
        }
        
        # Determine prediction range
        if last_date.day < last_day_of_month:
            # Prediksi sampai akhir bulan ini
            start_date = last_date + datetime.timedelta(days=1)
            forecast_year, forecast_month = year_ld, month_ld
            end_of_month = datetime.date(forecast_year, forecast_month, last_day_of_month)
            logger.info(f"Predicting from {start_date} to end of current month: {end_of_month}")
        else:
            # Prediksi untuk bulan berikutnya
            if month_ld == 12:
                forecast_year, forecast_month = year_ld + 1, 1
            else:
                forecast_year, forecast_month = year_ld, month_ld + 1
            
            start_date = datetime.date(forecast_year, forecast_month, 1)
            end_day = calendar.monthrange(forecast_year, forecast_month)[1]
            end_of_month = datetime.date(forecast_year, forecast_month, end_day)
            logger.info(f"Predicting for next month from {start_date} to {end_of_month}")
        
        # Prepare sliding window for prediction
        results = []
        df_hist = df_daily_scaled.sort_values('time').reset_index(drop=True)
        last_window = df_hist.tail(PAST_DAYS)[DAILY_FEATS].reset_index(drop=True).copy()
        
        logger.info(f"Initial window shape: {last_window.shape}")
        logger.info(f"Window features: {last_window.columns.tolist()}")
        
        # Generate predictions day by day
        current_date = start_date
        while current_date <= end_of_month:
            # Prepare input untuk model
            X_input = last_window.values.reshape(1, PAST_DAYS, len(DAILY_FEATS))
            predictions = model.predict(X_input, verbose=0)
            
            # Process classification output
            class_probs = predictions[0][0, 0, :]
            class_idx = np.argmax(class_probs)
            desc_pred = label_encoder.inverse_transform([class_idx])[0]
            confidence = float(np.max(class_probs))
            
            # Process regression output dengan proper inverse transform
            reg_scaled = predictions[1][0, 0, :]
            
            # Create dummy array untuk inverse transform 
            dummy = np.zeros((1, len(DAILY_FEATS)))
            dummy[0, 0] = reg_scaled[0]  # temp_max
            dummy[0, 1] = reg_scaled[1]  # temp_min  
            dummy[0, 2] = reg_scaled[2]  # precipitation
            dummy[0, 3] = reg_scaled[0] - reg_scaled[1]  # delta_temp
            dummy[0, 4] = current_date.day  # day_of_month
            dummy[0, 5] = current_date.month  # month
            
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
            
            # Update sliding window dengan prediksi baru 
            dom = current_date.day
            month_val = current_date.month
            delta_val = pred_temp_max - pred_temp_min
            
            new_feature = np.array([[pred_temp_max, pred_temp_min, pred_precip_sum, delta_val, dom, month_val]])
            new_feature_scaled = scaler.transform(new_feature)[0]
            
            # Slide the window
            last_window = last_window.drop(index=0).reset_index(drop=True).copy()
            last_window.loc[PAST_DAYS - 1] = new_feature_scaled
            
            current_date += datetime.timedelta(days=1)
        
        logger.info(f"Extended prediction completed: {len(results)} days")
        return current_weather, results
        
    except Exception as e:
        logger.error(f"Error in extended daily prediction: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ================== SAVE RESULTS ==================
def save_daily_predictions_to_json(current_weather, extended_predictions):
    """Save daily prediction results to JSON"""
    try:
        result = {
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "predictions": extended_predictions,
            "prediction_summary": {
                "total_days_predicted": len(extended_predictions),
                "prediction_start_date": extended_predictions[0]["date"],
                "prediction_end_date": extended_predictions[-1]["date"],
                "covers_full_month": True
            },
            "metadata": {
                "model_type": "LSTM Multi-output Daily Weather Prediction",
                "prediction_horizon": "Extended Daily (Until End of Month)",
                "features_used": DAILY_FEATS,
                "weather_categories": ["Berawan", "Cerah", "Gerimis", "Hujan"]
            }
        }
        
        # Save files
        timestamp = datetime.datetime.now(WITA).strftime("%Y%m%d_%H%M%S")
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
    """Main job untuk daily prediction - DEFAULT EXTENDED PREDICTION"""
    try:
        logger.info("========== Starting daily prediction job ==========")
        
        # Load model
        model, scaler, label_encoder = load_model_and_preprocessors()
        
        # Make extended predictions (DEFAULT)
        current_weather, extended_predictions = make_extended_daily_prediction(model, scaler, label_encoder)
        
        # Save results
        save_daily_predictions_to_json(current_weather, extended_predictions)
        
        logger.info("========== Daily prediction job completed successfully ==========")
        
    except Exception as e:
        logger.error(f"Daily prediction job failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ================== SCHEDULER ==================
def start_daily_scheduler():
    """Start scheduler untuk daily predictions"""
    scheduler = BlockingScheduler(timezone=WITA)
    
    # Run daily at 6:00 AM WITA
    scheduler.add_job(
        run_daily_prediction_job,
        CronTrigger(hour=6, minute=0, timezone=WITA),
        id='daily_weather_prediction',
        name='Daily Weather Prediction',
        misfire_grace_time=1800  # 30 minutes grace time
    )
    
    logger.info("Daily scheduler started. Predictions will run every day at 6:00 AM WITA.")
    logger.info("Press Ctrl+C to stop the scheduler.")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Daily scheduler stopped by user.")
        scheduler.shutdown()

# ================== DEBUGGING ==================
def test_daily_api_and_files():
    """Test function untuk debugging daily prediction"""
    output = []
    output.append("=== DAILY PREDICTION DEBUGGING TEST ===")
    
    # Test 1: File access
    output.append(f" Current directory: {os.getcwd()}")
    
    # Test 2: Model files
    output.append("\n Looking for daily model files...")
    model_files = find_model_files()
    for file_type, path in model_files.items():
        output.append(f"  {file_type}: {path}")
    
    # Test 3: API access
    output.append("\n Testing Daily Weather API...")
    try:
        end_date = datetime.datetime.now(WITA).date()
        start_date = end_date - datetime.timedelta(days=7)
        
        # Use the helper function to build URL
        test_url = build_daily_api_url(start_date, end_date)
        
        response = requests.get(test_url, timeout=10)
        output.append(f"  API Status: {response.status_code}")
        output.append(f"  API URL: {test_url}")
        if response.status_code == 200:
            data = response.json()
            output.append(f"  API Data keys: {list(data.keys())}")
            if 'daily' in data:
                daily_data = data['daily']
                output.append(f"  Daily data keys: {list(daily_data.keys())}")
                output.append(f"  Records count: {len(daily_data['time'])}")
                
                # Test weather code mapping
                if 'weather_code' in daily_data:
                    codes = daily_data['weather_code'][:5]  # First 5 codes
                    output.append(f"  Sample weather codes: {codes}")
                    mapped = [map_weather_group(int(code)) for code in codes if code is not None]
                    output.append(f"  Mapped weather groups: {mapped}")
        else:
            output.append(f"  API Error: {response.text}")
    except Exception as api_e:
        output.append(f"  API Exception: {api_e}")
    
    # Test 4: Feature consistency
    output.append(f"\n Features used in training: {DAILY_FEATS}")
    
    # Test 5: Data processing
    output.append("\n Testing data processing...")
    try:
        test_df = fetch_daily_weather_data(days_back=7)
        output.append(f"  Test DataFrame shape: {test_df.shape}")
        output.append(f"  Test DataFrame columns: {test_df.columns.tolist()}")
        output.append(f"  Weather group distribution: {test_df['weather_group'].value_counts().to_dict()}")
    except Exception as data_e:
        output.append(f"  Data processing error: {data_e}")
    
    return "\n".join(output)

# ================== FLASK ENDPOINTS ==================
@app.route('/predict', methods=['GET'])
@app.route('/predict/daily', methods=['GET'])
def predict_daily():
    """Main prediction endpoint"""
    try:
        # Load model dan preprocessors
        model, scaler, label_encoder = load_model_and_preprocessors()
        
        # Jalankan prediksi extended 
        current_weather, extended_predictions = make_extended_daily_prediction(model, scaler, label_encoder)
        
        # Simpan hasil ke JSON
        save_daily_predictions_to_json(current_weather, extended_predictions)
        
        # Format response
        response = {
            "status": "success",
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "current_weather": current_weather,
            "predictions": extended_predictions,
            "prediction_summary": {
                "total_days_predicted": len(extended_predictions),
                "prediction_start_date": extended_predictions[0]["date"],
                "prediction_end_date": extended_predictions[-1]["date"],
                "covers_full_month": True
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in daily prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }), 500

@app.route('/test', methods=['GET'])
@app.route('/test/daily', methods=['GET'])
def test_daily():
    """Test endpoint untuk debugging"""
    try:
        debug_output = test_daily_api_and_files()
        return jsonify({
            "status": "success",
            "debug_info": debug_output,
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        model_files = find_model_files()
        required_types = ["scaler", "model"]
        missing_files = [f for f in required_types if f not in model_files]
        
        status = "healthy" if not missing_files else "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_files_found": model_files,
            "missing_files": missing_files,
            "features_used": DAILY_FEATS,
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            print("Testing daily prediction system...")
            print(test_daily_api_and_files())
            
        elif command == "predict":
            print("Running extended daily prediction...")
            run_daily_prediction_job()
            
        elif command == "schedule":
            print("Starting daily prediction scheduler...")
            start_daily_scheduler()
            
        elif command == "flask":
            print("Starting Flask API server...")
            app.run(host='0.0.0.0', port=5000, debug=True)
            
        else:
            print("Available commands:")
            print("  python script.py test      - Test API and model files")
            print("  python script.py predict   - Run extended prediction")
            print("  python script.py schedule  - Start scheduled predictions")
            print("  python script.py flask     - Start Flask API server")
    else:
        print("Starting Flask API server by default...")
        app.run(host='0.0.0.0', port=5000, debug=True)
