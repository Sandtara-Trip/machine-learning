import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
import pickle
import logging
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# Flask app setup
app = Flask(__name__)

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
WITA = pytz.timezone('Asia/Makassar')

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

# Target features untuk inverse transform (sesuai dengan training)
TARGET_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "cloudcover",
    "windspeed_10m",
    "surface_pressure",
    "winddirection_10m"
]

# API URL untuk Open-Meteo
BASE_API_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_API_URL = (
    f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&current=temperature_2m,relative_humidity_2m,precipitation,cloudcover,"
    f"windspeed_10m,winddirection_10m,surface_pressure,weathercode"
)

# File paths
HISTORY_CSV = "weather_history.csv"
PAST_HOURS = 24
FUTURE_HOURS = 12

# Global variables untuk menyimpan hasil prediksi
current_predictions = None
last_prediction_time = None

# Function to convert wind direction to cardinal
def deg_to_compass(deg):
    dirs = ['Utara', 'Timur Laut', 'Timur', 'Tenggara',
            'Selatan', 'Barat Daya', 'Barat', 'Barat Laut']
    ix = int((deg + 22.5) // 45) % 8
    return dirs[ix]

# Function to map weather group (SAMA SEPERTI TRAINING) - DIPERBAIKI DENGAN FALLBACK
def map_weather_group(code):
    """
    Map weather code ke weather group dengan fallback untuk unknown codes
    """
    try:
        code = int(code)
    except (ValueError, TypeError):
        logger.warning(f"Invalid weather code: {code}, using default 'Cerah'")
        return 'Cerah'
    
    if code in [0, 1, 2]:
        return 'Cerah'
    elif code in [3]:
        return 'Berawan'
    elif code in [51, 53]:
        return 'Gerimis'
    elif code in [45, 48, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99]:
        # Semua weather code lainnya yang terkait hujan/presipitasi
        return 'Hujan'
    else:
        # Fallback untuk weather code yang tidak dikenal
        logger.warning(f"Unknown weather code: {code}, mapping to 'Cerah' as fallback")
        return 'Cerah'

# TAMBAHAN: Function untuk handle unknown labels dalam encoder
def safe_transform_weather_labels(encoder, weather_groups):
    """
    Safely transform weather groups to labels, dengan fallback untuk unknown labels
    """
    try:
        # Get known classes from encoder
        known_classes = set(encoder.classes_)
        
        # Check for unknown classes
        unknown_classes = set(weather_groups) - known_classes
        if unknown_classes:
            logger.warning(f"Unknown weather groups found: {unknown_classes}")
            logger.info(f"Known classes in encoder: {list(known_classes)}")
            
            # Replace unknown classes with most common known class (fallback)
            fallback_class = encoder.classes_[0]  # Use first class as fallback
            logger.info(f"Using '{fallback_class}' as fallback for unknown classes")
            
            # Replace unknown classes
            weather_groups_safe = [
                group if group in known_classes else fallback_class 
                for group in weather_groups
            ]
        else:
            weather_groups_safe = weather_groups
        
        # Transform safely
        return encoder.transform(weather_groups_safe)
        
    except Exception as e:
        logger.error(f"Error in safe_transform_weather_labels: {e}")
        # Create fallback labels (all zeros - first class)
        return np.zeros(len(weather_groups), dtype=int)

# Function to find model files
def find_model_files():
    """Find model files in possible locations"""
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
        "scaler": "hourly_scaler.pkl",
        "scaler_precip": "hourly_scaler_precip.pkl",
        "model": "best_hourly_model.h5",
        "encoder": "hourly_weather_label_encoder.pkl"
    }
    
    found_files = {}
    
    for base_path in possible_paths:
        abs_path = os.path.abspath(base_path)
        if os.path.exists(abs_path):
            for file_type, filename in required_files.items():
                if file_type in found_files:
                    continue
                full_path = os.path.join(abs_path, filename)
                if os.path.exists(full_path):
                    found_files[file_type] = full_path
                    logger.info(f"Found {file_type}: {full_path}")
    
    return found_files

# ================== LOAD MODEL & PREPROCESSORS ==================
def load_model_and_preprocessors():
    """Load trained model, scalers, and label encoder"""
    try:
        model_files = find_model_files()
        
        required_types = ["scaler", "scaler_precip", "model", "encoder"]
        missing_files = [f for f in required_types if f not in model_files]
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load the files
        logger.info("Loading scalers...")
        with open(model_files["scaler"], 'rb') as f:
            scaler = pickle.load(f)
        with open(model_files["scaler_precip"], 'rb') as f:
            scaler_precip = pickle.load(f)
        
        logger.info("Loading model...")
        try:
            model = load_model(model_files["model"], compile=False)
        except Exception as model_error:
            logger.warning(f"Failed to load model with compile=False: {model_error}")
            import tensorflow.keras.metrics as metrics
            custom_objects = {
                'mse': metrics.MeanSquaredError,
                'mae': metrics.MeanAbsoluteError,
                'accuracy': metrics.CategoricalAccuracy
            }
            model = load_model(model_files["model"], custom_objects=custom_objects, compile=False)
        
        logger.info("Loading encoder...")
        with open(model_files["encoder"], 'rb') as f:
            encoder = pickle.load(f)
        
        # LOG ENCODER INFO FOR DEBUGGING
        logger.info(f"Encoder classes: {encoder.classes_}")
        logger.info("Model dan preprocessors berhasil dimuat")
        return model, scaler, scaler_precip, encoder
    
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
        logger.info(f"Weather code dari API: {current['weathercode']}")  # DEBUG LOG
        return weather_data
        
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise

def fetch_historical_data():
    """Fetch historical hourly data"""
    try:
        now = datetime.datetime.utcnow()
        start_time = (now - datetime.timedelta(hours=PAST_HOURS + 1))
        end_time = now
        
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        
        historical_url = (
            f"{BASE_API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation,cloudcover,"
            f"windspeed_10m,winddirection_10m,surface_pressure,weathercode"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone=auto"
        )
        
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        response = requests.get(historical_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
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
        
        # Proper datetime conversion and filtering
        df['time'] = pd.to_datetime(df['time'])
        cutoff_time = pd.to_datetime(now - datetime.timedelta(hours=PAST_HOURS))
        df = df[df['time'] >= cutoff_time].copy()
        
        # Fill missing values properly
        numeric_cols = [col for col in df.columns if col != 'time']
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64'] and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        # Ensure we have exactly PAST_HOURS records
        df = df.tail(PAST_HOURS).reset_index(drop=True)
        
        # Convert time back to string for CSV storage
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M')
        
        # DEBUG: Log unique weather codes in historical data
        unique_codes = df['weathercode'].unique()
        logger.info(f"Unique weather codes in historical data: {unique_codes}")
        
        df.to_csv(HISTORY_CSV, index=False)
        logger.info(f"Historical data berhasil disimpan: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        create_dummy_historical_data()

def create_dummy_historical_data():
    """Create dummy historical data as fallback"""
    try:
        current_weather = fetch_current_weather()
        
        dummy_data = []
        base_time = datetime.datetime.now() - datetime.timedelta(hours=PAST_HOURS)
        
        # Use only known weather codes (from training) for dummy data
        known_codes = [0, 1, 2, 3, 51, 53, 61, 63, 65]  # Common codes from training
        
        for i in range(PAST_HOURS):
            time_stamp = (base_time + datetime.timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M')
            variation = np.random.normal(0, 0.1)
            
            dummy_record = {
                "time": time_stamp,
                "temperature_2m": current_weather["temperature_2m"] + variation * 5,
                "relative_humidity_2m": max(0, min(100, current_weather["relative_humidity_2m"] + variation * 10)),
                "precipitation": max(0, current_weather["precipitation"] + variation * 0.5),
                "cloudcover": max(0, min(100, current_weather["cloudcover"] + variation * 20)),
                "windspeed_10m": max(0, current_weather["windspeed_10m"] + variation * 2),
                "winddirection_10m": (current_weather["winddirection_10m"] + variation * 30) % 360,
                "surface_pressure": current_weather["surface_pressure"] + variation * 5,
                "weathercode": np.random.choice(known_codes)  # Use known codes only
            }
            dummy_data.append(dummy_record)
        
        df = pd.DataFrame(dummy_data)
        df.to_csv(HISTORY_CSV, index=False)
        logger.info(f"Dummy historical data created: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error creating dummy data: {e}")
        raise

def clean_and_validate_dataframe(df):
    """Clean and validate dataframe untuk memastikan tidak ada masalah dengan data"""
    try:
        # Ensure proper data types
        df = df.copy()
        
        # Handle time column properly
        if 'time' in df.columns:
            if df['time'].dtype == 'object':
                # Check if time strings are concatenated incorrectly
                time_values = df['time'].astype(str)
                # Split any concatenated timestamps
                fixed_times = []
                for time_val in time_values:
                    if len(time_val) > 19:  # Normal timestamp is 19 chars (YYYY-MM-DDTHH:MM)
                        # This might be concatenated timestamps, take only the first valid part
                        time_val = time_val[:19]
                    fixed_times.append(time_val)
                df['time'] = fixed_times
        
        # Ensure numeric columns are properly typed
        numeric_cols = [col for col in NUM_FEATS + ['weathercode'] if col in df.columns]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with column mean
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
                # If still NaN (all values were invalid), fill with 0
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
        
        # Remove any remaining rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure we have the required columns
        required_cols = NUM_FEATS + ['weathercode']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                df[col] = 0
        
        logger.info(f"Dataframe cleaned successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning dataframe: {e}")
        raise

# ================== PREDICTION ==================
def update_history_and_predict(model, scaler, scaler_precip, encoder):
    """Update history dengan data terbaru dan buat prediksi"""
    global current_predictions, last_prediction_time
    
    try:
        # 1. Fetch current weather
        current_weather = fetch_current_weather()
        
        # 2. Load history atau buat baru jika tidak ada
        if not os.path.exists(HISTORY_CSV):
            logger.info("History file tidak ditemukan, membuat data historis...")
            fetch_historical_data()
        
        # 3. Load history dan bersihkan data
        try:
            history_df = pd.read_csv(HISTORY_CSV)
            history_df = clean_and_validate_dataframe(history_df)
        except Exception as csv_error:
            logger.warning(f"Error reading history CSV: {csv_error}, creating new data")
            fetch_historical_data()
            history_df = pd.read_csv(HISTORY_CSV)
            history_df = clean_and_validate_dataframe(history_df)
        
        # 4. Create current data entry
        current_df = pd.DataFrame([{
            "time": current_weather["time"],
            **{feat: current_weather[feat] for feat in NUM_FEATS},
            "weathercode": current_weather["weathercode"]
        }])
        current_df = clean_and_validate_dataframe(current_df)
        
        # 5. Combine and ensure proper length
        updated_df = pd.concat([history_df, current_df], ignore_index=True)
        updated_df = updated_df.tail(PAST_HOURS + 1).reset_index(drop=True)
        
        # Save updated history
        updated_df.to_csv(HISTORY_CSV, index=False)
        
        # 6. PREPROCESSING SAMA SEPERTI TRAINING
        # Ambil 24 jam terakhir untuk input
        df_recent = updated_df.tail(PAST_HOURS).copy()
        df_recent = clean_and_validate_dataframe(df_recent)
        
        # Ensure we have exactly PAST_HOURS records
        if len(df_recent) < PAST_HOURS:
            logger.warning(f"Insufficient data: {len(df_recent)}, padding with mean values")
            # Pad with mean values
            mean_values = df_recent[NUM_FEATS + ['weathercode']].mean()
            while len(df_recent) < PAST_HOURS:
                new_row = {col: mean_values[col] for col in NUM_FEATS + ['weathercode']}
                new_row['time'] = df_recent['time'].iloc[-1] if len(df_recent) > 0 else current_weather["time"]
                df_recent = pd.concat([df_recent, pd.DataFrame([new_row])], ignore_index=True)
        
        # Map weather code ke weather group dan encode (DIPERBAIKI DENGAN SAFE TRANSFORM)
        df_recent['weathercode'] = df_recent['weathercode'].astype(int)
        df_recent['weather_group'] = df_recent['weathercode'].apply(map_weather_group)
        
        # DEBUG: Log weather groups before encoding
        unique_groups = df_recent['weather_group'].unique()
        logger.info(f"Weather groups to encode: {unique_groups}")
        
        # GUNAKAN SAFE TRANSFORM
        df_recent['weather_label'] = safe_transform_weather_labels(encoder, df_recent['weather_group'])
        
        # Normalize features (SAMA SEPERTI TRAINING)
        other_feats = [f for f in NUM_FEATS if f != 'precipitation']
        
        # Ensure all features exist and are numeric
        for feat in NUM_FEATS:
            if feat not in df_recent.columns:
                df_recent[feat] = 0
            df_recent[feat] = pd.to_numeric(df_recent[feat], errors='coerce').fillna(0)
        
        # Normalize fitur kecuali precipitation
        df_recent[other_feats] = scaler.transform(df_recent[other_feats])
        
        # Normalize precipitation secara terpisah
        df_recent[['precipitation']] = scaler_precip.transform(df_recent[['precipitation']])
        
        # Prepare input untuk model
        X_input = df_recent[NUM_FEATS].values.reshape(1, PAST_HOURS, len(NUM_FEATS))
        
        # Validate input
        if np.isnan(X_input).any() or np.isinf(X_input).any():
            logger.error("Invalid input data contains NaN or Inf")
            raise ValueError("Input data contains invalid values")
        
        # 7. Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(X_input, verbose=0)
        
        # 8. Process predictions
        y_class_pred = predictions[0]  # weather classification
        y_reg_pred = predictions[1]    # regression features
        
        # Get weather labels
        class_indices = np.argmax(y_class_pred[0], axis=-1)
        weather_labels = encoder.inverse_transform(class_indices)
        
        # === INVERSE TRANSFORM (SAMA SEPERTI TRAINING) ===
        y_reg_predictions = y_reg_pred[0]  # Shape: (future_hours, n_regression_features)
        
        # Based on the training code, regression output excludes winddirection_10m
        regression_features = [f for f in TARGET_FEATURES if f != 'winddirection_10m']
        
        # Separate precipitation and other features for inverse transform
        other_feats_for_inverse = [f for f in regression_features if f != 'precipitation']
        
        # The regression output should have len(regression_features) columns
        # Let's assume the order is the same as regression_features
        y_num_combined = np.zeros((FUTURE_HOURS, len(regression_features)))
        
        for i, feat in enumerate(regression_features):
            if feat == 'precipitation':
                # Inverse transform precipitation and clamp to positive
                y_precip = scaler_precip.inverse_transform(y_reg_predictions[:, [i]])
                y_num_combined[:, i] = np.maximum(y_precip.flatten(), 0)
            else:
                # Find the index in the scaler (other_feats)
                scaler_idx = other_feats.index(feat)
                # Create a temp array for inverse transform
                temp_data = np.zeros((FUTURE_HOURS, len(other_feats)))
                temp_data[:, scaler_idx] = y_reg_predictions[:, i]
                # Inverse transform
                temp_inverse = scaler.inverse_transform(temp_data)
                y_num_combined[:, i] = temp_inverse[:, scaler_idx]
        
        # Build prediction results
        prediction_results = []
        base_time = pd.to_datetime(current_weather["time"])
        
        for i in range(FUTURE_HOURS):
            pred_time = base_time + pd.Timedelta(hours=i+1)
            
            # Build prediction dictionary
            pred_data = {
                "time": pred_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "weather_label": weather_labels[i],
            }
            
            # Add regression features
            for j, feat in enumerate(regression_features):
                pred_data[feat] = float(y_num_combined[i, j])
            
            prediction_results.append(pred_data)
        
        # Update global variables
        current_predictions = {
            "current_weather": current_weather,
            "predictions": prediction_results,
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }
        last_prediction_time = datetime.datetime.now()
        
        logger.info(f"Prediksi berhasil dibuat untuk {FUTURE_HOURS} jam ke depan")
        return current_weather, prediction_results
        
    except Exception as e:
        logger.error(f"Error dalam prediksi: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ================== FLASK ROUTES ==================
@app.route('/predict/hourly', methods=['GET'])
def predict_hourly():
    """Endpoint untuk prediksi hourly"""
    try:
        global current_predictions, last_prediction_time
        
        # Check if we have recent predictions (less than 1 hour old)
        if (current_predictions and last_prediction_time and 
            (datetime.datetime.now() - last_prediction_time).seconds < 3600):
            logger.info("Returning cached predictions")
            response = {
                "status": "success",
                "generated_at": current_predictions["generated_at"],
                "location": {
                    "latitude": LATITUDE,
                    "longitude": LONGITUDE,
                    "city": "Denpasar, Bali"
                },
                "current_weather": current_predictions["current_weather"],
                "hourly_predictions": current_predictions["predictions"],
                "metadata": {
                    "model_type": "LSTM Multi-output",
                    "prediction_horizon_hours": FUTURE_HOURS,
                    "features_used": NUM_FEATS,
                    "data_source": "cached"
                }
            }
            return jsonify(response)
        
        # Generate new predictions
        model, scaler, scaler_precip, encoder = load_model_and_preprocessors()
        current_weather, predictions = update_history_and_predict(model, scaler, scaler_precip, encoder)
        
        response = {
            "status": "success",
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
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
                "features_used": NUM_FEATS,
                "data_source": "fresh"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in hourly prediction endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Cek kesehatan aplikasi"""
    try:
        model_files = find_model_files()
        status = "healthy" if len(model_files) == 4 else "unhealthy"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_files_found": model_files,
            "features_used": NUM_FEATS,
            "location": {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "city": "Denpasar, Bali"
            },
            "last_prediction": last_prediction_time.strftime("%Y-%m-%dT%H:%M:%S%z") if last_prediction_time else None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test/hourly', methods=['GET'])
def test_hourly():
    """Test endpoint untuk debugging"""
    try:
        output = []
        output.append("=== HOURLY PREDICTION DEBUGGING TEST ===")
        output.append(f"Current directory: {os.getcwd()}")
        
        output.append("\nLooking for hourly model files...")
        model_files = find_model_files()
        for file_type, path in model_files.items():
            output.append(f"  {file_type}: {path}")
        
        output.append("\nTesting Current Weather API...")
        try:
            response = requests.get(CURRENT_API_URL, timeout=10)
            output.append(f"  API Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                output.append(f"  API Data keys: {list(data.keys())}")
                if 'current' in data:
                    current = data['current']
                    output.append(f"  Current weather time: {current.get('time', 'N/A')}")
                    output.append(f"  Current weathercode: {current.get('weathercode', 'N/A')}")
                    # Test weather group mapping
                    weathercode = current.get('weathercode', 0)
                    weather_group = map_weather_group(weathercode)
                    output.append(f"  Weather group mapping: {weathercode} -> {weather_group}")
            else:
                output.append(f"  API Error: {response.text}")
        except Exception as api_e:
            output.append(f"  API Exception: {api_e}")
        
        # Test CSV loading
        output.append("\nTesting CSV operations...")
        if os.path.exists(HISTORY_CSV):
            try:
                df = pd.read_csv(HISTORY_CSV)
                output.append(f"  CSV loaded: {len(df)} rows, {len(df.columns)} columns")
                output.append(f"  CSV columns: {list(df.columns)}")
                output.append(f"  Sample time values: {df['time'].head(3).tolist() if 'time' in df.columns else 'No time column'}")
            except Exception as csv_e:
                output.append(f"  CSV Error: {csv_e}")
        else:
            output.append("  No CSV file found")
        
        # Test encoder
        try:
            model_files = find_model_files()
            if 'encoder' in model_files:
                with open(model_files['encoder'], 'rb') as f:
                    encoder = pickle.load(f)
                output.append(f"\nEncoder classes: {encoder.classes_}")
        except Exception as enc_e:
            output.append(f"\nEncoder test failed: {enc_e}")
        
        output.append(f"\nFeatures used in training: {NUM_FEATS}")
        output.append(f"Target features: {TARGET_FEATURES}")
        
        return jsonify({
            "status": "success",
            "debug_info": "\n".join(output),
            "timestamp": datetime.datetime.now(WITA).strftime("%Y-%m-%dT%H:%M:%S%z")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ================== SCHEDULER ==================
def run_prediction_job():
    """Background job yang dijalankan setiap jam"""
    try:
        logger.info("========== Starting hourly prediction background job ==========")
        model, scaler, scaler_precip, encoder = load_model_and_preprocessors()
        current_weather, predictions = update_history_and_predict(model, scaler, scaler_precip, encoder)
        logger.info("========== Prediction job completed successfully ==========")
        
    except Exception as e:
        logger.error(f"Background prediction job failed: {e}")

def start_background_scheduler():
    """Start background scheduler"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_prediction_job,
        CronTrigger(minute=5),  # Run every hour at minute 5
        id='hourly_weather_prediction',
        name='Hourly Weather Prediction',
        misfire_grace_time=300
    )
    scheduler.start()
    logger.info("Background scheduler started.")
    return scheduler

# ================== MAIN ==================
if __name__ == "__main__":
    # Start background scheduler
    scheduler = start_background_scheduler()
    
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
