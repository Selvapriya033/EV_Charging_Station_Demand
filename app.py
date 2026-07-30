import os
from pathlib import Path
import joblib
import pandas as pd
import sklearn
import streamlit as st

# Page Configuration MUST be the first Streamlit command executed
st.set_page_config(
    page_title="EV Charging Station Demand Prediction",
    page_icon="⚡",
    layout="wide",
)


# -----------------------------
# File Finder Helper
# -----------------------------
def find_file(target_name):
    """Searches for a file ignoring case in script dir and current working dir."""
    possible_dirs = [Path(__file__).resolve().parent, Path.cwd()]

    for d in possible_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.name.lower() == target_name.lower():
                    return f
    return None


# Locate required files
model_path = find_file("ev_model.pkl")
scaler_path = find_file("scaler.pkl")
columns_path = find_file("feature_columns.pkl")

# -----------------------------
# Load Saved Assets
# -----------------------------
model = None
scaler = None
expected_features = None

if model_path and model_path.exists():
    try:
        model = joblib.load(model_path)
        # Try extracting expected features directly from the model
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
    except Exception as e:
        st.error(f"Error loading `ev_model.pkl`: {e}")
else:
    current_files = os.listdir(Path(__file__).resolve().parent)
    st.error(
        f"❌ Could not locate `ev_model.pkl` in your app folder.\n\n"
        f"**Files found in folder:** `{current_files}`"
    )

if scaler_path and scaler_path.exists():
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.warning(f"Could not load `scaler.pkl`: {e}")

# If feature_columns.pkl exists, let it take precedence
if columns_path and columns_path.exists():
    try:
        expected_features = joblib.load(columns_path)
    except Exception as e:
        st.warning(f"Could not load `feature_columns.pkl`: {e}")

# -----------------------------
# UI Header
# -----------------------------
st.title("⚡ EV Charging Station Demand")
st.markdown("Predict EV charging demand using Machine Learning.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Features")

location_type = st.sidebar.selectbox(
    "Location Type", ["Urban", "Suburban", "Highway", "Rural"]
)

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type", ["Car", "SUV", "Bus", "Bike", "Truck"]
)

waiting_time = st.sidebar.number_input(
    "Waiting Time (minutes)", min_value=0.0, value=5.0
)

battery_capacity = st.sidebar.number_input(
    "Battery Capacity (kWh)", min_value=10.0, value=60.0
)

initial_soc = st.sidebar.slider("Initial SOC (%)", 0, 100, 30)

charging_power = st.sidebar.number_input(
    "Charging Power (kW)", min_value=1.0, value=22.0
)

queue_length = st.sidebar.number_input("Queue Length", min_value=0, value=2)

station_load = st.sidebar.slider("Station Load", 0.0, 1.0, 0.50)

electricity_price = st.sidebar.number_input(
    "Electricity Price", min_value=0.0, value=8.5
)

renewable_energy_ratio = st.sidebar.slider(
    "Renewable Energy Ratio", 0.0, 1.0, 0.40
)

traffic_density = st.sidebar.selectbox(
    "Traffic Density", ["Low", "Medium", "High"]
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition", ["Sunny", "Cloudy", "Rainy", "Foggy"]
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)

time_slot = st.sidebar.selectbox(
    "Time Slot", ["Morning", "Afternoon", "Evening", "Night"]
)

# -----------------------------
# Create Raw Input DataFrame
# -----------------------------
raw_input_df = pd.DataFrame(
    {
        "location_type": [location_type],
        "vehicle_type": [vehicle_type],
        "waiting_time": [waiting_time],
        "battery_capacity_kWh": [battery_capacity],
        "initial_soc": [initial_soc],
        "charging_power_kW": [charging_power],
        "queue_length": [queue_length],
        "station_load": [station_load],
        "electricity_price": [electricity_price],
        "renewable_energy_ratio": [renewable_energy_ratio],
        "traffic_density": [traffic_density],
        "weather_condition": [weather_condition],
        "day_of_week": [day_of_week],
        "time_slot": [time_slot],
    }
)

st.subheader("Input Data")
st.dataframe(raw_input_df)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict Charging Demand"):
    if model is None:
        st.error(
            "Prediction unavailable: `ev_model.pkl` is missing or failed to load."
        )
    else:
        try:
            # Step 1: One-Hot Encode user inputs
            encoded_df = pd.get_dummies(raw_input_df)

            # Step 2: Align with expected feature columns
            if expected_features is not None:
                encoded_df = encoded_df.reindex(
                    columns=expected_features, fill_value=0
                )

            # Step 3: Scale numeric features if scaler is present
            if scaler is not None:
                X_prepared = scaler.transform(encoded_df)
            else:
                X_prepared = encoded_df

            # Step 4: Predict demand
            prediction = model.predict(X_prepared)[0]

            st.success("Prediction Completed")
            st.metric(
                label="Predicted Charging Demand", value=f"{prediction:.2f}"
            )

            if prediction < 30:
                st.info("🔋 Low Charging Demand")
            elif prediction < 70:
                st.warning("⚡ Medium Charging Demand")
            else:
                st.error("🚗 High Charging Demand")

        except Exception as e:
            st.error(f"Error processing inputs or making prediction: {e}")

st.markdown("---")
st.write("Developed using Streamlit and Machine Learning")