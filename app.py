import datetime
import os
import pickle
import traceback
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OrderTime — Food Delivery ETA Predictor",
    page_icon="🛵",
    layout="wide",
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_delivery_modell.pkl")

TRAFFIC_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
LOAD_ORDER = {"Low": 0, "Medium": 1, "High": 2}
PRIORITY_ORDER = {"Normal": 0, "Priority": 1, "VIP": 2}
DEFAULT_VEHICLE_SPEEDS = {
    "Bike": 32.0,
    "Scooter": 28.0,
    "Bicycle": 15.0,
    "Car": 35.0,
}


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from `{MODEL_PATH}`: {e}")
    st.code(traceback.format_exc())
    st.stop()


def prepare_features(raw_df: pd.DataFrame, estimator) -> pd.DataFrame:
    """Transforms raw dispatch-time order fields into the exact numeric/encoded matrix expected by the raw model."""
    df = raw_df.copy()
    dt = pd.to_datetime(df["Order_Date"], errors="coerce")

    out = pd.DataFrame(index=df.index)
    hour = pd.to_numeric(df["Order_Hour"], errors="coerce")
    out["Order_Hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_rush_hour"] = (hour.between(8, 10) | hour.between(18, 21)).astype(float)

    out["month"] = dt.dt.month.fillna(1).astype(float)
    out["day_of_month"] = dt.dt.day.fillna(15).astype(float)
    out["Is_Weekend"] = (
        pd.to_numeric(df["Is_Weekend"], errors="coerce")
        if "Is_Weekend" in df.columns
        else (dt.dt.dayofweek >= 5).astype(float)
    )
    out["Is_Festival"] = pd.to_numeric(df["Is_Festival"], errors="coerce")
    out["Rider_Experience_Years"] = pd.to_numeric(df["Rider_Experience_Years"], errors="coerce")
    out["Rider_Rating"] = pd.to_numeric(df["Rider_Rating"], errors="coerce")
    out["Restaurant_Rating"] = pd.to_numeric(df["Restaurant_Rating"], errors="coerce")
    out["Order_Items"] = pd.to_numeric(df["Order_Items"], errors="coerce")
    out["Road_Distance_km"] = pd.to_numeric(df["Road_Distance_km"], errors="coerce")
    out["Number_of_Signals"] = pd.to_numeric(df["Number_of_Signals"], errors="coerce")
    out["Preparation_Time_Min"] = pd.to_numeric(df["Preparation_Time_Min"], errors="coerce")

    out["same_zone"] = (df["Pickup_Zone"].astype(str) == df["Dropoff_Zone"].astype(str)).astype(float)
    out["traffic_level_ord"] = df["Traffic_Level"].astype(str).map(TRAFFIC_ORDER).fillna(1.0)
    out["restaurant_load_ord"] = df["Restaurant_Load"].astype(str).map(LOAD_ORDER).fillna(1.0)
    out["priority_ord"] = df["Delivery_Priority"].astype(str).map(PRIORITY_ORDER).fillna(0.0)
    out["prep_per_item"] = out["Preparation_Time_Min"] / out["Order_Items"].replace(0, np.nan)

    speed = df["Vehicle_Type"].astype(str).map(DEFAULT_VEHICLE_SPEEDS).fillna(28.0)
    out["expected_travel_min"] = out["Road_Distance_km"] / speed * 60.0
    out["total_expected_min"] = out["expected_travel_min"] + out["Preparation_Time_Min"]

    # If the model was trained on one-hot encoded DataFrame columns, expand categoricals & align
    if hasattr(estimator, "feature_names_in_"):
        expected_cols = list(estimator.feature_names_in_)
        cat_cols = ["Day_of_Week", "Weather", "Pickup_Zone", "Dropoff_Zone", "Vehicle_Type", "Cuisine_Type"]
        for c in cat_cols:
            if c in df.columns:
                out[c] = df[c].astype(str)
        out = pd.get_dummies(out, columns=[c for c in cat_cols if c in out.columns], dtype=float)
        out = out.reindex(columns=expected_cols, fill_value=0.0)
    else:
        # Trained on numpy array from ColumnTransformer (numeric + OneHotEncoded categoricals)
        cat_spaces = {
            "Day_of_Week": ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"],
            "Weather": ["Clear", "Cloudy", "Fog", "Rain", "Storm"],
            "Pickup_Zone": ["Central", "East", "North", "South", "West"],
            "Dropoff_Zone": ["Central", "East", "North", "South", "West"],
            "Vehicle_Type": ["Bicycle", "Bike", "Car", "Scooter"],
            "Cuisine_Type": ["Chinese", "Continental", "Fast Food", "Indian", "Italian", "Mexican"],
        }
        v2_num = [
            "is_rush_hour", "Is_Festival", "Rider_Experience_Years", "Order_Items",
            "Road_Distance_km", "Number_of_Signals", "traffic_level_ord", "priority_ord",
            "expected_travel_min", "Preparation_Time_Min", "prep_per_item", "total_expected_min",
        ]
        v1_num = [
            "Order_Hour", "hour_sin", "hour_cos", "is_rush_hour", "Is_Weekend", "Is_Festival",
            "month", "day_of_month", "same_zone", "Rider_Experience_Years", "Rider_Rating",
            "Restaurant_Rating", "Order_Items", "Road_Distance_km", "Number_of_Signals",
            "traffic_level_ord", "restaurant_load_ord", "priority_ord", "expected_travel_min",
            "Preparation_Time_Min", "prep_per_item",
        ]

        def build_matrix(num_cols, cat_cols):
            parts = [out[num_cols].to_numpy(dtype=float)]
            for col in cat_cols:
                vals = df[col].astype(str).to_numpy()[:, None]
                cats = np.array(cat_spaces[col])[None, :]
                parts.append((vals == cats).astype(float))
            return np.hstack(parts)

        mat_v2 = build_matrix(v2_num, ["Weather", "Vehicle_Type", "Cuisine_Type", "Pickup_Zone", "Dropoff_Zone"])
        mat_v1 = build_matrix(v1_num, ["Day_of_Week", "Weather", "Pickup_Zone", "Dropoff_Zone", "Vehicle_Type", "Cuisine_Type"])

        n_req = getattr(estimator, "n_features_in_", mat_v2.shape[1])
        if mat_v2.shape[1] == n_req:
            return pd.DataFrame(mat_v2)
        if mat_v1.shape[1] == n_req:
            return pd.DataFrame(mat_v1)
        # Pad or trim gracefully if custom categories differ slightly
        base = mat_v2 if abs(mat_v2.shape[1] - n_req) <= abs(mat_v1.shape[1] - n_req) else mat_v1
        if base.shape[1] < n_req:
            base = np.hstack([base, np.zeros((base.shape[0], n_req - base.shape[1]))])
        else:
            base = base[:, :n_req]
        return pd.DataFrame(base)

    return out


# --- UI ---
st.sidebar.title("🛵 OrderTime ETA")
st.sidebar.caption(f"Model: `{type(model).__name__}` ({getattr(model, 'n_features_in_', '?')} features)")
page = st.sidebar.radio("Navigate", ["Single Order Prediction", "Batch CSV Prediction"])

if page == "Single Order Prediction":
    st.title("Dispatch-Time Food Delivery ETA Predictor")

    with st.form("single_order_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Order & Timing")
            order_date = st.date_input("Order Date", datetime.date(2025, 11, 20))
            order_hour = st.slider("Order Hour (0–23)", 0, 23, 19)
            is_festival = st.selectbox("Is Festival?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            cuisine_type = st.selectbox("Cuisine Type", ["Indian", "Chinese", "Italian", "Fast Food", "Mexican", "Continental"])
            order_items = st.number_input("Order Items", 1, 50, 3, step=1)
            delivery_priority = st.selectbox("Delivery Priority", ["Normal", "Priority", "VIP"])

        with col2:
            st.subheader("Route & Conditions")
            road_distance = st.number_input("Road Distance (km)", 0.1, 150.0, 12.5, step=0.5)
            pickup_zone = st.selectbox("Pickup Zone", ["North", "South", "East", "West", "Central"])
            dropoff_zone = st.selectbox("Dropoff Zone", ["North", "South", "East", "West", "Central"])
            traffic_level = st.selectbox("Traffic Level", ["Low", "Moderate", "High", "Severe"])
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog", "Storm"])
            num_signals = st.number_input("Number of Signals", 0, 100, 8, step=1)

        with col3:
            st.subheader("Restaurant & Rider")
            prep_time = st.number_input("Preparation Time (min)", 0, 120, 18, step=1)
            restaurant_load = st.selectbox("Restaurant Load", ["Low", "Medium", "High"])
            restaurant_rating = st.slider("Restaurant Rating", 1.0, 5.0, 4.3, step=0.1)
            vehicle_type = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Bicycle", "Car"])
            rider_exp = st.number_input("Rider Experience (Years)", 0.0, 40.0, 2.5, step=0.5)
            rider_rating = st.slider("Rider Rating", 1.0, 5.0, 4.6, step=0.1)

        submitted = st.form_submit_button("Predict Delivery ETA", use_container_width=True)

    if submitted:
        dt = pd.to_datetime(order_date)
        raw_input = pd.DataFrame([{
            "Order_Date": dt.strftime("%Y-%m-%d"),
            "Order_Hour": int(order_hour),
            "Day_of_Week": dt.day_name(),
            "Is_Weekend": int(dt.dayofweek >= 5),
            "Is_Festival": int(is_festival),
            "Weather": weather,
            "Pickup_Zone": pickup_zone,
            "Dropoff_Zone": dropoff_zone,
            "Vehicle_Type": vehicle_type,
            "Rider_Experience_Years": float(rider_exp),
            "Rider_Rating": float(rider_rating),
            "Restaurant_Rating": float(restaurant_rating),
            "Cuisine_Type": cuisine_type,
            "Order_Items": int(order_items),
            "Restaurant_Load": restaurant_load,
            "Preparation_Time_Min": float(prep_time),
            "Road_Distance_km": float(road_distance),
            "Traffic_Level": traffic_level,
            "Number_of_Signals": int(num_signals),
            "Delivery_Priority": delivery_priority,
        }])

        try:
            X_ready = prepare_features(raw_input, model)
            pred_min = float(model.predict(X_ready)[0])

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Predicted Delivery ETA", f"{pred_min:.1f} mins")
            c2.metric("Estimated Arrival Window", f"{max(1.0, pred_min - 6):.0f} – {pred_min + 6:.0f} mins")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.code(traceback.format_exc())

elif page == "Batch CSV Prediction":
    st.title("Batch Order ETA Prediction")
    uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        try:
            X_batch = prepare_features(raw_df, model)
            raw_df["Predicted_ETA_Min"] = np.round(model.predict(X_batch), 1)
            st.dataframe(raw_df.head(50), use_container_width=True)

            st.download_button(
                "Download Predictions CSV",
                data=raw_df.to_csv(index=False).encode("utf-8"),
                file_name="delivery_eta_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.code(traceback.format_exc())