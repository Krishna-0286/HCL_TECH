import streamlit as st
import pandas as pd
import pickle
import datetime

# 1. Page Configuration (Sets the browser tab title and layout)
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide"
)

# 2. Load the saved model
# @st.cache_resource tells Streamlit to load the model ONCE and keep it in memory
@st.cache_resource
def load_model():
    with open("models/best_delivery_modell.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please run your Jupyter Notebook to save 'models/best_delivery_model.pkl' first.")
    st.stop()

# 3. App Title and Description
st.title("🛵 Food Delivery Time Prediction")
st.markdown("Enter the order details at dispatch time to predict the total delivery duration (`Time_taken_min`).")
st.divider()

# 4. Create Input Fields for the User
# We organize inputs into 3 neat columns so the user doesn't have to scroll forever
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Location")
    order_date = st.date_input("Order Date", datetime.date(2025, 11, 20))
    order_hour = st.slider("Order Hour (0 - 23)", min_value=0, max_value=23, value=18)
    
    # Automatically derive Day_of_Week and Is_Weekend from the selected date
    day_of_week = order_date.strftime("%A")
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    st.caption(f"Detected: **{day_of_week}** ({'Weekend' if is_weekend else 'Weekday'})")
    
    is_festival = st.selectbox("Is it a Festival?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pickup_zone = st.selectbox("Pickup Zone", ["CBD", "Commercial", "Residential", "Industrial"])
    dropoff_zone = st.selectbox("Dropoff Zone", ["CBD", "Commercial", "Residential", "Industrial"])

with col2:
    st.subheader("🍔 Restaurant & Order")
    cuisine_type = st.selectbox("Cuisine Type", ["Pizza", "Desserts", "Biryani", "Burger", "Chinese", "Indian", "Healthy"])
    order_items = st.number_input("Number of Order Items", min_value=1, max_value=20, value=2)
    restaurant_rating = st.slider("Restaurant Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
    restaurant_load = st.selectbox("Restaurant Load", ["Low", "Medium", "High"])
    prep_time = st.number_input("Preparation Time (Min)", min_value=5, max_value=90, value=20)
    delivery_priority = st.selectbox("Delivery Priority", ["Normal", "High", "Express"])

with col3:
    st.subheader("🚦 Route, Weather & Rider")
    road_distance = st.number_input("Road Distance (km)", min_value=0.1, max_value=100.0, value=12.5, step=0.5)
    num_signals = st.number_input("Number of Traffic Signals", min_value=0, max_value=50, value=8)
    traffic_level = st.selectbox("Traffic Level", ["Low", "Moderate", "High", "Jam"])
    weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Stormy", "Fog"])
    vehicle_type = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Bicycle", "Electric Scooter"])
    rider_exp = st.number_input("Rider Experience (Years)", min_value=0.0, max_value=30.0, value=3.5, step=0.5)
    rider_rating = st.slider("Rider Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

st.divider()

# 5. Predict Button and Output
if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
    
    # Create a DataFrame matching the EXACT column names from your dataset
    input_data = pd.DataFrame([{
        "Order_Date": str(order_date),
        "Order_Hour": order_hour,
        "Day_of_Week": day_of_week,
        "Is_Weekend": is_weekend,
        "Is_Festival": is_festival,
        "Weather": weather,
        "Pickup_Zone": pickup_zone,
        "Dropoff_Zone": dropoff_zone,
        "Vehicle_Type": vehicle_type,
        "Rider_Experience_Years": rider_exp,
        "Rider_Rating": rider_rating,
        "Restaurant_Rating": restaurant_rating,
        "Cuisine_Type": cuisine_type,
        "Order_Items": order_items,
        "Restaurant_Load": restaurant_load,
        "Preparation_Time_Min": prep_time,
        "Road_Distance_km": road_distance,
        "Traffic_Level": traffic_level,
        "Number_of_Signals": num_signals,
        "Delivery_Priority": delivery_priority
    }])

    try:
        # Make prediction using the loaded pipeline
        prediction = model.predict(input_data)[0]
        
        # Display the result nicely
        st.success(f"### Estimated Delivery Time: **{round(prediction)} minutes** ({prediction:.1f} min)")
        
        # Show the input summary table for verification
        with st.expander("View Input Data Sent to Model"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Tip: Check that the dropdown options and column names match what your pipeline expects.")