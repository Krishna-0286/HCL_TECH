import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns a user must supply for a dispatch-time prediction
INPUT_COLUMNS = [
    "Order_Date", "Order_Hour", "Day_of_Week", "Is_Weekend", "Is_Festival", "Weather",
    "Pickup_Zone", "Dropoff_Zone", "Vehicle_Type", "Rider_Experience_Years", "Rider_Rating",
    "Restaurant_Rating", "Cuisine_Type", "Order_Items", "Restaurant_Load",
    "Preparation_Time_Min", "Road_Distance_km", "Traffic_Level", "Number_of_Signals",
    "Delivery_Priority",
]
NOMINAL_COLS = ["Day_of_Week", "Weather", "Pickup_Zone", "Dropoff_Zone", "Vehicle_Type", "Cuisine_Type"]
CATEGORICAL_INPUTS = NOMINAL_COLS + ["Restaurant_Load", "Traffic_Level", "Delivery_Priority"]
TRAFFIC_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
LOAD_ORDER = {"Low": 0, "Medium": 1, "High": 2}
PRIORITY_ORDER = {"Normal": 0, "Priority": 1, "VIP": 2}
GROUP_COLS = ["Vehicle_Type", "Traffic_Level", "Weather"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    # Builds model-ready columns from raw order columns. fit() learns typical speeds per
    # (vehicle, traffic, weather) group from TRAINING data only. transform() never needs the
    # individual trip's speed unless use_actual_speed=True (Scenario B, diagnostic only).

    def __init__(self, use_prep_time=True, use_actual_speed=False):
        self.use_prep_time = use_prep_time
        self.use_actual_speed = use_actual_speed

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        needed = GROUP_COLS + ["Average_Speed_kmph"]
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise ValueError(f"Training data must contain {missing} to learn typical speeds.")
        tmp = X[GROUP_COLS].copy()
        tmp["speed"] = pd.to_numeric(X["Average_Speed_kmph"], errors="coerce")
        self.global_speed_ = float(tmp["speed"].median())
        self.group_speed_ = tmp.groupby(GROUP_COLS)["speed"].median().rename("group_speed").reset_index()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        num = lambda c: pd.to_numeric(X[c], errors="coerce")
        out = pd.DataFrame(index=X.index)

        hour = num("Order_Hour")
        out["Order_Hour"] = hour
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_rush_hour"] = (hour.between(8, 10) | hour.between(18, 21)).astype(float)

        date = pd.to_datetime(X["Order_Date"], errors="coerce")
        out["month"] = date.dt.month
        out["day_of_month"] = date.dt.day

        for c in ["Is_Weekend", "Is_Festival", "Rider_Experience_Years", "Rider_Rating",
                  "Restaurant_Rating", "Order_Items", "Road_Distance_km", "Number_of_Signals",
                  "Preparation_Time_Min"]:
            out[c] = num(c)

        out["same_zone"] = (X["Pickup_Zone"] == X["Dropoff_Zone"]).astype(float)
        out["traffic_level_ord"] = X["Traffic_Level"].map(TRAFFIC_ORDER)
        out["restaurant_load_ord"] = X["Restaurant_Load"].map(LOAD_ORDER)
        out["priority_ord"] = X["Delivery_Priority"].map(PRIORITY_ORDER)
        out["prep_per_item"] = out["Preparation_Time_Min"] / out["Order_Items"].replace(0, np.nan)

        grp = X[GROUP_COLS].merge(self.group_speed_, on=GROUP_COLS, how="left")["group_speed"].to_numpy()
        grp = np.where(np.isnan(grp), self.global_speed_, grp)
        out["expected_travel_min"] = out["Road_Distance_km"].to_numpy() / grp * 60

        if self.use_actual_speed:
            out["Average_Speed_kmph"] = num("Average_Speed_kmph")

        for c in NOMINAL_COLS:
            out[c] = X[c].astype("object")
        return out


def numeric_feature_names(use_prep_time=True, use_actual_speed=False):
    cols = ["Order_Hour", "hour_sin", "hour_cos", "is_rush_hour", "Is_Weekend", "Is_Festival",
            "month", "day_of_month", "same_zone", "Rider_Experience_Years", "Rider_Rating",
            "Restaurant_Rating", "Order_Items", "Road_Distance_km", "Number_of_Signals",
            "traffic_level_ord", "restaurant_load_ord", "priority_ord", "expected_travel_min"]
    if use_prep_time:
        cols += ["Preparation_Time_Min", "prep_per_item"]
    if use_actual_speed:
        cols += ["Average_Speed_kmph"]
    return cols


def build_pipeline(model, scale=False, use_prep_time=True, use_actual_speed=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    preprocess = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric_feature_names(use_prep_time, use_actual_speed)),
         ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
          NOMINAL_COLS)],
        remainder="drop",
    )
    return Pipeline([
        ("features", FeatureBuilder(use_prep_time=use_prep_time, use_actual_speed=use_actual_speed)),
        ("preprocess", preprocess),
        ("model", model),
    ])
