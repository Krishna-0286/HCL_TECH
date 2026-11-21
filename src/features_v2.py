import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FeatureBuilder

# Lean feature set chosen from the Section 17 experiments (validation months only)
V2_NUMERIC = ["is_rush_hour", "Is_Festival", "Rider_Experience_Years", "Order_Items",
              "Road_Distance_km", "Number_of_Signals", "traffic_level_ord", "priority_ord",
              "expected_travel_min", "Preparation_Time_Min", "prep_per_item", "total_expected_min"]
V2_PREP_COLS = ["Preparation_Time_Min", "prep_per_item", "total_expected_min"]
V2_NOMINAL = ["Weather", "Vehicle_Type", "Cuisine_Type", "Pickup_Zone", "Dropoff_Zone"]


class FeatureBuilderV2(FeatureBuilder):
    # Same as FeatureBuilder plus total_expected_min = expected travel time + preparation time.
    def transform(self, X):
        out = super().transform(X)
        out["total_expected_min"] = out["expected_travel_min"] + out["Preparation_Time_Min"]
        return out


class ClippedPipeline(Pipeline):
    # A normal Pipeline whose predictions are clipped to the range of the TRAINING target
    # (read inside fit, so nothing is hard-coded). Parameter names (model__..., preprocess__...)
    # stay the same as in a plain Pipeline, so tuning and cloning work unchanged.
    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        self.low_, self.high_ = float(np.min(y)), float(np.max(y))
        return self

    def predict(self, X, **kwargs):
        return np.clip(super().predict(X, **kwargs), self.low_, self.high_)


def numeric_feature_names_v2(use_prep_time=True, use_actual_speed=False):
    cols = [c for c in V2_NUMERIC if use_prep_time or c not in V2_PREP_COLS]
    return cols + (["Average_Speed_kmph"] if use_actual_speed else [])


def build_pipeline_v2(model, scale=False, use_prep_time=True, use_actual_speed=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    preprocess = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric_feature_names_v2(use_prep_time, use_actual_speed)),
         ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), V2_NOMINAL)],
        remainder="drop")
    return ClippedPipeline([
        ("features", FeatureBuilderV2(use_prep_time=use_prep_time, use_actual_speed=use_actual_speed)),
        ("preprocess", preprocess),
        ("model", model),
    ])
