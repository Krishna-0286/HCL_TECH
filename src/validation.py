import numpy as np
import pandas as pd

# Hard rules: a value outside these limits is impossible, so the row is REJECTED.
HARD_LIMITS = {
    "Order_Hour": (0, 23), "Is_Weekend": (0, 1), "Is_Festival": (0, 1),
    "Rider_Experience_Years": (0, 60), "Rider_Rating": (1, 5), "Restaurant_Rating": (1, 5),
    "Order_Items": (1, None), "Preparation_Time_Min": (0, None),
    "Road_Distance_km": (0.001, None), "Number_of_Signals": (0, None),
}


def validate_orders(df, meta):
    # Returns a DataFrame with 'status' (Valid / Low confidence / Rejected) and 'reasons' per row.
    # Raises ValueError if required columns are missing (the whole file is unusable).
    df = df.reset_index(drop=True)
    missing = [c for c in meta["required_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n = len(df)
    reject = [[] for _ in range(n)]
    low = [[] for _ in range(n)]

    def add(store, mask, msg):
        for i in np.where(np.asarray(mask))[0]:
            store[i].append(msg)

    # Categorical columns must contain a category seen in training
    for col, allowed in meta["categories"].items():
        add(reject, ~df[col].isin(allowed), f"{col}: missing or unknown value")

    # Numeric columns must be numbers, inside hard limits; outside training range = low confidence
    for col, (lo_hard, hi_hard) in HARD_LIMITS.items():
        num = pd.to_numeric(df[col], errors="coerce")
        add(reject, num.isna(), f"{col}: missing or not a number")
        if lo_hard is not None:
            add(reject, num.notna() & (num < lo_hard), f"{col}: below allowed minimum {lo_hard}")
        if hi_hard is not None:
            add(reject, num.notna() & (num > hi_hard), f"{col}: above allowed maximum {hi_hard}")
        if col in meta["numeric_ranges"]:
            lo, hi = meta["numeric_ranges"][col]
            add(low, num.notna() & ((num < lo) | (num > hi)), f"{col}: outside training range [{lo}, {hi}]")

    # Date checks
    date = pd.to_datetime(df["Order_Date"], errors="coerce")
    add(reject, date.isna(), "Order_Date: missing or not a valid date")
    add(low, date.notna() & (date.dt.day_name() != df["Day_of_Week"]), "Day_of_Week does not match Order_Date")
    weekend = pd.to_numeric(df["Is_Weekend"], errors="coerce")
    add(low, df["Day_of_Week"].isin(["Saturday", "Sunday"]) != (weekend == 1), "Is_Weekend does not match Day_of_Week")

    status = ["Rejected" if reject[i] else ("Low confidence" if low[i] else "Valid") for i in range(n)]
    reasons = ["; ".join(reject[i] + low[i]) for i in range(n)]
    return pd.DataFrame({"status": status, "reasons": reasons})
