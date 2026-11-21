import numpy as np
import pandas as pd
from .validation import validate_orders


def predict_orders(df, bundle):
    # Validate, then predict only for non-rejected rows. Never labels a prediction right or wrong:
    # that is only possible when a true value exists (see compare_with_actual).
    meta = bundle["meta"]
    df = df.reset_index(drop=True)
    checks = validate_orders(df, meta)
    result = pd.concat([df, checks], axis=1)
    for c in ["predicted_min", "range_low_min", "range_high_min"]:
        result[c] = np.nan

    ok = (checks["status"] != "Rejected").to_numpy()
    if ok.any():
        X = df.loc[ok, meta["required_columns"]]
        pred = bundle["model"].predict(X)
        lo = np.minimum(bundle["q_low"].predict(X), pred)
        hi = np.maximum(bundle["q_high"].predict(X), pred)
        t_lo, t_hi = meta["target_range"]
        result.loc[ok, "predicted_min"] = np.clip(pred, t_lo, t_hi).round(1)
        result.loc[ok, "range_low_min"] = np.clip(lo, t_lo, t_hi).round(1)
        result.loc[ok, "range_high_min"] = np.clip(hi, t_lo, t_hi).round(1)

        wide = ok & ((result["range_high_min"] - result["range_low_min"]) > meta["wide_interval_threshold"]).to_numpy()
        for i in np.where(wide & (result["status"] == "Valid").to_numpy())[0]:
            result.loc[i, "status"] = "Low confidence"
            result.loc[i, "reasons"] = "prediction range is unusually wide for this kind of order"
    return result


def compare_with_actual(result, tolerance=10.0, target="Time_taken_min"):
    # Only valid when the true value is present in the uploaded data.
    if target not in result.columns:
        raise ValueError(f"'{target}' is not in the data, so correctness cannot be judged.")
    out = result.copy()
    out["abs_error_min"] = (out["predicted_min"] - out[target]).abs().round(1)
    out["result"] = np.where(out["predicted_min"].isna(), "Not predicted",
                             np.where(out["abs_error_min"] <= tolerance, f"Correct (within {tolerance:g} min)",
                                      f"Wrong (off by more than {tolerance:g} min)"))
    return out
