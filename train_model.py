# train_model.py
import pandas as pd
import numpy as np
import datetime
import pytz
import joblib
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import NotFittedError

print("--- Starting Advanced Model Training Script ---")

# --- Configuration ---
RANDOM_SEED = 42
CHRONOLOGICAL_SPLIT_DATE = datetime.datetime(2021, 2, 1, tzinfo=pytz.UTC)
MODEL_SAVE_PATH = "models/flight_prediction_models.joblib"


# --- Helper Functions ---
def assign_pandemic_phase(row):
    year, month = row["YEAR"], row["MONTH_NUM"]
    if year < 2020:
        return "Pre-Pandemic (2016-2019)"
    if year == 2020:
        if 3 <= month <= 5:
            return "Pandemic Peak Drop (Spring 2020)"
        if 6 <= month <= 8:
            return "Pandemic Initial Recovery (Summer 2020)"
        return "Pandemic Second Wave (Fall/Winter 2020)"
    if year == 2021:
        if 1 <= month <= 4:
            return "Pandemic Continued Stagnation (Early 2021)"
        if 5 <= month <= 8:
            return "Pandemic Strong Recovery (Summer 2021)"
        return "Pandemic Post-Summer Recovery (Late 2021)"
    if year == 2022:
        return "Post-Pandemic Recovery (2022)"
    if year >= 2023:
        return "Post-Pandemic Normalization (2023+)"
    return "Other"


def calculate_metrics(
    y_true_transformed, y_pred_transformed, y_true_original, y_pred_original
):
    y_pred_original_safe = np.maximum(0, y_pred_original)
    return {
        "log_scale": {
            "R2": r2_score(y_true_transformed, y_pred_transformed),
            "MAE": mean_absolute_error(y_true_transformed, y_pred_transformed),
            "RMSE": np.sqrt(mean_squared_error(y_true_transformed, y_pred_transformed)),
        },
        "original_scale": {
            "R2": r2_score(y_true_original, y_pred_original_safe),
            "MAE": mean_absolute_error(y_true_original, y_pred_original_safe),
            "RMSE": np.sqrt(mean_squared_error(y_true_original, y_pred_original_safe)),
        },
    }


def train_and_evaluate_model(
    model,
    model_name,
    X_train,
    y_train_transformed,
    y_train_original,
    X_test,
    y_test_transformed,
    y_test_original,
):
    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train_transformed)
    y_pred_train_transformed = model.predict(X_train)
    y_pred_train_original = np.expm1(y_pred_train_transformed)
    y_pred_test_transformed = model.predict(X_test)
    y_pred_test_original = np.expm1(y_pred_test_transformed)
    train_metrics = calculate_metrics(
        y_train_transformed,
        y_pred_train_transformed,
        y_train_original,
        y_pred_train_original,
    )
    test_metrics = calculate_metrics(
        y_test_transformed,
        y_pred_test_transformed,
        y_test_original,
        y_pred_test_original,
    )
    print(f"Evaluation for {model_name} complete.")
    return {"model": model, "metrics": {"train": train_metrics, "test": test_metrics}}


# --- 1. Data Loading & Feature Engineering ---
print("Step 1: Loading and processing data with advanced features...")
data_raw = pd.read_csv("flights.csv")
columns_to_drop_initial = [
    "FLT_DEP_IFR_2",
    "FLT_ARR_IFR_2",
    "FLT_TOT_IFR_2",
    "MONTH_MON",
    "Pivot Label",
    "APT_NAME",
]
data_cleaned = data_raw.drop(columns=columns_to_drop_initial)
data_cleaned["FLT_DATE"] = pd.to_datetime(data_cleaned["FLT_DATE"])
if data_cleaned["FLT_DATE"].dt.tz is None:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_localize(pytz.UTC)
else:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_convert(pytz.UTC)
data_cleaned["day_of_week"] = data_cleaned["FLT_DATE"].dt.dayofweek
data_cleaned["day_of_month"] = data_cleaned["FLT_DATE"].dt.day
data_cleaned["day_of_year"] = data_cleaned["FLT_DATE"].dt.dayofyear
data_cleaned["week_of_year"] = (
    data_cleaned["FLT_DATE"].dt.isocalendar().week.astype("int64")
)
data_cleaned["quarter"] = data_cleaned["FLT_DATE"].dt.quarter
data_cleaned["is_weekend"] = (data_cleaned["FLT_DATE"].dt.dayofweek >= 5).astype(int)
data_cleaned["day_of_year_sin"] = np.sin(
    2 * np.pi * data_cleaned["day_of_year"] / 365.25
)
data_cleaned["day_of_year_cos"] = np.cos(
    2 * np.pi * data_cleaned["day_of_year"] / 365.25
)
data_cleaned["month_sin"] = np.sin(2 * np.pi * data_cleaned["MONTH_NUM"] / 12)
data_cleaned["month_cos"] = np.cos(2 * np.pi * data_cleaned["MONTH_NUM"] / 12)
data_cleaned["pandemic_phase"] = data_cleaned.apply(assign_pandemic_phase, axis=1)
pandemic_phase_order = [
    "Pre-Pandemic (2016-2019)",
    "Pandemic Peak Drop (Spring 2020)",
    "Pandemic Initial Recovery (Summer 2020)",
    "Pandemic Second Wave (Fall/Winter 2020)",
    "Pandemic Continued Stagnation (Early 2021)",
    "Pandemic Strong Recovery (Summer 2021)",
    "Pandemic Post-Summer Recovery (Late 2021)",
    "Post-Pandemic Recovery (2022)",
    "Post-Pandemic Normalization (2023+)",
]
data_cleaned["pandemic_phase"] = pd.Categorical(
    data_cleaned["pandemic_phase"], categories=pandemic_phase_order, ordered=True
)
pre_pandemic_data = data_cleaned[data_cleaned["YEAR"] < 2020].copy()
avg_traffic_pre_pandemic_monthly = (
    pre_pandemic_data.groupby(["APT_ICAO", "MONTH_NUM"])["FLT_TOT_1"]
    .mean()
    .reset_index()
    .rename(columns={"FLT_TOT_1": "avg_pre_pandemic_monthly_traffic"})
)
data_cleaned = pd.merge(
    data_cleaned,
    avg_traffic_pre_pandemic_monthly,
    on=["APT_ICAO", "MONTH_NUM"],
    how="left",
)
overall_pre_pandemic_mean = pre_pandemic_data["FLT_TOT_1"].mean()
data_cleaned["avg_pre_pandemic_monthly_traffic"] = data_cleaned[
    "avg_pre_pandemic_monthly_traffic"
].fillna(overall_pre_pandemic_mean)
avg_traffic_pre_pandemic_overall = (
    pre_pandemic_data.groupby("APT_ICAO")["FLT_TOT_1"]
    .mean()
    .reset_index()
    .rename(columns={"FLT_TOT_1": "avg_traffic_pre_pandemic_overall"})
)
data_cleaned = pd.merge(
    data_cleaned, avg_traffic_pre_pandemic_overall, on="APT_ICAO", how="left"
)
data_cleaned["avg_traffic_pre_pandemic_overall"] = data_cleaned[
    "avg_traffic_pre_pandemic_overall"
].fillna(overall_pre_pandemic_mean)
_, bins_volume_category = pd.qcut(
    data_cleaned["avg_traffic_pre_pandemic_overall"].dropna(),
    q=4,
    labels=False,
    duplicates="drop",
    retbins=True,
)
labels_volume_category = ["Small", "Medium", "Large", "Very Large"]
data_cleaned["airport_volume_category"] = pd.cut(
    data_cleaned["avg_traffic_pre_pandemic_overall"],
    bins=bins_volume_category,
    labels=labels_volume_category,
    include_lowest=True,
    right=True,
).astype("category")
print("Data processing complete.")

# --- 2. Data Preparation for Models ---
print("Step 2: Preparing data for training...")
numerical_features = [
    "YEAR",
    "day_of_month",
    "is_weekend",
    "avg_pre_pandemic_monthly_traffic",
    "avg_traffic_pre_pandemic_overall",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]
categorical_features = [
    "APT_ICAO",
    "STATE_NAME",
    "day_of_week",
    "quarter",
    "pandemic_phase",
    "airport_volume_category",
]
features = numerical_features + categorical_features
train_df = data_cleaned[data_cleaned["FLT_DATE"] < CHRONOLOGICAL_SPLIT_DATE].copy()
test_df = data_cleaned[data_cleaned["FLT_DATE"] >= CHRONOLOGICAL_SPLIT_DATE].copy()
X_train, y_train_original = train_df[features], train_df["FLT_TOT_1"]
X_test, y_test_original = test_df[features], test_df["FLT_TOT_1"]
y_train_transformed, y_test_transformed = np.log1p(y_train_original), np.log1p(
    y_test_original
)
for col in categorical_features:
    X_train.loc[:, col] = X_train[col].astype("category")
    X_test.loc[:, col] = X_test[col].astype("category")
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# --- 3. Model Training & Evaluation ---
print("Step 3: Defining and training models...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ],
    remainder="passthrough",
)

# --- Model 1: Regularized Linear Model (Ridge) ---
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ]
)
ridge_results = train_and_evaluate_model(
    ridge_pipeline,
    "Ridge Regression",
    X_train,
    y_train_transformed,
    y_train_original,
    X_test,
    y_test_transformed,
    y_test_original,
)

# --- Model 2: Tuned & Regularized LightGBM ---
print("\n--- Training and Evaluating Tuned LightGBM with Early Stopping ---")
lgbm_tuned_params = {
    "objective": "regression_l1",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 20,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
    "boosting_type": "gbdt",
    "max_depth": 10,
}
lgbm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", lgb.LGBMRegressor(**lgbm_tuned_params)),
    ]
)

# ==============================================================================
# CORRECTED WORKFLOW FOR EARLY STOPPING WITH A PIPELINE
# ==============================================================================
# 1. Fit the entire pipeline on the training data. This fits the preprocessor.
lgbm_pipeline.fit(X_train, y_train_transformed)

# 2. Now that the preprocessor is fitted, use it to transform the validation set.
X_test_transformed = lgbm_pipeline.named_steps["preprocessor"].transform(X_test)

# 3. Re-fit the regressor part of the pipeline, but this time with the early stopping callback.
#    This will overwrite the regressor that was trained in step 1 with a better one.
lgbm_pipeline.named_steps["regressor"].fit(
    lgbm_pipeline.named_steps["preprocessor"].transform(
        X_train
    ),  # Use transformed train data
    y_train_transformed,
    eval_set=[(X_test_transformed, y_test_transformed)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(100, verbose=True)
    ],  # Set verbose=True to see it working
)

# 4. Now the pipeline is fully fitted and contains the early-stopped model. Proceed with evaluation.
y_pred_train_transformed = lgbm_pipeline.predict(X_train)
y_pred_train_original = np.expm1(y_pred_train_transformed)
y_pred_test_transformed = lgbm_pipeline.predict(X_test)
y_pred_test_original = np.expm1(y_pred_test_transformed)
train_metrics = calculate_metrics(
    y_train_transformed,
    y_pred_train_transformed,
    y_train_original,
    y_pred_train_original,
)
test_metrics = calculate_metrics(
    y_test_transformed, y_pred_test_transformed, y_test_original, y_pred_test_original
)
# ==============================================================================

lgbm_tuned_results = {
    "model": lgbm_pipeline,
    "metrics": {"train": train_metrics, "test": test_metrics},
}
print("Evaluation for Tuned LightGBM complete.")

# --- 4. Saving Models and Supporting Objects ---
print("Step 4: Saving models and artifacts...")
artifacts_to_save = {
    "data": {
        "processed_data": data_cleaned,
        "features": features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "pandemic_phase_order": pandemic_phase_order,
        "airport_volume_bins": bins_volume_category,
        "airport_volume_labels": labels_volume_category,
        "overall_pre_pandemic_mean": overall_pre_pandemic_mean,
        "pre_pandemic_monthly_stats": avg_traffic_pre_pandemic_monthly,
        "pre_pandemic_overall_stats": avg_traffic_pre_pandemic_overall,
    },
    "models": {"Ridge Regression": ridge_results, "Tuned LightGBM": lgbm_tuned_results},
    "helpers": {"assign_pandemic_phase": assign_pandemic_phase},
}

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(artifacts_to_save, MODEL_SAVE_PATH)

print(f"--- All artifacts saved to {MODEL_SAVE_PATH} ---")
print("--- Model Training Script Finished Successfully ---")
