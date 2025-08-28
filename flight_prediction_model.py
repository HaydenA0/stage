# flight_prediction_model.py

# %% Imports
import pandas as pd
import numpy as np
import datetime
import time
import pytz  # Used for timezone-aware datetime objects
import matplotlib.pyplot as plt  # Keep for initial script execution, though plots moved to Plotly in Dash
import seaborn as sns  # Keep for initial script execution
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set plot style for visualizations (these will be overridden by Plotly for Dash)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)  # Default figure size for plots

# %% Configuration & Helper Functions
RANDOM_SEED = 42
# Define a chronological split date for training/testing.
# Dates before this are training, dates from this onward are testing.
CHRONOLOGICAL_SPLIT_DATE = datetime.datetime(2021, 2, 1, tzinfo=pytz.UTC)

# Dictionary to store performance metrics for the linear regression model
results = {}

# Hardcoded airport coordinates for map visualization (Europe centric)
# These are a selection of European airports likely to be in the dataset
# For a real application, you'd fetch this from a more comprehensive database.
AIRPORT_COORDINATES = {
    "EDDF": {"lat": 50.033333, "lon": 8.570556},  # Frankfurt, Germany
    "EHAM": {"lat": 52.308611, "lon": 4.763889},  # Amsterdam Schiphol, Netherlands
    "EGLL": {"lat": 51.470025, "lon": -0.454295},  # London Heathrow, UK
    "LFPG": {"lat": 49.009663, "lon": 2.547925},  # Paris Charles de Gaulle, France
    "LSZH": {"lat": 47.464722, "lon": 8.549167},  # Zurich, Switzerland
    "ESSA": {"lat": 59.649998, "lon": 17.923056},  # Stockholm Arlanda, Sweden
    "EBAW": {"lat": 51.189444, "lon": 4.460278},  # Antwerp, Belgium
    "EFHK": {"lat": 60.317222, "lon": 24.963333},  # Helsinki, Finland
    "EIDW": {"lat": 53.426444, "lon": -6.249911},  # Dublin, Ireland
    "LEMD": {"lat": 40.471944, "lon": -3.562639},  # Madrid-Barajas, Spain
    "LIRF": {"lat": 41.800278, "lon": 12.238889},  # Rome Fiumicino, Italy
    "LOWW": {"lat": 48.110278, "lon": 16.569722},  # Vienna, Austria
    "LPPT": {"lat": 38.775556, "lon": -9.135833},  # Lisbon, Portugal
    "EBBR": {"lat": 50.901389, "lon": 4.484444},  # Brussels, Belgium
    "EDDH": {"lat": 53.630278, "lon": 9.988056},  # Hamburg, Germany
    "LEBL": {"lat": 41.2975, "lon": 2.078333},  # Barcelona, Spain
    "LTCK": {
        "lat": 40.9079,
        "lon": 29.3444,
    },  # Istanbul Sabiha Gökçen, Turkey (consider if it's in dataset)
    # Add more airports if they are present in your `APT_ICAO` column and you want them on the map.
}


def calculate_metrics(
    y_true_transformed, y_pred_transformed, y_true_original, y_pred_original_safe
):
    """
    Calculates R2, MAE, and RMSE on both log-transformed and original scales.
    Ensures predicted original values are non-negative.
    """
    # Metrics on log-transformed scale
    r2_log = r2_score(y_true_transformed, y_pred_transformed)
    mae_log = mean_absolute_error(y_true_transformed, y_pred_transformed)
    rmse_log = np.sqrt(mean_squared_error(y_true_transformed, y_pred_transformed))

    # Metrics on original scale
    r2_orig = r2_score(y_true_original, y_pred_original_safe)
    mae_orig = mean_absolute_error(y_true_original, y_pred_original_safe)
    rmse_orig = np.sqrt(mean_squared_error(y_true_original, y_pred_original_safe))

    return {
        "metrics_log_scale": {"R2": r2_log, "MAE": mae_log, "RMSE": rmse_log},
        "metrics_original_scale": {"R2": r2_orig, "MAE": mae_orig, "RMSE": rmse_orig},
    }


def record_final_model_performance(
    model,
    model_name,
    X_train_full,
    y_train_transformed_full,
    y_train_original_full,
    X_test_full,
    y_test_transformed_full,
    y_test_original_full,
    model_params=None,
):
    """
    Trains the model, makes predictions, and records performance metrics
    on both training and test sets.
    """
    print(f"\n--- Training {model_name} ---")
    start_train_time = time.time()
    model.fit(X_train_full, y_train_transformed_full)
    end_train_time = time.time()
    train_time_sec = end_train_time - start_train_time

    # Predict on train set
    start_predict_train_time = time.time()
    y_pred_train_transformed = model.predict(X_train_full)
    end_predict_train_time = time.time()
    predict_train_time_sec = end_predict_train_time - start_predict_train_time
    y_pred_train_original_safe = np.expm1(y_pred_train_transformed)
    y_pred_train_original_safe[y_pred_train_original_safe < 0] = (
        0  # Ensure non-negative predictions
    )
    train_metrics = calculate_metrics(
        y_train_transformed_full,
        y_pred_train_transformed,
        y_train_original_full,
        y_pred_train_original_safe,
    )

    # Predict on test set
    start_predict_test_time = time.time()
    y_pred_test_transformed = model.predict(X_test_full)
    end_predict_test_time = time.time()
    predict_test_time_sec = end_predict_test_time - start_predict_test_time
    y_pred_test_original_safe = np.expm1(y_pred_test_transformed)
    y_pred_test_original_safe[y_pred_test_original_safe < 0] = (
        0  # Ensure non-negative predictions
    )
    test_metrics = calculate_metrics(
        y_test_transformed_full,
        y_pred_test_transformed,
        y_test_original_full,
        y_pred_test_original_safe,
    )

    global results  # Access the global results dictionary
    results[model_name] = {
        "model_params": model_params if model_params is not None else {},
        "train_time_sec": train_time_sec,
        "predict_time_sec_train": predict_train_time_sec,
        "predict_time_sec_test": predict_test_time_sec,
        **train_metrics,  # Unpack train metrics
        "metrics_log_scale_test": test_metrics["metrics_log_scale"],
        "metrics_original_scale_test": test_metrics["metrics_original_scale"],
    }
    print(f"{model_name} Training Complete.")


# %% Data Loading & Comprehensive Feature Engineering
data_raw = pd.read_csv("flights.csv")

# Drop columns that are highly correlated, redundant, or have many missing values
columns_to_drop_initial = [
    "FLT_DEP_IFR_2",
    "FLT_ARR_IFR_2",
    "FLT_TOT_IFR_2",  # IFR_2 series has many NaNs and is not the target
    "MONTH_MON",  # Redundant with MONTH_NUM
    "Pivot Label",  # Redundant information
    "APT_NAME",  # Redundant with APT_ICAO
]
data_cleaned = data_raw.drop(columns=columns_to_drop_initial)

# Convert FLT_DATE to datetime objects, ensuring timezone-awareness for comparison with CHRONOLOGICAL_SPLIT_DATE
data_cleaned["FLT_DATE"] = pd.to_datetime(data_cleaned["FLT_DATE"])
# Robust timezone handling: If FLT_DATE is naive, localize it to UTC; if already aware, convert to UTC.
if data_cleaned["FLT_DATE"].dt.tz is None:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_localize(pytz.UTC)
else:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_convert(pytz.UTC)


# Feature Engineering from FLT_DATE
data_cleaned.loc[:, "day_of_week"] = data_cleaned[
    "FLT_DATE"
].dt.dayofweek  # Monday=0, Sunday=6
data_cleaned.loc[:, "day_of_month"] = data_cleaned["FLT_DATE"].dt.day
data_cleaned.loc[:, "day_of_year"] = data_cleaned["FLT_DATE"].dt.dayofyear
data_cleaned.loc[:, "week_of_year"] = (
    data_cleaned["FLT_DATE"].dt.isocalendar().week
)  # Returns UInt32Dtype
data_cleaned.loc[:, "quarter"] = data_cleaned["FLT_DATE"].dt.quarter
data_cleaned.loc[:, "is_weekend"] = (data_cleaned["FLT_DATE"].dt.dayofweek >= 5).astype(
    int
)


# Create a 'pandemic_phase' feature based on specific date ranges - EXTENDED FOR FUTURE DATES
def assign_pandemic_phase(row):
    year = row["YEAR"]
    month = row["MONTH_NUM"]
    if year < 2020:
        return "Pre-Pandemic (2016-2019)"
    elif year == 2020:
        if month >= 3 and month <= 5:  # March-May 2020: Initial severe drop
            return "Pandemic Peak Drop (Spring 2020)"
        elif month >= 6 and month <= 8:  # June-Aug 2020: Slight summer recovery
            return "Pandemic Initial Recovery (Summer 2020)"
        else:  # Sep-Dec 2020: Second wave / Fall-Winter stagnation
            return "Pandemic Second Wave (Fall/Winter 2020)"
    elif year == 2021:
        if month >= 1 and month <= 4:  # Jan-April 2021: Continued stagnation
            return "Pandemic Continued Stagnation (Early 2021)"
        elif month >= 5 and month <= 8:  # May-Aug 2021: Stronger recovery
            return "Pandemic Strong Recovery (Summer 2021)"
        else:  # Sep-Dec 2021: Post-summer recovery continues
            return "Pandemic Post-Summer Recovery (Late 2021)"
    elif year == 2022:  # Up to May 2022 in dataset
        return "Post-Pandemic Recovery (2022)"
    elif year >= 2023:  # NEW: For dates beyond 2022 (e.g., 2023, 2024, 2025+)
        return "Post-Pandemic Normalization (2023+)"
    return "Other"  # Should not happen with this data


data_cleaned.loc[:, "pandemic_phase"] = data_cleaned.apply(
    assign_pandemic_phase, axis=1
)

# Ensure the order of pandemic phases for better visualization and consistent encoding
pandemic_phase_order = [
    "Pre-Pandemic (2016-2019)",
    "Pandemic Peak Drop (Spring 2020)",
    "Pandemic Initial Recovery (Summer 2020)",
    "Pandemic Second Wave (Fall/Winter 2020)",
    "Pandemic Continued Stagnation (Early 2021)",
    "Pandemic Strong Recovery (Summer 2021)",
    "Pandemic Post-Summer Recovery (Late 2021)",
    "Post-Pandemic Recovery (2022)",
    "Post-Pandemic Normalization (2023+)",  # NEW: Added to order
]
data_cleaned.loc[:, "pandemic_phase"] = pd.Categorical(
    data_cleaned["pandemic_phase"], categories=pandemic_phase_order, ordered=True
)

# Calculate average pre-pandemic traffic for each airport and month
pre_pandemic_data = data_cleaned[
    data_cleaned["YEAR"] < 2020
].copy()  # Ensure working on a copy
avg_traffic_pre_pandemic_monthly = (
    pre_pandemic_data.groupby(["APT_ICAO", "MONTH_NUM"])["FLT_TOT_1"]
    .mean()
    .reset_index()
)
avg_traffic_pre_pandemic_monthly.rename(
    columns={"FLT_TOT_1": "avg_pre_pandemic_monthly_traffic"}, inplace=True
)

# Merge this back into the main DataFrame
data_cleaned = pd.merge(
    data_cleaned,
    avg_traffic_pre_pandemic_monthly,
    on=["APT_ICAO", "MONTH_NUM"],
    how="left",
)

# Handle cases where an airport/month might not have pre-pandemic data by filling with overall mean
overall_pre_pandemic_mean = pre_pandemic_data["FLT_TOT_1"].mean()
data_cleaned.loc[:, "avg_pre_pandemic_monthly_traffic"] = data_cleaned[
    "avg_pre_pandemic_monthly_traffic"
].fillna(overall_pre_pandemic_mean)

# Calculate 'airport_volume_category' based on overall pre-pandemic average traffic
avg_traffic_pre_pandemic_overall = (
    pre_pandemic_data.groupby("APT_ICAO")["FLT_TOT_1"].mean().reset_index()
)
avg_traffic_pre_pandemic_overall.rename(
    columns={"FLT_TOT_1": "avg_traffic_pre_pandemic_overall"}, inplace=True
)

data_cleaned = pd.merge(
    data_cleaned, avg_traffic_pre_pandemic_overall, on="APT_ICAO", how="left"
)
data_cleaned.loc[:, "avg_traffic_pre_pandemic_overall"] = data_cleaned[
    "avg_traffic_pre_pandemic_overall"
].fillna(overall_pre_pandemic_mean)

# Define airport volume categories based on quartiles of avg_traffic_pre_pandemic_overall
# Use pd.qcut on the full series to determine consistent bins for both training and new examples
s_avg_traffic_overall_full_non_null = data_cleaned[
    "avg_traffic_pre_pandemic_overall"
].dropna()
_, bins_volume_category = pd.qcut(
    s_avg_traffic_overall_full_non_null,
    q=4,
    labels=False,
    duplicates="drop",
    retbins=True,
)
labels_volume_category = [
    "Small",
    "Medium",
    "Large",
    "Very Large",
]  # Define explicit labels

data_cleaned.loc[:, "airport_volume_category"] = pd.cut(
    data_cleaned["avg_traffic_pre_pandemic_overall"],
    bins=bins_volume_category,
    labels=labels_volume_category,
    include_lowest=True,
    right=True,
).astype("category")

print("Data Cleaning and Feature Engineering Complete.")
print(
    "Columns in data_cleaned after feature engineering:", data_cleaned.columns.tolist()
)

# %% Data Preparation (X, y, Train/Test Split)
# Define numerical and categorical features for the model
numerical_features = [
    "YEAR",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "avg_pre_pandemic_monthly_traffic",
    "avg_traffic_pre_pandemic_overall",
]
categorical_features = [
    "APT_ICAO",
    "STATE_NAME",
    "MONTH_NUM",
    "day_of_week",
    "quarter",
    "pandemic_phase",
    "airport_volume_category",
]

# Combine all desired features for the model
final_model_features = numerical_features + categorical_features

# Target variable (total flights)
y_original = data_cleaned["FLT_TOT_1"]
# Apply log1p transformation to reduce skewness and improve linearity
y_transformed = np.log1p(y_original)

# Chronological split of data into training and testing sets using FLT_DATE
train_df = data_cleaned[data_cleaned["FLT_DATE"] < CHRONOLOGICAL_SPLIT_DATE].copy()
test_df = data_cleaned[data_cleaned["FLT_DATE"] >= CHRONOLOGICAL_SPLIT_DATE].copy()

# Select features and target for training and testing
X_train = train_df[final_model_features].copy()
y_train_original = train_df["FLT_TOT_1"].copy()
y_train_transformed = np.log1p(y_train_original)

X_test = test_df[final_model_features].copy()
y_test_original = test_df["FLT_TOT_1"].copy()
y_test_transformed = np.log1p(y_test_original)

# Ensure categorical features are of 'category' dtype for consistent handling by scikit-learn preprocessors
for col in categorical_features:
    if col in X_train.columns:
        X_train.loc[:, col] = X_train[col].astype("category")
    if col in X_test.columns:
        X_test.loc[:, col] = X_test[col].astype("category")

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(
    f"Train/Test chronological split ratio: {len(X_train) / (len(X_train) + len(X_test)):.2f}"
)

# %% Linear Regression Model Training & Evaluation
# Preprocessing pipeline for Linear Regression:
# - StandardScaler for numerical features (scaling)
# - OneHotEncoder for categorical features (dummy variables)
preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),  # 'ignore' handles unseen categories in test set gracefully
    ],
    remainder="passthrough",  # Keep other columns if any, though not expected here
)

# Create the full Linear Regression pipeline
lr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_lr),
        ("regressor", LinearRegression(n_jobs=-1)),
    ]
)  # n_jobs=-1 uses all available CPU cores

# Train and evaluate the Linear Regression model
record_final_model_performance(
    lr_model,
    "LinearRegression",
    X_train,
    y_train_transformed,
    y_train_original,
    X_test,
    y_test_transformed,
    y_test_original,
)

print("\n--- Model Training & Performance Summary ---")
for model_name, model_data in results.items():
    print(f"\nModel: {model_name}")
    print(f"  Training Time: {model_data['train_time_sec']:.2f} seconds")
    print(
        f"  Prediction Time (Train Set): {model_data['predict_time_sec_train']:.4f} seconds"
    )
    print(
        f"  Prediction Time (Test Set): {model_data['predict_time_sec_test']:.4f} seconds"
    )

    print("  Train Metrics (Log Scale):")
    for metric, value in model_data["metrics_log_scale"].items():
        print(f"    {metric}: {value:.4f}")
    print("  Train Metrics (Original Scale):")
    for metric, value in model_data["metrics_original_scale"].items():
        print(f"    {metric}: {value:.4f}")

    print("  Test Metrics (Log Scale):")
    for metric, value in model_data["metrics_log_scale_test"].items():
        print(f"    {metric}: {value:.4f}")
    print("  Test Metrics (Original Scale):")
    for metric, value in model_data["metrics_original_scale_test"].items():
        print(f"    {metric}: {value:.4f}")

# %% Visualization (Matplotlib/Seaborn calls are commented out as Plotly is used in app.py)
print(
    "\n--- Visualizing Relationships between Features and Log-Transformed Target (from training data) ---"
)

# Add the log-transformed target to the training DataFrame for plotting
train_df.loc[:, "FLT_TOT_1_log"] = np.log1p(train_df["FLT_TOT_1"])

# (Removed plt.show() and plt.savefig() from original script as Plotly will generate these for Dash)


# Calculate and print Pearson Linear Correlation Coefficients with Log-Transformed Target
print("\n--- Pearson Linear Correlation Coefficients with Log-Transformed Target ---")
numerical_data_for_corr = train_df[numerical_features + ["FLT_TOT_1_log"]]
correlation_matrix = numerical_data_for_corr.corr(
    numeric_only=True
)  # Ensure only numeric columns are used for correlation
print(correlation_matrix["FLT_TOT_1_log"].sort_values(ascending=False))


# %% Prediction with Example Inputs
# Function to create sample input DataFrame for prediction
def create_sample_input(
    airport_icao: str,
    date_str: str,
    source_data_cleaned: pd.DataFrame,
    pre_pandemic_stats_monthly: pd.DataFrame,
    pre_pandemic_stats_overall: pd.DataFrame,
    overall_pre_pandemic_mean_val: float,
    volume_bins,
    volume_labels,
):
    """
    Constructs a DataFrame for a single prediction example with all required features,
    mimicking the feature engineering steps.
    """
    # Parse date and ensure it's timezone-aware (UTC)
    input_date = pd.to_datetime(date_str).replace(tzinfo=pytz.UTC)

    # Basic time features
    sample_data = {
        "YEAR": input_date.year,
        "MONTH_NUM": input_date.month,
        "day_of_week": input_date.dayofweek,
        "day_of_month": input_date.day,
        "day_of_year": input_date.dayofyear,
        "week_of_year": input_date.isocalendar().week,
        "quarter": input_date.quarter,
        "is_weekend": int(input_date.dayofweek >= 5),
        "APT_ICAO": airport_icao,
    }

    # Derive STATE_NAME from existing data or set a default
    # Filter source_data_cleaned based on airport_icao and select STATE_NAME, then get first value
    state_rows = source_data_cleaned[source_data_cleaned["APT_ICAO"] == airport_icao][
        "STATE_NAME"
    ]
    sample_data["STATE_NAME"] = (
        state_rows.iloc[0] if not state_rows.empty else "UNKNOWN_STATE"
    )  # Fallback for new airports

    # Pandemic phase for the input date
    temp_series = pd.Series({"YEAR": input_date.year, "MONTH_NUM": input_date.month})
    sample_data["pandemic_phase"] = assign_pandemic_phase(temp_series)

    # Look up avg_pre_pandemic_monthly_traffic
    matching_row_monthly = pre_pandemic_stats_monthly[
        (pre_pandemic_stats_monthly["APT_ICAO"] == airport_icao)
        & (pre_pandemic_stats_monthly["MONTH_NUM"] == input_date.month)
    ]
    sample_data["avg_pre_pandemic_monthly_traffic"] = (
        matching_row_monthly["avg_pre_pandemic_monthly_traffic"].iloc[0]
        if not matching_row_monthly.empty
        else overall_pre_pandemic_mean_val
    )

    # Look up avg_traffic_pre_pandemic_overall
    matching_row_overall = pre_pandemic_stats_overall[
        pre_pandemic_stats_overall["APT_ICAO"] == airport_icao
    ]
    avg_traffic_overall = (
        matching_row_overall["avg_traffic_pre_pandemic_overall"].iloc[0]
        if not matching_row_overall.empty
        else overall_pre_pandemic_mean_val
    )
    sample_data["avg_traffic_pre_pandemic_overall"] = avg_traffic_overall

    # Airport volume category based on the pre-calculated bins
    category = pd.cut(
        [avg_traffic_overall],
        bins=volume_bins,
        labels=volume_labels,
        include_lowest=True,
        right=True,
    )[
        0
    ]  # [0] to get the scalar category value from the CategoricalIndex
    sample_data["airport_volume_category"] = category

    # Create DataFrame in the correct feature order and dtypes
    input_df = pd.DataFrame([sample_data], columns=final_model_features)
    for col in categorical_features:
        if col in input_df.columns:
            # Important: Ensure the category levels match those seen during training.
            # Convert to Categorical with the full set of categories from the training data.
            # handle_unknown='ignore' in OneHotEncoder will set new categories to all zeros.
            input_df.loc[:, col] = input_df[col].astype("category")
            if col == "APT_ICAO" or col == "STATE_NAME":
                # For high cardinality categories, OneHotEncoder handles unseen by 'ignore'
                pass
            elif col == "pandemic_phase":
                input_df.loc[:, col] = pd.Categorical(
                    input_df[col], categories=pandemic_phase_order, ordered=True
                )
            elif col == "airport_volume_category":
                input_df.loc[:, col] = pd.Categorical(
                    input_df[col], categories=labels_volume_category, ordered=True
                )

    return input_df


# --- Pre-calculate / store required components for create_sample_input ---
# These values are derived from the *full* `data_cleaned` before train/test split,
# representing the knowledge available at feature engineering time.
overall_pre_pandemic_mean_for_lookup = (
    overall_pre_pandemic_mean  # From feature engineering
)

# Re-calculate these from the full `data_cleaned` for consistent lookup in prediction function
avg_traffic_pre_pandemic_monthly_for_lookup = (
    data_cleaned.groupby(["APT_ICAO", "MONTH_NUM"])["FLT_TOT_1"].mean().reset_index()
)
avg_traffic_pre_pandemic_monthly_for_lookup.rename(
    columns={"FLT_TOT_1": "avg_pre_pandemic_monthly_traffic"}, inplace=True
)

avg_traffic_pre_pandemic_overall_for_lookup = (
    data_cleaned.groupby("APT_ICAO")["FLT_TOT_1"].mean().reset_index()
)
avg_traffic_pre_pandemic_overall_for_lookup.rename(
    columns={"FLT_TOT_1": "avg_traffic_pre_pandemic_overall"}, inplace=True
)

# `bins_volume_category` and `labels_volume_category` are already defined globally
# from the feature engineering section based on the full dataset.

print("\n--- Model Predictions for Example Inputs ---")

# Example 1: Busy airport, pre-pandemic (Friday)
airport_1 = "EDDF"
date_1 = "2019-03-15"
input_eddf_2019 = create_sample_input(
    airport_1,
    date_1,
    data_cleaned,
    avg_traffic_pre_pandemic_monthly_for_lookup,
    avg_traffic_pre_pandemic_overall_for_lookup,
    overall_pre_pandemic_mean_for_lookup,
    bins_volume_category,
    labels_volume_category,
)
print(f"\nExample 1: {airport_1}, {date_1} (Pre-Pandemic Friday)")
# print("Input Features:") # Uncomment to see the raw input DF
# print(input_eddf_2019)
predicted_log_1 = lr_model.predict(input_eddf_2019)
predicted_flights_1 = np.expm1(predicted_log_1)[0]
print(f"Predicted Total Flights: {predicted_flights_1:.2f}")

# Example 2: Busy airport, pandemic peak (Wednesday)
airport_2 = "EDDF"
date_2 = "2020-04-15"
input_eddf_2020 = create_sample_input(
    airport_2,
    date_2,
    data_cleaned,
    avg_traffic_pre_pandemic_monthly_for_lookup,
    avg_traffic_pre_pandemic_overall_for_lookup,
    overall_pre_pandemic_mean_for_lookup,
    bins_volume_category,
    labels_volume_category,
)
print(f"\nExample 2: {airport_2}, {date_2} (Pandemic Peak Wednesday)")
# print("Input Features:")
# print(input_eddf_2020)
predicted_log_2 = lr_model.predict(input_eddf_2020)
predicted_flights_2 = np.expm1(predicted_log_2)[0]
print(f"Predicted Total Flights: {predicted_flights_2:.2f}")

# Example 3: Less busy airport, post-pandemic recovery weekend (Saturday)
airport_3 = "EBAW"
date_3 = "2021-07-10"
input_ebaw_2021 = create_sample_input(
    airport_3,
    date_3,
    data_cleaned,
    avg_traffic_pre_pandemic_monthly_for_lookup,
    avg_traffic_pre_pandemic_overall_for_lookup,
    overall_pre_pandemic_mean_for_lookup,
    bins_volume_category,
    labels_volume_category,
)
print(f"\nExample 3: {airport_3}, {date_3} (Post-Pandemic Recovery Saturday)")
# print("Input Features:")
# print(input_ebaw_2021)
predicted_log_3 = lr_model.predict(input_ebaw_2021)
predicted_flights_3 = np.expm1(predicted_log_3)[0]
print(f"Predicted Total Flights: {predicted_flights_3:.2f}")
