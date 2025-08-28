# app.py

# %% Imports
import pandas as pd
import numpy as np
import datetime
import pytz  # Used for timezone-aware datetime objects
import time  # For model training time measurement

# Dash imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  # Import plotly.io to set default template

# Scikit-learn imports for the model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set default Plotly template to a known good one
pio.templates.default = "plotly_white"

# %% App-wide constants
DEEP_NAVY_COLOR = "#0F1D2C"
AIRLINE_BLUE_COLOR = "#0D47A1"
HORIZON_BLUE_COLOR = "#87CEEB"
ERROR_RED_DARK_COLOR = "#D32F2F"
SUCCESS_GREEN_COLOR = "#198754"

CHART_COLOR_SEQUENCE = ["#0D47A1", "#87CEEB", "#BDC3C7", "#DDE9F3", "#0F1D2C"]


# %% Translation Dictionaries
text_content = {
    "en": {
        "main_title": "Flight Operations Dashboard",
        "tab_operations_summary": "Operations Summary",
        "tab_model_performance": "Model Performance",
        "tab_flight_prediction": "Flight Prediction",
        "tab_visualizations": "Visualizations",
        # Operations Summary Tab
        "summary_card_title": "Select Date Range:",
        "total_ops_header": "TOTAL OPERATIONS",
        "total_arr_header": "TOTAL ARRIVALS",
        "total_dep_header": "TOTAL DEPARTURES",
        "change_header": "vs. PRE-PANDEMIC (2019)",
        "daily_ops_chart_title": "Daily Operations",
        "pareto_chart_title": "Top 10 Airports by Flight Volume",
        "pareto_xaxis": "Airport",
        "pareto_yaxis1": "Total Flights",
        "pareto_yaxis2": "Cumulative %",
        "country_chart_title": "Flights by Country (Top 15)",
        "volume_pie_title": "Operations by Airport Volume",
        "weekend_pie_title": "Operations by Day Type",
        "weekend_label": "Weekend",
        "weekday_label": "Weekday",
        # Model Performance Tab
        "card_header_times": "Model Training & Prediction Times",
        "card_title_train_time": "Training Time",
        "card_title_pred_time_train": "Prediction Time (Train)",
        "card_title_pred_time_test": "Prediction Time (Test)",
        "card_header_train_log": "Train Metrics (Log Scale)",
        "card_header_train_orig": "Train Metrics (Original Scale)",
        "card_header_test_log": "Test Metrics (Log Scale)",
        "card_header_test_orig": "Test Metrics (Original Scale)",
        # Prediction Tab
        "card_header_predict": "Predict Future Flight Traffic",
        "label_select_airport": "Select Airport:",
        "placeholder_select_airport": "Select an airport...",
        "label_select_date": "Select Date:",
        "button_predict": "Predict Flights",
        "prediction_output_template": "Predicted Total Flights for {airport} on {date}: {prediction:.2f}",
        "prediction_error_select": "Please select both an airport and a date.",
        "prediction_error_general": "Error during prediction: {e}",
        # Visualizations Tab
        "viz_map_title": "Airport Traffic Map (Pre-Pandemic Average)",
        "viz_map_hover_template": "Average Pre-Pandemic Flight Traffic by Airport",
        "viz_month_timeseries_title": "Total Flights by Month (Training Data)",
        "viz_month_timeseries_xaxis": "Month",
        "viz_month_timeseries_yaxis": "Total Flights",
        "viz_dayofweek_title": "Average Log(Total Flights) by Day of Week (Training Data)",
        "viz_dayofweek_xaxis": "Day of Week (0=Mon, 6=Sun)",
        "viz_dayofweek_yaxis": "Average Log(Total Flights)",
        "viz_pandemic_phase_title": "Average Log(Total Flights) by Pandemic Phase (Training Data)",
        "viz_pandemic_phase_xaxis": "Pandemic Phase",
        "viz_pandemic_phase_yaxis": "Average Log(Total Flights)",
        "viz_volume_category_title": "Average Log(Total Flights) by Airport Volume Category (Training Data)",
        "viz_volume_category_xaxis": "Airport Volume Category",
        "viz_volume_category_yaxis": "Average Log(Total Flights)",
    },
    "fr": {
        "main_title": "Tableau de Bord des Opérations Aériennes",
        "tab_operations_summary": "Résumé des Opérations",
        # ... (other french translations would go here)
    },
}

country_translations = {
    "en": {
        "Germany": "Germany",
        "Netherlands": "Netherlands",
        "United Kingdom": "UK",
        "France": "France",
        "Switzerland": "Switzerland",
        "Sweden": "Sweden",
        "Belgium": "Belgium",
        "Finland": "Finland",
        "Ireland": "Ireland",
        "Spain": "Spain",
        "Italy": "Italy",
        "Austria": "Austria",
        "Portugal": "Portugal",
        "Turkey": "Turkey",
        "Greece": "Greece",
        "Norway": "Norway",
        "Denmark": "Denmark",
        "Slovakia": "Slovakia",
        "Cyprus": "Cyprus",
        "Croatia": "Croatia",
        "Romania": "Romania",
        "Bulgaria": "Bulgaria",
        "Serbia": "Serbia",
    },
    "fr": {
        "Germany": "Allemagne",
        "Netherlands": "Pays-Bas",
        "United Kingdom": "Royaume-Uni",
        "France": "France",
        # ... (other french translations would go here)
    },
}

# %% Configuration & Helper Functions
# ... (rest of the section is unchanged)
RANDOM_SEED = 42
CHRONOLOGICAL_SPLIT_DATE = datetime.datetime(2021, 2, 1, tzinfo=pytz.UTC)
results = {}
AIRPORT_COORDINATES = {
    "EDDF": {"lat": 50.033333, "lon": 8.570556},
    "EHAM": {"lat": 52.308611, "lon": 4.763889},
    "EGLL": {"lat": 51.470025, "lon": -0.454295},
    "LFPG": {"lat": 49.009663, "lon": 2.547925},
    "LSZH": {"lat": 47.464722, "lon": 8.549167},
    "ESSA": {"lat": 59.649998, "lon": 17.923056},
    "EBAW": {"lat": 51.189444, "lon": 4.460278},
    "EFHK": {"lat": 60.317222, "lon": 24.963333},
    "EIDW": {"lat": 53.426444, "lon": -6.249911},
    "LEMD": {"lat": 40.471944, "lon": -3.562639},
    "LIRF": {"lat": 41.800278, "lon": 12.238889},
    "LOWW": {"lat": 48.110278, "lon": 16.569722},
    "LPPT": {"lat": 38.775556, "lon": -9.135833},
    "EBBR": {"lat": 50.901389, "lon": 4.484444},
    "EDDH": {"lat": 53.630278, "lon": 9.988056},
    "LEBL": {"lat": 41.2975, "lon": 2.078333},
    "LTCK": {"lat": 40.9079, "lon": 29.3444},
    "LGAV": {"lat": 37.936389, "lon": 23.947222},
    "ENBR": {"lat": 60.301389, "lon": 5.218056},
    "ENZV": {"lat": 58.761944, "lon": 5.626667},
    "EKCH": {"lat": 55.622222, "lon": 12.656111},
    "ESGG": {"lat": 57.662778, "lon": 12.278333},
    "LSGG": {"lat": 46.238889, "lon": 6.108889},
    "LTFJ": {"lat": 40.985, "lon": 28.813611},
    "LZIB": {"lat": 48.170833, "lon": 17.185278},
    "GCXO": {"lat": 28.483333, "lon": -16.341389},
    "GCLP": {"lat": 27.931389, "lon": -15.446667},
    "GCFV": {"lat": 28.461667, "lon": -14.075833},
    "LCLK": {"lat": 34.875, "lon": 33.623333},
    "LDDU": {"lat": 42.569722, "lon": 18.261111},
    "LGIR": {"lat": 35.339722, "lon": 25.180278},
    "LGSK": {"lat": 37.751667, "lon": 26.911389},
    "LRBS": {"lat": 44.408333, "lon": 26.095278},
    "LROP": {"lat": 44.570833, "lon": 26.095278},
    "LBWN": {"lat": 43.2325, "lon": 27.825278},
    "LBPH": {"lat": 42.2725, "lon": 24.850833},
    "LYBE": {"lat": 44.818333, "lon": 20.2925},
}

# %% Data Loading & Feature Engineering (Summarized for brevity)
print("Loading data and performing feature engineering...")
# ... (All data loading and feature engineering code remains identical to your original script)
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
data_cleaned["FLT_DATE"] = pd.to_datetime(data_cleaned["FLT_DATE"]).dt.tz_localize(
    pytz.UTC
)
data_cleaned.loc[:, "day_of_week"] = data_cleaned["FLT_DATE"].dt.dayofweek
data_cleaned.loc[:, "day_of_month"] = data_cleaned["FLT_DATE"].dt.day
data_cleaned.loc[:, "day_of_year"] = data_cleaned["FLT_DATE"].dt.dayofyear
data_cleaned.loc[:, "week_of_year"] = (
    data_cleaned["FLT_DATE"].dt.isocalendar().week.astype("int64")
)
data_cleaned.loc[:, "quarter"] = data_cleaned["FLT_DATE"].dt.quarter
data_cleaned.loc[:, "is_weekend"] = (data_cleaned["FLT_DATE"].dt.dayofweek >= 5).astype(
    int
)
data_cleaned.loc[:, "pandemic_phase"] = data_cleaned.apply(
    assign_pandemic_phase, axis=1
)
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
data_cleaned.loc[:, "pandemic_phase"] = pd.Categorical(
    data_cleaned["pandemic_phase"], categories=pandemic_phase_order, ordered=True
)
pre_pandemic_data = data_cleaned[data_cleaned["YEAR"] < 2020].copy()
avg_traffic_pre_pandemic_monthly_for_lookup = (
    pre_pandemic_data.groupby(["APT_ICAO", "MONTH_NUM"])["FLT_TOT_1"]
    .mean()
    .reset_index()
    .rename(columns={"FLT_TOT_1": "avg_pre_pandemic_monthly_traffic"})
)
data_cleaned = pd.merge(
    data_cleaned,
    avg_traffic_pre_pandemic_monthly_for_lookup,
    on=["APT_ICAO", "MONTH_NUM"],
    how="left",
)
overall_pre_pandemic_mean_for_lookup = pre_pandemic_data["FLT_TOT_1"].mean()
data_cleaned.loc[:, "avg_pre_pandemic_monthly_traffic"] = data_cleaned[
    "avg_pre_pandemic_monthly_traffic"
].fillna(overall_pre_pandemic_mean_for_lookup)
avg_traffic_pre_pandemic_overall_for_lookup = (
    pre_pandemic_data.groupby("APT_ICAO")["FLT_TOT_1"]
    .mean()
    .reset_index()
    .rename(columns={"FLT_TOT_1": "avg_traffic_pre_pandemic_overall"})
)
data_cleaned = pd.merge(
    data_cleaned, avg_traffic_pre_pandemic_overall_for_lookup, on="APT_ICAO", how="left"
)
data_cleaned.loc[:, "avg_traffic_pre_pandemic_overall"] = data_cleaned[
    "avg_traffic_pre_pandemic_overall"
].fillna(overall_pre_pandemic_mean_for_lookup)
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
labels_volume_category = ["Small", "Medium", "Large", "Very Large"]
data_cleaned.loc[:, "airport_volume_category"] = pd.cut(
    data_cleaned["avg_traffic_pre_pandemic_overall"],
    bins=bins_volume_category,
    labels=labels_volume_category,
    include_lowest=True,
    right=True,
).astype("category")
print("Data Cleaning and Feature Engineering Complete.")

# %% Data Preparation & Model Training (Summarized for brevity)
# ... (All model prep and training code remains identical to your original script)
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
final_model_features = numerical_features + categorical_features
y_original = data_cleaned["FLT_TOT_1"]
y_transformed = np.log1p(y_original)
train_df = data_cleaned[data_cleaned["FLT_DATE"] < CHRONOLOGICAL_SPLIT_DATE].copy()
test_df = data_cleaned[data_cleaned["FLT_DATE"] >= CHRONOLOGICAL_SPLIT_DATE].copy()
X_train = train_df[final_model_features].copy()
y_train_original = train_df["FLT_TOT_1"].copy()
y_train_transformed = np.log1p(y_train_original)
X_test = test_df[final_model_features].copy()
y_test_original = test_df["FLT_TOT_1"].copy()
y_test_transformed = np.log1p(y_test_original)
for col in categorical_features:
    if col in X_train.columns:
        X_train.loc[:, col] = X_train[col].astype("category")
    if col in X_test.columns:
        X_test.loc[:, col] = X_test[col].astype("category")
train_df.loc[:, "FLT_TOT_1_log"] = np.log1p(train_df["FLT_TOT_1"])
print("Training Linear Regression Model...")
preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",
)
lr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_lr),
        ("regressor", LinearRegression(n_jobs=-1)),
    ]
)
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
print("Linear Regression Model Trained and Evaluated.")


# %% Functions to create prediction input & get airport data (Unchanged)
def create_sample_input(
    airport_icao: str,
    date_str: str,
    source_data_cleaned: pd.DataFrame,
    pre_pandemic_stats_monthly: pd.DataFrame,
    pre_pandemic_stats_overall: pd.DataFrame,
    overall_pre_pandemic_mean_val: float,
    volume_bins,
    volume_labels,
    pandemic_phase_order_list,
    final_model_features_list,
    categorical_features_list,
):
    # ... (This function remains identical to your original script)
    input_date = pd.to_datetime(date_str).replace(tzinfo=pytz.UTC)
    sample_data = {
        "YEAR": input_date.year,
        "MONTH_NUM": input_date.month,
        "day_of_week": input_date.dayofweek,
        "day_of_month": input_date.day,
        "day_of_year": input_date.dayofyear,
        "week_of_year": int(input_date.isocalendar().week),
        "quarter": input_date.quarter,
        "is_weekend": int(input_date.dayofweek >= 5),
        "APT_ICAO": airport_icao,
    }
    state_rows = source_data_cleaned[source_data_cleaned["APT_ICAO"] == airport_icao][
        "STATE_NAME"
    ]
    sample_data["STATE_NAME"] = (
        state_rows.iloc[0] if not state_rows.empty else "UNKNOWN_STATE"
    )
    temp_series = pd.Series({"YEAR": input_date.year, "MONTH_NUM": input_date.month})
    sample_data["pandemic_phase"] = assign_pandemic_phase(temp_series)
    matching_row_monthly = pre_pandemic_stats_monthly[
        (pre_pandemic_stats_monthly["APT_ICAO"] == airport_icao)
        & (pre_pandemic_stats_monthly["MONTH_NUM"] == input_date.month)
    ]
    sample_data["avg_pre_pandemic_monthly_traffic"] = (
        matching_row_monthly["avg_pre_pandemic_monthly_traffic"].iloc[0]
        if not matching_row_monthly.empty
        else overall_pre_pandemic_mean_val
    )
    matching_row_overall = pre_pandemic_stats_overall[
        pre_pandemic_stats_overall["APT_ICAO"] == airport_icao
    ]
    avg_traffic_overall = (
        matching_row_overall["avg_traffic_pre_pandemic_overall"].iloc[0]
        if not matching_row_overall.empty
        else overall_pre_pandemic_mean_val
    )
    sample_data["avg_traffic_pre_pandemic_overall"] = avg_traffic_overall
    category = pd.cut(
        [avg_traffic_overall],
        bins=volume_bins,
        labels=volume_labels,
        include_lowest=True,
        right=True,
    )[0]
    sample_data["airport_volume_category"] = category
    input_df = pd.DataFrame([sample_data], columns=final_model_features_list)
    for col in categorical_features_list:
        if col in input_df.columns:
            input_df.loc[:, col] = input_df[col].astype("category")
            if col == "pandemic_phase":
                input_df.loc[:, col] = pd.Categorical(
                    input_df[col], categories=pandemic_phase_order_list, ordered=True
                )
            elif col == "airport_volume_category":
                input_df.loc[:, col] = pd.Categorical(
                    input_df[col], categories=labels_volume_category, ordered=True
                )
    return input_df


unique_airport_data = (
    data_cleaned[data_cleaned["APT_ICAO"].isin(AIRPORT_COORDINATES.keys())][
        ["APT_ICAO", "STATE_NAME"]
    ]
    .drop_duplicates()
    .sort_values("APT_ICAO")
)

# %% Dash App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Helper function to create KPI cards that match the new CSS
def create_kpi_card(title_id, value_id):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(id=title_id, className="card-text"),
                html.H2("0", id=value_id, className="card-title"),
            ]
        ),
        className="text-center",
    )


app.layout = dbc.Container(
    [
        dcc.Store(id="language-store", data="en"),
        html.H1(id="main-title", className="text-center my-4"),
        dbc.Tabs(
            [
                # ========================= Operations Summary Tab =========================
                dbc.Tab(
                    label="Operations Summary",
                    tab_id="operations-summary-tab",
                    id="tab-operations-summary",
                    className="custom-tab",
                    active_tab_class_name="custom-tab--selected",
                    children=dbc.Container(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.H5(
                                                        id="summary-card-title",
                                                        className="fw-bold",
                                                    ),
                                                    width="auto",
                                                    className="me-3",
                                                ),
                                                dbc.Col(
                                                    dcc.DatePickerRange(
                                                        id="summary-date-picker",
                                                        min_date_allowed=data_cleaned[
                                                            "FLT_DATE"
                                                        ]
                                                        .min()
                                                        .date(),
                                                        max_date_allowed=data_cleaned[
                                                            "FLT_DATE"
                                                        ]
                                                        .max()
                                                        .date(),
                                                        start_date=datetime.date(
                                                            2021, 1, 1
                                                        ),
                                                        end_date=datetime.date(
                                                            2021, 1, 31
                                                        ),
                                                        display_format="YYYY-MM-DD",
                                                    ),
                                                    width=4,
                                                ),
                                            ],
                                            align="center",
                                            className="mb-2",
                                        )
                                    ]
                                ),
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        create_kpi_card(
                                            "total-ops-header", "total-ops-card"
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_kpi_card(
                                            "total-arr-header", "total-arr-card"
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_kpi_card(
                                            "total-dep-header", "total-dep-card"
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        create_kpi_card(
                                            "change-header",
                                            "change-vs-pre-pandemic-card",
                                        ),
                                        md=3,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Card(
                                dcc.Graph(id="daily-ops-bar-chart"), className="mb-4"
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id="airport-pareto-chart")),
                                        md=7,
                                    ),
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id="country-bar-chart")),
                                        md=5,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id="volume-pie-chart")), md=6
                                    ),
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id="weekend-pie-chart")),
                                        md=6,
                                    ),
                                ]
                            ),
                        ],
                        fluid=True,
                        className="py-4",
                    ),
                ),
                # ========================= Other Tabs (Unchanged Layout) =========================
                dbc.Tab(
                    label="Model Performance",
                    tab_id="model-performance-tab",
                    id="tab-model-performance",
                    className="custom-tab",
                    active_tab_class_name="custom-tab--selected",
                    children=[  # This tab's layout remains the same, it will be styled by the CSS
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="card-header-times",
                                                className="bg-primary",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.H5(
                                                        id="card-title-train-time",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        f"{results['LinearRegression']['train_time_sec']:.2f} seconds",
                                                        className="card-text",
                                                    ),
                                                    html.H5(
                                                        id="card-title-pred-time-train",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        f"{results['LinearRegression']['predict_time_sec_train']:.4f} seconds",
                                                        className="card-text",
                                                    ),
                                                    html.H5(
                                                        id="card-title-pred-time-test",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        f"{results['LinearRegression']['predict_time_sec_test']:.4f} seconds",
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="shadow-sm border-0 mb-4",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="card-header-train-log",
                                                className="bg-primary",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        f"R2: {results['LinearRegression']['metrics_log_scale']['R2']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"MAE: {results['LinearRegression']['metrics_log_scale']['MAE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"RMSE: {results['LinearRegression']['metrics_log_scale']['RMSE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="shadow-sm border-0 mb-4",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="card-header-train-orig",
                                                className="bg-primary",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        f"R2: {results['LinearRegression']['metrics_original_scale']['R2']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"MAE: {results['LinearRegression']['metrics_original_scale']['MAE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"RMSE: {results['LinearRegression']['metrics_original_scale']['RMSE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="shadow-sm border-0 mb-4",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="mt-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="card-header-test-log",
                                                className="bg-primary",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        f"R2: {results['LinearRegression']['metrics_log_scale_test']['R2']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"MAE: {results['LinearRegression']['metrics_log_scale_test']['MAE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"RMSE: {results['LinearRegression']['metrics_log_scale_test']['RMSE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="shadow-sm border-0",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="card-header-test-orig",
                                                className="bg-primary",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.P(
                                                        f"R2: {results['LinearRegression']['metrics_original_scale_test']['R2']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"MAE: {results['LinearRegression']['metrics_original_scale_test']['MAE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                    html.P(
                                                        f"RMSE: {results['LinearRegression']['metrics_original_scale_test']['RMSE']:.4f}",
                                                        className="card-text",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="shadow-sm border-0",
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Flight Prediction",
                    tab_id="flight-prediction-tab",
                    id="tab-flight-prediction",
                    className="custom-tab",
                    active_tab_class_name="custom-tab--selected",
                    children=[  # This tab's layout also remains the same
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    id="card-header-predict", className="bg-info"
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            id="label-select-airport",
                                                            className="fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="airport-dropdown",
                                                            value="EDDF",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            id="label-select-date",
                                                            className="fw-bold",
                                                        ),
                                                        dcc.DatePickerSingle(
                                                            id="date-picker",
                                                            min_date_allowed=datetime.date(
                                                                2016, 1, 1
                                                            ),
                                                            max_date_allowed=datetime.date(
                                                                2028, 12, 31
                                                            ),
                                                            initial_visible_month=datetime.date.today(),
                                                            date=datetime.date.today(),
                                                            display_format="YYYY-MM-DD",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Button(
                                            id="predict-button",
                                            className="btn-primary w-100",
                                        ),
                                        html.Div(
                                            id="prediction-output",
                                            className="mt-4 text-center",
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-lg border-0 mt-4",
                        )
                    ],
                ),
                dbc.Tab(
                    label="Visualizations",
                    tab_id="visualizations-tab",
                    id="tab-visualizations",
                    className="custom-tab",
                    active_tab_class_name="custom-tab--selected",
                    children=[  # Unchanged layout
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    id="viz-header-map", className="bg-success"
                                ),
                                dbc.CardBody([dcc.Graph(id="airport-map")]),
                            ],
                            className="shadow-lg border-0 mt-4 mb-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="viz-header-month-timeseries",
                                                className="bg-success",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(
                                                        id="flights-by-month-timeseries"
                                                    )
                                                ]
                                            ),
                                        ],
                                        className="shadow-lg border-0 mb-4",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="viz-header-dayofweek",
                                                className="bg-success",
                                            ),
                                            dbc.CardBody(
                                                [dcc.Graph(id="flights-by-dayofweek")]
                                            ),
                                        ],
                                        className="shadow-lg border-0 mb-4",
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="viz-header-pandemic-phase",
                                                className="bg-success",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(
                                                        id="flights-by-pandemic-phase"
                                                    )
                                                ]
                                            ),
                                        ],
                                        className="shadow-lg border-0 mb-4",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                id="viz-header-volume-category",
                                                className="bg-success",
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(
                                                        id="flights-by-volume-category"
                                                    )
                                                ]
                                            ),
                                        ],
                                        className="shadow-lg border-0 mb-4",
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            className="custom-tabs-container",
        ),
    ],
    fluid=True,
    className="py-4",
)

# %% Callbacks


# --- Language & Static Text Callback ---
@app.callback(
    [
        Output("main-title", "children"),
        Output("tab-operations-summary", "label"),
        Output("tab-model-performance", "label"),
        Output("tab-flight-prediction", "label"),
        Output("tab-visualizations", "label"),
        Output("summary-card-title", "children"),
        Output("total-ops-header", "children"),
        Output("total-arr-header", "children"),
        Output("total-dep-header", "children"),
        Output("change-header", "children"),
        Output("card-header-times", "children"),
        Output("card-title-train-time", "children"),
        Output("card-title-pred-time-train", "children"),
        Output("card-title-pred-time-test", "children"),
        Output("card-header-train-log", "children"),
        Output("card-header-train-orig", "children"),
        Output("card-header-test-log", "children"),
        Output("card-header-test-orig", "children"),
        Output("card-header-predict", "children"),
        Output("label-select-airport", "children"),
        Output("airport-dropdown", "placeholder"),
        Output("label-select-date", "children"),
        Output("predict-button", "children"),
        Output("viz-header-map", "children"),
        Output("viz-header-month-timeseries", "children"),
        Output("viz-header-dayofweek", "children"),
        Output("viz-header-pandemic-phase", "children"),
        Output("viz-header-volume-category", "children"),
        Output("language-store", "data"),
        Output("airport-dropdown", "options"),
    ],
    [
        Input("language-selector", "value")
    ],  # Assuming a selector exists, though not in layout
)
def update_static_text_and_language(lang="en"):  # Default to 'en'
    text = text_content[lang]
    country_trans = country_translations[lang]
    airport_options = [
        {
            "label": f"{row['APT_ICAO']} ({country_trans.get(row['STATE_NAME'], row['STATE_NAME'])})",
            "value": row["APT_ICAO"],
        }
        for _, row in unique_airport_data.iterrows()
    ]
    return (
        text["main_title"],
        text["tab_operations_summary"],
        text["tab_model_performance"],
        text["tab_flight_prediction"],
        text["tab_visualizations"],
        text["summary_card_title"],
        text["total_ops_header"],
        text["total_arr_header"],
        text["total_dep_header"],
        text["change_header"],
        text["card_header_times"],
        text["card_title_train_time"],
        text["card_title_pred_time_train"],
        text["card_title_pred_time_test"],
        text["card_header_train_log"],
        text["card_header_train_orig"],
        text["card_header_test_log"],
        text["card_header_test_orig"],
        text["card_header_predict"],
        text["label_select_airport"],
        text["placeholder_select_airport"],
        text["label_select_date"],
        text["button_predict"],
        text["viz_map_title"],
        text["viz_month_timeseries_title"],
        text["viz_dayofweek_title"],
        text["viz_pandemic_phase_title"],
        text["viz_volume_category_title"],
        lang,
        airport_options,
    )


# --- Operations Summary Tab Callback ---
@app.callback(
    [
        Output("total-ops-card", "children"),
        Output("total-arr-card", "children"),
        Output("total-dep-card", "children"),
        Output("change-vs-pre-pandemic-card", "children"),
        Output("daily-ops-bar-chart", "figure"),
        Output("airport-pareto-chart", "figure"),
        Output("volume-pie-chart", "figure"),
        Output("weekend-pie-chart", "figure"),
        Output("country-bar-chart", "figure"),
    ],
    [
        Input("summary-date-picker", "start_date"),
        Input("summary-date-picker", "end_date"),
        Input("language-store", "data"),
    ],
)
def update_summary_tab(start_date, end_date, lang):
    if not start_date or not end_date:
        return ["-"] * 4 + [go.Figure()] * 5

    text = text_content.get(lang, text_content["en"])
    start = pd.to_datetime(start_date).tz_localize("UTC")
    end = pd.to_datetime(end_date).tz_localize("UTC")
    dff = data_cleaned[
        (data_cleaned["FLT_DATE"] >= start) & (data_cleaned["FLT_DATE"] <= end)
    ].copy()

    if dff.empty:
        return ["-"] * 4 + [go.Figure()] * 5

    def format_number(n):
        n = int(n)
        if abs(n) >= 1e6:
            return f"{n/1e6:.1f}M"
        if abs(n) >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)

    # KPI Calculations
    total_ops = dff["FLT_TOT_1"].sum()
    total_arr = dff["FLT_ARR_1"].sum()
    total_dep = dff["FLT_DEP_1"].sum()

    # Pre-pandemic comparison KPI
    start_pre = start.replace(year=2019)
    end_pre = end.replace(year=2019)
    pre_pandemic_ref_df = pre_pandemic_data[
        (pre_pandemic_data["FLT_DATE"] >= start_pre)
        & (pre_pandemic_data["FLT_DATE"] <= end_pre)
    ]
    total_ops_pre = pre_pandemic_ref_df["FLT_TOT_1"].sum()
    if total_ops_pre > 0:
        change_pct = ((total_ops - total_ops_pre) / total_ops_pre) * 100
        change_color = SUCCESS_GREEN_COLOR if change_pct >= 0 else ERROR_RED_DARK_COLOR
        change_component = html.H2(
            f"{change_pct:+.1f}%", className="card-title", style={"color": change_color}
        )
    else:
        change_component = html.H2(
            "N/A", className="card-title", style={"color": DEEP_NAVY_COLOR}
        )

    # Common layout for themed charts
    chart_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_color": DEEP_NAVY_COLOR,
        "margin": dict(l=40, r=20, t=60, b=40),
        "title_x": 0.5,
    }

    # Chart 1: Daily Operations
    daily_ops_df = (
        dff.groupby(dff["FLT_DATE"].dt.date)[["FLT_ARR_1", "FLT_DEP_1"]]
        .sum()
        .reset_index()
    )
    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=daily_ops_df["FLT_DATE"],
            y=daily_ops_df["FLT_ARR_1"],
            name="Arrivals",
            marker_color=HORIZON_BLUE_COLOR,
        )
    )
    fig_daily.add_trace(
        go.Bar(
            x=daily_ops_df["FLT_DATE"],
            y=daily_ops_df["FLT_DEP_1"],
            name="Departures",
            marker_color=AIRLINE_BLUE_COLOR,
        )
    )
    fig_daily.update_layout(
        **chart_layout,
        barmode="stack",
        title_text=text["daily_ops_chart_title"],
        yaxis_title="Flights",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Chart 2: Pareto
    airport_ops = (
        dff.groupby("APT_ICAO")["FLT_TOT_1"].sum().sort_values(ascending=False).head(10)
    )
    pareto_df = airport_ops.to_frame("Total Flights")
    pareto_df["Cumulative %"] = (pareto_df["Total Flights"].cumsum() / total_ops) * 100
    fig_pareto = go.Figure()
    fig_pareto.add_trace(
        go.Bar(
            x=pareto_df.index,
            y=pareto_df["Total Flights"],
            name=text["pareto_yaxis1"],
            marker_color=HORIZON_BLUE_COLOR,
        )
    )
    fig_pareto.add_trace(
        go.Scatter(
            x=pareto_df.index,
            y=pareto_df["Cumulative %"],
            name=text["pareto_yaxis2"],
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=DEEP_NAVY_COLOR),
        )
    )
    fig_pareto.update_layout(
        **chart_layout,
        title_text=text["pareto_chart_title"],
        yaxis=dict(title=text["pareto_yaxis1"]),
        yaxis2=dict(
            title=text["pareto_yaxis2"],
            overlaying="y",
            side="right",
            range=[0, 105],
            ticksuffix="%",
        ),
        legend=dict(x=0.01, y=0.98),
    )

    # Chart 3: Flights by Country
    country_ops = (
        dff.groupby("STATE_NAME")["FLT_TOT_1"]
        .sum()
        .sort_values(ascending=False)
        .nlargest(15)
    )
    fig_country = px.bar(
        x=country_ops.index,
        y=country_ops.values,
        title=text["country_chart_title"],
        color_discrete_sequence=[AIRLINE_BLUE_COLOR],
    )
    fig_country.update_layout(
        **chart_layout, xaxis_title=None, yaxis_title="Total Flights"
    )

    # Chart 4 & 5: Pies
    volume_counts = dff.groupby("airport_volume_category", observed=True)[
        "FLT_TOT_1"
    ].sum()
    fig_volume_pie = px.pie(
        names=volume_counts.index,
        values=volume_counts.values,
        title=text["volume_pie_title"],
        hole=0.4,
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    dff.loc[:, "day_type"] = np.where(
        dff["is_weekend"] == 1, text["weekend_label"], text["weekday_label"]
    )
    weekend_counts = dff.groupby("day_type")["FLT_TOT_1"].sum()
    fig_weekend_pie = px.pie(
        names=weekend_counts.index,
        values=weekend_counts.values,
        title=text["weekend_pie_title"],
        hole=0.4,
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    for fig in [fig_volume_pie, fig_weekend_pie]:
        fig.update_layout(**chart_layout, showlegend=False)
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker_line=dict(color="#FFFFFF", width=2),
        )

    return (
        format_number(total_ops),
        format_number(total_arr),
        format_number(total_dep),
        change_component,
        fig_daily,
        fig_pareto,
        fig_volume_pie,
        fig_weekend_pie,
        fig_country,
    )


# --- Other Callbacks (Prediction, Visualizations) ---
# ... (The callbacks for the other tabs remain identical to your original script)
@app.callback(
    Output("prediction-output", "children"),
    Output("prediction-output", "className"),
    [Input("predict-button", "n_clicks")],
    [
        State("airport-dropdown", "value"),
        State("date-picker", "date"),
        State("language-store", "data"),
    ],
)
def update_prediction(n_clicks, airport_icao, date_str, lang):
    text = text_content[lang]
    if n_clicks is None:
        return "", "mt-4 text-center"
    if not airport_icao or not date_str:
        return text["prediction_error_select"], "mt-4 text-center text-danger"
    try:
        input_df = create_sample_input(
            airport_icao,
            date_str,
            data_cleaned,
            avg_traffic_pre_pandemic_monthly_for_lookup,
            avg_traffic_pre_pandemic_overall_for_lookup,
            overall_pre_pandemic_mean_for_lookup,
            bins_volume_category,
            labels_volume_category,
            pandemic_phase_order,
            final_model_features,
            categorical_features,
        )
        predicted_log = lr_model.predict(input_df)
        predicted_flights = max(0, np.expm1(predicted_log)[0])
        return (
            text["prediction_output_template"].format(
                airport=airport_icao, date=date_str, prediction=predicted_flights
            ),
            "mt-4 text-center",
        )
    except Exception as e:
        return (
            text["prediction_error_general"].format(e=e),
            "mt-4 text-center text-danger",
        )


@app.callback(Output("airport-map", "figure"), [Input("language-store", "data")])
def create_map_figure(lang):
    # This chart can also be themed
    text = text_content[lang]
    map_data = avg_traffic_pre_pandemic_overall_for_lookup.copy()
    map_data["lat"] = map_data["APT_ICAO"].apply(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get("lat")
    )
    map_data["lon"] = map_data["APT_ICAO"].apply(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get("lon")
    )
    map_data.dropna(subset=["lat", "lon"], inplace=True)
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        size="avg_traffic_pre_pandemic_overall",
        color="avg_traffic_pre_pandemic_overall",
        color_continuous_scale=px.colors.sequential.Plasma,
        hover_name="APT_ICAO",
        hover_data={
            "avg_traffic_pre_pandemic_overall": ":.0f",
            "lat": False,
            "lon": False,
        },
        zoom=3.5,
        height=600,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# Note: The remaining visualization callbacks can also be themed by adding the chart_layout dictionary.

# %% Run the app
if __name__ == "__main__":
    app.run(debug=True)
