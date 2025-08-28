# app.py
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import datetime
import pytz


# --- Helper functions ---


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


# This is the app's own, vectorized version used for dashboard calculations.
def assign_pandemic_phase_vectorized(df: pd.DataFrame) -> pd.Series:
    conditions = [
        (df["YEAR"] < 2020),
        (df["YEAR"] == 2020) & (df["MONTH_NUM"].between(3, 5)),
        (df["YEAR"] == 2020) & (df["MONTH_NUM"].between(6, 8)),
        (df["YEAR"] == 2020),
        (df["YEAR"] == 2021) & (df["MONTH_NUM"].between(1, 4)),
        (df["YEAR"] == 2021) & (df["MONTH_NUM"].between(5, 8)),
        (df["YEAR"] == 2021),
        (df["YEAR"] == 2022),
        (df["YEAR"] >= 2023),
    ]
    choices = [
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
    return np.select(conditions, choices, default="Other")


# --- 1. Load Pre-trained Models and Artifacts ---
print("Loading pre-trained models and data artifacts...")
try:
    artifacts = joblib.load("models/flight_prediction_models.joblib")
    model_artifacts = artifacts["models"]
    data_artifacts = artifacts["data"]
    print("Artifacts loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file not found. Please run your training script first.")
    exit()

# --- 2. Load and Preprocess Data for Dashboard Visuals ---
print("Loading and processing data for dashboard...")
data_raw = pd.read_csv("flights.csv")
data_cleaned = data_raw.drop(
    columns=[
        "FLT_DEP_IFR_2",
        "FLT_ARR_IFR_2",
        "FLT_TOT_IFR_2",
        "MONTH_MON",
        "Pivot Label",
        "APT_NAME",
    ]
)
data_cleaned["FLT_DATE"] = pd.to_datetime(data_cleaned["FLT_DATE"])
if data_cleaned["FLT_DATE"].dt.tz is None:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_localize(pytz.UTC)
else:
    data_cleaned["FLT_DATE"] = data_cleaned["FLT_DATE"].dt.tz_convert(pytz.UTC)
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
data_cleaned["avg_pre_pandemic_monthly_traffic"].fillna(
    overall_pre_pandemic_mean, inplace=True
)
avg_traffic_pre_pandemic_overall = (
    pre_pandemic_data.groupby("APT_ICAO")["FLT_TOT_1"]
    .mean()
    .reset_index()
    .rename(columns={"FLT_TOT_1": "avg_traffic_pre_pandemic_overall"})
)
data_cleaned = pd.merge(
    data_cleaned, avg_traffic_pre_pandemic_overall, on="APT_ICAO", how="left"
)
data_cleaned["avg_traffic_pre_pandemic_overall"].fillna(
    overall_pre_pandemic_mean, inplace=True
)

top_airports = data_cleaned["APT_ICAO"].value_counts().nlargest(40).index
data_cleaned["APT_ICAO"] = data_cleaned["APT_ICAO"].where(
    data_cleaned["APT_ICAO"].isin(top_airports), "Other"
)
top_states = data_cleaned["STATE_NAME"].value_counts().nlargest(15).index
data_cleaned["STATE_NAME"] = data_cleaned["STATE_NAME"].where(
    data_cleaned["STATE_NAME"].isin(top_states), "Other"
)
data_cleaned["day_of_week"] = data_cleaned["FLT_DATE"].dt.dayofweek
data_cleaned["is_weekend"] = (data_cleaned["FLT_DATE"].dt.dayofweek >= 5).astype(int)
data_cleaned["pandemic_phase"] = assign_pandemic_phase_vectorized(data_cleaned)
data_cleaned["airport_volume_category"] = pd.cut(
    data_cleaned["avg_traffic_pre_pandemic_overall"],
    bins=data_artifacts["airport_volume_bins"],
    labels=data_artifacts["airport_volume_labels"],
    include_lowest=True,
)
train_df = data_cleaned[
    data_cleaned["FLT_DATE"] < datetime.datetime(2021, 2, 1, tzinfo=pytz.UTC)
].copy()
train_df["FLT_TOT_1_log"] = np.log1p(train_df["FLT_TOT_1"])

unique_airport_data_for_dropdown = (
    data_raw[["APT_ICAO", "STATE_NAME"]].drop_duplicates().sort_values("APT_ICAO")
)
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

# --- Full Translation Dictionaries ---
text_content = {
    "en": {
        "main_title": "Flight Traffic Dashboard",
        "tab_operations_summary": "Operations Summary",
        "tab_model_performance": "Model Performance",
        "tab_flight_prediction": "Flight Prediction",
        "tab_visualizations": "Visualizations",
        "summary_card_title": "Select Date Range:",
        "total_ops_header": "TOTAL OPERATIONS",
        "total_arr_header": "TOTAL ARRIVALS",
        "total_dep_header": "TOTAL DEPARTURES",
        "change_header": "vs. PRE-PANDEMIC (2019)",
        "daily_ops_chart_title": "Daily Operations (Arrivals vs Departures)",
        "pareto_chart_title": "Top 10 Airports by Contribution to Total Flights",
        "pareto_xaxis": "Airport",
        "pareto_yaxis1": "Total Flights",
        "pareto_yaxis2": "Cumulative Percentage",
        "country_chart_title": "Flights by Country (Top 15)",
        "volume_pie_title": "Operations by Airport Volume",
        "weekend_pie_title": "Operations by Day Type",
        "weekend_label": "Weekend",
        "weekday_label": "Weekday",
        "select_model_label": "Select Model to Evaluate:",
        "card_header_train_log": "Train Metrics (Log Scale)",
        "card_header_train_orig": "Train Metrics (Original Scale)",
        "card_header_test_log": "Test Metrics (Log Scale)",
        "card_header_test_orig": "Test Metrics (Original Scale)",
        "card_header_predict": "Predict Future Flight Traffic",
        "label_select_model_predict": "Select Model for Prediction:",
        "label_select_airport": "Select Airport:",
        "placeholder_select_airport": "Select an airport...",
        "label_select_date": "Select Date:",
        "button_predict": "Predict Flights",
        "prediction_success_title": "Prediction Successful",
        "prediction_output_template": "Predicted Total Flights: {prediction:.0f}",
        "historical_average_template": "The historical average for this airport in {month_name} is around {avg:.0f} flights per day.",
        "prediction_error_select": "Please select a model, an airport, and a date.",
        "prediction_error_general": "An error occurred during prediction.",
        "viz_map_title": "Airport Traffic Map (Pre-Pandemic Average)",
        "viz_map_hover_template": "Average Pre-Pandemic Flight Traffic by Airport",
        "viz_month_timeseries_title": "Total Flights by Month (Training Data)",
        "viz_month_timeseries_xaxis": "Month",
        "viz_month_timeseries_yaxis": "Total Flights",
        "viz_dayofweek_title": "Average Log(Total Flights) by Day of Week",
        "viz_dayofweek_xaxis": "Day of Week (0=Mon, 6=Sun)",
        "viz_dayofweek_yaxis": "Average Log(Total Flights)",
        "viz_pandemic_phase_title": "Average Log(Total Flights) by Pandemic Phase",
        "viz_pandemic_phase_xaxis": "Pandemic Phase",
        "viz_pandemic_phase_yaxis": "Average Log(Total Flights)",
        "viz_volume_category_title": "Average Log(Total Flights) by Airport Volume Category",
        "viz_volume_category_xaxis": "Airport Volume Category",
        "viz_volume_category_yaxis": "Average Log(Total Flights)",
    },
    "fr": {
        "main_title": "Tableau de Bord du Trafic Aérien",
        "tab_operations_summary": "Résumé des Opérations",
        "tab_model_performance": "Performance du Modèle",
        "tab_flight_prediction": "Prédiction de Vols",
        "tab_visualizations": "Visualisations",
        "summary_card_title": "Sélectionnez la Plage de Dates :",
        "total_ops_header": "OPÉRATIONS TOTALES",
        "total_arr_header": "ARRIVÉES TOTALES",
        "total_dep_header": "DÉPARTS TOTAUX",
        "change_header": "vs. PRÉ-PANDÉMIE (2019)",
        "daily_ops_chart_title": "Opérations Journalières (Arrivées vs Départs)",
        "pareto_chart_title": "Top 10 des Aéroports par Contribution aux Vols Totaux",
        "pareto_xaxis": "Aéroport",
        "pareto_yaxis1": "Vols Totaux",
        "pareto_yaxis2": "Pourcentage Cumulé",
        "country_chart_title": "Vols par Pays (Top 15)",
        "volume_pie_title": "Opérations par Volume d'Aéroport",
        "weekend_pie_title": "Opérations par Type de Jour",
        "weekend_label": "Weekend",
        "weekday_label": "Jour de semaine",
        "select_model_label": "Sélectionner le Modèle à Évaluer :",
        "card_header_train_log": "Métriques d'Entraînement (Échelle Log)",
        "card_header_train_orig": "Métriques d'Entraînement (Échelle Originale)",
        "card_header_test_log": "Métriques de Test (Échelle Log)",
        "card_header_test_orig": "Métriques de Test (Échelle Originale)",
        "card_header_predict": "Prédire le Trafic Aérien Futur",
        "label_select_model_predict": "Sélectionner le Modèle de Prédiction :",
        "label_select_airport": "Sélectionner l'Aéroport :",
        "placeholder_select_airport": "Sélectionnez un aéroport...",
        "label_select_date": "Sélectionner la Date :",
        "button_predict": "Prédire les Vols",
        "prediction_success_title": "Prédiction Réussie",
        "prediction_output_template": "Vols totaux prévus : {prediction:.0f}",
        "historical_average_template": "La moyenne historique pour cet aéroport en {month_name} est d'environ {avg:.0f} vols par jour.",
        "prediction_error_select": "Veuillez sélectionner un modèle, un aéroport et une date.",
        "prediction_error_general": "Une erreur est survenue lors de la prédiction.",
        "viz_map_title": "Carte du Trafic Aéroportuaire (Moyenne pré-pandémique)",
        "viz_map_hover_template": "Trafic aérien moyen pré-pandémique par aéroport",
        "viz_month_timeseries_title": "Total des Vols par Mois (Données d'entraînement)",
        "viz_month_timeseries_xaxis": "Mois",
        "viz_month_timeseries_yaxis": "Total des Vols",
        "viz_dayofweek_title": "Moyenne Log(Vols) par Jour de la Semaine",
        "viz_dayofweek_xaxis": "Jour de la semaine (0=Lun, 6=Dim)",
        "viz_dayofweek_yaxis": "Moyenne Log(Vols)",
        "viz_pandemic_phase_title": "Moyenne Log(Vols) par Phase Pandémique",
        "viz_pandemic_phase_xaxis": "Phase Pandémique",
        "viz_pandemic_phase_yaxis": "Moyenne Log(Vols)",
        "viz_volume_category_title": "Moyenne Log(Vols) par Catégorie de Volume",
        "viz_volume_category_xaxis": "Catégorie de Volume d'Aéroport",
        "viz_volume_category_yaxis": "Moyenne Log(Vols)",
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
        "Switzerland": "Suisse",
        "Sweden": "Suède",
        "Belgium": "Belgique",
        "Finland": "Finlande",
        "Ireland": "Irlande",
        "Spain": "Espagne",
        "Italy": "Italie",
        "Austria": "Autriche",
        "Portugal": "Portugal",
        "Turkey": "Turquie",
        "Greece": "Grèce",
        "Norway": "Norvège",
        "Denmark": "Danemark",
        "Slovakia": "Slovaquie",
        "Cyprus": "Chypre",
        "Croatia": "Croatie",
        "Romania": "Roumanie",
        "Bulgaria": "Bulgarie",
        "Serbia": "Serbie",
    },
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
)
server = app.server


# --- Helper Functions ---
def create_kpi_card(title_id, value_id, icon_class="fas fa-plane-departure"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(
                    [html.I(className=f"{icon_class} me-2"), html.Span(id=title_id)],
                    className="kpi-card-header",
                ),
                dbc.Spinner(html.H2("0", className="display-4", id=value_id)),
            ]
        ),
        className="text-center kpi-card",
    )


def create_sample_input(airport_icao, date_str):
    """
    Creates a single-row DataFrame for prediction, mirroring the feature
    engineering from the training script.
    """
    input_date = pd.to_datetime(date_str).replace(tzinfo=pytz.UTC)

    # Base DataFrame
    sample_df = pd.DataFrame([{"FLT_DATE": input_date}])

    # --- Create all features the model expects ---
    # Date-based features
    sample_df["YEAR"] = sample_df["FLT_DATE"].dt.year
    sample_df["MONTH_NUM"] = sample_df["FLT_DATE"].dt.month
    sample_df["day_of_month"] = sample_df["FLT_DATE"].dt.day
    sample_df["day_of_week"] = sample_df["FLT_DATE"].dt.dayofweek
    sample_df["day_of_year"] = sample_df["FLT_DATE"].dt.dayofyear
    sample_df["quarter"] = sample_df["FLT_DATE"].dt.quarter
    sample_df["is_weekend"] = (sample_df["FLT_DATE"].dt.dayofweek >= 5).astype(int)

    # Cyclical features
    sample_df["day_of_year_sin"] = np.sin(2 * np.pi * sample_df["day_of_year"] / 365.25)
    sample_df["day_of_year_cos"] = np.cos(2 * np.pi * sample_df["day_of_year"] / 365.25)
    sample_df["month_sin"] = np.sin(2 * np.pi * sample_df["MONTH_NUM"] / 12)
    sample_df["month_cos"] = np.cos(2 * np.pi * sample_df["MONTH_NUM"] / 12)

    # Other categorical features
    sample_df["pandemic_phase"] = assign_pandemic_phase_vectorized(sample_df)

    # Get state name from the airport ICAO using the pre-loaded lookup table
    state_rows = unique_airport_data_for_dropdown[
        unique_airport_data_for_dropdown["APT_ICAO"] == airport_icao
    ]
    state_name = state_rows["STATE_NAME"].iloc[0] if not state_rows.empty else "Other"

    sample_df["APT_ICAO"] = airport_icao
    sample_df["STATE_NAME"] = state_name

    # --- Merge historical data to create traffic-based features ---
    monthly_stats = data_artifacts["pre_pandemic_monthly_stats"]
    overall_stats = data_artifacts["pre_pandemic_overall_stats"]

    merged_df = pd.merge(
        sample_df, monthly_stats, on=["APT_ICAO", "MONTH_NUM"], how="left"
    )
    merged_df = pd.merge(merged_df, overall_stats, on="APT_ICAO", how="left")

    # Fill missing stats with the global average (same logic as training)
    merged_df["avg_pre_pandemic_monthly_traffic"].fillna(
        data_artifacts["overall_pre_pandemic_mean"], inplace=True
    )
    merged_df["avg_traffic_pre_pandemic_overall"].fillna(
        data_artifacts["overall_pre_pandemic_mean"], inplace=True
    )

    # Create airport_volume_category feature (same logic as training)
    merged_df["airport_volume_category"] = pd.cut(
        merged_df["avg_traffic_pre_pandemic_overall"],
        bins=data_artifacts["airport_volume_bins"],
        labels=data_artifacts["airport_volume_labels"],
        include_lowest=True,
        right=True,
    )

    # --- Finalize DataFrame for prediction ---
    # Ensure all columns the model expects are present and in the correct order
    final_df = merged_df[data_artifacts["features"]].copy()

    # Set categorical dtypes as expected by the model pipeline
    for col in data_artifacts["categorical_features"]:
        final_df[col] = final_df[col].astype("category")

    return final_df


# --- Layout & Callbacks ---
app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="language-store", data="en"),
        dbc.NavbarSimple(
            brand=html.Span(id="main-title"),
            children=[
                dbc.RadioItems(
                    id="language-selector",
                    options=[
                        {"label": "🇬🇧 EN", "value": "en"},
                        {"label": "🇫🇷 FR", "value": "fr"},
                    ],
                    value="en",
                    inline=True,
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                )
            ],
            color="primary",
            dark=False,
            className="mb-4",
        ),
        dbc.Tabs(
            [
                dbc.Tab(
                    id="tab-component-ops",
                    label="Operations Summary",
                    tab_id="tab-ops",
                    className="custom-tab",
                    children=[
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
                                                        2022, 1, 1
                                                    ),
                                                    end_date=datetime.date(2022, 1, 31),
                                                    display_format="YYYY-MM-DD",
                                                ),
                                                width=4,
                                            ),
                                        ],
                                        align="center",
                                        className="mb-4",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                create_kpi_card(
                                                    "total-ops-header",
                                                    "total-ops-card",
                                                    "fas fa-exchange-alt",
                                                ),
                                                md=3,
                                            ),
                                            dbc.Col(
                                                create_kpi_card(
                                                    "total-arr-header",
                                                    "total-arr-card",
                                                    "fas fa-plane-arrival",
                                                ),
                                                md=3,
                                            ),
                                            dbc.Col(
                                                create_kpi_card(
                                                    "total-dep-header",
                                                    "total-dep-card",
                                                    "fas fa-plane-departure",
                                                ),
                                                md=3,
                                            ),
                                            dbc.Col(
                                                create_kpi_card(
                                                    "change-header",
                                                    "change-vs-pre-pandemic-card",
                                                    "fas fa-chart-line",
                                                ),
                                                md=3,
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    dbc.Spinner(html.Div(id="summary-charts-content")),
                                ]
                            )
                        )
                    ],
                ),
                dbc.Tab(
                    id="tab-component-model",
                    label="Model Performance",
                    tab_id="tab-model",
                    className="custom-tab",
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label(
                                        id="select_model_label", className="fw-bold"
                                    ),
                                    dcc.Dropdown(
                                        id="model-performance-selector",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in model_artifacts.keys()
                                        ],
                                        value="Ridge Regression",
                                        clearable=False,
                                        className="mb-4",
                                    ),
                                    dbc.Spinner(
                                        html.Div(id="model-performance-content")
                                    ),
                                ]
                            )
                        )
                    ],
                ),
                dbc.Tab(
                    id="tab-component-predict",
                    label="Flight Prediction",
                    tab_id="tab-predict",
                    className="custom-tab",
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader(id="card-header-predict"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            id="label-select-model-predict",
                                                            className="fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="model-predict-selector",
                                                            options=[
                                                                {
                                                                    "label": name,
                                                                    "value": name,
                                                                }
                                                                for name in model_artifacts.keys()
                                                            ],
                                                            value="Ridge Regression",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            id="label-select-airport",
                                                            className="fw-bold",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="airport-dropdown",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            id="label-select-date",
                                                            className="fw-bold",
                                                        ),
                                                        dcc.DatePickerSingle(
                                                            id="date-picker",
                                                            date=datetime.date.today()
                                                            + datetime.timedelta(
                                                                days=7
                                                            ),
                                                            display_format="YYYY-MM-DD",
                                                        ),
                                                    ],
                                                    md=4,
                                                ),
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Button(
                                            id="predict-button",
                                            className="w-100",
                                            n_clicks=0,
                                        ),
                                        dbc.Spinner(
                                            html.Div(
                                                id="prediction-output", className="mt-4"
                                            )
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dbc.Tab(
                    id="tab-component-viz",
                    label="Visualizations",
                    tab_id="tab-viz",
                    className="custom-tab",
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(dbc.Col(dcc.Graph(id="airport-map"))),
                                    html.Hr(className="my-4"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Graph(
                                                    id="flights-by-month-timeseries"
                                                ),
                                                md=6,
                                            ),
                                            dbc.Col(
                                                dcc.Graph(id="flights-by-dayofweek"),
                                                md=6,
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Graph(
                                                    id="flights-by-pandemic-phase"
                                                ),
                                                md=6,
                                            ),
                                            dbc.Col(
                                                dcc.Graph(
                                                    id="flights-by-volume-category"
                                                ),
                                                md=6,
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        )
                    ],
                ),
            ],
            id="tabs",
            active_tab="tab-ops",
            className="custom-tabs-container",
        ),
    ],
)


@app.callback(
    [
        Output("main-title", "children"),
        Output("tab-component-ops", "label"),
        Output("tab-component-model", "label"),
        Output("tab-component-predict", "label"),
        Output("tab-component-viz", "label"),
        Output("summary-card-title", "children"),
        Output("select_model_label", "children"),
        Output("card-header-predict", "children"),
        Output("label-select-model-predict", "children"),
        Output("label-select-airport", "children"),
        Output("label-select-date", "children"),
        Output("predict-button", "children"),
        Output("airport-dropdown", "placeholder"),
    ],
    Input("language-selector", "value"),
)
def update_static_text(lang):
    text = text_content[lang]
    return (
        text["main_title"],
        text["tab_operations_summary"],
        text["tab_model_performance"],
        text["tab_flight_prediction"],
        text["tab_visualizations"],
        text["summary_card_title"],
        text["select_model_label"],
        text["card_header_predict"],
        text["label_select_model_predict"],
        text["label_select_airport"],
        text["label_select_date"],
        text["button_predict"],
        text["placeholder_select_airport"],
    )


@app.callback(Output("language-store", "data"), Input("language-selector", "value"))
def update_language_store(lang):
    return lang


@app.callback(Output("airport-dropdown", "options"), Input("language-store", "data"))
def update_airport_dropdown(lang):
    if not lang:
        return []

    def translate_state(state_name):
        return country_translations[lang].get(state_name, state_name)

    return [
        {
            "label": f"{row['APT_ICAO']} ({translate_state(row['STATE_NAME'])})",
            "value": row["APT_ICAO"],
        }
        for _, row in unique_airport_data_for_dropdown.iterrows()
    ]


@app.callback(
    [
        Output("total-ops-header", "children"),
        Output("total-arr-header", "children"),
        Output("total-dep-header", "children"),
        Output("change-header", "children"),
        Output("total-ops-card", "children"),
        Output("total-arr-card", "children"),
        Output("total-dep-card", "children"),
        Output("change-vs-pre-pandemic-card", "children"),
        Output("summary-charts-content", "children"),
    ],
    [
        Input("summary-date-picker", "start_date"),
        Input("summary-date-picker", "end_date"),
        Input("language-store", "data"),
    ],
)
def update_summary_data(start_date, end_date, lang):
    if not start_date or not end_date or not lang:
        return ["-"] * 8 + [None]
    text = text_content[lang]
    start, end = pd.to_datetime(start_date).tz_localize("UTC"), pd.to_datetime(
        end_date
    ).tz_localize("UTC")
    dff = data_cleaned[
        (data_cleaned["FLT_DATE"] >= start) & (data_cleaned["FLT_DATE"] <= end)
    ].copy()
    if dff.empty:
        return (
            [
                text[k]
                for k in [
                    "total_ops_header",
                    "total_arr_header",
                    "total_dep_header",
                    "change_header",
                ]
            ]
            + ["0"] * 3
            + ["-"]
            + [html.P("No data for selected range.")]
        )

    def format_number(n):
        n = int(n)
        return (
            f"{n/1e6:.1f}M"
            if abs(n) >= 1e6
            else f"{n/1e3:.1f}K" if abs(n) >= 1e3 else str(n)
        )

    total_ops, total_arr, total_dep = (
        dff["FLT_TOT_1"].sum(),
        dff["FLT_ARR_1"].sum(),
        dff["FLT_DEP_1"].sum(),
    )
    pre_pandemic_ref_df = data_cleaned[
        (data_cleaned["FLT_DATE"] >= start.replace(year=2019))
        & (data_cleaned["FLT_DATE"] <= end.replace(year=2019))
    ]
    total_ops_pre = pre_pandemic_ref_df["FLT_TOT_1"].sum()
    change_text, change_class = ("-", "kpi-change-neutral")
    if total_ops_pre > 0:
        change_pct = ((total_ops - total_ops_pre) / total_ops_pre) * 100
        change_text = f"{change_pct:+.1f}%"
        if change_pct > 0:
            change_class = "kpi-change-positive"
        elif change_pct < 0:
            change_class = "kpi-change-negative"
    change_component = html.H2(change_text, className=f"display-4 {change_class}")
    daily_ops_df = (
        dff.groupby(dff["FLT_DATE"].dt.date)[["FLT_ARR_1", "FLT_DEP_1"]]
        .sum()
        .reset_index()
    )
    fig_daily = go.Figure(
        data=[
            go.Bar(
                x=daily_ops_df["FLT_DATE"], y=daily_ops_df["FLT_ARR_1"], name="Arrivals"
            ),
            go.Bar(
                x=daily_ops_df["FLT_DATE"],
                y=daily_ops_df["FLT_DEP_1"],
                name="Departures",
            ),
        ]
    ).update_layout(
        barmode="stack",
        title_text=text["daily_ops_chart_title"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    airport_ops = (
        dff.groupby("APT_ICAO")["FLT_TOT_1"].sum().sort_values(ascending=False).head(10)
    )
    pareto_df = airport_ops.to_frame("Total Flights")
    pareto_df["Cumulative Percentage"] = (
        pareto_df["Total Flights"].cumsum() / total_ops
    ) * 100
    fig_pareto = go.Figure(
        data=[
            go.Bar(
                x=pareto_df.index,
                y=pareto_df["Total Flights"],
                name=text["pareto_yaxis1"],
            ),
            go.Scatter(
                x=pareto_df.index,
                y=pareto_df["Cumulative Percentage"],
                name=text["pareto_yaxis2"],
                mode="lines+markers",
                yaxis="y2",
            ),
        ]
    ).update_layout(
        title_text=text["pareto_chart_title"],
        yaxis2=dict(
            title=text["pareto_yaxis2"],
            overlaying="y",
            side="right",
            range=[0, 105],
            ticksuffix="%",
        ),
        legend=dict(x=0.01, y=0.98),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    country_ops = (
        dff.groupby("STATE_NAME")["FLT_TOT_1"]
        .sum()
        .sort_values(ascending=False)
        .nlargest(15)
    )
    fig_country = px.bar(
        x=country_ops.index, y=country_ops.values, title=text["country_chart_title"]
    ).update_layout(
        xaxis_title=None,
        yaxis_title="Total Flights",
        margin=dict(l=40, r=20, t=60, b=20),
    )
    volume_counts = dff.groupby("airport_volume_category", observed=False)[
        "FLT_TOT_1"
    ].sum()
    fig_volume_pie = (
        px.pie(
            names=volume_counts.index,
            values=volume_counts.values,
            title=text["volume_pie_title"],
            hole=0.4,
        )
        .update_traces(textposition="inside", textinfo="percent+label")
        .update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    )
    dff.loc[:, "day_type"] = np.where(
        dff["is_weekend"] == 1, text["weekend_label"], text["weekday_label"]
    )
    weekend_counts = dff.groupby("day_type")["FLT_TOT_1"].sum()
    fig_weekend_pie = (
        px.pie(
            names=weekend_counts.index,
            values=weekend_counts.values,
            title=text["weekend_pie_title"],
            hole=0.4,
        )
        .update_traces(textposition="inside", textinfo="percent+label")
        .update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    )
    charts_layout = html.Div(
        [
            dbc.Row(
                [dbc.Col(dbc.Card(dcc.Graph(figure=fig_daily)), md=12)],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_pareto)), md=7),
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_country)), md=5),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_volume_pie)), md=6),
                    dbc.Col(dbc.Card(dcc.Graph(figure=fig_weekend_pie)), md=6),
                ]
            ),
        ]
    )
    headers, kpis = [
        text[k]
        for k in [
            "total_ops_header",
            "total_arr_header",
            "total_dep_header",
            "change_header",
        ]
    ], [
        format_number(total_ops),
        format_number(total_arr),
        format_number(total_dep),
        change_component,
    ]
    return headers + kpis + [charts_layout]


@app.callback(
    Output("model-performance-content", "children"),
    [Input("model-performance-selector", "value"), Input("language-store", "data")],
)
def update_model_performance_content(selected_model, lang):
    if not selected_model or not lang:
        return ""
    text, metrics = text_content[lang], model_artifacts[selected_model]["metrics"]

    def create_body(metric_set):
        return [
            html.P(f"R² Score: {metric_set['R2']:.4f}"),
            html.P(f"MAE: {metric_set['MAE']:.4f}"),
            html.P(f"RMSE: {metric_set['RMSE']:.4f}"),
        ]

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(text["card_header_train_log"]),
                            dbc.CardBody(create_body(metrics["train"]["log_scale"])),
                        ]
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(text["card_header_train_orig"]),
                            dbc.CardBody(
                                create_body(metrics["train"]["original_scale"])
                            ),
                        ],
                        className="mt-4",
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(text["card_header_test_log"]),
                            dbc.CardBody(create_body(metrics["test"]["log_scale"])),
                        ]
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(text["card_header_test_orig"]),
                            dbc.CardBody(
                                create_body(metrics["test"]["original_scale"])
                            ),
                        ],
                        className="mt-4",
                    ),
                ],
                md=6,
            ),
        ]
    )


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [
        State("model-predict-selector", "value"),
        State("airport-dropdown", "value"),
        State("date-picker", "date"),
        State("language-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_prediction(n_clicks, model_name, airport_icao, date_str, lang):
    if not lang:
        return ""
    text = text_content[lang]
    if not all([model_name, airport_icao, date_str]):
        return dbc.Alert(text["prediction_error_select"], color="danger")
    try:
        input_df = create_sample_input(airport_icao, date_str)
        model_obj = model_artifacts[model_name]["model"]
        predicted_log = model_obj.predict(input_df)
        predicted_flights = max(0, np.expm1(predicted_log)[0])
        input_date = pd.to_datetime(date_str)
        hist_data = data_artifacts["pre_pandemic_monthly_stats"]
        hist_avg_row = hist_data[
            (hist_data["APT_ICAO"] == airport_icao)
            & (hist_data["MONTH_NUM"] == input_date.month)
        ]
        hist_avg_text = ""
        if not hist_avg_row.empty:
            avg_val, month_name = hist_avg_row["avg_pre_pandemic_monthly_traffic"].iloc[
                0
            ], input_date.strftime("%B")
            hist_avg_text = text["historical_average_template"].format(
                month_name=month_name, avg=avg_val
            )
        return dbc.Alert(
            [
                html.H4(text["prediction_success_title"], className="alert-heading"),
                html.P(
                    text["prediction_output_template"].format(
                        prediction=predicted_flights
                    ),
                    className="mb-0 fs-5",
                ),
                html.Hr(),
                html.P(hist_avg_text, className="mb-0"),
            ],
            color="success",
        )
    except Exception as e:
        print(f"Prediction Error: {e}")
        return dbc.Alert(f"Error during prediction: {e}", color="danger")


@app.callback(Output("airport-map", "figure"), Input("language-store", "data"))
def create_map_figure(lang):
    if not lang:
        return go.Figure()
    text = text_content[lang]
    map_data = data_artifacts["pre_pandemic_overall_stats"].copy()
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
    return fig.update_layout(
        mapbox_style="carto-positron",
        title_text=text["viz_map_title"],
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )


@app.callback(
    Output("flights-by-month-timeseries", "figure"), Input("language-store", "data")
)
def create_month_timeseries_chart(lang):
    if not lang:
        return go.Figure()
    text = text_content[lang]
    df_plot = (
        train_df.groupby(pd.Grouper(key="FLT_DATE", freq="M"))["FLT_TOT_1"]
        .sum()
        .reset_index()
    )
    return px.line(
        df_plot,
        x="FLT_DATE",
        y="FLT_TOT_1",
        title=text["viz_month_timeseries_title"],
        markers=True,
        labels={
            "FLT_DATE": text["viz_month_timeseries_xaxis"],
            "FLT_TOT_1": text["viz_month_timeseries_yaxis"],
        },
    )


@app.callback(Output("flights-by-dayofweek", "figure"), Input("language-store", "data"))
def create_dayofweek_bar_chart(lang):
    if not lang:
        return go.Figure()
    text = text_content[lang]
    df_plot = train_df.groupby("day_of_week")["FLT_TOT_1_log"].mean().reset_index()
    fig = px.bar(
        df_plot,
        x="day_of_week",
        y="FLT_TOT_1_log",
        title=text["viz_dayofweek_title"],
        labels={
            "day_of_week": text["viz_dayofweek_xaxis"],
            "FLT_TOT_1_log": text["viz_dayofweek_yaxis"],
        },
    )
    return fig.update_xaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )


@app.callback(
    Output("flights-by-pandemic-phase", "figure"), Input("language-store", "data")
)
def create_pandemic_phase_bar_chart(lang):
    if not lang:
        return go.Figure()
    text = text_content[lang]
    df_plot = (
        train_df.groupby("pandemic_phase", observed=True)["FLT_TOT_1_log"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        df_plot,
        x="pandemic_phase",
        y="FLT_TOT_1_log",
        title=text["viz_pandemic_phase_title"],
        labels={
            "pandemic_phase": text["viz_pandemic_phase_xaxis"],
            "FLT_TOT_1_log": text["viz_pandemic_phase_yaxis"],
        },
        category_orders={"pandemic_phase": data_artifacts["pandemic_phase_order"]},
    )
    return fig.update_xaxes(tickangle=45)


@app.callback(
    Output("flights-by-volume-category", "figure"), Input("language-store", "data")
)
def create_volume_category_bar_chart(lang):
    if not lang:
        return go.Figure()
    text = text_content[lang]
    df_plot = (
        train_df.groupby("airport_volume_category", observed=False)["FLT_TOT_1_log"]
        .mean()
        .reset_index()
    )
    return px.bar(
        df_plot,
        x="airport_volume_category",
        y="FLT_TOT_1_log",
        title=text["viz_volume_category_title"],
        labels={
            "airport_volume_category": text["viz_volume_category_xaxis"],
            "FLT_TOT_1_log": text["viz_volume_category_yaxis"],
        },
        category_orders={
            "airport_volume_category": data_artifacts["airport_volume_labels"]
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
