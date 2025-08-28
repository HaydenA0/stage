 
# **Feature Engineering**

#### 1. Basic Time-Based Features
*   **Features:** `YEAR`, `MONTH_NUM`, `day_of_month`, `day_of_week`, `day_of_year`, `quarter`.
*   **Purpose:** To break down the date into components the model can use to learn trends and seasonality.
*   **How:** `data['day_of_week'] = data['FLT_DATE'].dt.dayofweek`

#### 2. Binary Flag Feature
*   **Feature:** `is_weekend` (1 for weekend, 0 for weekday).
*   **Purpose:** To explicitly signal the weekday vs. weekend pattern to the model.
*   **How:** `data['is_weekend'] = (data['FLT_DATE'].dt.dayofweek >= 5).astype(int)`

#### 3. Cyclical Feature Encoding (Sin/Cos)
*   **Features:** `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos`.
*   **Purpose:** To represent cyclical time data (like months) on a circle, removing artificial "cliffs" (e.g., Dec being far from Jan).
*   **How:** `data['month_sin'] = np.sin(2 * np.pi * data['MONTH_NUM'] / 12)`

#### 4. Historical Aggregate Features
*   **Features:** `avg_pre_pandemic_monthly_traffic`, `avg_traffic_pre_pandemic_overall`.
*   **Purpose:** To give the model a historical baseline of an airport's typical traffic volume.
*   **How:** `avg_traffic = pre_pandemic_data.groupby('APT_ICAO')['FLT_TOT_1'].mean()`

#### 5. Discretization (Binning)
*   **Feature:** `airport_volume_category` ("Small", "Medium", "Large", "Very Large").
*   **Purpose:** To group airports into high-level categories, helping the model generalize patterns across airports of similar sizes.
*   **How:** `data['category'] = pd.cut(data['avg_traffic'], bins=..., labels=...)`

#### 6. Custom Domain-Specific Feature
*   **Feature:** `pandemic_phase` ("Pre-Pandemic", "Peak Drop", "Recovery", etc.).
*   **Purpose:** To explicitly inform the model about the major real-world event (COVID-19) driving massive changes in the data.
*   **How:** Custom function with `if/elif` logic based on `YEAR` and `MONTH_NUM`.

Of course. Here is a quick markdown summary of the models used in the project.


# **Modeling**

#### Overall Strategy

*   **Target Transformation:** Models predict `log1p(flights)` to handle skewed data. The final output is converted back to flight counts using `expm1`.
*   **Preprocessing:** A `ColumnTransformer` pipeline is used to `StandardScale` numerical features and `OneHotEncode` categorical features.
*   **Validation:** Data is split chronologically (train on past, test on future), which is the correct method for time-series forecasting.


### 1. Ridge Regression

*   **Type:**
    *   Regularized Linear Model.
*   **Role in Project:**
    *   Acts as a strong, fast, and robust **baseline model**.
*   **Key Characteristic:**
    *   The L2 regularization (`alpha=1.0`) prevents the model from overfitting, which is especially important when one-hot encoding creates hundreds of new features.


### 2. LightGBM (Light Gradient Boosting Machine)

*   **Type:**
    *   Advanced Gradient Boosted Trees.
*   **Role in Project:**
    *   The **high-performance model**, designed to capture complex, non-linear relationships in the data that the linear Ridge model cannot.
*   **Key Characteristics:**
    *   **Tuned Hyperparameters:** Uses specific parameters (e.g., `learning_rate`, `num_leaves`) for better accuracy.
    *   **Early Stopping:** Training automatically stops when performance on a validation set stops improving, finding the optimal number of trees and preventing overfitting.

Of course. Here is a quick markdown summary of the application.

---

# **Application**

#### Core Technology
*   **Framework:** **Dash** (a Python framework for building analytical web apps).
*   **Plotting:** **Plotly** for creating interactive charts and maps.
*   **Styling:** **Dash Bootstrap Components** 

---

#### Main Purpose
To provide an interactive web interface that allows users to **explore historical flight data** and **use the pre-trained models to predict future flight traffic**.

---

### Key Features (Organized by Tab)

*   #### **Operations Summary Tab**
    *   **KPIs:** Shows key metrics like Total Flights, Arrivals, and Departures for a selected date range.
    *   **Charts:** Interactive charts showing daily operations, top airports (Pareto), and breakdowns by country and airport size.

*   #### **Model Performance Tab**
    *   **Transparency:** Displays the performance metrics (R², MAE, RMSE) for each trained model.
    *   **Comparison:** Allows users to see how well each model performed on both the training and test data.

*   #### **Flight Prediction Tab**
    *   **Core Tool:** The main predictive feature of the application.
    *   **Interaction:** User selects an airport, a date, and a model.
    *   **Output:** The app calls the selected model to forecast the total number of flights and displays the result.

*   #### **Visualizations Tab**
    *   **Data Exploration:** Provides deeper insights into the relationships within the data.
    *   **Interactive Map:** A key visual showing airport locations and their pre-pandemic traffic volume.
    *   **Box Plots:** Shows how flight volume varies by day of the week, pandemic phase, etc.



