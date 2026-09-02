import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Section spacing */
    h1 {
        margin-bottom: 0.5rem;
    }

    h2 {
        margin-top: 2rem;
    }

    h3 {
        margin-top: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #30363d;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/raw/train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df


df = load_data()


# ============================================================
# CREATE DAILY SALES TIME SERIES
# ============================================================

# Create daily sales series
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

daily_sales = (
    df.groupby("Order Date")["Sales"]
      .sum()
      .sort_index()
)

# Create a complete daily date range
all_dates = pd.date_range(
    start=df["Order Date"].min(),
    end=df["Order Date"].max(),
    freq="D"
)

# Fill missing dates with zero sales
daily_sales = daily_sales.reindex(
    all_dates,
    fill_value=0
)

daily_sales.index.name = "Order Date"

# ============================================================
# HEADER
# ============================================================

st.title("📈 Sales Forecasting Dashboard")

st.markdown(
    """
    **Time-Series Sales Analysis & Forecasting**

    An interactive forecasting application built using historical
    sales data and a seasonal ARIMA model to support
    data-driven sales planning.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📊 Analysis Type")
    st.write("Time-Series Forecasting")

with col2:
    st.caption("🤖 Final Model")
    st.write("SARIMA(0,0,1)(1,0,1,7)")

with col3:
    st.caption("📅 Forecast Frequency")
    st.write("Daily")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=30,
    max_value=90,
    value=90,
    step=30
)

st.sidebar.markdown("---")

st.sidebar.write("### Forecasting Model")

st.sidebar.markdown(
    """
    **SARIMA(0,0,1)(1,0,1,7)**

    - Non-seasonal order: (0,0,1)
    - Seasonal order: (1,0,1,7)
    - Seasonal period: 7 days
    - Forecast frequency: Daily
    """
)


# ============================================================
# HISTORICAL SALES OVERVIEW
# ============================================================

st.header("Historical Sales Overview")


total_sales = df["Sales"].sum()

average_transaction = df["Sales"].mean()

total_transactions = len(df)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Historical Sales",
        f"${total_sales:,.2f}"
    )


with col2:

    st.metric(
        "Average Transaction Sales",
        f"${average_transaction:,.2f}"
    )


with col3:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


# ============================================================
# HISTORICAL SALES CHART
# ============================================================

st.subheader("📈 Historical Daily Sales")

historical_chart = pd.DataFrame({
    "Daily Sales": daily_sales,
    "30-Day Rolling Average": daily_sales.rolling(30).mean()
})

st.line_chart(historical_chart)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

st.write(
    "The final forecasting model was selected after comparing "
    "multiple baseline and statistical forecasting approaches."
)

performance_data = pd.DataFrame({
    "Model": [
        "Naive",
        "Seasonal Naive",
        "ARIMA(1,0,0)",
        "ARIMA(1,0,1)",
        "SARIMA(0,0,1)(1,0,1,7)"
    ],
    "MAE": [
        2366.75,
        2140.04,
        1703.62,
        1700.68,
        1549.26
    ],
    "RMSE": [
        3382.21,
        3300.74,
        2632.48,
        2632.61,
        2401.91
    ]
})

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)

st.subheader("Walk-Forward Validation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Validation Folds",
        "12"
    )

with col2:
    st.metric(
        "Average Validation MAE",
        "1,439.41"
    )

with col3:
    st.metric(
        "Average Validation RMSE",
        "2,062.19"
    )

st.info(
    "Walk-forward validation evaluates the forecasting model "
    "across multiple historical periods, providing a more "
    "robust assessment than relying on a single train-test split."
)    

# ============================================================
# FORECAST GENERATION
# ============================================================

st.markdown("---")

st.header("🔮 Future Sales Forecast")

st.write(
    "Generate future sales estimates using the validated "
    "SARIMA forecasting model."
)


st.write(
    f"The selected model will forecast the next "
    f"**{forecast_days} days** using the complete "
    "historical daily sales series."
)

st.caption(
    "Choose the number of future days you want the model to forecast."
)

if st.button(
    "🔮 Generate Forecast",
    type="primary",
    use_container_width=False
):


    with st.spinner("Training SARIMA model and generating forecast..."):

        # ----------------------------------------------------
        # FINAL SARIMA MODEL
        # ----------------------------------------------------

        model = SARIMAX(
            daily_sales,
            order=(0, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)


        # ----------------------------------------------------
        # GENERATE FORECAST
        # ----------------------------------------------------

        forecast_result = model_fit.get_forecast(
            steps=forecast_days
        )

        forecast = forecast_result.predicted_mean
        confidence = forecast_result.conf_int()
        

        # Create a clean future date index starting after
        # the final historical observation
        forecast_dates = pd.date_range(
            start=daily_sales.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )


        # ----------------------------------------------------
        # CREATE FORECAST DATAFRAME
        # ----------------------------------------------------

        forecast_df = pd.DataFrame({

            "Date": forecast_dates,

            "Forecast": forecast.values,

            "Lower Bound": confidence.iloc[:, 0].values,

            "Upper Bound": confidence.iloc[:, 1].values

        })

        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

        # ----------------------------------------------------
        # SAVE FORECAST IN SESSION
        # ----------------------------------------------------

        st.session_state["forecast_df"] = forecast_df


        st.success(
            f"{forecast_days}-day forecast generated successfully."
        )


# ============================================================
# DISPLAY FORECAST
# ============================================================

if "forecast_df" in st.session_state:

    st.markdown("---")    

    forecast_df = st.session_state["forecast_df"]


    # ========================================================
    # FORECAST KPIs
    # ========================================================

    st.subheader("📌 Forecast Summary")


    total_forecast = forecast_df["Forecast"].sum()

    average_forecast = forecast_df["Forecast"].mean()

    highest_forecast = forecast_df["Forecast"].max()

    lowest_forecast = forecast_df["Forecast"].min()


    highest_date = forecast_df.loc[
        forecast_df["Forecast"].idxmax(),
        "Date"
    ]

    lowest_date = forecast_df.loc[
        forecast_df["Forecast"].idxmin(),
        "Date"
    ]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Expected Total Sales",
            f"${total_forecast:,.2f}"
        )


    with col2:

        st.metric(
            "Average Daily Forecast",
            f"${average_forecast:,.2f}"
        )


    with col3:

        st.metric(
            "Highest Forecast",
            f"${highest_forecast:,.2f}"
        )


    with col4:

        st.metric(
            "Lowest Forecast",
            f"${lowest_forecast:,.2f}"
        )


    # ========================================================
    # FORECAST VISUALIZATION
    # ========================================================

    st.subheader("📈 Historical Sales vs Future Forecast")

    # ------------------------------------------------------------
    # Prepare dates consistently for plotting
    # ------------------------------------------------------------

    historical_dates = pd.to_datetime(daily_sales.index)
    forecast_dates = pd.to_datetime(forecast_df["Date"])

    # Show the most recent 180 days of historical sales
    # so the future forecast is easier to compare.
    recent_history = daily_sales.tail(180)
    recent_history_dates = pd.to_datetime(recent_history.index)

    # ------------------------------------------------------------
    # Create chart
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical sales
    ax.plot(
        recent_history_dates,
        recent_history.values,
        label="Historical Sales",
        linewidth=1.5
    )

    # Forecast
    ax.plot(
        forecast_dates,
        forecast_df["Forecast"].values,
        label="Forecast",
        linewidth=2
    )

    # 95% prediction interval
    ax.fill_between(
        forecast_dates,
        forecast_df["Lower Bound"].values,
        forecast_df["Upper Bound"].values,
        alpha=0.2,
        label="95% Prediction Interval"
    )

    # Forecast start
    forecast_start = historical_dates[-1]

    ax.axvline(
        forecast_start,
        linestyle="--",
        label="Forecast Start"
    )

    # Titles and labels
    ax.set_title(
        "Historical Sales and Future Sales Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Make sure Matplotlib formats the dates properly
    fig.autofmt_xdate()

    plt.tight_layout()

    st.pyplot(fig)


    # ========================================================
    # FORECAST DATA
    # ========================================================

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )


    # ========================================================
    # MONTHLY FORECAST
    # ========================================================

    st.subheader("📊 Monthly Forecast")

    monthly_forecast = (
        forecast_df
        .assign(
            Month=forecast_df["Date"].dt.strftime("%b %Y")
        )
        .groupby("Month", sort=False)["Forecast"]
        .sum()
    )

    st.bar_chart(
        monthly_forecast,
        use_container_width=True
    )

    # ========================================================
    # WEEKDAY FORECAST
    # ========================================================

    st.subheader("📅 Forecast by Day of Week")

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_forecast = (
        forecast_df
        .assign(
            DayOfWeek=forecast_df["Date"].dt.day_name()
        )
        .groupby("DayOfWeek")["Forecast"]
        .mean()
        .reindex(weekday_order)
    )

    weekday_chart = pd.DataFrame({
        "Day": weekday_forecast.index,
        "Average Forecast": weekday_forecast.values
    })

    st.bar_chart(
        weekday_chart.set_index("Day"),
        use_container_width=True
    )


    # ========================================================
    # FORECAST INSIGHTS
    # ========================================================

    st.subheader("Forecast Insights")


    highest_weekday = weekday_forecast.idxmax()

    lowest_weekday = weekday_forecast.idxmin()


    st.write(
        f"• **{highest_weekday}** has the highest average "
        f"forecasted daily sales at approximately "
        f"**${weekday_forecast.max():,.2f}**."
    )


    st.write(
        f"• **{lowest_weekday}** has the lowest average "
        f"forecasted daily sales at approximately "
        f"**${weekday_forecast.min():,.2f}**."
    )


    st.write(
        f"• The model predicts approximately "
        f"**${total_forecast:,.2f}** in total sales "
        f"over the selected {forecast_days}-day horizon."
    )


    st.write(
        "• The prediction interval represents uncertainty "
        "around individual daily forecasts, so the results "
        "should be interpreted as planning estimates rather "
        "than exact future sales values."
    )


    # ========================================================
    # DOWNLOAD FORECAST
    # ========================================================

    st.subheader("Download Forecast")


    csv = forecast_df.to_csv(index=False)


    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name=f"sales_forecast_{forecast_days}_days.csv",
        mime="text/csv"
    )

st.markdown("---")

st.header("Methodology")

st.write(
    """
    The forecasting workflow begins by aggregating transaction-level
    sales into a daily time series. Exploratory analysis is then used
    to investigate trends, temporal patterns, stationarity, and
    weekly seasonality.

    Several forecasting approaches were evaluated, including naive
    and seasonal-naive baselines, ARIMA models, and SARIMA models.

    The selected SARIMA(0,0,1)(1,0,1,7) model incorporates a weekly
    seasonal cycle of seven days and was selected based on holdout
    performance and walk-forward validation.

    The final model is fitted using the available historical daily
    sales data and generates forecasts for the selected future horizon.
    """
)

st.caption(
    "Note: Forecasts represent statistical estimates based on "
    "historical sales patterns and should be used as planning "
    "guidance rather than exact future sales values."
)

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">

    **Sales Forecasting using Time-Series Analysis**

    Built with Python • Pandas • Statsmodels • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
