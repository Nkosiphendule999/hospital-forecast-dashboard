

import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

st.set_option('client.showErrorDetails', True)
# Set page configuration
st.set_page_config(
    page_title="Hospital Forecast Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏥 Hospital Visits Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Forecasting", "🔮 Future Predictions"])

# Helper function to load data
@st.cache_data
def read_csv_from_zip(zpath, filename):
    try:
        with zipfile.ZipFile(zpath) as z:
            with z.open(filename) as f:
                return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return pd.DataFrame()

# Function to generate sample data if no zip file is available
def generate_sample_data():
    # Generate sample dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample appointments data
    appointments_data = {
        'appointment_id': range(1, 1001),
        'patient_id': np.random.randint(1, 201, 1000),
        'doctor_id': np.random.randint(1, 11, 1000),
        'appointment_date': np.random.choice(dates, 1000),
        'status': np.random.choice(['Completed', 'Cancelled', 'No Show', 'Scheduled'], 1000, p=[0.7, 0.1, 0.1, 0.1])
    }
    appointments = pd.DataFrame(appointments_data)
    
    # Generate sample doctors data
    doctors_data = {
        'doctor_id': range(1, 11),
        'name': [f'Dr. {name}' for name in ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 
                                          'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']],
        'specialty': np.random.choice(['Cardiology', 'Pediatrics', 'Orthopedics', 'Neurology', 'Dermatology'], 10)
    }
    doctors = pd.DataFrame(doctors_data)
    
    # Generate sample patients data
    patients_data = {
        'patient_id': range(1, 201),
        'name': [f'Patient {i}' for i in range(1, 201)],
        'age': np.random.randint(18, 80, 200),
        'gender': np.random.choice(['Male', 'Female'], 200)
    }
    patients = pd.DataFrame(patients_data)
    
    return appointments, doctors, patients

# Load data
ZIP_PATH = "archive (3).zip"  # Update this path to your zip file location

with tab1:
    st.header("Data Overview")
    
    if not os.path.exists(ZIP_PATH):
        st.warning(f"Zip file not found at {ZIP_PATH}. Using sample data for demonstration.")
        appointments, doctors, patients = generate_sample_data()
    else:
        try:
            appointments = read_csv_from_zip(ZIP_PATH, "appointments.csv")
            doctors = read_csv_from_zip(ZIP_PATH, "doctors.csv")
            patients = read_csv_from_zip(ZIP_PATH, "patients.csv")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Using sample data instead.")
            appointments, doctors, patients = generate_sample_data()
    
    st.subheader("Appointments Data")
    st.dataframe(appointments.head())
    
    st.subheader("Doctors Data")
    st.dataframe(doctors.head())
    
    st.subheader("Patients Data")
    st.dataframe(patients.head())
    
    # Preprocessing
    appointments['appointment_date'] = pd.to_datetime(appointments['appointment_date'], errors='coerce')
    appointments['is_completed'] = appointments['status'].astype(str).str.lower().eq('completed').astype(int)

    # Aggregate: completed appointments per day
    daily = (appointments.groupby('appointment_date')['is_completed']
             .sum()
             .rename('demand')
             .reset_index()
             .rename(columns={'appointment_date':'date'}))

    # Create a continuous daily index from min to max date and fill missing days with 0 demand
    date_index = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(date_index).fillna(0.0).rename_axis('date').reset_index()

    st.subheader("Daily Completed Visits")
    st.line_chart(daily.set_index("date")["demand"])
    
    # Show some statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Appointments", len(appointments))
    col2.metric("Completed Appointments", appointments['is_completed'].sum())
    col3.metric("Completion Rate", f"{appointments['is_completed'].mean()*100:.1f}%")

with tab2:
    st.header("Forecasting Model")
    
    # Feature engineering
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    daily['day'] = daily['date'].dt.day
    daily['dow'] = daily['date'].dt.dayofweek
    daily['is_month_start'] = daily['date'].dt.is_month_start.astype(int)
    daily['is_month_end'] = daily['date'].dt.is_month_end.astype(int)

    # Create lag features and rolling means
    for lag in [1, 7, 14]:
        daily[f'lag_{lag}'] = daily['demand'].shift(lag)

    daily['roll7_mean'] = daily['demand'].rolling(7).mean()
    daily['roll14_mean'] = daily['demand'].rolling(14).mean()

    # Removing rows with NaN introduced by lags/rolling
    daily_model = daily.dropna().reset_index(drop=True)

    # Feature columns
    feature_cols = [
        'year','month','day','dow','is_month_start','is_month_end',
        'lag_1','lag_7','lag_14','roll7_mean','roll14_mean'
    ]

    X = daily_model[feature_cols]
    y = daily_model['demand']

    # Split: first 80% train, last 20% test (preserve time order)
    split_idx = int(len(daily_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    dates_test = daily_model['date'].iloc[split_idx:].copy()

    # Helper metric (MAPE)
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if not mask.any():
            return np.nan
        return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100)

    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ("Random Forest", "Linear Regression")
    )
    
    if model_option == "Random Forest":
        n_estimators = st.slider("Number of Trees", 100, 1000, 500, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    else:
        model = LinearRegression()
        
    # Train model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mp = mape(y_test, preds)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAPE", f"{mp:.2f}%")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_test, y_test.values, label='Actual', marker='o', markersize=3)
    ax.plot(dates_test, preds, label='Predicted', linestyle='--')
    ax.set_title("Actual vs Predicted Visits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Completed visits")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance for Random Forest
    if model_option == "Random Forest":
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(feature_importance['feature'], feature_importance['importance'])
        ax2.set_title('Feature Importance')
        ax2.set_xlabel('Importance')
        plt.tight_layout()
        st.pyplot(fig2)

with tab3:
    st.header("Future Predictions")
    
    # Function to forecast future days
    def forecast_future_days(daily_full_df, model, horizon=14):
        df = daily_full_df.copy().set_index('date').sort_index()
        last_date = df.index.max()
        preds = []

        # Extend index for horizon days
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1),
                                   last_date + pd.Timedelta(days=horizon),
                                   freq='D')

        for next_date in future_idx:
            # Create a temporary DF including next_date
            idx = pd.date_range(df.index.min(), next_date, freq='D')
            tmp = df.reindex(idx)

            # Features
            tmp['year'] = tmp.index.year
            tmp['month'] = tmp.index.month
            tmp['day'] = tmp.index.day
            tmp['dow'] = tmp.index.dayofweek
            tmp['is_month_start'] = tmp.index.is_month_start.astype(int)
            tmp['is_month_end'] = tmp.index.is_month_end.astype(int)

            tmp['lag_1'] = tmp['demand'].shift(1)
            tmp['lag_7'] = tmp['demand'].shift(7)
            tmp['lag_14'] = tmp['demand'].shift(14)
            tmp['roll7_mean'] = tmp['demand'].rolling(7).mean()
            tmp['roll14_mean'] = tmp['demand'].rolling(14).mean()

            # Get row for next_date
            feat_row = tmp.loc[next_date, [
                'year','month','day','dow','is_month_start','is_month_end',
                'lag_1','lag_7','lag_14','roll7_mean','roll14_mean'
            ]].values.reshape(1,-1)

            # Predict
            pred = model.predict(feat_row)[0]

            # Store and update demand in df
            preds.append((next_date, pred))
            df.loc[next_date, 'demand'] = pred

        return pd.DataFrame(preds, columns=['date','predicted_demand'])
    
    # Get forecast horizon from user
    horizon = st.slider("Forecast Horizon (days)", 7, 30, 14)
    
    # Train a model on all available data for forecasting
    forecast_model = RandomForestRegressor(n_estimators=500, random_state=42)
    forecast_model.fit(X, y)
    
    # Generate forecast
    future_preds = forecast_future_days(daily, forecast_model, horizon=horizon)
    
    # Display forecast
    st.subheader(f"Forecast for Next {horizon} Days")
    st.dataframe(future_preds)
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily['date'], daily['demand'], label='Historical Data')
    ax.plot(future_preds['date'], future_preds['predicted_demand'],
            label='Forecast', marker='o', linestyle='--')
    ax.set_title(f"Forecast of Next {horizon} Days of Completed Visits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Completed visits (demand)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show forecast statistics
    avg_forecast = future_preds['predicted_demand'].mean()
    max_forecast = future_preds['predicted_demand'].max()
    min_forecast = future_preds['predicted_demand'].min()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Forecast", f"{avg_forecast:.1f}")
    col2.metric("Maximum Forecast", f"{max_forecast:.1f}")
    col3.metric("Minimum Forecast", f"{min_forecast:.1f}")


# Footer
st.markdown("---")
st.markdown("### 🏥 Hospital Forecast Dashboard | Built with Streamlit")
