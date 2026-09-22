
import holidays
import numpy as np
import streamlit as st
import joblib
import pandas as pd

import data_analysis_prediction
import data_cleaning_validation

# 1. Load the pre-trained assets into memory on server start
model = joblib.load('campsite_prediction/camping_predictor_model.pkl')
park_tier_map = pd.read_pickle('campsite_prediction/park_scale_tier_map.pkl')


df = pd.read_csv('campsite_prediction/Natural_Resources_Camping_Parks_Reservation_Data.csv')
country_count = df.groupby(['Country'], as_index=False).agg(total_count=('Country', 'count')).sort_values(
        by=['total_count'], ascending=False)


# 1. Run your existing validation step exactly as it was
daily_df_raw = data_cleaning_validation.data_cleaning(df[df['Country'] == 'Canada'].copy())

# =====================================================================
# 🚨 CONVERSION PATCH (Fixes scale alignment using existing columns)
# =====================================================================
# Reverse-engineer the log values to get back clean raw person counts
daily_df_raw['PartyTotal_Raw_Est'] = np.expm1(daily_df_raw['PartyTotal_Log'])

# Condense down into true single daily summaries per park
daily_df = daily_df_raw.groupby(['Park', 'arrival_year', 'arrival_month', 'arrival_day'], as_index=False).agg({
    'PartyTotal_Raw_Est': 'sum',  # Sum raw counts to match model aggregates
    'day_of_week': 'first',
    'is_weekend': 'first',
    'is_holiday': 'first',
    'is_holiday_adjacent': 'first',
    'Region_Eastern': 'first',
    'Region_Western': 'first',
    'ArrivalDate': 'first'
})

# Re-calculate the clean log target variable on the aggregated totals
daily_df['PartyTotal_Log'] = np.log1p(daily_df['PartyTotal_Raw_Est'])
daily_df.drop(columns=['PartyTotal_Raw_Est'], inplace=True)

# Synchronize exact calendar types for lookup tracking
daily_df['ArrivalDate'] = pd.to_datetime(daily_df['ArrivalDate'])
daily_df['arrival_year'] = daily_df['ArrivalDate'].dt.year
daily_df['arrival_month'] = daily_df['ArrivalDate'].dt.month
daily_df['arrival_day'] = daily_df['ArrivalDate'].dt.day
daily_df['day_of_week'] = daily_df['ArrivalDate'].dt.dayofweek

# =====================================================================
# 2. BUILD PRODUCTION LOOKUPS ON PERFECT MATH SCALES
# =====================================================================
GLOBAL_LOOKUP = daily_df.set_index(['Park', 'arrival_year', 'arrival_month', 'arrival_day'])['PartyTotal_Log'].to_dict()
GLOBAL_BASELINE = daily_df.groupby(['Park', 'arrival_month', 'day_of_week'])['PartyTotal_Log'].median().to_dict()
PARK_TIER_MAP = daily_df.groupby('Park')['PartyTotal_Log'].median().to_dict()


def predict_traffic(target_date, park, ns_holidays):
    # Placeholder function for predicting traffic based on the date
    # In a real application, this would involve loading a trained model and making predictions
    print(f"Predicting traffic for park: {park} on date: {target_date}")

    target_date = pd.to_datetime(target_date)
    arrival_day = target_date.day
    arrival_month = target_date.month
    arrival_year = target_date.year
    day_of_week = target_date.dayofweek
    is_weekend = 1 if day_of_week in [4, 5] else 0

    # 2. Holiday flags
    is_holiday = 1 if target_date in ns_holidays else 0
    is_holiday_adjacent = 1 if (
            (target_date + pd.Timedelta(days=1)) in ns_holidays or
            (target_date - pd.Timedelta(days=1)) in ns_holidays
    ) else 0

    region_eastern = 1 if "Eastern" in park else 0  # Adjust according to your dataset rules
    region_western = 1 if "Western" in park else 0

    # 4. Resolve static park scale tier
    park_scale_tier = PARK_TIER_MAP.get(park, 0.0)

    # 5. Math-based 2-year lookback extraction
    date_2_years_ago = target_date - pd.DateOffset(years=2)
    y_2y, m_2y, d_2y = date_2_years_ago.year, date_2_years_ago.month, date_2_years_ago.day

    # Attempt to grab exact historical log value; fall back to the multi-year DOW median
    people_two_years_ago = GLOBAL_LOOKUP.get(
        (park, y_2y, m_2y, d_2y),
        GLOBAL_BASELINE.get((park, arrival_month, day_of_week), park_scale_tier)
    )

    # 6. Construct input payload matching the 11-column training matrix exactly
    features = pd.DataFrame([{
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_day': arrival_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_holiday_adjacent': is_holiday_adjacent,
        'Region_Eastern': region_eastern,
        'Region_Western': region_western,
        'people_two_years_ago': people_two_years_ago,
        'park_scale_tier': park_scale_tier
    }])

    # 7. Predict log target and handle potential scalar array wrapping
    prediction_log = model.predict(features)




    if hasattr(prediction_log, "__len__") and len(prediction_log) > 0:
        pred_val = float(prediction_log[0])
    else:
        pred_val = float(prediction_log)

    # 8. Inverse Log Transform (Transforms log prediction back into true person count)
    real_attendance = np.expm1(pred_val)

    return max(0.0, round(real_attendance, 2))


if __name__ == "__main__":

    # Remove all records which does not have Canada as Country
    df_canada = df[df['Country'] == 'Canada'].copy()
    df_canada['ArrivalDate'] = pd.to_datetime(df_canada['ArrivalDate'])

    ns_holidays = holidays.Canada(subdiv='NS', years=df_canada['ArrivalDate'].dt.year.unique())

    st.title('App to predict the number of people coming to a campsite in Canada on a specific day')

    park_names = list(df_canada['Park'].unique())

    # 1. Use st.form to hold your inputs and handle the submit cleanly
    with st.form("prediction_form"):

        # 2. Add keys if you want to use session state, or save them directly to variables
        chosen_date = st.date_input("Pick a date", value="today", key="camp_date")
        park_name = st.selectbox('select a park from the list', options=park_names, key="camp_name_option")

        # 3. Use the form submit button instead of a regular button
        submitted = st.form_submit_button("Submit")

    # 4. Handle execution AFTER the submit block runs
    if submitted:
        if not park_name.strip():
            st.warning("Please enter a valid park name.")
        else:
            with st.spinner(f"Running ML model for {park_name}..."):
                # Call your function using the local variables or st.session_state
                prediction = predict_traffic(chosen_date, park_name, ns_holidays)

            st.success(f"📈 Predicted attendance: **{prediction} people**")
