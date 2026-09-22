import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


import data_analysis_prediction
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, root_mean_squared_error


import holidays

import warnings
# Suppress the specific loky physical cores warning
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")



def check_remove_outliers(df, cols):
    # It seems from the below outlier check that it consist of almost 27% of whole dataset, so we cannot remove it.
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3-q1
        upper_limit = q3 + (1.5 * iqr)
        lower_limit = q1 - (1.5 * iqr)

        outlier_index = df[(df[c] < lower_limit) | (df[c] > upper_limit)].index
        print((len(outlier_index)))

def data_cleaning(df):

    # clean values in column 'BookingCategory'
    booking_category_map = {'GroupCampsite': 'Campsite', 'Group Campsite': 'Campsite'}
    df['BookingCategory'] = df['BookingCategory'].replace(booking_category_map)
    # we will just focus on 'campsite' booking category and drop other records
    category_index_to_drop = df[(df['BookingCategory'] == 'Hiking') | (df['BookingCategory'] == 'Kayaking') |
                                (df['BookingCategory'] == 'Yurt')].index

    df.drop(index=category_index_to_drop, axis=0, inplace=True)
    # as we have only 1 value as Campsite, we can drop column as it does not give much information
    df.drop(['BookingCategory'], axis=1, inplace=True)

    # convert date into datatime
    df['CreateDate'] = pd.to_datetime(df['CreateDate'])
    df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])

    df['create_year'] = df['CreateDate'].dt.year
    df['create_month'] = df['CreateDate'].dt.month
    df['create_day'] = df['CreateDate'].dt.day


    df['arrival_year'] = df['ArrivalDate'].dt.year
    df['arrival_month'] = df['ArrivalDate'].dt.month
    df['arrival_day'] = df['ArrivalDate'].dt.day

    df['day_of_week'] = df['ArrivalDate'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5]).astype(int)

    ns_holidays = holidays.Canada(subdiv='NS', years=df['ArrivalDate'].dt.year.unique())
    df['is_holiday'] = df['ArrivalDate'].apply(lambda x: 1 if x in ns_holidays else 0)

    # Check if a day is adjacent to a holiday weekend
    df['is_holiday_adjacent'] = df['ArrivalDate'].apply(
        lambda x: 1 if (x + pd.Timedelta(days=1) in ns_holidays or x - pd.Timedelta(days=1) in ns_holidays) else 0
    )

    df.drop(['CreateDate', 'CreateDayOfWeek', 'ArrivalMonth', 'ArrivalYear', 'ArrivalDayOfWeek',
                    'DepartureDate'], axis=1, inplace=True)

    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    df['DepartDayOfWeek'] = df['DepartDayOfWeek'].map(day_map)

    # drop unnecessary columns
    df.drop(['SiteName', 'City', 'County', 'Country', 'Origin'], axis=1, inplace=True)

    # covert Children column to int
    df.fillna({'Children': 0}, inplace=True)
    df['Children'] = df['Children'].astype(int)

    # Replace Province values to make it clean
    df['Province'] = df['Province'].replace('UNKNOWN', 'Unknown')

    # check outliers
    outlier_col = ['LengthOfStay(Nights)', 'PartyTotal', 'Adults', 'Children']
    fig, ax = plt.subplots(2,2, figsize=(14,6))
    ax = ax.flatten()

    for i, col in enumerate(outlier_col):
        ax[i].boxplot(x=df[col])
        ax[i].set_xlabel(col)
        ax[i].set_ylabel('Values')

    plt.suptitle('Outlier Check before log transformation')
    plt.tight_layout()
    plt.show()

    check_remove_outliers(df, outlier_col)

    fig, ax = plt.subplots(2,2, figsize=(14,6))
    ax = ax.flatten()

    for i, col in enumerate(outlier_col):
        sns.histplot(data=df, x=col, ax=ax[i], bins=500)
        ax[i].set_xlabel('Values')
        ax[i].set_ylabel('Frequency')
        ax[i].set_title(col)

    plt.suptitle('Data distribution before log transformation')
    plt.tight_layout()
    plt.show()


    # 1. Fill missing values (Crucial: ML models break on NaN)
    df["Children"] = df["Children"].fillna(0)

    # 2. Cap extreme outliers (Winsorization)
    # This prevents extreme values from warping the model's weights
    df["Adults_Capped"] = df["Adults"].clip(upper=10)
    df["Children_Capped"] = df["Children"].clip(upper=6)

    # 3. Structural consistency
    df["PartyTotal_Capped"] = (
            df["Adults_Capped"] + df["Children_Capped"]
    )

    # 4. Log-Transformations (Keep the originals, add log versions)
    # np.log1p handles 0 smoothly by calculating log(x + 1)
    df["LengthOfStay_Log"] = np.log1p(df["LengthOfStay(Nights)"])
    df["PartyTotal_Log"] = np.log1p(df["PartyTotal_Capped"])


    # 5. Drop the old un-capped columns to prevent multicollinearity
    # (Keep LengthOfStay(Nights) if using tree models, use Log if linear)
    columns_to_drop = ["Adults", "Children", "PartyTotal", "LengthOfStay(Nights)"]
    df.drop(columns=columns_to_drop, inplace=True)



    # Do one hot encoding
    #park_dummy = pd.get_dummies(df['Park'], dtype=int, prefix = 'Park', drop_first=True)
    region_dummy = pd.get_dummies(df['Region'], dtype=int, prefix='Region', drop_first=True)
    equipment_dummy = pd.get_dummies(df['Equipment'], dtype=int, prefix='Equipment', drop_first=True)
    tent_or_rv_dummy = pd.get_dummies(df['TentOrRV'], dtype=int, prefix='TentOrRV', drop_first=True)
    rate_category_dummy = pd.get_dummies(df['RateCategory'], dtype=int, prefix='RateCategory', drop_first=True)
    province_dummy = pd.get_dummies(df['Province'], dtype=int, prefix='Province', drop_first=True)

    df = pd.concat([df, region_dummy, equipment_dummy, tent_or_rv_dummy, rate_category_dummy,
                    province_dummy], axis=1)
    df.drop(['Region', 'Equipment', 'TentOrRV', 'RateCategory', 'Province'], axis=1, inplace=True)


    # 1. Group by Park and exact Date to get the total daily number of people
    # We preserve the calendar features by taking the 'first' since they are identical for the same day

    # 1. Structural consistency on the RAW df first
    df["PartyTotal_Capped"] = df["Adults_Capped"] + df["Children_Capped"]

    # 2. Correct daily aggregation (Summing raw numbers)
    daily_df = df.groupby(['Park', 'arrival_year', 'arrival_month', 'arrival_day'], as_index=False).agg({
        'PartyTotal_Capped': 'sum',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'is_holiday': 'first',
        'is_holiday_adjacent': 'first',
        'Region_Eastern': 'first',
        'Region_Western': 'first',
        'ArrivalDate': 'first'
    })

    # 3. Create the log target variable directly on the summary dataframe
    daily_df['PartyTotal_Log'] = np.log1p(daily_df['PartyTotal_Capped'])
    daily_df.drop(columns=['PartyTotal_Capped'], inplace=True)

    # 🚨 FIX: Remove the lines that read directly from raw 'df' here!
    # Ensure ArrivalDate components are parsed correctly from daily_df
    daily_df['ArrivalDate'] = pd.to_datetime(daily_df['ArrivalDate'])
    daily_df['arrival_year'] = daily_df['ArrivalDate'].dt.year
    daily_df['arrival_month'] = daily_df['ArrivalDate'].dt.month
    daily_df['arrival_day'] = daily_df['ArrivalDate'].dt.day
    daily_df['day_of_week'] = daily_df['ArrivalDate'].dt.dayofweek

    # 4. Recompute exact historical lookup map
    lookup_dict = daily_df.set_index(['Park', 'arrival_year', 'arrival_month', 'arrival_day'])[
        'PartyTotal_Log'].to_dict()

    # 5. Recompute historical lookbacks
    temp_dates = pd.to_datetime(pd.DataFrame({
        'year': daily_df['arrival_year'],
        'month': daily_df['arrival_month'],
        'day': daily_df['arrival_day']
    }))
    target_dates_2y = temp_dates - pd.DateOffset(years=2)

    lookup_years = target_dates_2y.dt.year
    lookup_months = target_dates_2y.dt.month
    lookup_days = target_dates_2y.dt.day

    daily_df['people_two_years_ago'] = [
        lookup_dict.get((park, dt.year, dt.month, dt.day))
        for park, dt in zip(daily_df['Park'], target_dates_2y)
    ]

    # 6. Recompute structural baselines and tiers
    baseline_map = daily_df.groupby(['Park', 'arrival_month', 'day_of_week'])['PartyTotal_Log'].median().to_dict()
    daily_df['historical_dow_median'] = [baseline_map.get((park, m, dow)) for park, m, dow in
                                         zip(daily_df['Park'], daily_df['arrival_month'], daily_df['day_of_week'])]

    # Fill missing lookbacks with stable baselines
    daily_df['people_two_years_ago'] = daily_df['people_two_years_ago'].fillna(daily_df['historical_dow_median'])

    # Finalize scale mapping assets
    park_traffic_tier = daily_df.groupby('Park')['PartyTotal_Log'].median().to_dict()
    daily_df['park_scale_tier'] = daily_df['Park'].map(park_traffic_tier)

    # Drop cleanup artifacts
    daily_df.drop(columns=['historical_dow_median'], inplace=True, errors='ignore')

    # ==========================================
    # Model Development
    # ==========================================
    # 7. Safety filter to completely protect XGBoost labels from rogue NaNs
    daily_df = daily_df.dropna(subset=['PartyTotal_Log'])

    y_grouped = daily_df['PartyTotal_Log']

    columns_to_drop = ['PartyTotal_Log', 'Park', 'ArrivalDate']
    features_df = daily_df.drop(columns=[col for col in columns_to_drop if col in daily_df.columns])

    # Create dummy categorical mappings
    X_grouped = pd.get_dummies(features_df, drop_first=True)
    X_grouped = X_grouped.fillna(0)

    # 8. Train/Test Split & Fit
    X_train, X_test, y_train, y_test = train_test_split(
        X_grouped, y_grouped, test_size=0.2, random_state=42
    )

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)

    preds = xgb.predict(X_test)
    print(f"Aggregated XGBoost RMSE: {root_mean_squared_error(y_test, preds):.4f}")
    print(f"Aggregated XGBoost R2 Score: {r2_score(y_test, preds):.4f}")


    print(X_grouped.columns)

    # Save the trained XGBoost model structure
    joblib.dump(xgb, 'campsite_prediction/camping_predictor_model.pkl')

    # Save your park rankings map so your app backend can lookup 'park_scale_tier'
    pd.Series(park_traffic_tier).to_pickle('campsite_prediction/park_scale_tier_map.pkl')
    print("Core app engines successfully exported!")

    return daily_df



if __name__ == "__main__":
    df_canada = data_analysis_prediction.df_canada
    data_cleaning(df_canada)
