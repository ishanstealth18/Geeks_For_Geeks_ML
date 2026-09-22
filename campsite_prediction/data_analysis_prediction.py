import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st

import sqlite3


df = pd.read_csv('campsite_prediction/Natural_Resources_Camping_Parks_Reservation_Data.csv')
country_count = df.groupby(['Country'], as_index=False).agg(total_count=('Country', 'count')).sort_values(
        by=['total_count'], ascending=False)

# Remove all records which does not have Canada as Country
df_canada = df[df['Country'] == 'Canada'].copy()

conn = sqlite3.connect('camping_data.db')
df_canada.to_sql('camping_data', conn, if_exists='replace', index=False)
conn.close()

def data_analysis(df):
    # we see children column has 5% approximate null values, lets check distribution
    df['Children'].hist(bins=50)
    plt.show()

    # from the above histogram, we see 'Children' column is right skewed, we can replace null values with median
    df_canada.fillna({'Children': df['Children'].median()}, inplace=True)

    # All other null values are negligible so rows can be dropped
    df_canada.dropna(inplace=True)

    # convert date into datatime
    df_canada['CreateDate'] = pd.to_datetime(df_canada['CreateDate'])
    df_canada['ArrivalDate'] = pd.to_datetime(df_canada['ArrivalDate'])

    df_canada['create_year'] = df_canada['CreateDate'].dt.year
    df_canada['create_month'] = df_canada['CreateDate'].dt.month
    df_canada['create_day'] = df_canada['CreateDate'].dt.day

    df_canada['arrival_year'] = df_canada['ArrivalDate'].dt.year
    df_canada['arrival_month'] = df_canada['ArrivalDate'].dt.month
    df_canada['arrival_day'] = df_canada['ArrivalDate'].dt.day

    df_canada.drop(['CreateDate', 'ArrivalDate', 'CreateDayOfWeek', 'ArrivalMonth', 'ArrivalYear', 'ArrivalDayOfWeek',
                    'DepartureDate'], axis=1, inplace=True)

    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    df_canada['DepartDayOfWeek'] = df_canada['DepartDayOfWeek'].map(day_map)

    # remove unnecessary columns
    df_canada.drop(['Equipment', 'TentOrRV', 'County', 'Origin', 'RateCategory'],
                   axis=1, inplace=True)

    print(df_canada.info())

    # exploratory analysis

    # From the plot we see that
    # Western Region : Month April and November has the highest number of people visiting parks
    # Eastern Region and Central Region: Month July and August has the highest number of people visiting parks

    region_western_df = df_canada[df_canada['Region'] == 'Western']
    region_eastern_df = df_canada[df_canada['Region'] == 'Eastern']
    region_central_df = df_canada[df_canada['Region'] == 'Central']

    fig, ax = plt.subplots(3, 2, figsize=(14, 8))
    ax = ax.flatten()
    arrival_col = ['arrival_year', 'arrival_month']
    i = 0
    for region in [region_western_df, region_eastern_df, region_central_df]:

        for col in arrival_col:
            sns.barplot(data=region, x=col, y='PartyTotal', ax=ax[i], hue='Region')
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('# of People')
            ax[i].tick_params(axis='x', rotation=45)
            i += 1

    plt.suptitle('People Traffic For All Regions')
    plt.tight_layout()
    plt.show()

    # check booking category for all 3 regions
    # from the plot below, we see that  Western region has maximum campsite available
    booking_per_region_df = df_canada.groupby(['BookingCategory', 'Region'], as_index=False).agg(
        booking_category_count=('BookingCategory', 'count'))

    sns.barplot(data=booking_per_region_df, x='BookingCategory', y='booking_category_count', hue='Region')
    plt.xlabel('Booking Category')
    plt.ylabel('# of Booking Category')
    plt.title('Booking category Count Per Region')
    plt.show()

    # check how many adult and children visited th campsite per region
    df_canada['Children'] = df_canada['Children'].astype(int)

    # check distribution => Both the columns have skewed distribution, median should be used to fill NaN values
    df_canada['Adults'].hist(bins=100)
    df_canada['Children'].hist(bins=100)
    plt.show()

    # check and remove children and adult records with both 0 values
    # from the below chart, it shows that both Adults and Children booked campsites in Western region more compared to
    # other regions
    children_adult_zero_record = list(df_canada[(df_canada['Adults'] == 0) & (df_canada['Children'] == 0)].index)
    df_canada.drop(index=children_adult_zero_record, axis=0, inplace=True)

    children_adult_count_df = df_canada.groupby(['Region'], as_index=False).agg(adult_count=('Adults', 'sum'),
                                                                                children_count=('Children', 'sum'))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax = ax.flatten()

    value_count = ['adult_count', 'children_count']
    for i, val in enumerate(value_count):
        sns.barplot(data=children_adult_count_df, x='Region', y=val, ax=ax[i])
        ax[i].set_xlabel('Region')
        ax[i].set_ylabel(val)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt='%.0f', padding=3)

    plt.suptitle('# of Adults v/s Children Per Region')
    plt.show()

    province_index = df_canada[(df_canada['Province'] == 'UNKNOWN') | (df_canada['Province'] == 'Unknown')].index
    df_canada.drop(index=province_index, axis=0, inplace=True)

    # check traffic per province
    # from the plot below we can see that Nova scotia has highest number of parks and highest number of people coming to
    # campsite
    province_df = df_canada.groupby(['Province'], as_index=False).agg(park_count=('Park', 'count'), people_count=
    ('PartyTotal', 'sum'))

    fig, ax = plt.subplots(2, 1, figsize=(16, 5))
    ax = ax.flatten()

    province_count = ['park_count', 'people_count']
    for i, val in enumerate(province_count):
        sns.barplot(data=province_df, y='Province', x=val, ax=ax[i])
        ax[i].set_xlabel('Province')
        ax[i].set_ylabel(val)
        ax[i].tick_params(axis='x', rotation=45)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt='%.0f', padding=3)

    plt.suptitle('# of Parks and People Count Per Province')
    plt.show()

    # check length of stay per region
    # From the plot below, we can see that people mostly choose 2 nights stay for camping except in Eastern region.
    st.write("From the plot below, we can see that people mostly choose 2 nights stay for camping except in Eastern region.")
    stay_length_df = df_canada.groupby(['Region', 'LengthOfStay(Nights)'], as_index=False).agg(
        stay_length_count=('LengthOfStay(Nights)', 'count'))

    sns.barplot(data=stay_length_df, x='LengthOfStay(Nights)', y='stay_length_count', hue='Region')
    plt.xlabel('Stay Length (Nights)')
    plt.ylabel('Total Count')
    plt.title('Stav Length Count Per Region')
    plt.show()




if __name__ == "__main__":
    print(df_canada.shape)
    data_analysis(df_canada)
