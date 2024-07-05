import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
import matplotlib.pyplot as plt

# Fetch data in batches
base_url = "https://data.gov.sg/api/action/datastore_search"
resource_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
url = f"{base_url}?resource_id={resource_id}"
batch_size = 10000


def fetch_data_in_batches(url, batch_size):
    all_records = []
    initial_response = requests.get(url)
    initial_data = initial_response.json()
    total_records = initial_data["result"]["total"]
    total_batches = (total_records // batch_size) + 1

    for batch_num in range(total_batches):
        offset = batch_num * batch_size
        request_url = f"{url}&limit={batch_size}&offset={offset}"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            records = data["result"]["records"]
            all_records.extend(records)
        else:
            print("Error fetching data:", response.status_code)
            break

    return all_records


# Fetch data
all_records = fetch_data_in_batches(url, batch_size)

# Convert to DataFrame
df = pd.DataFrame(all_records)


# Preprocessing function
def preprocess_data(df, town, flat_type):
    filtered_data = df[(df['town'] == town) & (df['flat_type'] == flat_type)]
    filtered_data['year'] = pd.to_datetime(filtered_data['month']).dt.year
    filtered_data['month_num'] = pd.to_datetime(filtered_data['month']).dt.month

    X = filtered_data[['year', 'month_num']]
    y = filtered_data['resale_price'].astype(float)

    return X, y


# Forecasting function (no test-train as I used full dataset for training)
def forecast_prices(town, flat_type):
    X, y = preprocess_data(df, town, flat_type)

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(start='2024-01-01', end='2028-12-01', freq='MS')
    future_df = pd.DataFrame({'year': future_dates.year, 'month_num': future_dates.month})

    future_prices = model.predict(future_df)

    return future_df, future_prices


# Get available flat types for each town
town_flat_type_mapping = df.groupby('town')['flat_type'].unique().to_dict()

# Streamlit app
st.title('HDB Resale Price Forecast')
st.write("Forecast HDB resale prices for different towns and flat types.")

towns = list(town_flat_type_mapping.keys())

# Select town
selected_town = st.selectbox('Select Town', towns)

# Update flat types based on selected town
flat_types = town_flat_type_mapping[selected_town]
selected_flat_type = st.selectbox('Select Flat Type', flat_types)

if st.button('Forecast'):
    future_df, future_prices = forecast_prices(selected_town, selected_flat_type)
    forecast_df = pd.DataFrame({
        'Date': future_df.apply(lambda row: datetime(row['year'], row['month_num'], 1), axis=1),
        'Forecasted Price': future_prices
    })
    st.write(f"Forecasted prices for {selected_flat_type} in {selected_town} from 2024 to 2028:")
    st.write(forecast_df)

    # Plot the forecasted prices
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df['Date'], forecast_df['Forecasted Price'], marker='o')
    plt.title(f"Forecasted Prices for {selected_flat_type} in {selected_town} (2024-2028)")
    plt.xlabel('Date')
    plt.ylabel('Forecasted Price')
    plt.grid(True)
    st.pyplot(plt)

    # Option to download as CSV
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download forecast as CSV",
        data=csv,
        file_name='forecast.csv',
        mime='text/csv',
    )
