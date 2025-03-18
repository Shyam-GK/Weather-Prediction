import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os
from io import StringIO
import statsmodels.api as sm

# Set page config
st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌤️",
    layout="wide"
)

# Popular cities list for dropdown
popular_cities = [
    # Major Indian Cities
    "Leh", "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Ahmedabad",
    "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal",
    "Visakhapatnam", "Coimbatore", "Kochi", "Guwahati", "Shimla", "Darjeeling", "Manali",
    "Amritsar", "Chandigarh", "Varanasi", "Patna", "Ranchi", "Raipur", "Surat", "Vadodara",
    "Ludhiana", "Agra", "Meerut", "Dehradun", "Jodhpur", "Udaipur", "Mysore", "Tiruchirapalli",
    "Madurai", "Nashik", "Aurangabad", "Vijayawada", "Gwalior", "Allahabad", "Jamshedpur",
    "Bhubaneswar", "Cuttack", "Srinagar", "Panaji", "Shillong", "Itanagar", "Aizawl", "Gangtok",
    "Kozhikode", "Thiruvananthapuram", "Rajkot", "Jalandhar", "Dhanbad", "Bokaro", "Faridabad",
    "Ghaziabad", "Noida", "Gandhinagar", "Mangalore", "Belgaum", "Jabalpur", "Bilaspur",
    "Haridwar", "Rishikesh", "Kota", "Ajmer", "Aligarh", "Bikaner", "Silchar", "Imphal", 
    "Tezpur", "Puducherry", "Port Blair",

    # Major Global Cities
    "New York", "London", "Tokyo", "Paris", "Sydney", "Dubai", "Singapore", "Hong Kong",
    "Los Angeles", "San Francisco", "Chicago", "Toronto", "Vancouver", "Berlin", "Rome",
    "Madrid", "Barcelona", "Moscow", "Istanbul", "Bangkok", "Seoul", "Beijing", "Shanghai",
    "Mexico City", "São Paulo", "Rio de Janeiro", "Buenos Aires", "Cairo", "Cape Town",
    "Amsterdam", "Vienna", "Zurich", "Lisbon", "Athens", "Prague", "Helsinki", "Stockholm",
    "Oslo", "Copenhagen", "Venice", "Milan", "Florence", "Edinburgh", "Dublin", "Brussels",
    "Manila", "Jakarta", "Ho Chi Minh City", "Kuala Lumpur", "Taipei", "Doha", "Riyadh",
    "Abu Dhabi", "Johannesburg"
]

all_cities = sorted(popular_cities)

# Function to get user's location
def get_user_location():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('city', 'Coimbatore')  
        return 'Coimbatore'
    except Exception as e:
        return 'Coimbatore'  

# API Configuration
with st.sidebar:
    st.header("Settings")
    
    # Location selection
    location_options = ["Auto-detect", "Enter manually"] 
    location_choice = st.selectbox("Select location method:", location_options, index=0)
    
    if location_choice == "Auto-detect":
        user_location = get_user_location()
        st.success(f"Auto-detected location: {user_location}")
        selected_location = user_location
    elif location_choice == "Enter manually":
        # Add the dropdown selector for cities
        city_selection_method = st.radio("Choose selection method:", ["Select from list", "Type city name"])
        
        if city_selection_method == "Select from list":
            selected_location = st.selectbox("Select city:", all_cities, index=all_cities.index("Coimbatore") if "Coimbatore" in all_cities else 0)
        else:
            manual_location = st.text_input("Enter location:", value="Coimbatore")
            if manual_location:
                selected_location = manual_location
            else:
                selected_location = "Coimbatore"
    else:
        selected_location = location_choice
        
    # API key configuration
    st.header("API Configuration")
    weatherapi_api_key = st.text_input("Enter WeatherAPI Key:", value="8bf450c8d0b94a44b78151729251503", type="password")
    
    # Add API usage information
    st.info("WeatherAPI Free Plan Limits: 750 calls per day\nCurrent implementation optimizes API call usage to avoid hitting this limit.")

# API Call Counter and Rate Limiter
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0

def log_api_call():
    st.session_state.api_calls += 1
    if st.session_state.api_calls >= 700:  # Set a safe threshold below the 750 limit
        st.sidebar.warning(f"⚠️ High API usage detected: {st.session_state.api_calls}/750 calls. Consider limiting refreshes.")

def reset_api_counter():
    st.session_state.api_calls = 0
    st.sidebar.success("API call counter reset for the new day!")

# Check if we need to reset the counter (new day)
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.now().date()
elif st.session_state.last_reset_date < datetime.now().date():
    reset_api_counter()
    st.session_state.last_reset_date = datetime.now().date()

# Function to handle API requests with error management
def make_weatherapi_request(endpoint, params):
    if not weatherapi_api_key:
        return None, "API key is missing"
    
    base_url = f"http://api.weatherapi.com/v1/{endpoint}"
    params['key'] = weatherapi_api_key
    
    try:
        log_api_call()
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            return None, "API authentication failed. Please check your API key."
        elif response.status_code == 403:
            return None, "API access forbidden. Your account may be suspended or your plan limits exceeded."
        elif response.status_code == 429:
            return None, "Too many API requests. You've reached the rate limit for your plan."
        else:
            return None, f"API request failed with status code: {response.status_code}. {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "API request timed out. WeatherAPI might be experiencing high traffic."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Function to fetch real-time weather data
def fetch_realtime_weather(location):
    data, error = make_weatherapi_request('current.json', {'q': location, 'aqi': 'yes'})
    
    if error:
        st.error(f"Failed to fetch real-time weather: {error}")
        return None
        
    try:
        # Extract relevant data
        weather_data = {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temp': data['current']['temp_c'],
            'tempmin': data['current']['temp_c'] - 2,  
            'tempmax': data['current']['temp_c'] + 2,  
            'humidity': data['current']['humidity'],
            'pressure': data['current']['pressure_mb'],
            'precip': data['current']['precip_mm'],
            'windspeed': data['current']['wind_kph'],
            'winddir': data['current']['wind_degree'],
            'cloudcover': data['current']['cloud'],
            'visibility': data['current']['vis_km'],
            'uv': data['current']['uv'],
            'description': data['current']['condition']['text'],
            'icon': data['current']['condition']['icon'],
            'air_quality': data['current'].get('air_quality', {}).get('us-epa-index', None)
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([weather_data])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        return df
    except KeyError as e:
        st.error(f"Error parsing real-time weather data: Missing key {e}")
        return None
    except Exception as e:
        st.error(f"Error processing real-time weather data: {str(e)}")
        return None

# Function to fetch historical data with expanded range
def fetch_historical_weather(location, days=30):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    all_data = []
    current_date = start_date
    
    with st.spinner(f"Fetching historical data for {location} - past {days} days..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        days_processed = 0
        total_days = (end_date - start_date).days + 1
        
        while current_date <= end_date:
            # Update progress
            days_processed += 1
            progress_value = days_processed / total_days
            progress_bar.progress(progress_value)
            
            # WeatherAPI for historical data
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Make API call
            data, error = make_weatherapi_request('history.json', {'q': location, 'dt': date_str})
            
            if error:
                st.warning(f"Could not fetch data for {date_str}: {error}")
                current_date += timedelta(days=1)
                continue
            
            try:
                # Extract day summary
                day_data = data['forecast']['forecastday'][0]['day']
                
                weather_data = {
                    'datetime': f"{date_str} 12:00:00",  # Add noon time to ensure consistent format
                    'temp': day_data['avgtemp_c'],
                    'tempmin': day_data['mintemp_c'],
                    'tempmax': day_data['maxtemp_c'],
                    'humidity': day_data['avghumidity'],
                    'precip': day_data['totalprecip_mm'],
                    'windspeed': day_data['maxwind_kph'],
                    'visibility': day_data['avgvis_km'],
                    'uv': day_data['uv'],
                    'description': data['forecast']['forecastday'][0]['day']['condition']['text']
                }
                
                # Add hourly data points to increase dataset size
                for hour_data in data['forecast']['forecastday'][0]['hour']:
                    hour_time = hour_data['time']
                    hour_weather = {
                        'datetime': hour_time,
                        'temp': hour_data['temp_c'],
                        'tempmin': day_data['mintemp_c'], 
                        'tempmax': day_data['maxtemp_c'],  
                        'humidity': hour_data['humidity'],
                        'pressure': hour_data['pressure_mb'],
                        'precip': hour_data['precip_mm'],
                        'windspeed': hour_data['wind_kph'],
                        'winddir': hour_data['wind_degree'],
                        'cloudcover': hour_data['cloud'],
                        'visibility': hour_data['vis_km'],
                        'uv': hour_data['uv'],
                        'description': hour_data['condition']['text']
                    }
                    all_data.append(hour_weather)
                
                # Add the daily summary too
                all_data.append(weather_data)
            except KeyError as ke:
                st.warning(f"Missing data in API response for {date_str}: {ke}")
            except Exception as e:
                st.warning(f"Error processing data for {date_str}: {str(e)}")
            
            current_date += timedelta(days=1)
        progress_bar.empty()
    
    if all_data:

        df = pd.DataFrame(all_data)
        
        # Convert datetime strings to datetime objects with consistent format
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Handle any problematic dates that didn't parse correctly
        mask = pd.isna(df['datetime'])
        if mask.any():
            # Try again with a more flexible format for problematic rows
            df.loc[mask, 'datetime'] = pd.to_datetime(df.loc[mask, 'datetime'], format='mixed', errors='coerce')
            
            # For any remaining NaT values, drop those rows
            df = df.dropna(subset=['datetime'])
        
        # Set as index
        df.set_index('datetime', inplace=True)
        
        # Clean data
        df = clean_data(df)
        return df
    else:
        return None

# Function to fetch forecast data
def fetch_forecast_weather(location, days=7):
    # WeatherAPI forecast
    data, error = make_weatherapi_request('forecast.json', {'q': location, 'days': min(days, 14), 'aqi': 'yes'})
    
    if error:
        st.error(f"Failed to fetch forecast: {error}")
        return None
    
    try:
        forecast_data = []
        hourly_forecast_data = []
        
        for day in data['forecast']['forecastday']:
            # Daily forecast - ensure datetime includes time component
            day_date = day['date']
            forecast_data.append({
                'datetime': f"{day_date} 12:00:00",  # Add noon time to ensure consistent format
                'temp': day['day']['avgtemp_c'],
                'tempmin': day['day']['mintemp_c'],
                'tempmax': day['day']['maxtemp_c'],
                'humidity': day['day']['avghumidity'],
                'precip': day['day']['totalprecip_mm'],
                'windspeed': day['day']['maxwind_kph'],
                'visibility': day['day']['avgvis_km'],
                'uv': day['day']['uv'],
                'description': day['day']['condition']['text'],
                'icon': day['day']['condition']['icon']
            })
            
            # Hourly forecast
            for hour in day['hour']:
                hourly_forecast_data.append({
                    'datetime': hour['time'], 
                    'temp': hour['temp_c'],
                    'humidity': hour['humidity'],
                    'pressure': hour['pressure_mb'],
                    'precip': hour['precip_mm'],
                    'windspeed': hour['wind_kph'],
                    'cloudcover': hour['cloud'],
                    'description': hour['condition']['text'],
                    'icon': hour['condition']['icon']
                })
        
        # Convert to DataFrames
        daily_df = pd.DataFrame(forecast_data)
        daily_df['datetime'] = pd.to_datetime(daily_df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        daily_df.set_index('datetime', inplace=True)
        
        hourly_df = pd.DataFrame(hourly_forecast_data)
        hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Handle any problematic hourly dates
        mask = pd.isna(hourly_df['datetime'])
        if mask.any():
            hourly_df.loc[mask, 'datetime'] = pd.to_datetime(hourly_df.loc[mask, 'datetime'], format='mixed', errors='coerce')
            hourly_df = hourly_df.dropna(subset=['datetime'])
        
        hourly_df.set_index('datetime', inplace=True)
        
        # Clean data
        daily_df = clean_data(daily_df)
        hourly_df = clean_data(hourly_df)
        
        return daily_df, hourly_df
    except KeyError as ke:
        st.error(f"Missing data in forecast API response: {ke}")
        return None
    except Exception as e:
        st.error(f"Error processing forecast data: {str(e)}")
        return None

def clean_data(df):
    df_clean = df.copy()
    # Get numeric columns
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Replace negative values where they don't make sense
        if col in ['temp', 'tempmin', 'tempmax', 'humidity', 'precip', 'visibility', 'uv']:
            df_clean[col] = df_clean[col].apply(lambda x: np.nan if (x < -50 or x > 100) else x)  
            
    # Fill NaN values with appropriate methods for each column
    for col in numeric_cols:
        if not df_clean[col].isna().any():
            continue

        if col in ['temp', 'tempmin', 'tempmax']:
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        elif col in ['humidity', 'cloudcover']:
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill').fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in numeric_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median() if not pd.isna(df_clean[col].median()) else 0)
    
    return df_clean

# Function to get location-specific filename
def get_location_filename(location):
    # Remove spaces and special characters, convert to lowercase
    safe_location = ''.join(e.lower() for e in location if e.isalnum())
    return f"{safe_location}_weather_data.csv"

# Function to save data to CSV
def save_to_csv(df, location):
    try:
        filename = get_location_filename(location)
        
        # Check if file exists
        if os.path.exists(filename):
            # Read existing data
            existing_df = pd.read_csv(filename)
            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
            existing_df.set_index('datetime', inplace=True)
            
            # Combine with new data
            combined_df = pd.concat([existing_df, df])
            # Remove duplicates based on index
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            # Sort by datetime
            combined_df = combined_df.sort_index()
            combined_df = clean_data(combined_df)
            combined_df.to_csv(filename)
            return combined_df
        else:
            # Create new file with clean data
            clean_df = clean_data(df)
            clean_df.to_csv(filename)
            return clean_df
    except Exception as e:
        st.error(f"Error saving to CSV: {str(e)}")
        return df

# Function to load data from CSV or fetch new data with expanded historical range
def load_or_fetch_data(location, force_refresh=False, initial_load=False):
    filename = get_location_filename(location)
    if force_refresh:
        st.info(f"Refreshing data for {location}...")
        # Get realtime data first
        realtime_df = fetch_realtime_weather(location)
        historical_df = fetch_historical_weather(location, days=30)
        
        if historical_df is not None:
            if realtime_df is not None:
                combined_df = pd.concat([historical_df, realtime_df])
            else:
                combined_df = historical_df
                
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            combined_df = save_to_csv(combined_df, location)
            st.success(f"Fetched {len(combined_df)} data points spanning {combined_df.index.nunique()} days.")
            return combined_df
        elif realtime_df is not None:
            # If only realtime data is available
            realtime_df = save_to_csv(realtime_df, location)
            st.warning("Only current weather data is available.")
            return realtime_df
            
        st.error("Failed to fetch any weather data. Please check API key and connection.")
        return None
    
    # First check if we have locally stored data
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            mask = pd.isna(df['datetime'])
            if mask.any():
                df.loc[mask, 'datetime'] = pd.to_datetime(df.loc[mask, 'datetime'], format='mixed', errors='coerce')
                df = df.dropna(subset=['datetime'])
            
            df.set_index('datetime', inplace=True)
            if (datetime.now() - df.index.max()).total_seconds() > 3600:
                st.info(f"Updating with latest weather data for {location}...")
                realtime_df = fetch_realtime_weather(location)
                if realtime_df is not None:
                    df = pd.concat([df, realtime_df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    df = save_to_csv(df, location)
            if initial_load and (len(df) < 30 or df.index.nunique() < 10):
                st.info(f"Not enough historical data for {location}. Fetching more data...")
                historical_df = fetch_historical_weather(location, days=30)  # Fetch 30 days of historical data
                if historical_df is not None:
                    df = pd.concat([historical_df, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    df = save_to_csv(df, location)
            
            st.success(f"Loaded {len(df)} weather data points for {location} spanning {df.index.nunique()} days.")
            return df
    except Exception as e:
        st.warning(f"Error loading existing data: {str(e)}. Will fetch new data.")
    
    st.info(f"Fetching new weather data for {location}...")

    realtime_df = fetch_realtime_weather(location)
    historical_df = fetch_historical_weather(location, days=30) 
    
    if historical_df is not None:
        if realtime_df is not None:
            combined_df = pd.concat([historical_df, realtime_df])
        else:
            combined_df = historical_df
            
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        combined_df = save_to_csv(combined_df, location)
        st.success(f"Fetched {len(combined_df)} data points spanning {combined_df.index.nunique()} days.")
        return combined_df
    elif realtime_df is not None:
        # If only realtime data is available
        realtime_df = save_to_csv(realtime_df, location)
        st.warning("Only current weather data is available.")
        return realtime_df
    
    st.error("Failed to fetch any weather data. Please check API key and connection.")
    return None

if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_location' not in st.session_state:
    st.session_state.current_location = "Coimbatore"  

location_changed = (st.session_state.current_location != selected_location)
if location_changed:
    st.session_state.current_location = selected_location
    st.session_state.df = None  

if st.sidebar.button("Refresh Weather Data"):
    st.session_state.df = load_or_fetch_data(selected_location, force_refresh=True)

st.sidebar.info(f"API calls today: {st.session_state.api_calls}/750")

# Display current location
st.sidebar.markdown(f"### Currently showing: **{selected_location}**")

# Load data if not already loaded
if st.session_state.df is None:
    st.session_state.df = load_or_fetch_data(selected_location, initial_load=True)

# Set df variable for compatibility with the rest of the code
df = st.session_state.df

# Check if data is available
if df is None or len(df) == 0:
    st.error(f"No weather data available for {selected_location}. Please check your API key and try again.")
    st.stop()
else:
    # Display data size info
    st.sidebar.success(f"Dataset: {len(df)} data points across {df.index.nunique()} unique days")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌤️ Current Weather", "📊 Data Overview", "🔍 Data Analysis", "🤖 Model Performance", "🔮 Weather Forecast"])

with tab1:
    st.header(f"Current Weather in {selected_location}")
    current_weather = df.iloc[-1] if not df.empty else None
    
    if current_weather is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display current temperature
            temp = current_weather.get('temp')
            if temp is not None:
                st.markdown(f"<h1 style='text-align: center; font-size: 48px;'>{temp:.1f}°C</h1>", unsafe_allow_html=True)
                
                # Display min and max temperature if available
                tempmin = current_weather.get('tempmin')
                tempmax = current_weather.get('tempmax')
                if tempmin is not None and tempmax is not None:
                    st.markdown(f"<p style='text-align: center; font-size: 20px;'>Min: {tempmin:.1f}°C | Max: {tempmax:.1f}°C</p>", unsafe_allow_html=True)
            
            # Display weather description if available
            description = current_weather.get('description')
            if description is not None:
                st.markdown(f"<p style='text-align: center; font-size: 24px;'>{description.title()}</p>", unsafe_allow_html=True)
            
            # Display updated time
            st.markdown(f"<p style='text-align: center;'>Last Updated: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
        
        with col2:
            metrics_data = [
                ("Humidity", current_weather.get('humidity', "N/A"), "%"),
                ("Wind Speed", current_weather.get('windspeed', "N/A"), "km/h"),
                ("Pressure", current_weather.get('pressure', "N/A"), "hPa"),
                ("Cloud Cover", current_weather.get('cloudcover', "N/A"), "%")
            ]
            
            # Add additional metrics if available
            if 'visibility' in current_weather:
                metrics_data.append(("Visibility", current_weather.get('visibility', "N/A"), "km"))
            if 'precip' in current_weather:
                metrics_data.append(("Precipitation", current_weather.get('precip', "N/A"), "mm"))
            if 'uv' in current_weather:
                metrics_data.append(("UV Index", current_weather.get('uv', "N/A"), ""))
            if 'air_quality' in current_weather and current_weather.get('air_quality') is not None:
                metrics_data.append(("Air Quality (US EPA)", current_weather.get('air_quality', "N/A"), ""))
            
            # Display metrics in a grid
            for i in range(0, len(metrics_data), 2):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    name, value, unit = metrics_data[i]
                    if isinstance(value, (int, float)):
                        st.metric(name, f"{value:.1f} {unit}")
                    else:
                        st.metric(name, f"{value} {unit}")

                if i + 1 < len(metrics_data):
                    with col_b:
                        name, value, unit = metrics_data[i + 1]
                        if isinstance(value, (int, float)):
                            st.metric(name, f"{value:.1f} {unit}")
                        else:
                            st.metric(name, f"{value} {unit}")
    else:
        st.error("No current weather data available.")

with tab2:
    st.header("Dataset Overview")
    
    # Get only numeric columns for visualization
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sample Data")
        # Select a representative sample of the data
        sample_size = min(5, len(df))
        if len(df) > 5:
            # Get first, middle and last rows for a better representation
            indices = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
            sample = df.iloc[indices]
        else:
            sample = df.head()
            
        st.dataframe(sample)
        
        st.subheader("Dataset Statistics")
        st.dataframe(numeric_df.describe().round(2))
        
        # Display data completeness
        st.subheader("Data Completeness")
        completeness = (numeric_df.count() / len(numeric_df) * 100).round(1).sort_values(ascending=False)
        completeness_df = pd.DataFrame(completeness, columns=['Completeness %'])
        st.dataframe(completeness_df)
    
    with col2:
        st.subheader("Temperature Trends")
        if all(col in df.columns for col in ['temp', 'tempmin', 'tempmax']):
            # Get daily average if we have hourly data
            if df.index.nunique() < len(df):
                daily_df = df.resample('D').mean()
            else:
                daily_df = df
                
            fig = px.line(daily_df, y=['temp', 'tempmin', 'tempmax'], 
                          labels={'value': 'Temperature (°C)', 'variable': 'Temperature Type'},
                          color_discrete_map={'temp': 'orange', 'tempmin': 'blue', 'tempmax': 'red'})
            fig.update_layout(legend_title_text='', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show temperature distribution by hour of day if we have enough hourly data
            if len(df) >= 24:
                st.subheader("Temperature by Hour of Day")
                df['hour'] = df.index.hour
                hourly_temp = df.groupby('hour')['temp'].mean().reset_index()
                
                fig = px.line(hourly_temp, x='hour', y='temp',
                             labels={'hour': 'Hour of Day', 'temp': 'Average Temperature (°C)'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required temperature columns not found. Cannot display temperature trends.")

with tab3:
    st.header("Data Analysis")
    
    # Get only numeric columns for analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if 'temp' in numeric_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Distribution")
            fig = px.histogram(numeric_df, x='temp', nbins=20, 
                              labels={'temp': 'Temperature (°C)', 'count': 'Frequency'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Correlation with Temperature")
            corr = numeric_df.corr()['temp'].sort_values(ascending=False)
            corr = corr[corr.index != 'temp']  # Remove self-correlation
            fig = px.bar(x=corr.index, y=corr.values,
                        labels={'x': 'Features', 'y': 'Correlation Coefficient'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Temperature vs. Other Factors
        st.subheader("Temperature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'humidity' in numeric_df.columns:
                fig = px.scatter(numeric_df, x='humidity', y='temp',
                                labels={'humidity': 'Humidity (%)', 'temp': 'Temperature (°C)'},
                                title='Temperature vs. Humidity')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            if 'windspeed' in numeric_df.columns:
                fig = px.scatter(numeric_df, x='windspeed', y='temp',
                                labels={'windspeed': 'Wind Speed (km/h)', 'temp': 'Temperature (°C)'},
                                title='Temperature vs. Wind Speed')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional relationship selection
        st.subheader("Explore Other Feature Relationships")
        available_features = [col for col in numeric_df.columns if col != 'temp' and numeric_df[col].nunique() > 1]
        if available_features:
            default_features = available_features[:2] if len(available_features) > 1 else available_features
            selected_features = st.multiselect("Select features to compare with temperature:", 
                                              options=available_features,
                                              default=default_features[:1])
            
            if selected_features:
                for feature in selected_features:
                    fig = px.scatter(numeric_df, x=feature, y='temp',
                                    labels={feature: feature.capitalize(), 'temp': 'Temperature (°C)'},
                                    title=f'Temperature vs. {feature.capitalize()}',
                                    trendline="ols")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No additional features available for comparison.")
    else:
        st.warning("Temperature column not found in numeric data. Cannot perform analysis.")

with tab4:
    st.header("Model Performance")
    
    # Check if we have sufficient data for modeling
    if len(df) >= 5:  # Reduced threshold for training
        # Train the model
        with st.spinner("Training model..."):
            try:
                @st.cache_resource
                def train_model(df):
                    # Get only numeric columns for correlation
                    numeric_df = df.select_dtypes(include=['float64', 'int64'])
                    
                    # Make sure 'temp' is in numeric_df
                    if 'temp' not in numeric_df.columns:
                        st.error("Temperature column not found in numeric data. Cannot train model.")
                        return None, None, None, None, None, None, None
                    
                    # Feature Selection based on correlation
                    correlation = numeric_df.corr()['temp'].abs().dropna()
                    important_features = correlation[correlation > 0.1].index.tolist()
                    
                    # Make sure we have at least one feature
                    if len(important_features) <= 1:  # Only temp itself
                        # If no strong correlations, use all numeric columns
                        important_features = numeric_df.columns.tolist()
                        
                    # Remove temp from features if it's there
                    if 'temp' in important_features:
                        important_features.remove('temp')
                    
                    # Check if we have any features left
                    if not important_features:
                        st.error("No suitable features for prediction found.")
                        return None, None, None, None, None, None, None
                    # Add temporal features
                    df_model = numeric_df.copy()
                    df_model['hour'] = df_model.index.hour
                    df_model['day'] = df_model.index.day
                    df_model['month'] = df_model.index.month
                    df_model['dayofweek'] = df_model.index.dayofweek
                    
                    # Add these to important features
                    important_features.extend(['hour', 'day', 'month', 'dayofweek'])
                    
                    # Prepare data for modeling
                    X = df_model[important_features].copy()
                    y = df_model['temp'].copy()
                    
                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    
                    # Get feature importances
                    feature_importance = pd.DataFrame({
                        'Feature': important_features,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    
                    return model, X_train, X_test, y_train, y_test, y_pred_test, feature_importance
                
                # Train the model
                model, X_train, X_test, y_train, y_test, y_pred_test, feature_importance = train_model(df)
                
                if model is not None:
                    # Display model performance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Accuracy")
                        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        train_r2 = r2_score(y_train, model.predict(X_train))
                        test_r2 = r2_score(y_test, y_pred_test)
                        
                        # Display metrics
                        metrics_df = pd.DataFrame({
                            'Metric': ['RMSE (Train)', 'RMSE (Test)', 'R² (Train)', 'R² (Test)'],
                            'Value': [train_rmse, test_rmse, train_r2, test_r2]
                        })
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Interpret results
                        if test_r2 > 0.7:
                            st.success(f"Model performs well with R² of {test_r2:.2f}")
                        elif test_r2 > 0.5:
                            st.info(f"Model has moderate performance with R² of {test_r2:.2f}")
                        else:
                            st.warning(f"Model performance is limited with R² of {test_r2:.2f}")
                    
                    with col2:
                        st.subheader("Feature Importance")
                        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Predictions vs Actual
                    st.subheader("Predictions vs Actual")
                    pred_vs_actual = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred_test
                    })
                    
                    fig = px.scatter(pred_vs_actual, x='Actual', y='Predicted',
                                   labels={'Actual': 'Actual Temperature (°C)', 'Predicted': 'Predicted Temperature (°C)'},
                                   title='Model Predictions vs Actual Values')
                    
                    # Add ideal prediction line
                    min_val = min(pred_vs_actual['Actual'].min(), pred_vs_actual['Predicted'].min())
                    max_val = max(pred_vs_actual['Actual'].max(), pred_vs_actual['Predicted'].max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                          mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show residuals
                    st.subheader("Residual Analysis")
                    pred_vs_actual['Residual'] = pred_vs_actual['Actual'] - pred_vs_actual['Predicted']
                    
                    fig = px.histogram(pred_vs_actual, x='Residual', nbins=20,
                                     labels={'Residual': 'Prediction Error (°C)', 'count': 'Frequency'},
                                     title='Distribution of Prediction Errors')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    else:
        st.warning("Not enough data points for modeling. Need at least 5 data points.")


with tab5:
    st.header("Weather Forecast")
    
    # Fetch weather forecast data
    with st.spinner("Fetching forecast data..."):
        try:
            daily_forecast, hourly_forecast = fetch_forecast_weather(selected_location, days=7)
            
            if daily_forecast is not None and hourly_forecast is not None:
                # Display daily forecast
                st.subheader("7-Day Weather Forecast")
                
                # Create forecast cards
                forecast_cols = st.columns(min(7, len(daily_forecast)))

                for i, (idx, day) in enumerate(daily_forecast.iterrows()):
                    if i < len(forecast_cols):
                        with forecast_cols[i]:
                            date_str = idx.strftime('%a, %b %d')
                            st.markdown(f"<h4 style='text-align: center;'>{date_str}</h4>", unsafe_allow_html=True)

                            # Display temperature
                            temp = day.get('temp')
                            if temp is not None:
                                st.markdown(f"<p style='text-align: center; font-size: 24px;'>{temp:.1f}°C</p>", unsafe_allow_html=True)

                            # Display min/max
                            tempmin = day.get('tempmin')
                            tempmax = day.get('tempmax')
                            if tempmin is not None and tempmax is not None:
                                st.markdown(f"<p style='text-align: center;'>Min: {tempmin:.1f}°C<br>Max: {tempmax:.1f}°C</p>", unsafe_allow_html=True)

                            # Display other metrics
                            humidity = day.get('humidity')
                            if humidity is not None:
                                st.markdown(f"<p style='text-align: center;'>Humidity: {humidity:.0f}%</p>", unsafe_allow_html=True)

                            precip = day.get('precip')
                            if precip is not None:
                                st.markdown(f"<p style='text-align: center;'>Precip: {precip:.1f}mm</p>", unsafe_allow_html=True)

                            # Display description
                            description = day.get('description')
                            if description is not None:
                                st.markdown(f"<p style='text-align: center;'>{description}</p>", unsafe_allow_html=True)

                st.subheader("Hourly Temperature Forecast")

                # Filter to next 24 hours
                next_24h = hourly_forecast.loc[hourly_forecast.index <= datetime.now() + timedelta(hours=24)]

                if not next_24h.empty:
                    fig = px.line(next_24h, y='temp',
                                  labels={'temp': 'Temperature (°C)', 'datetime': 'Date & Time'},
                                  title='Next 24 Hours Temperature Forecast')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Detailed Hourly Forecast")

                    hourly_display_cols = ['temp', 'humidity', 'precip', 'windspeed', 'description']
                    hourly_display = next_24h[hourly_display_cols].copy()

                    hourly_display.columns = ['Temperature (°C)', 'Humidity (%)', 'Precipitation (mm)',
                                              'Wind Speed (km/h)', 'Condition']

                    # Display with datetime index
                    st.dataframe(hourly_display)
                else:
                    st.warning("No hourly forecast data available for the next 24 hours.")
            else:
                st.error("Failed to fetch forecast data. Please check your API key and connection.")
        
        except Exception as e:
            st.error(f"Error displaying forecast: {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Weather prediction by Shyam GK | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
