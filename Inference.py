import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ================== LOAD MODEL, ENCODER, SCALER, CLEAN_DF ==================
@st.cache_resource
def load_model():
    with open('XGBoost_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('clean_df.pkl', 'rb') as f:
        clean_df = pickle.load(f)
    return model, encoder, scaler, clean_df

model, encoder, scaler, clean_df = load_model()

# ================== STREAMLIT UI ==================
st.title("Hotel Booking Cancellation Predictor")

no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
no_of_children = st.number_input("Number of Children", min_value=0, value=1)
no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
no_of_week_nights = st.number_input("Week Nights", min_value=0, value=3)
type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.selectbox("Car Parking Required", [0, 1])
room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = st.number_input("Lead Time", min_value=0, value=45)
arrival_year = st.selectbox("Arrival Year", [2017])  # Bisa diganti jika model support tahun lain
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings", min_value=0, value=1)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, value=80.5)
no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)

if st.button("Predict Booking Status"):
    # ================== INPUT DATA ==================
    new_data = pd.DataFrame([{
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }])

    # ================== ENCODING & SCALING ==================
    encoded_cats = encoder.transform(new_data[['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(), index=new_data.index)

    scaled_features = scaler.transform(new_data[[
        "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
        "lead_time", "arrival_year", "arrival_month", "arrival_date",
        "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
        "avg_price_per_room", "no_of_special_requests"
    ]])
    scaled_df = pd.DataFrame(scaled_features, columns=[
        "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
        "lead_time", "arrival_year", "arrival_month", "arrival_date",
        "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
        "avg_price_per_room", "no_of_special_requests"
    ], index=new_data.index)

    numerik_non_scaled = ['required_car_parking_space', 'repeated_guest']
    final_input = pd.concat([scaled_df, new_data[numerik_non_scaled], encoded_cats_df], axis=1)

    final_input = final_input[clean_df.drop(columns='booking_status').columns]

    # ================== PREDICTION ==================
    prediction = model.predict(final_input)[0]
    output_label = 'Not Canceled' if prediction == 1 else 'Canceled'
    st.success(f"Predicted Booking Status: **{output_label}**")
