import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ================== STEP 1: LOAD MODEL, ENCODER, SCALER, CLEAN_DF ==================
with open('XGBoost_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('clean_df.pkl', 'rb') as f:
    clean_df = pickle.load(f)

# ================== STEP 2: INPUT USER BARU ==================
new_data = pd.DataFrame([{
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 0,
    'room_type_reserved': 'Room_Type 1',
    'lead_time': 45,
    'arrival_year': 2017,
    'arrival_month': 9,
    'arrival_date': 18,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 1,
    'avg_price_per_room': 80.5,
    'no_of_special_requests': 0
}])

# ================== STEP 3: ENCODING ==================
encoded_cats = encoder.transform(new_data[['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']])
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(), index=new_data.index)

# ================== STEP 4: SCALING ==================
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

# ================== STEP 5: FINAL INPUT ==================
numerik_non_scaled = ['required_car_parking_space', 'repeated_guest']
final_input = pd.concat([
    scaled_df,
    new_data[numerik_non_scaled],
    encoded_cats_df
], axis=1)

# Pastikan urutan kolom sesuai dengan training set (clean_df)
final_input = final_input[clean_df.drop(columns='booking_status').columns]

# ================== STEP 6: PREDICTION ==================
prediction = model.predict(final_input)[0]
output_label = 'Not_Canceled' if prediction == 1 else 'Canceled'

print("Prediksi Booking Status:", output_label)
