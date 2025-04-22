import pandas as pd #pandas digunakan untuk manipulasi data
import numpy as np #numpy digunakan untuk operasi numerik
import matplotlib.pyplot as plt #matplotlib digunakan untuk visualisasi data
import seaborn as sns #seaborn digunakan untuk visualisasi data yang lebih baik
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler #sklearn digunakan untuk preprocessing data
from sklearn.model_selection import train_test_split #sklearn digunakan untuk membagi data menjadi data latih dan data uji
from sklearn.ensemble import RandomForestClassifier #sklearn digunakan untuk membuat model Random Forest
import xgboost as xgb #xgboost digunakan untuk membuat model XGBoost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #sklearn digunakan untuk evaluasi model
import pickle #pickle digunakan untuk menyimpan model

class HotelBookingModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_encoded = None
        self.df_scaled = None
        self.clean_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print("Data loaded successfully.")

        # Drop 'Booking_ID' column
        self.df.drop(columns=['Booking_ID'], inplace=True)

        print("First 10 rows:")
        print(self.df.head(10))

        print("Last 10 rows:")
        print(self.df.tail(10))

        print("Dataset Description:")
        print(self.df.describe())

        print("Dataset Info:")
        print(self.df.info())

        print("Duplicate rows:")
        print(self.df[self.df.duplicated()])

        print("Missing values:")
        print(self.df.isnull().sum())

        null_count = self.df.isnull().sum()
        null_percent = self.df.isnull().mean() * 100
        data_type = self.df.dtypes
        df_unique = self.df.nunique()

        MisVal = pd.DataFrame({
            'Missing Values': null_count,
            'Percentage': null_percent,
            'Data Type': data_type,
            'Unique Values': df_unique
        })

        print("Missing Values Information:")
        print(MisVal.sort_values(by='Percentage', ascending=False))    
        
    def preprocess(self):
        # Fill missing values
        self.df['type_of_meal_plan'] = self.df['type_of_meal_plan'].fillna(self.df['type_of_meal_plan'].mode()[0])
        self.df['required_car_parking_space'] = self.df['required_car_parking_space'].fillna(0)
        self.df['avg_price_per_room'] = self.df['avg_price_per_room'].fillna(self.df['avg_price_per_room'].median())

        # Convert int64 columns to float64
        itg = self.df.select_dtypes(include=["int64"]).columns
        self.df[itg] = self.df[itg].astype('float64')
        self.df[['repeated_guest','required_car_parking_space']] = self.df[['repeated_guest','required_car_parking_space']].astype('int64')

        # Visualizations
        sns.histplot(x='booking_status', data=self.df)
        plt.xlabel('Booking Status')
        plt.ylabel('Count')
        plt.title('Booking Status Distribution')
        plt.show()

        numerics = self.df.select_dtypes(include=["float64", "int64"])
        n_cols = 3
        n_rows = (len(numerics.columns) + n_cols - 1) // n_cols

        plt.figure(figsize=(n_cols * 10, n_rows * 8))
        for i, col in enumerate(numerics.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,8))
        sns.heatmap(numerics.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        for i, col in enumerate(numerics.columns, 1):
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df[col], color='skyblue')
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            plt.show()

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(self.df[['type_of_meal_plan','room_type_reserved','market_segment_type']])
        df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['type_of_meal_plan','room_type_reserved','market_segment_type']))
        self.df_encoded = pd.concat([self.df.drop(['type_of_meal_plan','room_type_reserved','market_segment_type'], axis=1), df_encoded], axis=1)

        scaler = StandardScaler()
        nums_scaled = scaler.fit_transform(self.df_encoded[[
            "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
            "lead_time", "arrival_year", "arrival_month", "arrival_date",
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
            "avg_price_per_room", "no_of_special_requests"]])

        self.df_scaled = pd.DataFrame(nums_scaled, columns=[
            "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
            "lead_time", "arrival_year", "arrival_month", "arrival_date",
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
            "avg_price_per_room", "no_of_special_requests"])

        # Merge scaled numerics and rest of encoded dataframe
        self.clean_df = pd.concat([
            self.df_scaled,
            self.df_encoded.drop([
                "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
                "lead_time", "arrival_year", "arrival_month", "arrival_date",
                "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
                "avg_price_per_room", "no_of_special_requests"
            ], axis=1)
        ], axis=1)

    def train_and_evaluate(self):
        X = self.clean_df.drop(columns='booking_status')
        Y = self.clean_df['booking_status']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Label encoding for target
        encoder = LabelEncoder()
        Y_train_encoded = encoder.fit_transform(Y_train)
        Y_test_encoded = encoder.transform(Y_test)

        # Train XGBoost classifier
        xg_boost = xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        xg_boost.fit(X_train, Y_train_encoded)
        preds = xg_boost.predict(X_test)

        accuracy = accuracy_score(Y_test_encoded, preds)
        report = classification_report(Y_test_encoded, preds)
        confusion = confusion_matrix(Y_test_encoded, preds)

        print("Accuracy: ", accuracy)
        print("Classification Report: \n", report)
        print("Confusion Matrix: \n", confusion)

    def save_model(self, filename='best_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved as {filename}")
