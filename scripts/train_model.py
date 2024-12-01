# Standard Library Imports
import os
import time
import pickle

# Third-Party Library Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import kagglehub

# Define the main training function
def train_and_save_model():
    # Download the dataset
    path = kagglehub.dataset_download("maddiegupta/walmart-sales-forecasting")
    print("Path to dataset files:", path)

    # Load datasets
    train = pd.read_csv(os.path.join(path, "train.csv"))
    features = pd.read_csv(os.path.join(path, "features.csv"))
    stores = pd.read_csv(os.path.join(path, "stores.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))

    # Split the dataset
    split_ratio = 0.8
    train, test = train_test_split(
        train,
        test_size=(1 - split_ratio),
        random_state=42,
        stratify=train['Store']
    )
    print("Training Data Shape (80%):", train.shape)
    print("Testing Data Shape (20%):", test.shape)

    # Define preprocessing function
    def preprocess_data(data, features, stores):
        data = data.merge(features[['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']], on=["Store", "Date"], how="left")
        data = data.merge(stores[['Store', 'Type', 'Size']], on="Store", how="left")
        data['Date'] = pd.to_datetime(data['Date'])
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['month'] = data['Date'].dt.month

        # Calculate CPI and Unemployment per store
        data['CPI_per_store'] = data.groupby('Store')['CPI'].transform('mean')
        data['Unemployment_per_store'] = data.groupby('Store')['Unemployment'].transform('mean')

        # Handle IsHoliday merging issues
        if 'IsHoliday_x' in data.columns and 'IsHoliday_y' in data.columns:
            data['IsHoliday'] = data['IsHoliday_x'].fillna(data['IsHoliday_y'])
            data.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)
        elif 'IsHoliday' not in data.columns:
            data['IsHoliday'] = data['IsHoliday_x'] if 'IsHoliday_x' in data.columns else data['IsHoliday_y']
            data.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)

        data['IsHoliday'] = data['IsHoliday'].astype(int)  # Convert boolean to int if necessary
        data = pd.get_dummies(data, columns=['Type'])  # One-hot encode the 'Type' column

        return data

    # Apply preprocessing
    train = preprocess_data(train, features, stores)
    test = preprocess_data(test, features, stores)

    # Feature Engineering for XGBoost Model
    feature_columns = ['Temperature', 'day_of_week', 'month', 'Fuel_Price', 'CPI_per_store', 'Unemployment_per_store', 'Size', 'IsHoliday', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    feature_columns += [col for col in train.columns if col.startswith('Type_')]  # Include one-hot encoded 'Type' columns
    xgboost_train_data = train[feature_columns]
    xgboost_test_data = test[feature_columns]

    X_train = xgboost_train_data
    X_test = xgboost_test_data

    # Setup parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9]
    }

    # Setup the XGBoost model for RandomizedSearchCV
    xgboost_model = XGBRegressor(objective='reg:squarederror')
    random_search = RandomizedSearchCV(
        estimator=xgboost_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Run Randomized Search
    start_time = time.time()
    random_search.fit(X_train, train['Weekly_Sales'])
    elapsed_time = time.time() - start_time

    best_model = random_search.best_estimator_
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "optimized_model.pkl")
    with open(output_path, 'wb') as file:
        pickle.dump(best_model, file)

    print(f"Random grid search completed in {elapsed_time:.2f} seconds.")
    print(f"Optimized model saved to: {output_path}")

    return elapsed_time
