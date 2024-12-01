from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import sys
import pandas as pd  # Ensure pandas is imported

# Add the 'scripts' directory to the Python path
scripts_path = os.path.join(os.getcwd(), "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

# Try to import the training function from train_model.py
try:
    from train_model import train_and_save_model
except ImportError as e:
    raise ImportError(f"Failed to import 'train_and_save_model': {str(e)}")

app = FastAPI()

# Define the path to the saved model (updated to the full feature model)
# MODEL_PATH = os.path.join(os.getcwd(), "models", "full_feature_model.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "models", "optimized_model.pkl")


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}

@app.get("/train")
def train_model():
    try:
        # Call the modularized training function
        elapsed_time = train_and_save_model()
        return {
            "message": f"Model training completed in {elapsed_time:.2f} seconds. Model saved to '{MODEL_PATH}'"
        }
    except FileNotFoundError as e:
        return {"error": f"File not found: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred during training: {str(e)}"}

class PredictInput(BaseModel):
    Temperature: float
    day_of_week: int
    month: int
    Fuel_Price: float
    CPI_per_store: float
    Unemployment_per_store: float
    Type_A: int
    Type_B: int
    Type_C: int  # Adjust types as per your specific types used during training
    Size: int
    IsHoliday: bool
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float

@app.post("/predict")
def predict(input_data: PredictInput):
    try:
        # Load the trained model
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model file not found. Please train the model first.")

        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        # List of feature names as used in the model training
        feature_names = [
            'Temperature', 'day_of_week', 'month', 'Fuel_Price', 'CPI_per_store', 'Unemployment_per_store',
            'Size', 'IsHoliday', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'
        ] + [col for col in model.get_booster().feature_names if 'Type_' in col]  # Dynamically include Type columns

        # Create a list of input values in the exact order of feature_names
        input_values = [
            input_data.Temperature, input_data.day_of_week, input_data.month,
            input_data.Fuel_Price, input_data.CPI_per_store, input_data.Unemployment_per_store,
            input_data.Size, input_data.IsHoliday,
            input_data.MarkDown1, input_data.MarkDown2, input_data.MarkDown3,
            input_data.MarkDown4, input_data.MarkDown5
        ] + [getattr(input_data, col) for col in model.get_booster().feature_names if 'Type_' in col]

        # Convert input data to dataframe expected by the model
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Make predictions using the model
        predictions = model.predict(input_df)
        return {"predictions": predictions.tolist()}  # Convert to list if it's a numpy array
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
