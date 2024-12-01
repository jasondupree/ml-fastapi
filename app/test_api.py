import warnings
import urllib3
from fastapi.testclient import TestClient
from app.main import app  # Import FastAPI app from main.py

# Suppress specific warning from urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

# Create a TestClient instance to interact with the FastAPI app
client = TestClient(app)

def test_unit_price():
    """
    Test the '/predict' endpoint.
    """
    # Arrange: Define the request body with all required fields
    request_body = {
        "Temperature": 72.5,
        "day_of_week": 3,
        "month": 6,
        "Fuel_Price": 3.45,
        "CPI_per_store": 142.6,
        "Unemployment_per_store": 7.8,
        "Type_A": 1,
        "Type_B": 0,
        "Type_C": 0,
        "Size": 104000,
        "IsHoliday": False,
        "MarkDown1": 2532.32,
        "MarkDown2": 292.34,
        "MarkDown3": 244.57,
        "MarkDown4": 1235.5,
        "MarkDown5": 3099.38
    }  # Example values for all the required fields

    # Act: Send a POST request to the '/predict' endpoint
    response = client.post("/predict", json=request_body)
    
    # Assert: Check if the status code is 200 (OK) and the response contains 'predictions'
    assert response.status_code == 200
    assert "predictions" in response.json()