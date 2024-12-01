import random
from locust import HttpUser, constant, task

class UnitPriceUser(HttpUser):
    wait_time = constant(1)  # Simulates 1 second between requests
    
    @task
    def unit_price(self):
        # Arrange: Create a random distance between 1 and 6500 meters
        distance = 1.0 + random.random() * 6499.0  # Distance in meters

        # Send a POST request to the /predict endpoint with the generated random distance
        response = self.client.post(
            "/predict",
            json={
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
                "MarkDown5": 3099.38,
                "Distance": distance  # Pass the random distance as part of the payload
            }
        )

        # Optionally, you can print the response for debugging
        # print(f"Response status code: {response.status_code}, Response body: {response.text}")