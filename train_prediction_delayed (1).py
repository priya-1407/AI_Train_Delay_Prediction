# AI-Based Train Delay Prediction System

import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1: Training Data
# Features: Distance, Weather, DayType
X = np.array([
    [300, 0, 0],   # Sunny, Weekday
    [500, 2, 1],   # Rainy, Weekend
    [200, 1, 0],   # Cloudy, Weekday
    [400, 2, 0],   # Rainy, Weekday
    [600, 1, 1]    # Cloudy, Weekend
])

# Target: Delay in minutes
y = np.array([5, 25, 8, 18, 20])

# Step 2: Create AI Model
model = LinearRegression()

# Step 3: Train the Model
model.fit(X, y)

# Step 4: User Input
print("Enter Train Details for Delay Prediction")

distance = int(input("Enter distance in km: "))
weather = int(input("Enter weather (0-Sunny, 1-Cloudy, 2-Rainy): "))
day_type = int(input("Enter day type (0-Weekday, 1-Weekend): "))

# Step 5: Prediction
input_data = np.array([[distance, weather, day_type]])
predicted_delay = model.predict(input_data)

# Step 6: Output
print("\nPredicted Train Delay:")
print(round(predicted_delay[0]), "minutes")
