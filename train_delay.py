# AI Based Train Delay Prediction System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create dataset
data = {
    'Distance': [100, 250, 300, 150, 500, 400, 120, 350],
    'Weather': [0, 1, 0, 0, 1, 1, 0, 1],
    'DayType': [0, 1, 0, 1, 1, 0, 0, 1],
    'PastDelay': [0, 1, 0, 1, 1, 0, 0, 1],
    'Result': [0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Split input and output
X = df[['Distance', 'Weather', 'DayType', 'PastDelay']]
y = df['Result']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Take new input
# Example: Distance=200km, Bad weather, Weekday, Past delay happened
new_data = [[200, 1, 0, 1]]

prediction = model.predict(new_data)

# Step 6: Show result
if prediction[0] == 1:
    print("🚨 Train is likely to be DELAYED")
else:
    print("✅ Train is likely to be ON TIME")
