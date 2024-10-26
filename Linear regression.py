# Importing the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Simulating a dataset (you can replace this with real-world data)
data = {
    'weather': [1, 2, 1, 3, 1, 2, 3, 1, 3, 2],  # 1 = Clear, 2 = Rainy, 3 = Foggy
    'speed': [60, 85, 70, 90, 40, 55, 80, 72, 88, 50],  # Speed of the vehicle
    'road_type': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # 1 = Highway, 2 = Dry weather Road
    'severity': [1, 3, 2, 4, 1, 2, 3, 2, 4, 1]  # Accident severity: 1 = low, 4 = high
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Defining the dependent variable (y) and independent variables (X)
X = df[['weather', 'speed', 'road_type']]  # Independent variables
y = df['severity']  # Dependent variable (Accident severity)

# Spliting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the accident severity on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Saving the model
joblib.dump(model, 'accident_severity_model.pkl')

# Visualizing actual vs predicted severity (optional for analysis)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.xlabel('Actual Severity')
plt.ylabel('Predicted Severity')
plt.title('Actual vs Predicted Severity')
plt.show()
# Loading the saved model
model = joblib.load('accident_severity_model.pkl')

# Defining a hypothetical set of independent variables
# Hypothetical scenario: Clear weather (1), speed of 80, highway (1)
hypothetical_data = [[1, 80, 1]]

# Predicting accident severity for the hypothetical scenario
predicted_severity = model.predict(hypothetical_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
