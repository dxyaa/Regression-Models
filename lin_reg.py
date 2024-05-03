# Define a function to encode categorical variables
def encode_categorical_variables(year, kms_driven, fuel_type, seller_type, transmission, owner):
    # Encode fuel_type
    fuel_type_encoded = 1 if fuel_type == 'Petrol' else 0  # Example encoding

    # Encode seller_type
    seller_type_encoded = 1 if seller_type == 'Individual' else 0  # Example encoding

    # Encode transmission
    transmission_encoded = 1 if transmission == 'Manual' else 0  # Example encoding

    # Return encoded values
    return year, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\Diya\Downloads\archive\datacar.csv")
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=["Fuel_Type", "Seller_Type", "Transmission"])
X = data.drop(['Selling_Price', 'Car_Name'], axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Example of making a prediction with new data
new_data_categorical = [2016, 80000, 'Petrol', 'Individual', 'Manual', 1]  # Categorical variables
new_data_numerical = encode_categorical_variables(*new_data_categorical)  # Encode categorical variables
new_data = np.array([new_data_numerical])  # Combine numerical features with categorical features
new_data = np.hstack((new_data, np.zeros((new_data.shape[0], X_train.shape[1] - new_data.shape[1]))))  # Pad with zeros for missing features
prediction = model.predict(new_data)  # Make prediction
print("Predicted Selling Price:", prediction[0])

# Plotting the graph for actual vs predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()
