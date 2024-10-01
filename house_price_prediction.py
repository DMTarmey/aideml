import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
data_path = "/home/dmtarmey/AiderML/house_prices.csv"  # Update with your actual data file
data = pd.read_csv(data_path)

# Preprocess the data
# Assuming 'Price' is the target variable and others are features
features = data.drop('Price', axis=1)
target = data['Price']

# Handle categorical variables and missing values if necessary
features = pd.get_dummies(features).fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
