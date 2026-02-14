import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample weather data
data = {
    "Day": [1,2,3,4,5,6,7,8,9,10],
    "Temperature": [30,32,31,29,28,35,36,34,33,37]
}

df = pd.DataFrame(data)

# Input (Day)
X = df[["Day"]]

# Output (Temperature)
y = df["Temperature"]

# Create model
model = LinearRegression()
model.fit(X, y)

# Predict temperature for Day 11
prediction = model.predict([[11]])

print("Predicted Temperature for Day 11:", prediction[0])

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Weather Forecasting Model")
plt.show()
