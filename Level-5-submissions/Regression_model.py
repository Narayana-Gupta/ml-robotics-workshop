#task1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('robot_movements.csv')
X, y = data[['speed', 'acceleration', 'rotation']], data['future_movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse, r2 = np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}\nR-squared: {r2:.2f}")

pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv('robot_movement_predictions.csv', index=False)

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Actual vs Predicted')
plt.show()
