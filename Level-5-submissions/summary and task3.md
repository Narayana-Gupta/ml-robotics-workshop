# Summary of Findings
# Task 1: Regression Model
- Performance Metrics:

  - Root Mean Squared Error (RMSE): This metric quantifies the typical prediction error. A lower RMSE indicates better predictive accuracy.
  - R-squared (R²): Represents the proportion of variance in the target variable (future_movement) explained by the model. Higher values suggest a stronger fit.
- Observations:

  - The regression model's RMSE and R² suggest reasonable predictive performance, with the model able to capture some of the trends in future movements.
  - The scatter plot comparing actual and predicted values indicates that the model is generally able to approximate future movements, but certain outliers suggest areas for improvement.
# Task 2: Classification Model
- Evaluation Metrics:

  - Accuracy: The percentage of correct predictions made by the model.
  - Precision: The ratio of true positive predictions for each class, averaged across all classes.
  - Recall: The model’s ability to identify all relevant instances of each class.
# Confusion Matrix Insights:

- The confusion matrix provides a visual breakdown of the model’s performance across the various classes (normal, warning, and critical).
Findings:
- The classification model demonstrated good accuracy and was effective in predicting the correct labels.
- Precision and recall scores show balanced performance, but some challenges were observed in distinguishing between warning and critical labels.
- The confusion matrix confirmed that while the model excelled at identifying normal and critical states, it had a few misclassifications in the warning category.


![image](https://github.com/user-attachments/assets/71a89932-f0a4-4d2e-aed9-9e30aeae6ec8)
![image](https://github.com/user-attachments/assets/4e07d828-681a-4c5a-87bd-8c01f7f87bdd)
