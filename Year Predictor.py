import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


# This dataset contains ~515,345 data points and 91 columns (1 target + 90 features).
df_reg = pd.read_csv("year_prediction.csv", low_memory=False)

# Separate features and target.
# The first column is the target (release year), and the remaining columns are features.
X_reg = df_reg.iloc[:, 1:].to_numpy().astype(float)
y_reg = df_reg.iloc[:, 0].to_numpy().astype(float)

# Manually normalize the features and target.
# For each feature: subtract the mean and divide by the standard deviation.
X_reg = (X_reg - np.mean(X_reg, axis=0)) / np.std(X_reg, axis=0)
y_reg = (y_reg - np.mean(y_reg)) / np.std(y_reg)

# Create a train/test split that is highly unbalanced (10% training, 90% test)
# This small training set increases the chance of overfitting.
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, train_size=0.1, random_state=42
)

# Set up the Ridge regression model and grid search for optimal alpha.
# The cost function minimized by Ridge is: ||y - Xβ||² + α||β||².
ridge_reg = Ridge()
# Define a grid of α values (regularization strength).
alpha_grid = {"alpha": np.linspace(0.01, 0.05, num=30)}
grid_search_reg = GridSearchCV(ridge_reg, param_grid=alpha_grid, cv=5, scoring="r2")
grid_search_reg.fit(X_reg_train, y_reg_train)

# Get the best alpha from grid search.
best_alpha = grid_search_reg.best_params_["alpha"]
print("=== Regression using Ridge ===")
print("Best alpha from GridSearchCV:", best_alpha)

# Evaluate the model on the test set.
ridge_best = grid_search_reg.best_estimator_
y_reg_pred = ridge_best.predict(X_reg_test)
r2_reg = r2_score(y_reg_test, y_reg_pred)
rmse_reg = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print("Test R^2 score:", r2_reg)
print("Test RMSE:", rmse_reg)


# Test R^2 score is always around .25
#  -- this means that my model explains about 25% of the varience
#  -- this makes sense becuase of the 10/90 split (to make the problem harder)

# Test RMSE score is always around .85
#  -- Average magnitude of errors between predicted and actual
#  -- Both of these demonstrate the effect of using a very limited training set
#  -- However, overfitting was likely as requested by the assignment