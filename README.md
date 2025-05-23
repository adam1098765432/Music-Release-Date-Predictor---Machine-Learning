# Music Release Date Prediction using Ridge Regression

This project uses Ridge Regression to predict the release year of music tracks based on a set of features. The model attempts to find the best fit line that minimizes the cost function while also adding regularization to prevent overfitting. This implementation includes hyperparameter tuning using GridSearchCV to find the optimal regularization parameter (alpha). The dataset is highly imbalanced, with only 10% of the data used for training, which introduces the challenge of overfitting.

## Dataset Description
The dataset used in this project is year_prediction.csv, which contains approximately 515,345 data points and 91 columns. The columns include:
- Target (Release Year): The first column represents the release year of each music track (this is the target variable we aim to predict).
- Features (Columns 2-91): These columns represent various features that describe the music tracks. The features may include attributes like track length, genre, popularity, or other metadata (specifics of these features are not detailed in the code).  

The cost function minimized by Ridge is: ||y - Xβ||² + α||β||².

## Results
=== Regression using Ridge ===
 - Best alpha from GridSearchCV: 0.05
 - Test R^2 score: 0.2354281309002343
 - Test RMSE: 0.874767739670711  
<br>

Interpretation of Results: 
 - Overfitting: Due to the small training set (only 10% of the data), the model likely overfits, meaning it performs well on the test set but may not generalize well to new, unseen data.

Model Performance: 
 - The R^2 score average of 0.25 suggests that the model has some predictive power but is far from perfect. A significant portion of the variance in the target variable is unexplained, which aligns with the expectation of using a limited training set.

RMSE: 
 - The RMSE average of 0.85 indicates that, on average, the model’s predictions are off by about 0.85 units (on the normalized scale), which could be considered a reasonable error given the limited training data and the challenge of predicting the release year of music.
