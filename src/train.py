import numpy as np
import joblib
import os
import logging
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

MODEL_DIR = "models"
MODEL_NAME = "linear_model.joblib"
TEST_DATA_NAME = "test_data.joblib"
SEED = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_and_prepare_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    logging.info("Dataset shape: %s, Target shape: %s", X.shape, y.shape)
    logging.info("Features: %s", housing.feature_names)
    return X, y

def train_model(X, y, test_size=TEST_SIZE, seed=SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logging.info("Training R²: %.4f, RMSE: %.4f", train_r2, train_rmse)
    logging.info("Testing  R²: %.4f, RMSE: %.4f", test_r2, test_rmse)
    return model, (X_test, y_test)

def save_model_and_test(model, test_data):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    test_path = os.path.join(MODEL_DIR, TEST_DATA_NAME)

    joblib.dump(model, model_path)
    joblib.dump(test_data, test_path)

    logging.info("Model saved to %s", model_path)
    logging.info("Test data saved to %s", test_path)
    logging.debug("Model coefficients shape: %s", model.coef_.shape)
    logging.debug("Model intercept: %.6f", model.intercept_)
    logging.debug("First 5 coefficients: %s", model.coef_[:5])

def main():
    logging.info("Starting California Housing Linear Regression pipeline.")
    X, y = load_and_prepare_data()
    model, test_data = train_model(X, y, test_size=TEST_SIZE, seed=SEED)
    save_model_and_test(model, test_data)
    logging.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()