import joblib
import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model():
    print("Zapocinjem evaluaciju modela...")

    # 1. Ucitavamo podatke (treba nam test skup X_test i y_test)
    _, X_test, _, y_test = preprocess_data("../data/cars_features.csv")

    # 2. Ucitavamo nas sacuvani model
    model = joblib.load("../models/car_price_model.joblib")

    # 3. Pravimo predikcije nas test podacima 
    y_pred = model.predict(X_test)

    # 4. Racunamo metrike
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Rezultati evaluacije modela ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} USD")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R2 Score (Koeficijent determinacije): {r2:.4f}")

    return mae, mse, r2

if __name__ == "__main__":
    evaluate_model()