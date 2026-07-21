import joblib
from data_preprocessing import preprocess_data
from sklearn.linear_model import LinearRegression

def train_model():
    print("Zapocinjemo treniranje modela...")

    # 1. Ucitavamo i preprocesiramo podatke pomocu funkcije iz prethodne skripte
    X_train, X_test, y_train, y_test = preprocess_data("../data/cars_features.csv")

    # 2. Inicijalizujemo model Linearne Regresije
    model = LinearRegression()

    # 3. Treniramo model na podacima za trening
    model.fit(X_train, y_train)
    print("Model je uspesno istreniran!")

    # 4. Cuvamo istreniran moodel u "models" folder kako bismo ga koristili kasnije
    model_path = "../models/car_price_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model je sacuvan u: {model_path}")

    return model

if __name__ == "__main__":
    train_model()