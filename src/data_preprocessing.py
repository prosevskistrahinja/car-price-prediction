import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(input_path):
    print("Zapocinjemo predobradu podataka (Data Preprocessing)...")

    # Ucitavanje podataka sa novim karakteristikama
    df = pd.read_csv(input_path)

    # 1. Odabir kljucnih kolona koje cemo koristiti za predikciju cene
    # Izbacujemo "priceusd" jer je to cilj (y), i tekstualne kolone sa previse jedinstvenih vrednosti poput "model"
    X_columns = ["year", "condition", "mileage_kilometers", "fuel_type",
                 "volume_cm3", "color", "transmission", "drive_unit",
                 "segment", "car_age", "mileage_per_year"]
    X = df[X_columns]
    y = df["priceusd"]

    # 2. Transformacija kategorickih varijabli u numericke (One-Hot Encoding)
    # Ovo pretvara tekstualne kolone (npr. fuel_type: gasoline, diesel) u kolone sa jedinicama i nulama
    categorical_cols = ["condition", "fuel_type", "color", "transmission", "drive_unit", "segment"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 3. Podela na trening i test skup (80% za trening, 20% za test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Predobrada zavrsena uspesno!")
    print(f"Dimenzije trening skupa (X_train): {X_train.shape}")
    print(f"Dimenzije test skupa (X_test): {X_test.shape}")

    return X_train, X_test, y_train, y_test 

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("../data/cars_features.csv")