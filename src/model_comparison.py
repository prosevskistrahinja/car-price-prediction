import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import preprocess_data

def compare_and_visualize():
    print("Zapocinjem vizuelnu evaluaciju modela...")

    # 1. Ucitavamo test podatke
    _, X_test, _, y_test = preprocess_data("../data/cars_features.csv")

    # 2. Ucitavamo nas istrenirrani model 
    model = joblib.load("../models/car_price_model.joblib")

    # 3. Pravimo predikcije
    y_pred = model.predict(X_test)

    # 4. Kreiranje grafikona: Stvarne vs. Predvidjene cene
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="teal")

    # Savrsena linija predikcije (gde je stvarna cena jednaka predvidjenoj)
    ideal_line_min = min(y_test.min(), y_pred.min())
    ideal_line_max = max(y_test.max(), y_pred.max())
    plt.plot([ideal_line_min, ideal_line_max], [ideal_line_min, ideal_line_max], color="red", linestyle="--", lw=2)

    plt.title("Stvarne cene automobila vs. Predvidjene cene (Linearr Regression)")
    plt.xlabel("Stvarne cene (USD)")
    plt.ylabel("Predvidjene cene (USD)")
    plt.grid(True)

    # Cuvanje grafikona kao slike u koren foldera ili unutar notebooks
    plt.savefig("predictions_vs_actual.png")
    print("Grafikon uspesno generisan i sacuvan kao prediction_vs_actual.png!")
    plt.show()

if __name__ == "__main__":
    compare_and_visualize()