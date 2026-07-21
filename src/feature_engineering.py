import pandas as pd

def engineer_features(input_path, output_path):
    print("Zapocinjem kreiranje novih karakteristika(Feature Engineering)...")

    # Ucitavanje ociscenih podataka
    df = pd.read_csv(input_path)

    # Kreiramo novu kolonu "car_age" (starost auta)
    # Posto radimo u 2026. godini, starost racunamo u odnosu na nju
    df["car_age"] = 2026 - df["year"]

    # Kreiramo jos jednu korisnu kolonu: prosecna kilometraza po godini
    # Dodajemo 1 na car_age da izbegnemo deljenje sa nulom za nova vozila
    df["mileage_per_year"] = df["mileage_kilometers"] / (df["car_age"] + 1)

    # Cuvanje podataka sa novim kolonama
    df.to_csv(output_path, index=False)
    print(f"Feature engineering zavrsen! Podaci sacuvani u: {output_path}")
    print(f"Nove kolone u tabeli: car_age i mileage_per_year")
    return df

if __name__ == "__main__":
    # Pokretanje skripte nad ociscenim podacima
    engineer_features("../data/cars_cleaned.csv", "../data/cars_features.csv")