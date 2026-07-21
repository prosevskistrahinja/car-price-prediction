import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    print("Zapocinjem ciscenje podataka...")

    # Ucitavanje podataka
    df = pd.read_csv(input_path)

    # 1. Standardizacija naziva kolona (pretvaramo u mala slova i menjamo cudne znakove)
    df.columns = df.columns.str.lower().str.replace("(", "_", regex=False).str.replace(")","", regex=False)
    # Sada su kolone: mileage_kilometers i volume_cm3 umesto sa zagradama

    # 2. Resavanje nedostajucih vrednosti 
    # Za numericku kolonu "volume_cm3" menjamo prazna polja medijanom 
    if "volume_cm3" in df.columns:
        median_volume = df["volume_cm3"].median()
        df["volume_cm3"] = df["volume_cm3"].fillna(median_volume)

    # Za kategoricke kolone "drive_unit" i "segment" menjamo prazna polja sa "unknown"
    if "drive_unit" in df.columns:
        df["drive_unit"] = df["drive_unit"].fillna("unknown")
    if "segment" in df.columns:
        df["segment"] = df["segment"].fillna("unknown")

    # 3. Filtriranje ekstremnih/nevalidnih vrednosti (provera anomalija)
    # Uklanjamo automobile koji imaju nerealnu cenu (npr. manju od 100 dolara) ili nerealnu godinu
    df = df[df["priceusd"] > 100]
    df = df[df["year"] > 195]

    # Cuvanje ociscenih podataka u novi CSV fajl
    df.to_csv(output_path, index=False)
    print(f"Ciscenje zavrseno! Ocisceni podaci sacuvani su u: {output_path}")
    print(f"Preostalo redova nakon ciscenja: {df.shape[0]}")
    return df

if __name__ == "__main__":
    # Pokretanje skripte kada se pokrene direktno
    clean_data("../data/cars.csv", "../data/cars_cleaned.sv")