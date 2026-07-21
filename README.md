# Predikcija Cena Automobila (Car Price Prediction)
Ovaj projekat predstavlja kompletan *Machine Learning* pipeline za predikciju cena polovnih automobila na osnovu njihovih karakteristika (godina proizvodnje, kilometraža, tip pogona, vrsta goriva, segment vozila itd.).

## 📂 Struktura Projekta
- "data/" - Sadrži originalne podatke ("cars.csv") i generisane privremene skupove podataka.
- "models/" - Sačuvan istrenirani finalni model u ".joblib" formatu.
- "src/" - Izvorni Python kod podeljen po fazama razvoja:
- "data_cleaning.py" - Čišćenje podataka, obrada nedostajućih vrednosti i uklanjanje anomalija.
- "feature_engineering.py" - Kreiranje novih kolona ("car_age" i "mileage_per_year").
- "data_preprocessing.py" - Enkodiranje kategoričkih varijabli i podela na train/test skupove.
- "model_training.py" - Treniranje modela Linearne Regresije i čuvanje modela.
- "model_evaluation.py" - Evaluacija modela kroz MAE, MSE i R2 metrike.
- "model_comparison.py" - Generisanje grafikona stvarnih naspram predviđenih vrednosti.
- "requirements.txt" - Spisak potrebnih Python biblioteka.
- "predictions_vs_actual.png" - Vizuelni prikaz tačnosti modela.

## 📊 Rezultati Modelovanja
Nakon uspešnog treniranja i evaluacije modela Linearne Regresije na test skupu, dobijeni su sledeći rezultati:
- **Mean Absolute Error (MAE):** 3023.69 USD (prosečno odstupanje predviđene cene od stvarne)
- **R2 Score:** 0.5081 (model uspešno objašnjava oko 51% varijanse u cenama automobila)

## 🚀 Kako pokrenuti projekat
1. Instalirajte potrebne biblioteke:
```bash
pip install -r requirements.txt