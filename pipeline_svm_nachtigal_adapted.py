# pipeline_svm_nachtigal_adapted.py
"""
Pipeline SVM appliqué aux données SCADA + DGA du site Nachtigal.
Fichier d'entrée : Simulated_DGA_SCADA_Sample.csv
Sorties : modèle optimisé, figures (matrice de confusion, etc.)
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# === CONFIGURATION ===
DATA_CSV = "Simulated_DGA_SCADA_Sample.csv"   # fichier CSV uploadé
OUTPUT_DIR = "SVM_Pipeline_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colonnes sélectionnées (features)
FEATURES = [
    "H2", "CH4", "C2H2", "C2H4", "C2H6",
    "CO", "CO2", "TDCG",
    "oil_temp_mean", "oil_temp_max", "moisture", "load_MVA"
]

TARGET = "label"            # colonne des classes
GROUP_COL = "transformer_id" # utile pour GroupKFold si besoin

# === 1. Chargement des données ===
df = pd.read_csv(DATA_CSV, parse_dates=["date"])
print("Aperçu des données :", df.head())

X = df[FEATURES]
y = df[TARGET]

# === 2. Split temporel (75% train, 25% test) ===
n = len(df)
split_idx = int(n * 0.75)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# === 3. Pipeline ===
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

# === 4. Optimisation GridSearchCV ===
param_grid = {
    "clf__C": [1, 10, 50],
    "clf__gamma": [0.001, 0.01, 0.1]
}
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Meilleurs paramètres :", grid.best_params_)

# Sauvegarde du modèle
best_model = grid.best_estimator_
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "SVM_Nachtigal_Model.joblib"))

# === 5. Évaluation ===
y_pred = best_model.predict(X_test)
print("\nRapport classification :\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title("Matrice de confusion – SVM Nachtigal")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}")
