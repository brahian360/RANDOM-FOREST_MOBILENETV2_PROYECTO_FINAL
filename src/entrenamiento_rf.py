"""
entrenamiento_rf.py
Este script entrena y evalúa un Random Forest sobre las características extraídas del dataset balanceado.
Todos los archivos generados por este modelo se guardan dentro de analysis_results/RandomForest para mantener todo organizado.
Incluye el resumen en TXT, el modelo entrenado, las métricas y los datos clave para visualizaciones avanzadas y comparaciones.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Defino las rutas principales del proyecto, asumiendo la estructura estándar
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(root_dir, "extraccion_features", "caracteristicas.csv")
output_dir = os.path.join(root_dir, "analysis_results", "RandomForest")
os.makedirs(output_dir, exist_ok=True)

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"No encontré el archivo de características en: {csv_path}")

# Cargo el dataset de features
df = pd.read_csv(csv_path)
X = df.drop(columns=["filename", "class", "class_index"])
y = df["class"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Resumen del dataset
resumen = []
resumen.append("--- Resumen del dataset de características ---")
resumen.append(f"Total de muestras: {len(df)}")
resumen.append(f"Número de clases: {len(np.unique(y))}")
resumen.append(f"Clases: {list(le.classes_)}")
resumen.append(f"Shape de X: {X.shape}")

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
resumen.append(f"\nTamaño train: {X_train.shape[0]}")
resumen.append(f"Tamaño test: {X_test.shape[0]}")

# Entrenamiento del modelo
resumen.append("\nEntrenando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluación
start = time.time()
probs_rf = rf.predict_proba(X_test)
pred_rf = rf.predict(X_test)
t_rf = (time.time() - start) / len(X_test)

acc_rf = accuracy_score(y_test, pred_rf)
f1_rf = f1_score(y_test, pred_rf, average="weighted")
cm_rf = confusion_matrix(y_test, pred_rf)
report_rf = classification_report(y_test, pred_rf, target_names=le.classes_, output_dict=True)
top_feats = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)

# Métricas detalladas por clase
df_report = pd.DataFrame(report_rf).T
resumen.append("\n--- Métricas detalladas por clase ---")
resumen.append(df_report[["precision", "recall", "f1-score", "support"]].round(3).to_string())

# Matriz de confusión
cm_df = pd.DataFrame(cm_rf, index=le.classes_, columns=le.classes_)
resumen.append("\n--- Matriz de confusión ---")
resumen.append(cm_df.to_string())

# Top 10 features más importantes
resumen.append("\n--- Top 10 características más importantes según el Random Forest ---")
resumen.append(top_feats.to_string())

# Resumen general
resumen.append("\n--- Resumen general del modelo Random Forest ---")
resumen.append(f"Precisión global en test: {acc_rf:.2%}")
resumen.append(f"F1-score ponderado: {f1_rf:.3f}")
resumen.append(f"Tiempo promedio de inferencia por muestra: {t_rf:.5f} segundos")

# Guardar el resultado en un archivo TXT dentro de la carpeta del modelo
output_txt = os.path.join(output_dir, "resultados_rf.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(resumen))

print(f"\n✅ El análisis completo del Random Forest se guardó en:\n   {output_txt}\n")

# --- BLOQUE EXTRA PARA VISUALIZACIONES AUTOMÁTICAS ---

# Guarda el modelo Random Forest entrenado
joblib.dump(rf, os.path.join(output_dir, "random_forest_model.joblib"))

# Guarda los datos y métricas clave para visualizaciones
np.save(os.path.join(output_dir, "y_test_rf.npy"), y_test)
np.save(os.path.join(output_dir, "pred_rf.npy"), pred_rf)
np.save(os.path.join(output_dir, "cm_rf.npy"), cm_rf)
np.save(os.path.join(output_dir, "probs_rf.npy"), probs_rf)
top_feats.to_csv(os.path.join(output_dir, "top_feats_rf.csv"))

# Guarda los valores de precisión, f1 y tiempo de inferencia en un txt
with open(os.path.join(output_dir, "metrics_rf.txt"), "w") as f:
    f.write(f"{acc_rf}\n{f1_rf}\n{t_rf}")

# Guarda el reporte de clasificación completo en TXT (opcional, para visualizaciones)
rf_report_txt = classification_report(y_test, pred_rf, target_names=le.classes_)
with open(os.path.join(output_dir, "classification_report_rf.txt"), "w", encoding="utf-8") as f:
    f.write(rf_report_txt)
