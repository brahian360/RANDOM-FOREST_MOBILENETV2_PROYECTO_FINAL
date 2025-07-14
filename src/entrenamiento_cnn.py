"""
entrenamiento_cnn.py
Este script entrena y evalúa un modelo MobileNetV2 sobre las imágenes balanceadas.
Todos los archivos generados por este modelo se guardan dentro de analysis_results/MobileNetV2 para mantener todo organizado.
Incluye el resumen en TXT, el modelo entrenado, las métricas, las probabilidades y los datos clave para visualizaciones y comparaciones avanzadas.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import time
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Defino las rutas principales del proyecto, asumiendo la estructura estándar
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(root_dir, "dataset_balanceado")
output_dir = os.path.join(root_dir, "analysis_results", "MobileNetV2")
os.makedirs(output_dir, exist_ok=True)

img_size = (224, 224)
batch_size = 32

# Generador de datos con validación del 20%
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    subset="training", class_mode="categorical", shuffle=True
)
val_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    subset="validation", class_mode="categorical", shuffle=False
)

# Defino la arquitectura MobileNetV2 congelando la base
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size,3))
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Entreno el modelo con early stopping
history = model.fit(
    train_gen, validation_data=val_gen, epochs=10,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

# Evaluación del modelo en el set de validación
val_loss, val_acc = model.evaluate(val_gen)
resumen = []
resumen.append("🔍 Evaluación del modelo MobileNetV2")
resumen.append(f"Precisión en validación final: {val_acc:.2%}")
resumen.append(f"Pérdida en validación final: {val_loss:.4f}")

# Reporte de clasificación y matriz de confusión
probs_cnn = model.predict(val_gen, verbose=0)
pred_cnn  = np.argmax(probs_cnn, axis=1)
class_report = classification_report(val_gen.classes, pred_cnn, target_names=list(train_gen.class_indices.keys()))
cm = confusion_matrix(val_gen.classes, pred_cnn)

resumen.append("\n--- Reporte de clasificación por clase ---")
resumen.append(class_report)
resumen.append("\n--- Matriz de confusión ---")
cm_df = pd.DataFrame(cm, index=list(train_gen.class_indices.keys()), columns=list(train_gen.class_indices.keys()))
resumen.append(cm_df.to_string())

# Resumen general
f1_cnn = f1_score(val_gen.classes, pred_cnn, average="weighted")
resumen.append("\n--- Resumen general del modelo MobileNetV2 ---")
resumen.append(f"Precisión global en validación: {val_acc:.2%}")
resumen.append(f"F1-score ponderado: {f1_cnn:.3f}")

# Guardar el resultado en un archivo TXT dentro de la carpeta del modelo
output_txt = os.path.join(output_dir, "resultados_cnn.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(resumen))

print(f"\n✅ El análisis completo del modelo MobileNetV2 se guardó en:\n   {output_txt}\n")

# --- BLOQUE EXTRA PARA VISUALIZACIONES AUTOMÁTICAS ---

# Guarda el history de entrenamiento (usa joblib para objetos grandes)
joblib.dump(history.history, os.path.join(output_dir, "history_cnn.joblib"))

# Guarda las predicciones y etiquetas verdaderas
np.save(os.path.join(output_dir, "val_classes_cnn.npy"), val_gen.classes)
np.save(os.path.join(output_dir, "pred_cnn.npy"), pred_cnn)
np.save(os.path.join(output_dir, "cm_cnn.npy"), cm)
np.save(os.path.join(output_dir, "probs_cnn.npy"), probs_cnn)

# Guarda los valores de precisión, f1, pérdida y tiempo en un txt
start = time.time()
_ = model.predict(val_gen, verbose=0)
t_cnn = (time.time() - start) / len(val_gen.filepaths)
with open(os.path.join(output_dir, "metrics_cnn.txt"), "w") as f:
    f.write(f"{val_acc}\n{f1_cnn}\n{val_loss}\n{t_cnn}")

# Guarda el reporte de clasificación completo en TXT (opcional, para visualizaciones)
with open(os.path.join(output_dir, "classification_report_cnn.txt"), "w", encoding="utf-8") as f:
    f.write(class_report)
