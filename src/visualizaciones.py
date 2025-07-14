"""
visualizaciones.py
Script completo: genera todas las gráficas y tablas relevantes para comparar Random Forest y MobileNetV2.
- La tabla combinada de métricas por clase se guarda como TXT (no CSV).
- Se generan más ejemplos visuales de errores para ambos modelos y se guardan imágenes en carpetas separadas.
- En la carpeta de errores de RF se guarda un TXT con los nombres de los archivos mal clasificados.
Guarda todo en analysis_results/Graficos y analysis_results/ErroresEjemplos.
Solo ejecuta el script después de entrenar ambos modelos.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve
)
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

# Configuración global de estilo profesional
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight'
})
sns.set_palette("colorblind")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
graphs_dir = os.path.join(root_dir, "analysis_results", "Graficos")
errors_dir = os.path.join(root_dir, "analysis_results", "ErroresEjemplos")
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(errors_dir, exist_ok=True)
os.makedirs(os.path.join(errors_dir, "MobileNetV2"), exist_ok=True)
os.makedirs(os.path.join(errors_dir, "RandomForest"), exist_ok=True)

# ------------------- CARGA DE DATOS Y MODELOS --------------------

# Random Forest
rf_dir = os.path.join(root_dir, "analysis_results", "RandomForest")
rf = joblib.load(os.path.join(rf_dir, "random_forest_model.joblib"))
y_test = np.load(os.path.join(rf_dir, "y_test_rf.npy"))
pred_rf = np.load(os.path.join(rf_dir, "pred_rf.npy"))
cm_rf = np.load(os.path.join(rf_dir, "cm_rf.npy"))
probs_rf = np.load(os.path.join(rf_dir, "probs_rf.npy"))
top_feats = pd.read_csv(os.path.join(rf_dir, "top_feats_rf.csv"), index_col=0).iloc[:, 0]
with open(os.path.join(rf_dir, "metrics_rf.txt")) as f:
    acc_rf, f1_rf, t_rf = map(float, f.read().splitlines())

# MobileNetV2
cnn_dir = os.path.join(root_dir, "analysis_results", "MobileNetV2")
history_dict = joblib.load(os.path.join(cnn_dir, "history_cnn.joblib"))
class DummyHistory: pass
history = DummyHistory()
history.history = history_dict
val_classes_cnn = np.load(os.path.join(cnn_dir, "val_classes_cnn.npy"))
pred_cnn = np.load(os.path.join(cnn_dir, "pred_cnn.npy"))
cm_cnn = np.load(os.path.join(cnn_dir, "cm_cnn.npy"))
probs_cnn = np.load(os.path.join(cnn_dir, "probs_cnn.npy"))
with open(os.path.join(cnn_dir, "metrics_cnn.txt")) as f:
    acc_cnn, f1_cnn, loss_cnn, t_cnn = map(float, f.read().splitlines())

csv_path = os.path.join(root_dir, "extraccion_features", "caracteristicas.csv")
df = pd.read_csv(csv_path)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["class"])
classes = list(le.classes_)
X = df.drop(columns=["filename", "class", "class_index"])
y_enc = le.transform(df["class"])

# Reconstrucción de val_gen para errores visuales
dataset_path = os.path.join(root_dir, "dataset_balanceado")
img_size = (224, 224)
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    subset="validation", class_mode="categorical", shuffle=False
)

# ------------------- FUNCIONES DE VISUALIZACIÓN Y ERRORES --------------------------

def combined_classification_table_and_plot(y_true_rf, pred_rf, y_true_cnn, pred_cnn, classes, graphs_dir):
    report_rf = classification_report(y_true_rf, pred_rf, target_names=classes, output_dict=True)
    report_cnn = classification_report(y_true_cnn, pred_cnn, target_names=classes, output_dict=True)
    df_rf = pd.DataFrame(report_rf).T.loc[classes, ["precision", "recall", "f1-score"]]
    df_cnn = pd.DataFrame(report_cnn).T.loc[classes, ["precision", "recall", "f1-score"]]
    df_rf.columns = [f"{col}_RF" for col in df_rf.columns]
    df_cnn.columns = [f"{col}_CNN" for col in df_cnn.columns]
    combined = pd.concat([df_rf, df_cnn], axis=1)
    combined.index.name = "Clase"
    combined_rounded = combined.round(3)

    # Guardar como TXT en vez de CSV
    txt_path = os.path.join(graphs_dir, "combined_classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(combined_rounded.to_string())
    print("\n=== Tabla combinada de métricas por clase (RF vs MobileNetV2) ===")
    print(combined_rounded)

    # Gráfica comparativa de métricas por clase
    melted = combined_rounded.reset_index().melt(id_vars="Clase")
    plt.figure(figsize=(15, 7))
    sns.barplot(data=melted, x="Clase", y="value", hue="variable")
    plt.title("Comparación de Precisión, Recall y F1-score por Clase\nRandom Forest vs MobileNetV2")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title="Métrica_Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "combined_metrics_per_class.png"))
    plt.close()

def plot_learning_curve_cnn(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], 'o-', label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], 'o-', label='Validación')
    plt.title('Curva de Precisión por Época - MobileNetV2')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'learning_curve_cnn.png'))
    plt.close()

def plot_loss_curve_cnn(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], 'o-', label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], 'o-', label='Pérdida Validación')
    plt.title('Curva de Pérdida por Época - MobileNetV2')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'loss_curve_cnn.png'))
    plt.close()

def plot_loss_curve_rf(rf, X, y_enc):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        rf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
    )
    train_loss = 1 - np.mean(train_scores, axis=1)
    val_loss = 1 - np.mean(val_scores, axis=1)
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_loss, 'o-', label='Pérdida Entrenamiento')
    plt.plot(train_sizes, val_loss, 'o-', label='Pérdida Validación')
    plt.title('Curva de Pérdida vs Tamaño de Entrenamiento - Random Forest')
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Pérdida (1 - Precisión)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'loss_curve_rf.png'))
    plt.close()

def plot_confusion_matrix(cm, classes, modelo, acc, color):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color,
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusión - {modelo}\nPrecisión: {acc:.2%}")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'confusion_matrix_{modelo.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_top_features(top_feats):
    plt.figure(figsize=(10, 6))
    top_feats.sort_values().plot(kind='barh')
    plt.title('Top 10 Características Más Importantes - Random Forest')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'feature_importance.png'))
    plt.close()

def plot_accuracy_comparison(acc_rf, acc_cnn):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Random Forest', 'MobileNetV2'], y=[acc_rf, acc_cnn])
    plt.title('Comparación de Precisión Global')
    plt.ylim(0, 1)
    plt.ylabel('Precisión')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'accuracy_comparison.png'))
    plt.close()

def plot_roc_pr_curves(y_true, y_score, classes, model_name):
    from sklearn.preprocessing import label_binarize
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.title(f'Curvas ROC por Clase - {model_name}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'{classes[i]} (AUC={pr_auc[i]:.2f})')
    plt.title(f'Curvas Precision-Recall por Clase - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'pr_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_correlation_heatmap(X):
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Heatmap de Correlación entre Características')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'correlation_heatmap.png'))
    plt.close()

def plot_class_distribution(y_true, pred, classes, model_name):
    plt.figure(figsize=(10, 6))
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(pred).value_counts().sort_index()
    width = 0.35
    indices = np.arange(len(classes))
    plt.bar(indices - width/2, true_counts, width=width, label="Reales")
    plt.bar(indices + width/2, pred_counts, width=width, label="Predichas")
    plt.xticks(indices, classes, rotation=45)
    plt.ylabel("Cantidad")
    plt.title(f'Distribución de Predicciones por Clase - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'class_distribution_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def mostrar_errores_imagenes(val_gen, pred_cnn, clases, base_dir, n=12, modelo="MobileNetV2"):
    # Guarda n ejemplos de errores en analysis_results/ErroresEjemplos/MobileNetV2/
    error_dir = os.path.join(base_dir, modelo)
    os.makedirs(error_dir, exist_ok=True)
    misclassified_idx = np.where(val_gen.classes != pred_cnn)[0]
    if len(misclassified_idx) == 0:
        print(f"No hay errores de clasificación en el set de validación para {modelo}.")
        return
    n = min(n, len(misclassified_idx))
    for i, idx in enumerate(misclassified_idx[:n]):
        img_path = val_gen.filepaths[idx]
        img = load_img(img_path)
        real_label = clases[val_gen.classes[idx]]
        pred_label = clases[pred_cnn[idx]]
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Real: {real_label}\nPred: {pred_label}", fontsize=10)
        fname = os.path.basename(img_path)
        plt.savefig(os.path.join(error_dir, f"error_{i+1}_{real_label}_as_{pred_label}_{fname}"))
        plt.close()
    print(f"✅ Guardados {n} ejemplos de errores en: {error_dir}")

def guardar_errores_rf(df, y_true, y_pred, clases, base_dir, n=12):
    # Guarda n ejemplos de errores de RF en analysis_results/ErroresEjemplos/RandomForest/ y un TXT con los nombres
    error_dir = os.path.join(base_dir, "RandomForest")
    os.makedirs(error_dir, exist_ok=True)
    errores = []
    count = 0
    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        if true_idx != pred_idx:
            fname = df.iloc[i]["filename"]
            real_label = clases[true_idx]
            pred_label = clases[pred_idx]
            errores.append(f"{fname} - Real: {real_label} - Pred: {pred_label}")
            count += 1
            if count <= n:
                # Busca la imagen en el dataset_balanceado
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    possible = os.path.join(root_dir, "dataset_balanceado", real_label, fname)
                    if os.path.isfile(possible):
                        img_path = possible
                        break
                if img_path:
                    img = load_img(img_path)
                    plt.figure(figsize=(4, 4))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Real: {real_label}\nPred: {pred_label}", fontsize=10)
                    plt.savefig(os.path.join(error_dir, f"error_{count}_{real_label}_as_{pred_label}_{fname}"))
                    plt.close()
    # Guarda TXT con los nombres de los errores
    with open(os.path.join(error_dir, "errores_rf.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(errores))
    print(f"✅ Guardados {min(count, n)} imágenes y listado de errores en: {error_dir}")

# ------------------- GENERACIÓN DE GRÁFICAS Y TABLAS --------------------

plot_learning_curve_cnn(history)
plot_loss_curve_cnn(history)
plot_loss_curve_rf(rf, X, y_enc)
combined_classification_table_and_plot(y_test, pred_rf, val_classes_cnn, pred_cnn, classes, graphs_dir)
plot_confusion_matrix(cm_rf, classes, "Random Forest", acc_rf, "Blues")
plot_confusion_matrix(cm_cnn, classes, "MobileNetV2", acc_cnn, "Greens")
plot_top_features(top_feats)
plot_accuracy_comparison(acc_rf, acc_cnn)
plot_roc_pr_curves(y_test, probs_rf, classes, "Random Forest")
plot_roc_pr_curves(val_classes_cnn, probs_cnn, classes, "MobileNetV2")
plot_correlation_heatmap(X)
plot_class_distribution(y_test, pred_rf, classes, "Random Forest")
plot_class_distribution(val_classes_cnn, pred_cnn, classes, "MobileNetV2")

# Ejemplos visuales de errores (12 por modelo, puedes ajustar n)
mostrar_errores_imagenes(val_gen, pred_cnn, classes, errors_dir, n=12, modelo="MobileNetV2")
guardar_errores_rf(df, y_test, pred_rf, classes, errors_dir, n=12)

print(f"\n✅ Todas las gráficas, tablas, reportes y ejemplos de errores se guardaron automáticamente en:\n   {graphs_dir}\n   {errors_dir}\n")
print("Revisa los archivos PNG, TXT y las carpetas de ejemplos para tu análisis, documentación y presentaciones.")
