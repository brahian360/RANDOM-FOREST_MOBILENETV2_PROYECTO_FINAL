"""
extraccion_features.py
Este script recorre todas las imágenes del dataset balanceado, extrae más de 300 características por imagen y guarda el resultado en un CSV dentro de la carpeta extraccion_features. Lo hago así para tener el resumen de features listo para cualquier modelo clásico que quiera probar después.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte

# Esta función se encarga de sacar todas las estadísticas y features relevantes de cada imagen.
def extract_image_features(image_path, class_name, class_index):
    features = {
        "filename": os.path.basename(image_path),
        "class": class_name,
        "class_index": class_index
    }
    try:
        img = Image.open(image_path).convert("RGB")
        np_img = np.array(img)
        height, width, _ = np_img.shape
        features.update({
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 2),
            "is_square": width == height
        })
        # Saco estadísticas de color RGB
        for i, ch_name in enumerate(['R', 'G', 'B']):
            ch = np_img[:, :, i].flatten()
            features[f"mean_{ch_name}"] = np.mean(ch)
            features[f"std_{ch_name}"] = np.std(ch)
            features[f"min_{ch_name}"] = np.min(ch)
            features[f"max_{ch_name}"] = np.max(ch)
            features[f"skew_{ch_name}"] = skew(ch)
            features[f"kurt_{ch_name}"] = kurtosis(ch)
            hist, _ = np.histogram(ch, bins=16, range=(0, 255), density=True)
            for j, val in enumerate(hist):
                features[f"hist_{ch_name}_{j}"] = round(val, 5)
        # También calculo estadísticas en HSV
        hsv = rgb2hsv(np_img)
        for i, ch_name in enumerate(['H', 'S', 'V']):
            ch = hsv[:, :, i].flatten()
            features[f"mean_{ch_name}"] = np.mean(ch)
            features[f"std_{ch_name}"] = np.std(ch)
            features[f"skew_{ch_name}"] = skew(ch)
            features[f"kurt_{ch_name}"] = kurtosis(ch)
        # Ahora paso a escala de grises para textura y entropía
        gray = rgb2gray(np_img)
        gray_u8 = img_as_ubyte(gray)
        features["entropy"] = shannon_entropy(gray)
        features["sobel"] = np.mean(sobel(gray))
        # GLCM: saco varias propiedades de textura en 4 ángulos
        glcm = graycomatrix(
            gray_u8, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            symmetric=True, normed=True
        )
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            vals = graycoprops(glcm, prop)[0]
            for i, angle in enumerate(["0", "45", "90", "135"]):
                features[f"{prop}_{angle}"] = round(vals[i], 5)
    except Exception as e:
        print(f"❌ Error procesando {image_path}: {e}")
    return features

# Voy a buscar el dataset balanceado en la raíz del proyecto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(root_dir, "dataset_balanceado")
output_dir = os.path.join(root_dir, "extraccion_features")
os.makedirs(output_dir, exist_ok=True)

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"No encuentro la carpeta del dataset balanceado en: {dataset_path}")

# Recorro todas las carpetas de clase y proceso cada imagen
dataset = []
class_names = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])
class_map = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, fname)
            feats = extract_image_features(img_path, class_name, class_map[class_name])
            dataset.append(feats)

df = pd.DataFrame(dataset)
output_csv = os.path.join(output_dir, "caracteristicas_300.csv")
df.to_csv(output_csv, index=False)

print(f"\n✅  creado CSV con {len(df)} imágenes y {df.shape[1]} características en:")
print(f"   {output_csv}")
