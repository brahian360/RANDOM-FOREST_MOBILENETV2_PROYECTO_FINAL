"""
balanceo_dataset.py
Este script balancea el dataset de imágenes por clase. Antes de hacer nada, reviso cuántas imágenes tiene cada clase en el dataset original. Luego, copio todas las imágenes originales a la carpeta balanceada y, si alguna clase no llega al mínimo que quiero (target_min), genero imágenes aumentadas solo para esa clase. Al final, muestro una tabla resumen con el conteo inicial, cuántas se generaron y el total final por cada categoría.
"""

import os
import random
import shutil
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Estoy asumiendo que este script está dentro de la carpeta src/
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(root_dir, "dataset_original")
output_dir = os.path.join(root_dir, "dataset_balanceado")
os.makedirs(output_dir, exist_ok=True)

# Aquí defino el mínimo de imágenes que quiero por clase
target_min = 350

# Configuro el generador de aumento de imágenes con transformaciones variadas
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Antes de balancear, hago un conteo de imágenes originales por clase
conteo_inicial = {}
for class_name in sorted(os.listdir(input_dir)):
    src_class_path = os.path.join(input_dir, class_name)
    num_imgs = len([f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    conteo_inicial[class_name] = num_imgs

# Imprimo la tabla con el conteo inicial
print("\n--- Conteo inicial de imágenes por clase ---")
print(f"{'Clase':<20} {'Originales':>10}")
print("-" * 32)
for clase, n in conteo_inicial.items():
    print(f"{clase:<20} {n:>10}")
print("-" * 32)

# Copio todas las imágenes originales a la carpeta balanceada
for class_name in sorted(os.listdir(input_dir)):
    src_class_path = os.path.join(input_dir, class_name)
    dst_class_path = os.path.join(output_dir, class_name)
    os.makedirs(dst_class_path, exist_ok=True)
    for fname in os.listdir(src_class_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_file = os.path.join(src_class_path, fname)
            dst_file = os.path.join(dst_class_path, fname)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

# Ahora hago el aumento solo para las clases que lo necesitan
generadas_por_clase = defaultdict(int)
for class_name in sorted(os.listdir(input_dir)):
    class_output_path = os.path.join(output_dir, class_name)
    original_images = [f for f in os.listdir(class_output_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_count = len(original_images)

    if original_count >= target_min:
        print(f"✅ {class_name} ya tiene {original_count} imágenes (OK)")
        continue

    to_generate = target_min - original_count
    print(f"➕ Generando {to_generate} imágenes aumentadas para clase '{class_name}'...")

    generated = 0
    while generated < to_generate:
        random.shuffle(original_images)
        for img_name in original_images:
            img_path = os.path.join(class_output_path, img_name)
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                for batch in datagen.flow(
                    x, batch_size=1,
                    save_to_dir=class_output_path,
                    save_prefix='aug',
                    save_format='jpg'
                ):
                    generated += 1
                    generadas_por_clase[class_name] += 1
                    break  # Solo una imagen por batch
                if generated >= to_generate:
                    break
            except Exception as e:
                print(f"Error con {img_path}: {e}")

# Después de balancear, cuento cuántas imágenes hay por clase en la carpeta balanceada
conteo_final = {}
for class_name in sorted(os.listdir(output_dir)):
    class_output_path = os.path.join(output_dir, class_name)
    num_imgs = len([f for f in os.listdir(class_output_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    conteo_final[class_name] = num_imgs

# Imprimo la tabla final con los resultados
print("\n--- Resumen del balanceo por clase ---")
print(f"{'Clase':<20} {'Originales':>10} {'Aumentadas':>12} {'Total final':>12}")
print("-" * 58)
for clase in sorted(conteo_inicial.keys()):
    originales = conteo_inicial.get(clase, 0)
    generadas = generadas_por_clase.get(clase, 0)
    total = conteo_final.get(clase, 0)
    print(f"{clase:<20} {originales:>10} {generadas:>12} {total:>12}")
print("-" * 58)
print("🎯 Dataset balanceado y aumentado correctamente.")

