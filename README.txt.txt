PROYECTO DE CLASIFICACIÓN DE IMÁGENES: RANDOM FOREST Y MOBILENETV2

Fabián Quiñones Zuñiga
Ricardo Cuesta
Brahian Calderón


PROYECTO FINAL - FUNDAMENTOS DE INTELIGENCIA COMPUTACIONAL

Julio - 2025
UNIVERSIDAD DE ANTIOQUIA

===================================================================

Este documento explica la estructura del proyecto, el flujo de ejecución y dónde se almacenan todos los resultados, gráficas, tablas y ejemplos visuales generados por el pipeline de clasificación de imágenes.

-------------------------------------------------------------------
ESTRUCTURA DEL PROYECTO
-------------------------------------------------------------------

mi_proyecto_clasificacion/
│
├── dataset_original/            # Imágenes originales por clase
├── dataset_balanceado/          # Imágenes balanceadas y aumentadas
├── extraccion_features/         # CSVs de características extraídas
├── analysis_results/
│   ├── RandomForest/            # Resultados y errores RF
│   ├── MobileNetV2/             # Resultados y errores MobileNetV2
│   ├── Graficos/                # Todas las gráficas y tablas clave
│   └── ErroresEjemplos/
│        ├── MobileNetV2/        # Imágenes mal clasificadas por MobileNetV2
│        └── RandomForest/       # Imágenes y TXT de errores de RF
├── src/
│   ├── balanceo_dataset.py
│   ├── extraccion_features.py
│   ├── entrenamiento_rf.py
│   ├── entrenamiento_cnn.py
│   └── visualizaciones.py
├── requirements.txt
└── README.txt

-------------------------------------------------------------------
¿CÓMO EJECUTAR EL PIPELINE?
-------------------------------------------------------------------

1. Instala las dependencias:
   pip install -r requirements.txt

2. Ejecuta los scripts en este orden desde la raíz del proyecto:

   a) Balanceo del dataset:
      python src/balanceo_dataset.py

   b) Extracción de características:
      python src/extraccion_features.py

   c) Entrenamiento de modelos:
      - Random Forest:
        python src/entrenamiento_rf.py
      - MobileNetV2:
        python src/entrenamiento_cnn.py

   d) Visualización y comparaciones:
      python src/visualizaciones.py

-------------------------------------------------------------------
¿QUÉ RESULTADOS Y GRÁFICAS SE GENERAN?
-------------------------------------------------------------------

- Curvas de aprendizaje y pérdida (MobileNetV2 y Random Forest)
- Matrices de confusión para ambos modelos
- Importancia de características (Random Forest)
- Curvas ROC y Precision-Recall por clase
- Boxplots y gráficas comparativas de precisión, recall y F1-score por clase
- Distribución de predicciones por clase
- Heatmap de correlación de features
- Comparación visual de errores (imágenes mal clasificadas por ambos modelos)
- Tabla combinada de métricas por clase (TXT y gráfica comparativa)

-------------------------------------------------------------------
¿DÓNDE SE GUARDAN LOS RESULTADOS?
-------------------------------------------------------------------

| Tipo de Resultado                           | Carpeta/Archivo de Salida                                  |
|---------------------------------------------|------------------------------------------------------------|
| Gráficas y tablas comparativas              | analysis_results/Graficos/                                 |
| Ejemplos visuales de errores MobileNetV2    | analysis_results/ErroresEjemplos/MobileNetV2/              |
| Ejemplos visuales y TXT de errores RF       | analysis_results/ErroresEjemplos/RandomForest/             |
| Reportes y métricas detalladas por modelo   | analysis_results/RandomForest/ y analysis_results/MobileNetV2/ |
| Tabla combinada de métricas por clase (TXT) | analysis_results/Graficos/combined_classification_report.txt |

-------------------------------------------------------------------
¿CÓMO INTERPRETAR Y USAR LOS RESULTADOS?
-------------------------------------------------------------------

- Las gráficas y tablas permiten comparar el desempeño, identificar fortalezas y debilidades de cada modelo y detectar posibles problemas de sobreajuste, desbalance o sesgo.
- Los ejemplos visuales de errores facilitan el análisis cualitativo y ayudan a entender por qué los modelos fallan en ciertas clases.
- El archivo TXT combinado de métricas por clase resume precisión, recall y F1-score para cada clase y modelo, ideal para documentación y presentaciones.
- Todos los outputs quedan organizados y listos para análisis, reporte o presentación profesional.

-------------------------------------------------------------------
RECOMENDACIONES FINALES
-------------------------------------------------------------------

- Ejecuta cada script desde la raíz del proyecto para que las rutas funcionen correctamente.
- Si agregas nuevas clases o imágenes, repite el flujo desde el balanceo.
- Puedes ajustar el número de ejemplos de errores visualizados modificando el parámetro n en las funciones correspondientes.
- El archivo requirements.txt contiene todas las dependencias necesarias.
- Consulta las carpetas Graficos/ y ErroresEjemplos/ para acceder rápidamente a los resultados clave y ejemplos visuales.

-------------------------------------------------------------------
