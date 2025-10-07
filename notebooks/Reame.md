
# Descripción detallada de archivos en la carpeta notebooks

Este directorio contiene todos los artefactos, configuraciones y resultados generados durante el desarrollo y experimentación del sistema de recomendación de Market Basket Analysis. A continuación se describe el propósito y uso de cada archivo relevante:

## Archivos principales

- **market-basket-analysis.ipynb**: Notebook principal que documenta todo el flujo del proyecto, desde la carga de datos, exploración, ingeniería de características, entrenamiento de modelos (LightGBM, Deep Learning), evaluación y generación de recomendaciones. Incluye visualizaciones, análisis y celdas ejecutables.

- **recommender_config.json**: Archivo de configuración en formato JSON. Define parámetros clave del sistema de recomendación, como rutas de entrada/salida, hiperparámetros, umbrales de predicción, número de recomendaciones a mostrar, y opciones de filtrado. Permite modificar el comportamiento del pipeline sin cambiar el código fuente.

- **Final_recommendation.txt**: Archivo de texto con las recomendaciones finales generadas por el modelo para cada usuario. Suele contener el listado de productos recomendados, el score de predicción y el identificador del usuario.

- **baseline_lgb_model.txt**: Registro de resultados, métricas y parámetros del modelo baseline basado en LightGBM. Puede incluir logs de entrenamiento, scores de validación y observaciones sobre el rendimiento inicial.

- **best_ncf_model.pt**: Archivo de checkpoint del mejor modelo de Neural Collaborative Filtering entrenado con PyTorch. Guarda los pesos y el estado del optimizador para poder recargar el modelo y continuar el entrenamiento o realizar inferencia.

## Archivos de datos y features (.pkl)

- **train_data.pkl**: DataFrame con los datos de entrenamiento ya procesados y listos para alimentar el modelo. Incluye las features seleccionadas, las etiquetas y los identificadores necesarios para el entrenamiento supervisado.

- **val_data_with_lgb.pkl**: DataFrame con los datos de validación y las predicciones generadas por el modelo LightGBM. Permite comparar el rendimiento y realizar análisis de errores sobre el conjunto de validación.

- **model_features.pkl**: DataFrame con el conjunto de características (features) seleccionadas para el modelo. Incluye variables de usuario, producto y contexto, como frecuencia de compra, recencia, popularidad, etc.

- **product_features.pkl**: DataFrame con las features calculadas para cada producto, como número de veces comprado, ratio de reorden, pertenencia a categorías, y otras métricas relevantes para la recomendación.

- **user_features.pkl**: DataFrame con las features calculadas para cada usuario, como frecuencia de compra, diversidad de productos, patrones de compra, recencia, y métricas de engagement.

- **user_product_features.pkl**: DataFrame con las features calculadas para cada combinación usuario-producto. Permite capturar la relación histórica entre cada usuario y cada producto, como número de compras, tiempo desde la última compra, ratio de reorden, etc.

## Archivos de predicciones y evaluación (.pkl)

- **val_predictions_baseline.pkl**: DataFrame con las predicciones del modelo baseline (LightGBM) sobre el conjunto de validación. Incluye los scores de predicción, etiquetas verdaderas y los identificadores de usuario y producto. Se usa para calcular métricas como precision@k, recall@k y para análisis de errores.

- **val_predictions_dl.pkl**: DataFrame con las predicciones del modelo deep learning (Neural Collaborative Filtering) sobre el conjunto de validación. Permite comparar el rendimiento entre modelos y realizar análisis detallados de los resultados.

- **ensemble_predictions.pkl**: DataFrame con las predicciones combinadas de varios modelos (ensemble). Suele contener los scores finales obtenidos al promediar o ponderar las salidas de diferentes modelos, mejorando la robustez y precisión de las recomendaciones.

## Otros archivos

- **model_config.pkl**: Diccionario o DataFrame con la configuración y parámetros del modelo entrenado, como hiperparámetros, estructura de red, criterios de parada, etc. Permite reproducir experimentos y documentar el setup utilizado.

- **Final_recommendation.txt**: Archivo de texto con las recomendaciones finales para cada usuario, útil para reportes o integración con sistemas externos.

- **best_ncf_model.pt**: Checkpoint del mejor modelo de deep learning, útil para recargar el modelo y realizar inferencia sin necesidad de reentrenar.

---

**Ejemplo de uso de archivos .pkl en el notebook:**

```python
# Cargar predicciones del modelo baseline
import pandas as pd
val_preds = pd.read_pickle('val_predictions_baseline.pkl')
print(val_preds.head())

# Cargar features de usuario
user_feats = pd.read_pickle('user_features.pkl')
print(user_feats.describe())
```

Estos archivos permiten acelerar el flujo de trabajo, guardar resultados intermedios y facilitar la reproducibilidad de los experimentos.
