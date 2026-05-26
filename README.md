# NATICUSdroid Android Permissions Dataset

Proyecto curso modelos y simulacion de sistemas II

Grupo 8:

- Claudia Rocha
- David Naranjo
- Sebastian Bedoya

## Entrega final

Informe: [Ver documento](./Informe_Final_NATICUSDroid.pdf)

Video: 

## Estructura

- `data/raw/`: CSV original del dataset.
- `data/processed/`: particiones generadas por el script de preparación.
- `src/prepare_data.py`: carga, validación y split reproducible 70/30 estratificado.
- `src/replay_update.py`: utilidades para simulación incremental, memoria replay y persistencia del vector en random forest.

## Índice de notebooks

Todos los notebooks están enlazados directamente a Colab haga click sobre el notebook que desea abrir:

### Preprocesamiento

- [01_preprocesamiento.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/01_preprocesamiento.ipynb) - preprocesamiento y generación de artefactos.

### Modelos

- [02_random_forest.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/02_random_forest.ipynb) - Random Forest.
- [03_logistic_regression.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/03_logistic_regression.ipynb) - Regresión Logística.
- [04_knn.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/04_knn.ipynb) - KNN.
- [05_SVM.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/05_SVM.ipynb) - SVM.
- [06_red_neuronal.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/06_red_neuronal.ipynb) - Red neuronal.

### Reducción de dimensión

- [07_análisis_variables.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/07_análisis_variables.ipynb) - análisis de variables previo a la reducción de dimensión.
- [08_extraccion_caracteristicas_lineal_pca.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/08_extraccion_caracteristicas_lineal_pca.ipynb) - extracción de características lineal con PCA.
- [09_extraccion_caracteristicas_no_lineal_umap.ipynb](https://colab.research.google.com/github/DavidNaranjo1237/NATICUSdroid/blob/main/notebooks/09_extraccion_caracteristicas_no_lineal_umap.ipynb) - extracción de características no lineal con UMAP.

## Cómo ejecutar

1. Abrir el notebook correspondiente desde el índice anterior o desde `notebooks/`.
2. Ejecutar la primera celda de arranque para detectar la ruta del repositorio o clonar el proyecto en Colab.
3. Ejecutar la celda de preprocesado interna del notebook, que llama al script compartido en `src/prepare_data.py`.
4. Continuar con el entrenamiento y evaluación del modelo correspondiente.

## Reproducibilidad

- Partición: 70/30.
- Estratificación: sí.
- Semilla fija: 42.
- Variable objetivo: `Result`.
