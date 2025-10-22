# trabajo_final_mlops

Proyecto Final MLOps – BMW Sales con MLflow
Descripción general

Este proyecto implementa un pipeline reproducible de Machine Learning utilizando el dataset “BMW Sales Data (2010–2024)”, el cual contiene información sobre las ventas de vehículos BMW en distintos años y regiones.
El objetivo fue entrenar, evaluar y registrar un modelo predictivo de ventas, aplicando prácticas de MLOps con MLflow y automatización CI/CD con GitHub Actions.

Estructura del proyecto

trabajo_final_mlops/
│
├── data/
│   └── BMW_sales_data_2010_2024.csv        # Dataset de entrenamiento
│
├── src/
│   ├── preprocess.py                       # Carga, limpieza y escalamiento de datos
│   ├── train.py                            # Entrenamiento, evaluación y registro MLflow
│   └── evaluate.py                         # Validación adicional del modelo
│
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml                   # Pipeline CI/CD automatizado en GitHub Actions
│
├── config.yaml                             # Parámetros de configuración
├── Makefile                                # Comandos automatizados (install, train, test)
├── requirements.txt                        # Dependencias del entorno
└── README.md                               # Instrucciones y documentación

Requisitos previos

Python 3.10 o superior
Git instalado y configurado
Entorno virtual activo (venv)
MLflow instalado
Cuenta en GitHub (para CI/CD)

Instalación y ejecución local

Clonar el repositorio
git clone https://github.com/ccjaviers777/trabajo_final_mlops.git
cd trabajo_final_mlops

Crear y activar el entorno virtual
python -m venv venv
venv\Scripts\activate      # En Windows

Instalar dependencias
pip install -r requirements.txt o make install

Ejecutar el entrenamiento
python src/train.py o make train
Esto entrenará el modelo y registrará los resultados en mlruns/ con MLflow.

Abrir la interfaz de MLflow
mlflow ui
Luego abre en el navegador: http://127.0.0.1:5000
------------------------------------------------------------------------------EAN-----------------------------------------------------------------------
Automatización CI/CD (GitHub Actions)

El archivo .github/workflows/mlflow-ci.yml automatiza todo el pipeline:
Instala dependencias
Ejecuta el entrenamiento
Registra el modelo
Guarda los artefactos (trained-model)

Cada vez que se hace un push a la rama main, el flujo se ejecuta automáticamente.
Evidencia del pipeline exitoso:
GitHub Actions muestra el estado Success con el modelo guardado como artefacto
-----------------------------------------------------------------------------------EAN----------------------------------
Resultados del modelo

El modelo final fue un Random Forest Regressor, con las siguientes métricas:

Métrica	Valor
MSE	3,080,813.31
MAE	1,441.95
R²	0.62

Estos resultados se registran automáticamente en MLflow bajo el experimento “BMW-Sales-Model”.
-------------------------------------------------------------------------------------------EAN----------------------------
Autor

Javier Callejas Cardozo
Maestria en ciencia de datos – Universidad EAN
Bogotá, Colombia – 2025

