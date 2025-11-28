import pandas as pd
import numpy as np
from scipy.stats import poisson

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    # Convertir la columna Fecha a datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    return df


def entrenar_poisson(df):
    # Filtrar septiembre y octubre
    df_train = df[df["Fecha"].dt.month.isin([9, 10])]
    ventas = df_train["Unidades"].values
    lambda_poisson = np.mean(ventas)
    return lambda_poisson

def predecir_noviembre(lambda_poisson, dias=30):
    predicciones = poisson.rvs(mu=lambda_poisson, size=dias)
    return predicciones