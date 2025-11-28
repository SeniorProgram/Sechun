from flask import Flask, render_template
from src import modelos   # Importar modelos desde la carpeta src
import os
import numpy as np

app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    # Cargar datos desde la carpeta data en la raíz
    df = modelos.cargar_datos("data/ventas.csv")

    # Entrenar con septiembre y octubre
    lambda_poisson = modelos.entrenar_poisson(df)

    # Predicciones para noviembre
    predicciones = modelos.predecir_noviembre(lambda_poisson, dias=30)

    # Ventas por mes
    df["Mes"] = df["Fecha"].dt.month
    ventas_sep = df[df["Mes"] == 9][["Fecha", "Unidades"]].to_dict(orient="records")
    ventas_oct = df[df["Mes"] == 10][["Fecha", "Unidades"]].to_dict(orient="records")
    ventas_nov = df[df["Mes"] == 11][["Fecha", "Unidades"]].to_dict(orient="records")

    # Validación con noviembre
    df_nov = df[df["Mes"] == 11]
    reales = df_nov["Unidades"].values
    min_len = min(len(reales), len(predicciones))
    reales = reales[:min_len]
    predicciones = predicciones[:min_len]

    mae = np.mean(np.abs(reales - predicciones))
    rmse = np.sqrt(np.mean((reales - predicciones) ** 2))
    mape = np.mean(np.abs((reales - predicciones) / reales)) * 100

    # Fechas históricas (sept + oct) y reales noviembre para gráfico comparativo
    fechas_hist = df[df["Mes"].isin([9, 10])]["Fecha"].dt.strftime("%d/%m/%Y").tolist()
    ventas_hist = df[df["Mes"].isin([9, 10])]["Unidades"].tolist()
    fechas_nov = [f"Día {i+1}" for i in range(len(predicciones))]

    return render_template(
        "index.html",
        predicciones=predicciones.tolist(),
        lambda_poisson=lambda_poisson,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2),
        ventas_sep=ventas_sep,
        ventas_oct=ventas_oct,
        ventas_nov=ventas_nov,
        fechas_hist=fechas_hist,
        ventas_hist=ventas_hist,
        fechas_nov=fechas_nov,
        reales=reales.tolist()
    )

if __name__ == "__main__":
    app.run(debug=True)