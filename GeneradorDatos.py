import pandas as pd
import numpy as np

# Creamos datos ficticios pero realistas
data = {
    "Fecha": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Empresa": np.random.choice(["Apple", "Google", "Microsoft", "Amazon"], 100),
    "Precio_Cierre": np.random.uniform(100, 200, 100).round(2),
    "Volumen": np.random.randint(1000, 5000, 100),
    "Sector": "Tecnología",
}

df = pd.DataFrame(data)
df.to_csv("datos_prueba.csv", index=False)
print("✅ Archivo 'datos_prueba.csv' creado con éxito.")
