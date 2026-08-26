# Proyecto Final - Riesgo Crediticio

## Objetivo

Construir un modelo de clasificación que prediga el riesgo crediticio de un
crédito otorgado, es decir, si el cliente va a pagar a tiempo o no.

## Dataset

El dataset se encuentra en `Base_de_datos.xlsx` y tiene 23 columnas, donde
cada fila representa un crédito otorgado. Los montos están expresados en
pesos colombianos (COP) y la columna `puntaje_datacredito` corresponde al
score del buró de crédito colombiano DataCrédito.

**Variable objetivo:** `Pago_atiempo`
- `1` = el cliente pagó a tiempo.
- `0` = el cliente no pagó a tiempo.

## Estructura de carpetas

```
proyecto-final-riesgo-crediticio/
├── data/
│   ├── raw/           # Datos originales, sin modificar
│   └── processed/     # Datos limpios/transformados, listos para modelar
├── notebooks/         # Notebooks de exploración, EDA y experimentación
├── src/                # Código fuente reutilizable (funciones, config)
├── models/             # Modelos entrenados y serializados
├── requirements.txt    # Librerías necesarias para correr el proyecto
├── set_up.bat          # Script para armar el entorno virtual en Windows
├── .gitignore
└── README.md
```

## Cómo levantar el entorno

1. Clonar el repositorio y pararse en la raíz del proyecto.
2. Correr `set_up.bat` (Windows). El script lee el nombre del proyecto
   desde `src/config.json`, crea un entorno virtual (`pf-riesgo-venv`) e
   instala las dependencias de `requirements.txt`.
3. Activar el entorno virtual manualmente si hace falta:
   ```
   call pf-riesgo-venv\Scripts\activate.bat
   ```
4. Colocar `Base_de_datos.xlsx` dentro de `data/raw/`.
5. Abrir los notebooks de la carpeta `notebooks/` con Jupyter para
   reproducir el análisis.
