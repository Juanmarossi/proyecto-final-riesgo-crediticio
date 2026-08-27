"""Módulo único de ingeniería de características del proyecto.

Los notebooks importan las funciones de acá, no reimplementan la lógica
(regla del Avance 2 en CLAUDE.md). Contiene `crear_features` (variables
derivadas) y `construir_preprocesador` (el ColumnTransformer que arma el
pipeline de preprocesamiento).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Clasificación de columnas fijada en ft_engineering.ipynb (sección
# "Clasificación de columnas"). Vive acá para que preparar_datos() no
# dependa de que el notebook la redefina.
NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
]
CATEGORICAS_NOMINALES = ["tipo_credito", "tipo_laboral"]
CATEGORICAS_ORDINALES = ["tendencia_ingresos"]
ORDEN_ORDINALES = {"tendencia_ingresos": ["Decreciente", "Estable", "Creciente"]}

COLUMNA_OBJETIVO = "Pago_atiempo"


def _dividir_seguro(numerador, denominador):
    """Divide dos series manejando denominador en cero.

    Un denominador en 0 con numerador != 0 da +-inf en pandas (no un error),
    así que hay que reemplazarlo por NaN a mano: la regla del proyecto pide
    que ninguna división devuelva infinito.
    """
    resultado = numerador / denominador
    return resultado.replace([np.inf, -np.inf], np.nan)


def crear_features(df):
    """Agrega variables derivadas de riesgo, perfil crediticio y temporales.

    No modifica `df`: trabaja sobre una copia y la devuelve. No imputa
    ningún nulo (ni los que ya traía el dataset ni los que puedan surgir
    de las divisiones) — la imputación es responsabilidad del
    `ColumnTransformer` que se arma después del split, para no filtrar
    información del conjunto de test hacia el de entrenamiento.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset crudo, con al menos las columnas cuota_pactada,
        salario_cliente, total_otros_prestamos, capital_prestado,
        creditos_sectorFinanciero, creditos_sectorCooperativo,
        creditos_sectorReal, promedio_ingresos_datacredito y
        fecha_prestamo.

    Returns
    -------
    pandas.DataFrame
        Copia de `df` con las columnas nuevas agregadas.
    """
    datos = df.copy()

    # --- Ratios de riesgo -------------------------------------------------
    # Hipótesis: la capacidad de pago no depende del monto absoluto de la
    # cuota, sino de qué proporción del ingreso mensual se lleva.
    datos["ratio_cuota_salario"] = _dividir_seguro(datos["cuota_pactada"], datos["salario_cliente"])

    # Hipótesis: el endeudamiento previo (otras deudas) relativo al ingreso
    # es un indicador clásico de sobreendeudamiento, más informativo que el
    # monto de otras deudas en pesos.
    datos["ratio_endeudamiento"] = _dividir_seguro(datos["total_otros_prestamos"], datos["salario_cliente"])

    # Hipótesis: a cuántos meses de sueldo equivale el crédito otorgado;
    # créditos grandes respecto del ingreso son más riesgosos.
    datos["ratio_capital_salario"] = _dividir_seguro(datos["capital_prestado"], datos["salario_cliente"])

    # Hipótesis: la carga financiera total (esta cuota + las otras deudas)
    # sobre el ingreso resume en un solo número la presión de pago mensual
    # del cliente, más completa que mirar cuota u otras deudas por separado.
    datos["carga_total"] = _dividir_seguro(
        datos["cuota_pactada"] + datos["total_otros_prestamos"], datos["salario_cliente"]
    )

    # --- Perfil crediticio --------------------------------------------------
    # Hipótesis: la cantidad total de créditos vigentes en el sistema
    # (sumando los tres sectores) resume la exposición crediticia total del
    # cliente en un solo número, útil cuando lo que importa es el volumen
    # total y no en qué sector está cada crédito.
    datos["total_creditos_sectores"] = (
        datos["creditos_sectorFinanciero"] + datos["creditos_sectorCooperativo"] + datos["creditos_sectorReal"]
    )

    # Hipótesis: una brecha grande entre lo que el cliente declara ganar y
    # lo que el buró reporta como su ingreso promedio puede señalar un
    # ingreso declarado poco confiable (sobreestimado o subestimado).
    datos["diferencia_ingresos"] = datos["salario_cliente"] - datos["promedio_ingresos_datacredito"]

    # Hipótesis: la razón entre ambos ingresos captura la misma idea que la
    # diferencia pero en términos relativos, más comparable entre clientes
    # de distinto nivel de ingreso.
    datos["ratio_ingresos_declarados"] = _dividir_seguro(
        datos["salario_cliente"], datos["promedio_ingresos_datacredito"]
    )

    # Hipótesis: no tener historial de ingresos en DataCrédito (nulo en
    # promedio_ingresos_datacredito) es en sí misma una señal de riesgo —
    # clientes sin huella suficiente en el buró — y quedaría oculta si solo
    # se imputa el valor faltante sin dejar constancia de que faltaba.
    datos["sin_historial_ingresos"] = datos["promedio_ingresos_datacredito"].isna().astype(int)

    # --- Temporales, derivadas de fecha_prestamo -----------------------------
    fecha = pd.to_datetime(datos["fecha_prestamo"])

    # Hipótesis: el mes de originación puede capturar estacionalidad en el
    # comportamiento de pago (campañas, aguinaldo, etc.).
    datos["mes_prestamo"] = fecha.dt.month

    # Hipótesis: el trimestre agrupa la estacionalidad en una variable de
    # menor cardinalidad que el mes, útil si la señal mensual es ruidosa.
    datos["trimestre_prestamo"] = fecha.dt.quarter

    # Hipótesis: el año de originación permite detectar un efecto de
    # cosecha (vintage) - créditos otorgados en un año distinto pueden
    # comportarse distinto por cambios macro o de política de originación.
    datos["anio_prestamo"] = fecha.dt.year

    # Hipótesis: el día de la semana en que se otorga el crédito podría
    # asociarse a un perfil de cliente distinto (por ejemplo, trámites de
    # fin de semana vs. de mitad de semana).
    datos["dia_semana_prestamo"] = fecha.dt.dayofweek

    return datos


def construir_preprocesador(numericas, nominales, ordinales, orden_ordinales):
    """Arma el ColumnTransformer de preprocesamiento (estructura DSM5L4).

    Tres ramas independientes, cada una un Pipeline propio:
    - "num": SimpleImputer(strategy="median") + StandardScaler, para las
      columnas numéricas continuas o de conteo.
    - "nom": SimpleImputer(strategy="most_frequent") + OneHotEncoder
      (handle_unknown="ignore", drop="if_binary"), para categóricas sin
      orden.
    - "ord": SimpleImputer(strategy="most_frequent") + OrdinalEncoder con
      las categorías pasadas explícitamente en el orden correcto
      (handle_unknown="use_encoded_value", unknown_value=-1), para
      categóricas con orden de negocio.

    `remainder="drop"` para que ninguna columna sin clasificar se cuele sin
    transformar, y `verbose_feature_names_out=False` para que
    `get_feature_names_out()` devuelva nombres legibles (sin el prefijo
    "num__"/"nom__"/"ord__").

    Parameters
    ----------
    numericas : list[str]
        Nombres de las columnas numéricas.
    nominales : list[str]
        Nombres de las columnas categóricas nominales.
    ordinales : list[str]
        Nombres de las columnas categóricas ordinales.
    orden_ordinales : dict[str, list[str]]
        Para cada columna de `ordinales`, la lista de sus categorías en el
        orden correcto (de menor a mayor).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        El preprocesador, SIN ajustar.
    """
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_nominal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    categorias_ordinales_en_orden = [orden_ordinales[columna] for columna in ordinales]
    pipeline_ordinal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=categorias_ordinales_en_orden,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, numericas),
            ("nom", pipeline_nominal, nominales),
            ("ord", pipeline_ordinal, ordinales),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocesador


# Columnas numéricas que agrega crear_features, para sumarlas a NUMERICAS
# al armar el preprocesador dentro de preparar_datos().
NUMERICAS_DERIVADAS = [
    "ratio_cuota_salario", "ratio_endeudamiento", "ratio_capital_salario", "carga_total",
    "total_creditos_sectores", "diferencia_ingresos", "ratio_ingresos_declarados", "sin_historial_ingresos",
    "mes_prestamo", "trimestre_prestamo", "anio_prestamo", "dia_semana_prestamo",
]


def preparar_datos(ruta_datos, columnas_excluidas=None, test_size=0.2, random_state=42):
    """Arma X_train, X_test, y_train, y_test y el preprocesador para modelar.

    Hace, en orden: carga el Excel, aplica `crear_features`, separa X de y
    (target `Pago_atiempo`), elimina `columnas_excluidas` (las que el
    análisis de fuga de `ft_engineering.ipynb` marcó como sospechosas),
    parte en train/test de forma estratificada por y, y arma el
    `ColumnTransformer` con `construir_preprocesador`.

    El preprocesador se devuelve **sin ajustar, a propósito**: si se
    ajustara acá, sobre todo el dataset (train + test), el `SimpleImputer`
    y el `StandardScaler` verían estadísticos (medianas, medias, desvíos)
    calculados con filas de test — eso es fuga de información del test
    hacia el train, aunque no toque directamente la variable objetivo. El
    ajuste correcto ocurre más adelante, dentro del `Pipeline` de cada
    modelo (`Pipeline([("preprocesador", preprocesador), ("modelo", ...)])`),
    llamando a `.fit()` con `X_train` únicamente — así el preprocesador
    solo ve estadísticos del conjunto de entrenamiento.

    Parameters
    ----------
    ruta_datos : str o pathlib.Path
        Ruta al Excel crudo (`data/raw/Base_de_datos.xlsx`).
    columnas_excluidas : list[str], optional
        Columnas a eliminar de X antes de modelar (por ejemplo, las
        sospechosas de fuga: `puntaje`). Por defecto no excluye ninguna.
    test_size : float, optional
        Proporción del test split. Por defecto 0.2.
    random_state : int, optional
        Semilla para el split. Por defecto 42.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocesador) — el
        preprocesador sin ajustar.
    """
    columnas_excluidas = columnas_excluidas or []

    df_crudo = pd.read_excel(ruta_datos, engine="openpyxl")
    df_con_features = crear_features(df_crudo)

    y = df_con_features[COLUMNA_OBJETIVO]
    X = df_con_features.drop(columns=[COLUMNA_OBJETIVO, "fecha_prestamo"])
    X = X.drop(columns=[c for c in columnas_excluidas if c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    numericas = [c for c in NUMERICAS + NUMERICAS_DERIVADAS if c not in columnas_excluidas]
    nominales = [c for c in CATEGORICAS_NOMINALES if c not in columnas_excluidas]
    ordinales = [c for c in CATEGORICAS_ORDINALES if c not in columnas_excluidas]

    preprocesador = construir_preprocesador(numericas, nominales, ordinales, ORDEN_ORDINALES)

    return X_train, X_test, y_train, y_test, preprocesador
