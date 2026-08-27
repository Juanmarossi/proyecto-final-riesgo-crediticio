"""Módulo de monitoreo de modelo y detección de data drift.

Genera la tabla de scoring, calcula métricas de drift período a período
contra una ventana de referencia fija, y produce recomendaciones
accionables. La app de Streamlit (`app/app.py`) solo lee lo que este
módulo deja escrito en `data/processed/` — no recalcula nada.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from src.ft_engineering import crear_features

# Umbrales de alerta, en un único diccionario (regla del Avance 3 en
# CLAUDE.md: nada de umbrales repartidos por el código).
#
# PSI y Jensen-Shannon: umbrales estándar de la industria de riesgo/scoring
# para monitoreo de modelos (la misma escala que usan bancos y bureaus de
# crédito: < 0.10 sin acción, 0.10-0.25 revisar, > 0.25 alerta — ver por
# ejemplo las guías de model risk management tipo SR 11-7/Basel y la
# práctica estándar de proveedores de scoring).
# KS y Chi2: se evalúan por p-valor (no por el estadístico crudo), con el
# nivel de significancia estadística convencional del 5%.
UMBRALES = {
    "PSI": {"moderado": 0.10, "significativo": 0.25},
    "Jensen-Shannon": {"moderado": 0.10, "significativo": 0.20},
    "KS": {"p_valor_significativo": 0.05},
    "Chi2": {"p_valor_significativo": 0.05},
}


def generar_tabla_scoring(ruta_datos, ruta_modelo, ruta_salida):
    """Genera la tabla de scoring histórica para el monitoreo.

    Carga el dataset crudo, aplica `crear_features` (las mismas features
    del entrenamiento, importadas de `src.ft_engineering` para no
    reimplementar la lógica), carga el pipeline entrenado y predice sobre
    todo el dataset. El resultado se guarda en Parquet y se devuelve.

    Nota: `Pago_atiempo` (el resultado real) se guarda en la tabla
    únicamente porque este es un ejercicio con datos históricos, donde ya
    se conoce el desenlace de cada crédito. En producción, al momento de
    monitorear, ese resultado real todavía no existiría (los créditos
    recientes no maduraron) — se sabría mucho después, si es que se
    vuelve a cargar. Guardarlo acá es solo para poder demostrar el
    monitoreo con datos ya cerrados.

    Parameters
    ----------
    ruta_datos : str o pathlib.Path
        Ruta al Excel crudo (`data/raw/Base_de_datos.xlsx`).
    ruta_modelo : str o pathlib.Path
        Ruta al pipeline serializado (`models/modelo_final.pkl`).
    ruta_salida : str o pathlib.Path
        Ruta del Parquet de salida (`data/processed/tabla_scoring.parquet`).

    Returns
    -------
    pandas.DataFrame
        La tabla de scoring, la misma que se guardó en `ruta_salida`.

    Raises
    ------
    ValueError
        Si las columnas de entrada no coinciden exactamente con las que
        espera el modelo, según `models/modelo_final_metadata.json`.
    """
    df_crudo = pd.read_excel(ruta_datos, engine="openpyxl")
    df_features = crear_features(df_crudo)

    modelo = joblib.load(ruta_modelo)

    ruta_metadata = Path(ruta_modelo).with_name("modelo_final_metadata.json")
    with open(ruta_metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    columnas_esperadas = metadata["columnas_entrada_esperadas"]
    columnas_excluidas_por_fuga = metadata.get("columnas_excluidas_por_fuga", [])

    faltantes = [c for c in columnas_esperadas if c not in df_features.columns]
    if faltantes:
        raise ValueError(
            f"generar_tabla_scoring: faltan columnas que el modelo espera: {faltantes}. "
            "Revisar que crear_features no haya cambiado."
        )

    X = df_features[columnas_esperadas]

    # Columnas que crear_features produce pero el modelo NO espera: es
    # esperable para las que el entrenamiento excluyó por fuga (quedan
    # registradas en el metadata) y para fecha_prestamo/Pago_atiempo, que
    # nunca son features. Cualquier otra columna "de más" sí es una señal
    # de que crear_features cambió después de entrenar el modelo.
    columnas_ignorables = set(columnas_excluidas_por_fuga) | {"Pago_atiempo", "fecha_prestamo"}
    columnas_inesperadas = [
        c for c in df_features.columns if c not in columnas_esperadas and c not in columnas_ignorables
    ]
    if columnas_inesperadas:
        raise ValueError(
            f"generar_tabla_scoring: crear_features generó columnas nuevas que el modelo no "
            f"conoce y que tampoco están en columnas_excluidas_por_fuga del metadata: "
            f"{columnas_inesperadas}. Revisar antes de continuar."
        )

    probabilidad_predicha = modelo.predict_proba(X)[:, 1]
    prediccion = (probabilidad_predicha >= 0.5).astype(int)

    tabla_scoring = X.copy()

    # tendencia_ingresos, sobre datos crudos, mezcla strings ("Creciente")
    # con el ruido numérico fuera de dominio ya documentado en el EDA
    # (valores como 8315). Parquet no admite una columna object con tipos
    # mixtos, así que se uniforma a texto conservando los nulos reales
    # como nulos (no como el string "nan").
    for columna in tabla_scoring.select_dtypes(include=["object", "str"]).columns:
        tabla_scoring[columna] = tabla_scoring[columna].apply(
            lambda v: str(v) if pd.notna(v) else None
        )

    tabla_scoring["fecha_prestamo"] = df_features["fecha_prestamo"]
    tabla_scoring["periodo"] = df_features["fecha_prestamo"].dt.to_period("M").astype(str)
    tabla_scoring["probabilidad_predicha"] = probabilidad_predicha
    tabla_scoring["prediccion"] = prediccion
    tabla_scoring["Pago_atiempo"] = df_features["Pago_atiempo"]

    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    tabla_scoring.to_parquet(ruta_salida, index=False)

    return tabla_scoring


def calcular_ks(referencia, actual):
    """Test de Kolmogorov-Smirnov de dos muestras, para variables numéricas.

    Mide la distancia máxima entre las funciones de distribución acumulada
    de `referencia` y `actual` — 0 significa que las dos muestras podrían
    venir de la misma distribución, 1 es la separación máxima posible.

    Interpretación: con las muestras grandes que suele haber en monitoreo
    de producción, el test detecta como "significativas" (p-valor chico)
    hasta diferencias mínimas que no le importan a nadie en la práctica.
    Por eso el **estadístico** (la distancia en sí, entre 0 y 1) importa
    más que el p-valor para decidir si hay drift real; el p-valor sirve
    más como corroboración de que la diferencia no es pura casualidad de
    muestra chica.

    Parameters
    ----------
    referencia, actual : array-like
        Valores numéricos de la ventana de referencia y del período actual.

    Returns
    -------
    dict
        {"estadistico": float, "p_valor": float}
    """
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()
    resultado = ks_2samp(referencia, actual)
    return {"estadistico": float(resultado.statistic), "p_valor": float(resultado.pvalue)}


def calcular_psi(referencia, actual, n_bins=10):
    """Population Stability Index, para variables numéricas.

    Cuantifica cuánto cambió la distribución de una variable entre la
    referencia y el período actual, en bins fijos calculados **sobre la
    referencia únicamente** (deciles) y aplicados tal cual al período
    actual — nunca se recalculan bins sobre el período actual, o se
    estarían comparando dos escalas distintas.

    Interpretación estándar de industria: PSI < 0.10 estable, entre 0.10
    y 0.25 cambio moderado (revisar), > 0.25 cambio significativo (alerta).

    Parameters
    ----------
    referencia, actual : array-like
        Valores numéricos de la ventana de referencia y del período actual.
    n_bins : int, optional
        Cantidad de bins (deciles por defecto, 10). Por defecto 10.

    Returns
    -------
    float
        El PSI. Siempre >= 0.
    """
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()

    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf

    prop_ref = referencia.pipe(lambda s: pd.cut(s, bins=cortes)).value_counts(sort=False) / len(referencia)
    prop_act = actual.pipe(lambda s: pd.cut(s, bins=cortes)).value_counts(sort=False) / len(actual)

    # Proporciones en 0 rompen el logaritmo (da infinito): se reemplazan
    # por un valor chico pero positivo antes de calcular la fórmula.
    prop_ref = prop_ref.replace(0, 1e-6)
    prop_act = prop_act.replace(0, 1e-6)

    psi = ((prop_act - prop_ref) * np.log(prop_act / prop_ref)).sum()
    return float(psi)


def calcular_jensen_shannon(referencia, actual, n_bins=20):
    """Distancia de Jensen-Shannon, para variables numéricas.

    Mismo criterio de bins que `calcular_psi`: se calculan por deciles
    (percentiles equiespaciados) **sobre la referencia** y se aplican tal
    cual al período actual, solo que con más bins por defecto (20) para
    mayor resolución. Se usa `base=2` para que el resultado quede acotado
    entre 0 (distribuciones idénticas) y 1 (distribuciones totalmente
    distintas), a diferencia del PSI, que no tiene techo.

    Interpretación: < 0.10 estable, entre 0.10 y 0.20 cambio moderado,
    > 0.20 cambio significativo (mismos umbrales estilo PSI, adaptados a
    esta escala 0-1).

    Parameters
    ----------
    referencia, actual : array-like
        Valores numéricos de la ventana de referencia y del período actual.
    n_bins : int, optional
        Cantidad de bins calculados sobre la referencia. Por defecto 20.

    Returns
    -------
    float
        La distancia de Jensen-Shannon, entre 0 y 1.
    """
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()

    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf

    prop_ref = referencia.pipe(lambda s: pd.cut(s, bins=cortes)).value_counts(sort=False) / len(referencia)
    prop_act = actual.pipe(lambda s: pd.cut(s, bins=cortes)).value_counts(sort=False) / len(actual)

    distancia = jensenshannon(prop_ref.to_numpy(), prop_act.to_numpy(), base=2)
    return float(distancia)


def calcular_chi2(referencia, actual):
    """Test chi-cuadrado de independencia, para variables categóricas.

    Arma una tabla de contingencia 2xK uniendo las categorías que
    aparecen en la referencia o en el período actual (con 0 donde una
    categoría no aparece en alguno de los dos) y corre el test de
    independencia sobre esa tabla.

    Interpretación: igual que KS, con el p-valor — < 0.05 indica una
    diferencia significativa entre las dos distribuciones de categorías.
    Si alguna frecuencia esperada de la tabla queda por debajo de 5, el
    test pierde validez estadística (es la condición estándar para
    chi-cuadrado); el resultado se devuelve igual, pero con
    `advertencia=True` para que se use con cautela.

    Parameters
    ----------
    referencia, actual : array-like
        Valores categóricos de la ventana de referencia y del período actual.

    Returns
    -------
    dict
        {"estadistico": float, "p_valor": float, "grados_libertad": int,
        "advertencia": bool}
    """
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()

    categorias = sorted(set(referencia.unique()) | set(actual.unique()), key=str)
    conteo_ref = referencia.value_counts().reindex(categorias, fill_value=0)
    conteo_act = actual.value_counts().reindex(categorias, fill_value=0)
    tabla_contingencia = pd.DataFrame({"referencia": conteo_ref, "actual": conteo_act}).T

    estadistico, p_valor, grados_libertad, esperadas = chi2_contingency(tabla_contingencia)
    advertencia = bool((esperadas < 5).any())

    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "grados_libertad": int(grados_libertad),
        "advertencia": advertencia,
    }


def ejecutar_monitoreo(tabla_scoring, fecha_corte_referencia, columnas_numericas, columnas_categoricas, ruta_salida):
    """Corre el monitoreo de drift, período a período, contra una referencia fija.

    Parte `tabla_scoring` en referencia (todo lo anterior a
    `fecha_corte_referencia`) y períodos de monitoreo (uno por cada
    año-mes posterior). Para cada período calcula, contra la MISMA
    referencia siempre: las tres métricas numéricas (KS, PSI,
    Jensen-Shannon) para cada columna de `columnas_numericas`,
    chi-cuadrado para cada columna de `columnas_categoricas`, PSI y
    Jensen-Shannon sobre `probabilidad_predicha` (la salida del modelo,
    la primera señal de que algo cambió), y la diferencia en puntos
    porcentuales de la tasa de aprobación (`prediccion`) contra la
    referencia.

    Es reproducible: no usa aleatoriedad en ningún paso, así que correrlo
    dos veces con los mismos datos da exactamente el mismo resultado.

    Parameters
    ----------
    tabla_scoring : pandas.DataFrame
        La tabla devuelta por `generar_tabla_scoring` (con `fecha_prestamo`,
        `periodo`, `probabilidad_predicha`, `prediccion` y las features).
    fecha_corte_referencia : str o datetime-like
        Todo lo anterior a esta fecha es referencia; todo lo posterior se
        parte en períodos de monitoreo por año-mes.
    columnas_numericas : list[str]
        Columnas numéricas a monitorear (KS, PSI, Jensen-Shannon).
    columnas_categoricas : list[str]
        Columnas categóricas a monitorear (chi-cuadrado).
    ruta_salida : str o pathlib.Path
        Ruta del CSV de salida (`data/processed/historico_drift.csv`).

    Returns
    -------
    pandas.DataFrame
        Formato largo: una fila por período+variable+métrica, con columnas
        `periodo`, `variable`, `tipo_variable`, `metrica`, `valor`,
        `p_valor`, `n_referencia`, `n_actual`.
    """
    fecha_corte_referencia = pd.Timestamp(fecha_corte_referencia)
    referencia = tabla_scoring[tabla_scoring["fecha_prestamo"] < fecha_corte_referencia]
    monitoreo = tabla_scoring[tabla_scoring["fecha_prestamo"] >= fecha_corte_referencia]
    n_referencia = len(referencia)

    tasa_aprobacion_referencia = referencia["prediccion"].mean()

    filas = []
    for periodo in sorted(monitoreo["periodo"].unique()):
        datos_periodo = monitoreo[monitoreo["periodo"] == periodo]
        n_actual = len(datos_periodo)

        if n_actual < 100:
            print(f"[model_monitoring] período {periodo} salteado: {n_actual} registros (< 100 mínimo).")
            continue

        for columna in columnas_numericas:
            ref_col, act_col = referencia[columna], datos_periodo[columna]

            ks = calcular_ks(ref_col, act_col)
            filas.append({"periodo": periodo, "variable": columna, "tipo_variable": "numerica",
                          "metrica": "KS", "valor": ks["estadistico"], "p_valor": ks["p_valor"],
                          "n_referencia": n_referencia, "n_actual": n_actual})

            filas.append({"periodo": periodo, "variable": columna, "tipo_variable": "numerica",
                          "metrica": "PSI", "valor": calcular_psi(ref_col, act_col), "p_valor": np.nan,
                          "n_referencia": n_referencia, "n_actual": n_actual})

            filas.append({"periodo": periodo, "variable": columna, "tipo_variable": "numerica",
                          "metrica": "Jensen-Shannon", "valor": calcular_jensen_shannon(ref_col, act_col),
                          "p_valor": np.nan, "n_referencia": n_referencia, "n_actual": n_actual})

        for columna in columnas_categoricas:
            chi2 = calcular_chi2(referencia[columna], datos_periodo[columna])
            filas.append({"periodo": periodo, "variable": columna, "tipo_variable": "categorica",
                          "metrica": "Chi2", "valor": chi2["estadistico"], "p_valor": chi2["p_valor"],
                          "n_referencia": n_referencia, "n_actual": n_actual})

        # probabilidad_predicha: no es una feature de entrada, es la salida
        # del modelo — la primera señal de que algo está cambiando.
        ref_prob, act_prob = referencia["probabilidad_predicha"], datos_periodo["probabilidad_predicha"]
        filas.append({"periodo": periodo, "variable": "probabilidad_predicha", "tipo_variable": "prediccion",
                      "metrica": "PSI", "valor": calcular_psi(ref_prob, act_prob), "p_valor": np.nan,
                      "n_referencia": n_referencia, "n_actual": n_actual})
        filas.append({"periodo": periodo, "variable": "probabilidad_predicha", "tipo_variable": "prediccion",
                      "metrica": "Jensen-Shannon", "valor": calcular_jensen_shannon(ref_prob, act_prob),
                      "p_valor": np.nan, "n_referencia": n_referencia, "n_actual": n_actual})

        # tasa de aprobación: promedio de la clase predicha, en puntos
        # porcentuales de diferencia contra la referencia (no es una
        # distribución para binear, es un agregado puntual).
        diferencia_pp = (datos_periodo["prediccion"].mean() - tasa_aprobacion_referencia) * 100
        filas.append({"periodo": periodo, "variable": "tasa_aprobacion", "tipo_variable": "agregado",
                      "metrica": "diferencia_pct_puntos", "valor": diferencia_pp, "p_valor": np.nan,
                      "n_referencia": n_referencia, "n_actual": n_actual})

    historico_drift = pd.DataFrame(filas)

    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    historico_drift.to_csv(ruta_salida, index=False)

    return historico_drift


def clasificar_severidad(metrica, valor, p_valor=None, umbrales=None):
    """Clasifica un valor de métrica de drift en "verde"/"amarillo"/"rojo".

    Para PSI y Jensen-Shannon usa `valor` contra los cortes "moderado"/
    "significativo" de `umbrales`. Para KS y Chi2 usa `p_valor` contra el
    corte de significancia: no hay nivel "amarillo" para estas dos (la
    consigna define un solo corte, no un rango), así que quedan en
    "rojo" o "verde".

    Parameters
    ----------
    metrica : str
        Una de "PSI", "Jensen-Shannon", "KS", "Chi2".
    valor : float
        El valor de la métrica (para PSI/Jensen-Shannon).
    p_valor : float, optional
        El p-valor (para KS/Chi2). Si falta o es NaN, se clasifica como
        "verde" (sin evidencia de diferencia, no se puede afirmar drift).
    umbrales : dict, optional
        Diccionario de umbrales a usar; por defecto, el `UMBRALES` del
        módulo. La app de Streamlit pasa acá los umbrales que el usuario
        ajustó en la barra lateral, para no reimplementar esta lógica.

    Returns
    -------
    str
        "verde", "amarillo" o "rojo".
    """
    umbrales = umbrales if umbrales is not None else UMBRALES

    if metrica in ("PSI", "Jensen-Shannon"):
        cortes = umbrales[metrica]
        if valor >= cortes["significativo"]:
            return "rojo"
        if valor >= cortes["moderado"]:
            return "amarillo"
        return "verde"

    if metrica in ("KS", "Chi2"):
        if p_valor is None or (isinstance(p_valor, float) and np.isnan(p_valor)):
            return "verde"
        corte = umbrales[metrica]["p_valor_significativo"]
        return "rojo" if p_valor < corte else "verde"

    raise ValueError(f"clasificar_severidad: métrica desconocida '{metrica}'")


def generar_recomendaciones(historico_drift):
    """Traduce el histórico de drift en mensajes accionables.

    Cada mensaje dice qué pasó, en qué variable, desde cuándo y qué
    conviene hacer — nunca un genérico "se detectó drift". Reglas:

    1. Si 3 o más variables están en rojo en el último período: recomienda
       reentrenamiento completo.
    2. Si una variable está en amarillo o rojo durante 3 períodos
       consecutivos: la señala como deterioro sostenido.
    3. Si `probabilidad_predicha` está en rojo: avisa que cambió la
       distribución de salida del modelo (la señal más urgente).
    4. Si `tasa_aprobacion` se movió más de 10 puntos porcentuales contra
       la referencia: lo marca como posible cambio de política de riesgo.
    5. Si no se dispara ninguna de las anteriores, lo dice explícitamente
       en vez de devolver una lista vacía.

    Parameters
    ----------
    historico_drift : pandas.DataFrame
        El DataFrame devuelto por `ejecutar_monitoreo`.

    Returns
    -------
    list[str]
        Uno o más mensajes accionables.
    """
    datos = historico_drift.copy()

    # clasificar_severidad solo sabe de PSI/Jensen-Shannon/KS/Chi2.
    # "diferencia_pct_puntos" (la métrica de tasa_aprobacion) no es una
    # métrica de drift de distribución, así que no entra en este semáforo
    # genérico: la regla 4, más abajo, la evalúa aparte con su propio
    # umbral (10 puntos porcentuales).
    metricas_clasificables = {"PSI", "Jensen-Shannon", "KS", "Chi2"}
    datos["severidad"] = datos.apply(
        lambda fila: (
            clasificar_severidad(fila["metrica"], fila["valor"], fila.get("p_valor"))
            if fila["metrica"] in metricas_clasificables
            else None
        ),
        axis=1,
    )

    periodos_ordenados = sorted(datos["periodo"].unique())
    mensajes = []

    if not periodos_ordenados:
        return ["No hay períodos de monitoreo para analizar todavía."]

    ultimo_periodo = periodos_ordenados[-1]

    # Regla 1: 3+ variables en rojo en el último período -> reentrenar
    datos_ultimo_periodo = datos[datos["periodo"] == ultimo_periodo]
    variables_rojo = sorted(datos_ultimo_periodo.loc[datos_ultimo_periodo["severidad"] == "rojo", "variable"].unique())
    if len(variables_rojo) >= 3:
        mensajes.append(
            f"🔴 Reentrenamiento completo recomendado: en el período {ultimo_periodo} hay "
            f"{len(variables_rojo)} variables en rojo ({', '.join(variables_rojo)}). Con tantas "
            f"variables fuera de rango a la vez, ajustar el modelo variable por variable ya no "
            f"alcanza — conviene reentrenar con una ventana de datos que incluya {ultimo_periodo} "
            f"en adelante."
        )

    # Regla 2: deterioro sostenido - variable en amarillo/rojo 3 períodos seguidos
    for variable in sorted(datos["variable"].unique()):
        datos_variable = datos[datos["variable"] == variable]
        peor_severidad_por_periodo = {}
        for periodo in periodos_ordenados:
            severidades_periodo = datos_variable.loc[datos_variable["periodo"] == periodo, "severidad"]
            if (severidades_periodo == "rojo").any():
                peor_severidad_por_periodo[periodo] = "rojo"
            elif (severidades_periodo == "amarillo").any():
                peor_severidad_por_periodo[periodo] = "amarillo"
            else:
                peor_severidad_por_periodo[periodo] = "verde"

        largo_racha, inicio_racha, ya_reportada = 0, None, False
        for periodo in periodos_ordenados:
            if peor_severidad_por_periodo[periodo] in ("amarillo", "rojo"):
                if largo_racha == 0:
                    inicio_racha = periodo
                largo_racha += 1
            else:
                largo_racha, inicio_racha, ya_reportada = 0, None, False

            if largo_racha >= 3 and not ya_reportada:
                mensajes.append(
                    f"🟡 Deterioro sostenido en '{variable}': viene en amarillo o rojo desde "
                    f"{inicio_racha} hasta {periodo} ({largo_racha} períodos consecutivos). "
                    f"Revisar el proceso de originación/captura de esta variable en ese rango de "
                    f"fechas antes de que siga empeorando."
                )
                ya_reportada = True

    # Regla 3: probabilidad_predicha en rojo -> señal más urgente
    filas_prob_rojo = datos[(datos["variable"] == "probabilidad_predicha") & (datos["severidad"] == "rojo")]
    if not filas_prob_rojo.empty:
        periodos_afectados = sorted(filas_prob_rojo["periodo"].unique())
        mensajes.append(
            f"🔴 Señal más urgente: la distribución de probabilidad_predicha (la salida del "
            f"modelo, no una variable de entrada) cambió de forma significativa en "
            f"{', '.join(periodos_afectados)}. Es más grave que el drift de una sola variable de "
            f"entrada porque indica que el modelo en conjunto está scoreando distinto a como "
            f"scoreaba en la referencia — revisar esto antes que cualquier otra alerta."
        )

    # Regla 4: tasa de aprobación con más de 10pp de diferencia vs referencia
    filas_tasa = datos[(datos["variable"] == "tasa_aprobacion") & (datos["valor"].abs() > 10)]
    for _, fila in filas_tasa.iterrows():
        direccion = "subió" if fila["valor"] > 0 else "bajó"
        mensajes.append(
            f"🟠 Posible cambio de política de riesgo: la tasa de aprobación {direccion} "
            f"{abs(fila['valor']):.1f} puntos porcentuales respecto de la referencia en el "
            f"período {fila['periodo']}. Puede ser un cambio deliberado del umbral de negocio, no "
            f"necesariamente un problema del modelo — confirmar con el área de riesgo antes de "
            f"asumir que el modelo se deterioró."
        )

    # Regla 5: si no se disparó nada, decirlo explícitamente
    if not mensajes:
        mensajes.append(
            f"🟢 Todo estable: en los {len(periodos_ordenados)} períodos monitoreados (hasta "
            f"{ultimo_periodo}) no se detectó ninguna alerta de drift, deterioro sostenido, ni "
            f"cambios relevantes en la salida del modelo o en la tasa de aprobación. No se "
            f"requiere ninguna acción por ahora."
        )

    return mensajes


def simular_drift(datos, tipo, variable, intensidad):
    """Genera una copia de `datos` con drift SINTÉTICO inyectado en una columna.

    Sirve únicamente para probar que el sistema de alertas dispara de
    verdad cuando hay un cambio real — nunca se usa con datos reales.
    Quien llama a esta función es responsable de dejar en claro, en
    cualquier lugar donde se muestre el resultado, que es una simulación.

    Tres tipos:

    - "desplazamiento" (variable numérica): corre la media `intensidad`
      desvíos estándar de la propia distribución de la columna. Ej.:
      `intensidad=2` mueve la media dos desvíos hacia arriba (negativo
      la mueve hacia abajo).
    - "dispersion" (variable numérica): multiplica el desvío estándar
      por `intensidad`, sin mover la media — la variable queda más
      dispersa (intensidad > 1) o más concentrada (intensidad < 1)
      alrededor del mismo centro.
    - "categorico" (variable categórica): concentra una proporción
      `intensidad` (0 a 1) de los casos en la categoría más frecuente
      de la columna, redistribuyendo el resto entre las demás
      categorías en sus proporciones originales. Usa una semilla fija
      para que la simulación sea reproducible.

    Parameters
    ----------
    datos : pandas.DataFrame
        Datos originales. No se modifican: se devuelve una copia.
    tipo : str
        "desplazamiento", "dispersion" o "categorico".
    variable : str
        Columna a alterar.
    intensidad : float
        Desvíos estándar (desplazamiento), factor multiplicativo
        (dispersion), o proporción 0-1 (categorico).

    Returns
    -------
    pandas.DataFrame
        Copia de `datos` con la columna `variable` alterada.
    """
    datos_simulados = datos.copy()

    if tipo == "desplazamiento":
        desvio = datos_simulados[variable].std()
        datos_simulados[variable] = datos_simulados[variable] + intensidad * desvio

    elif tipo == "dispersion":
        media = datos_simulados[variable].mean()
        datos_simulados[variable] = media + (datos_simulados[variable] - media) * intensidad

    elif tipo == "categorico":
        conteo = datos_simulados[variable].value_counts(normalize=True, dropna=True)
        if conteo.empty:
            raise ValueError(f"simular_drift: la columna '{variable}' no tiene valores no nulos.")
        categoria_frecuente = conteo.index[0]
        n_total = len(datos_simulados)
        n_categoria_frecuente = int(round(n_total * intensidad))

        otras_categorias = conteo.index[1:]
        rng = np.random.default_rng(42)
        nuevos_valores = [categoria_frecuente] * n_categoria_frecuente
        if len(otras_categorias) > 0 and n_total > n_categoria_frecuente:
            proporciones_otras = conteo[otras_categorias] / conteo[otras_categorias].sum()
            resto = rng.choice(otras_categorias, size=n_total - n_categoria_frecuente, p=proporciones_otras.to_numpy())
            nuevos_valores += list(resto)
        nuevos_valores = nuevos_valores[:n_total]
        rng.shuffle(nuevos_valores)
        datos_simulados[variable] = nuevos_valores

    else:
        raise ValueError(
            f"simular_drift: tipo desconocido '{tipo}' (usar 'desplazamiento', 'dispersion' o 'categorico')"
        )

    return datos_simulados


if __name__ == "__main__":
    from src.ft_engineering import CATEGORICAS_NOMINALES, CATEGORICAS_ORDINALES, NUMERICAS, NUMERICAS_DERIVADAS

    RAIZ = Path(__file__).resolve().parent.parent
    RUTA_EXCEL = RAIZ / "data" / "raw" / "Base_de_datos.xlsx"
    RUTA_MODELO = RAIZ / "models" / "modelo_final.pkl"
    RUTA_TABLA_SCORING = RAIZ / "data" / "processed" / "tabla_scoring.parquet"
    RUTA_HISTORICO = RAIZ / "data" / "processed" / "historico_drift.csv"

    # Fecha de corte y ventana de referencia acordadas en monitoreo.ipynb:
    # referencia = 2024-11 a 2025-01 (3.319 registros).
    FECHA_CORTE_REFERENCIA = "2025-02-01"

    print("Generando tabla de scoring...")
    tabla_scoring = generar_tabla_scoring(RUTA_EXCEL, RUTA_MODELO, RUTA_TABLA_SCORING)
    print(f"  tabla_scoring: {tabla_scoring.shape}, guardada en {RUTA_TABLA_SCORING}")

    # Solo se monitorean las columnas que el modelo realmente usa (así
    # "puntaje", excluida por fuga, queda afuera sin tener que
    # hardcodearla acá). Además se excluyen mes_prestamo/trimestre_prestamo/
    # anio_prestamo: son funciones deterministas del propio período de
    # partición (cada período de monitoreo ES un mes calendario), así que
    # su PSI siempre da altísimo sin que eso sea drift real — taparían las
    # señales genuinas. dia_semana_prestamo sí queda, porque no está atado
    # al límite de período.
    COLUMNAS_TRIVIALMENTE_ATADAS_AL_PERIODO = {"mes_prestamo", "trimestre_prestamo", "anio_prestamo"}
    columnas_numericas = [
        c for c in NUMERICAS + NUMERICAS_DERIVADAS
        if c in tabla_scoring.columns and c not in COLUMNAS_TRIVIALMENTE_ATADAS_AL_PERIODO
    ]
    columnas_categoricas = [c for c in CATEGORICAS_NOMINALES + CATEGORICAS_ORDINALES if c in tabla_scoring.columns]

    print("Ejecutando monitoreo de drift...")
    historico_drift = ejecutar_monitoreo(
        tabla_scoring, FECHA_CORTE_REFERENCIA, columnas_numericas, columnas_categoricas, RUTA_HISTORICO,
    )
    print(f"  historico_drift: {historico_drift.shape}, guardado en {RUTA_HISTORICO}")
    print(f"  períodos procesados: {historico_drift['periodo'].nunique()}")
