"""App de Streamlit para monitoreo de riesgo crediticio.

Esta app NO calcula drift: solo lee lo que `src/model_monitoring.py` ya
dejó escrito en `data/processed/` (historico_drift.csv y
tabla_scoring.parquet). Si esos archivos no existen, hay que correr
`python -m src.model_monitoring` primero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.model_monitoring import (
    UMBRALES,
    calcular_chi2,
    calcular_jensen_shannon,
    calcular_ks,
    calcular_psi,
    clasificar_severidad,
    generar_recomendaciones,
    simular_drift,
)

COLOR_SIMULADO = "#8C564B"

# Colores consistentes en toda la app: la referencia siempre el mismo
# color, el período actual siempre otro. Semáforo de severidad aparte.
COLOR_REFERENCIA = "#4C72B0"
COLOR_ACTUAL = "#DD8452"
COLOR_VERDE = "#2E7D32"
COLOR_AMARILLO = "#F9A825"
COLOR_ROJO = "#C62828"

RUTA_HISTORICO = RAIZ / "data" / "processed" / "historico_drift.csv"
RUTA_TABLA_SCORING = RAIZ / "data" / "processed" / "tabla_scoring.parquet"

st.set_page_config(
    page_title="Monitoreo de riesgo crediticio",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def cargar_datos():
    """Carga historico_drift.csv y tabla_scoring.parquet desde data/processed/.

    Decorada con @st.cache_data para no releer los archivos en cada
    interacción del usuario con la app. Si alguno de los dos no existe,
    corta la ejecución con un mensaje claro: hay que correr el job de
    monitoreo antes de levantar la app.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        (historico_drift, tabla_scoring)
    """
    if not RUTA_HISTORICO.exists() or not RUTA_TABLA_SCORING.exists():
        st.error(
            "Faltan los archivos de monitoreo. Antes de levantar esta app, corré desde la "
            "raíz del proyecto:\n\n```\npython -m src.model_monitoring\n```\n\n"
            f"Se esperaba encontrar:\n- `{RUTA_HISTORICO.relative_to(RAIZ)}`\n"
            f"- `{RUTA_TABLA_SCORING.relative_to(RAIZ)}`"
        )
        st.stop()

    historico_drift = pd.read_csv(RUTA_HISTORICO)
    tabla_scoring = pd.read_parquet(RUTA_TABLA_SCORING)
    return historico_drift, tabla_scoring


historico_drift, tabla_scoring = cargar_datos()

# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
st.sidebar.header("Filtros")

periodos_disponibles = sorted(historico_drift["periodo"].unique())
periodo_seleccionado = st.sidebar.selectbox(
    "Período a analizar", periodos_disponibles, index=len(periodos_disponibles) - 1,
)

variables_disponibles = sorted(historico_drift["variable"].unique())
variables_seleccionadas = st.sidebar.multiselect(
    "Variables", variables_disponibles, default=variables_disponibles[:5],
)

st.sidebar.subheader("Umbrales de alerta")
umbral_psi_moderado = st.sidebar.slider(
    "PSI - moderado desde", 0.0, 1.0, UMBRALES["PSI"]["moderado"], step=0.01,
)
umbral_psi_significativo = st.sidebar.slider(
    "PSI - significativo desde", 0.0, 1.0, UMBRALES["PSI"]["significativo"], step=0.01,
)
umbral_js_moderado = st.sidebar.slider(
    "Jensen-Shannon - moderado desde", 0.0, 1.0, UMBRALES["Jensen-Shannon"]["moderado"], step=0.01,
)
umbral_js_significativo = st.sidebar.slider(
    "Jensen-Shannon - significativo desde", 0.0, 1.0, UMBRALES["Jensen-Shannon"]["significativo"], step=0.01,
)

umbrales_usuario = {
    "PSI": {"moderado": umbral_psi_moderado, "significativo": umbral_psi_significativo},
    "Jensen-Shannon": {"moderado": umbral_js_moderado, "significativo": umbral_js_significativo},
    "KS": UMBRALES["KS"],
    "Chi2": UMBRALES["Chi2"],
}

# ----------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------
st.title("📊 Monitoreo de riesgo crediticio")

datos_periodo = historico_drift[historico_drift["periodo"] == periodo_seleccionado].copy()
datos_periodo["severidad"] = datos_periodo.apply(
    lambda fila: clasificar_severidad(fila["metrica"], fila["valor"], fila.get("p_valor"), umbrales_usuario)
    if fila["metrica"] in ("PSI", "Jensen-Shannon", "KS", "Chi2")
    else None,
    axis=1,
)

variables_en_rojo = datos_periodo.loc[datos_periodo["severidad"] == "rojo", "variable"].nunique()
variables_en_amarillo = datos_periodo.loc[datos_periodo["severidad"] == "amarillo", "variable"].nunique()
psi_maximo = datos_periodo.loc[datos_periodo["metrica"] == "PSI", "valor"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Período analizado", periodo_seleccionado)
col2.metric("Variables en rojo", int(variables_en_rojo))
col3.metric("Variables en amarillo", int(variables_en_amarillo))
col4.metric("PSI máximo del período", f"{psi_maximo:.3f}" if pd.notna(psi_maximo) else "—")

# ----------------------------------------------------------------------
# Pestañas
# ----------------------------------------------------------------------
tab_distribuciones, tab_metricas, tab_evolucion, tab_recomendaciones, tab_simulacion = st.tabs(
    ["Distribuciones", "Métricas por variable", "Evolución temporal", "Recomendaciones", "Simulación"]
)

with tab_distribuciones:
    st.caption(
        "Referencia (nov 2024 - ene 2025) vs. período seleccionado, para cada variable elegida "
        "en la barra lateral."
    )

    # La referencia son todas las filas de tabla_scoring anteriores al
    # primer período monitoreado — se deriva así, sin hardcodear la
    # fecha de corte acá (esta app no calcula drift, solo reconstruye
    # qué filas corresponden a cada ventana para graficarlas).
    primer_periodo_monitoreo = min(periodos_disponibles)
    referencia_filas = tabla_scoring[tabla_scoring["periodo"] < primer_periodo_monitoreo]
    actual_filas = tabla_scoring[tabla_scoring["periodo"] == periodo_seleccionado]

    columnas_por_fila = st.columns(2)
    for indice, variable in enumerate(variables_seleccionadas):
        columna_destino = columnas_por_fila[indice % 2]

        info_variable = datos_periodo[datos_periodo["variable"] == variable]
        tipo_variable = info_variable["tipo_variable"].iloc[0] if not info_variable.empty else None

        with columna_destino:
            st.markdown(f"**{variable}**")

            if tipo_variable == "agregado":
                st.info(
                    f"'{variable}' es un agregado (un solo número por período, no una "
                    "distribución) — se analiza en la pestaña 'Evolución temporal'."
                )
                continue

            columna_datos = "prediccion" if tipo_variable == "agregado" else variable
            serie_referencia = referencia_filas[columna_datos].dropna()
            serie_actual = actual_filas[columna_datos].dropna()

            if tipo_variable in ("numerica", "prediccion"):
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=serie_referencia, name="Referencia", histnorm="probability density",
                    marker_color=COLOR_REFERENCIA, opacity=0.55,
                ))
                fig.add_trace(go.Histogram(
                    x=serie_actual, name=periodo_seleccionado, histnorm="probability density",
                    marker_color=COLOR_ACTUAL, opacity=0.55,
                ))
                fig.add_vline(x=serie_referencia.mean(), line_dash="dash", line_color=COLOR_REFERENCIA,
                               annotation_text="media referencia", annotation_position="top left",
                               annotation_y=1.0)
                fig.add_vline(x=serie_actual.mean(), line_dash="dash", line_color=COLOR_ACTUAL,
                               annotation_text=f"media {periodo_seleccionado}", annotation_position="top left",
                               annotation_y=0.9)
                fig.update_layout(
                    barmode="overlay", xaxis_title=variable, yaxis_title="Densidad",
                    title=f"Distribución de {variable}", height=350, margin=dict(t=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            elif tipo_variable == "categorica":
                pct_referencia = (serie_referencia.value_counts(normalize=True) * 100).rename("Referencia")
                pct_actual = (serie_actual.value_counts(normalize=True) * 100).rename(periodo_seleccionado)
                tabla_pct = pd.concat([pct_referencia, pct_actual], axis=1).fillna(0).sort_index()

                fig = go.Figure()
                fig.add_trace(go.Bar(x=tabla_pct.index.astype(str), y=tabla_pct["Referencia"],
                                      name="Referencia", marker_color=COLOR_REFERENCIA))
                fig.add_trace(go.Bar(x=tabla_pct.index.astype(str), y=tabla_pct[periodo_seleccionado],
                                      name=periodo_seleccionado, marker_color=COLOR_ACTUAL))
                fig.update_layout(
                    barmode="group", xaxis_title="Categoría", yaxis_title="% de casos",
                    title=f"Distribución de {variable}", height=350, margin=dict(t=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Línea de métricas + severidad, debajo del gráfico.
            partes = []
            for _, fila in info_variable.iterrows():
                if pd.notna(fila.get("severidad")):
                    icono = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}.get(fila["severidad"], "")
                    valor_texto = f"{fila['valor']:.3f}" if fila["metrica"] != "Chi2" else f"{fila['valor']:.1f}"
                    partes.append(f"{fila['metrica']}={valor_texto} {icono}")
            if partes:
                st.caption(" · ".join(partes) + f" (n_referencia={info_variable['n_referencia'].iloc[0]}, "
                           f"n_actual={info_variable['n_actual'].iloc[0]})")

with tab_metricas:
    st.caption(
        f"Métricas de cada variable seleccionada en el período {periodo_seleccionado}, con los "
        "umbrales de la barra lateral. Ordenado por PSI descendente."
    )

    ORDEN_SEVERIDAD = {"verde": 0, "amarillo": 1, "rojo": 2}
    SIMBOLO_SEVERIDAD = {"verde": "✓ Verde", "amarillo": "▲ Amarillo", "rojo": "✕ Rojo"}
    COLOR_FILA = {"verde": "#173318", "amarillo": "#4a3a08", "rojo": "#451414"}

    filas_tabla = []
    for variable in variables_seleccionadas:
        info = datos_periodo[datos_periodo["variable"] == variable]
        if info.empty:
            continue
        tipo_variable = info["tipo_variable"].iloc[0]

        def _valor_de(metrica, columna="valor"):
            sub = info.loc[info["metrica"] == metrica, columna]
            return sub.iloc[0] if not sub.empty else np.nan

        psi_valor = _valor_de("PSI")
        js_valor = _valor_de("Jensen-Shannon")
        ks_estadistico = _valor_de("KS")
        # p-valor: el de KS si es numérica, el de Chi2 si es categórica.
        p_valor = _valor_de("KS", "p_valor")
        if pd.isna(p_valor):
            p_valor = _valor_de("Chi2", "p_valor")

        severidades = [
            clasificar_severidad(fila["metrica"], fila["valor"], fila.get("p_valor"), umbrales_usuario)
            for _, fila in info.iterrows()
            if fila["metrica"] in ("PSI", "Jensen-Shannon", "KS", "Chi2")
        ]
        severidad_final = max(severidades, key=lambda s: ORDEN_SEVERIDAD[s]) if severidades else "verde"

        filas_tabla.append({
            "variable": variable, "tipo": tipo_variable, "PSI": psi_valor,
            "Jensen-Shannon": js_valor, "KS": ks_estadistico, "p_valor": p_valor,
            "n": int(info["n_actual"].iloc[0]), "severidad": severidad_final,
        })

    if not filas_tabla:
        st.info("No hay variables seleccionadas en la barra lateral.")
    else:
        tabla_metricas = pd.DataFrame(filas_tabla).sort_values("PSI", ascending=False, na_position="last")
        tabla_metricas["Severidad"] = tabla_metricas["severidad"].map(SIMBOLO_SEVERIDAD)

        columnas_mostrar = ["variable", "tipo", "PSI", "Jensen-Shannon", "KS", "p_valor", "n", "Severidad"]

        def _pintar_fila(fila):
            color = COLOR_FILA.get(fila["severidad"], "")
            return [f"background-color: {color}"] * len(fila)

        tabla_estilizada = tabla_metricas[columnas_mostrar + ["severidad"]].style.apply(_pintar_fila, axis=1)

        st.dataframe(
            tabla_estilizada,
            column_order=columnas_mostrar,
            column_config={
                "PSI": st.column_config.ProgressColumn("PSI", min_value=0.0, max_value=0.5, format="%.3f"),
                "Jensen-Shannon": st.column_config.NumberColumn("Jensen-Shannon", format="%.3f"),
                "KS": st.column_config.NumberColumn("KS", format="%.3f"),
                "p_valor": st.column_config.NumberColumn("p-valor", format="%.4f"),
                "severidad": None,
            },
            use_container_width=True,
            hide_index=True,
        )

        conteo_severidad = tabla_metricas["severidad"].value_counts()
        peores = tabla_metricas.head(3)["variable"].tolist()
        st.markdown(
            f"**Resumen de {periodo_seleccionado}:** "
            f"🟢 {conteo_severidad.get('verde', 0)} en verde · "
            f"🟡 {conteo_severidad.get('amarillo', 0)} en amarillo · "
            f"🔴 {conteo_severidad.get('rojo', 0)} en rojo. "
            f"Las 3 peores del período: **{', '.join(peores)}**."
        )

        st.download_button(
            "⬇️ Descargar tabla del período (CSV)",
            data=tabla_metricas[columnas_mostrar].to_csv(index=False).encode("utf-8"),
            file_name=f"metricas_drift_{periodo_seleccionado}.csv",
            mime="text/csv",
        )

with tab_evolucion:
    st.caption("PSI a lo largo de todos los períodos, para las variables seleccionadas en la barra lateral.")

    datos_psi_todas = historico_drift[
        (historico_drift["metrica"] == "PSI") & (historico_drift["variable"].isin(variables_seleccionadas))
    ]

    # 1. Líneas de PSI con bandas de umbral de fondo
    psi_maximo_grafico = max(datos_psi_todas["valor"].max() * 1.1, umbral_psi_significativo * 1.2) if not datos_psi_todas.empty else 1.0
    fig_lineas = go.Figure()
    fig_lineas.add_hrect(y0=0, y1=umbral_psi_moderado, fillcolor=COLOR_VERDE, opacity=0.10, line_width=0)
    fig_lineas.add_hrect(y0=umbral_psi_moderado, y1=umbral_psi_significativo, fillcolor=COLOR_AMARILLO, opacity=0.10, line_width=0)
    fig_lineas.add_hrect(y0=umbral_psi_significativo, y1=psi_maximo_grafico, fillcolor=COLOR_ROJO, opacity=0.10, line_width=0)
    for variable in variables_seleccionadas:
        serie = datos_psi_todas[datos_psi_todas["variable"] == variable].sort_values("periodo")
        if not serie.empty:
            fig_lineas.add_trace(go.Scatter(x=serie["periodo"], y=serie["valor"], mode="lines+markers", name=variable))
    fig_lineas.update_layout(xaxis_title="Período", yaxis_title="PSI", title="Evolución de PSI por variable", height=450)
    st.plotly_chart(fig_lineas, use_container_width=True)

    if not datos_psi_todas.empty:
        ultimo = datos_psi_todas[datos_psi_todas["periodo"] == periodo_seleccionado].sort_values("valor", ascending=False)
        if not ultimo.empty:
            peor_variable = ultimo.iloc[0]
            st.markdown(
                f"**Conclusión operativa:** en {periodo_seleccionado}, `{peor_variable['variable']}` es la que "
                f"más lejos está de la banda verde (PSI={peor_variable['valor']:.3f}) — es la primera candidata "
                f"a revisar si hay que priorizar."
            )

    # 2. Mapa de calor variable x período
    st.subheader("Mapa de calor: variable x período")
    pivot_psi = datos_psi_todas.pivot(index="variable", columns="periodo", values="valor").reindex(columns=periodos_disponibles)
    if not pivot_psi.empty:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_psi.values, x=pivot_psi.columns, y=pivot_psi.index,
            colorscale="Reds", colorbar=dict(title="PSI"),
        ))
        fig_heatmap.update_layout(title="PSI por variable y período", height=max(300, 40 * len(pivot_psi)))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown(
            "**Conclusión operativa:** las columnas más oscuras de punta a punta (si las hay) muestran un "
            "problema general del período, no de una variable puntual; las filas oscuras solo hacia la "
            "derecha muestran una variable que se fue deteriorando con el tiempo, no que arrancó mal."
        )

    # 3. Detección de cambios abruptos
    st.subheader("Cambios abruptos (> 2 desvíos estándar de la propia serie)")
    saltos_detectados = []
    for variable in variables_seleccionadas:
        serie = datos_psi_todas[datos_psi_todas["variable"] == variable].sort_values("periodo")
        if len(serie) < 3:
            continue
        diferencias = serie["valor"].diff().dropna()
        umbral_salto = 2 * diferencias.std()
        if pd.isna(umbral_salto) or umbral_salto == 0:
            continue
        periodos_serie = serie["periodo"].tolist()
        for posicion, diferencia in zip(range(1, len(periodos_serie)), diferencias):
            if abs(diferencia) > umbral_salto:
                saltos_detectados.append({
                    "variable": variable, "periodo": periodos_serie[posicion],
                    "magnitud_del_salto": round(diferencia, 4),
                })

    if saltos_detectados:
        tabla_saltos = pd.DataFrame(saltos_detectados).sort_values("magnitud_del_salto", key=abs, ascending=False)
        st.dataframe(tabla_saltos, hide_index=True, use_container_width=True)
        peor_salto = tabla_saltos.iloc[0]
        st.markdown(
            f"**Conclusión operativa:** el salto más grande es en `{peor_salto['variable']}` en "
            f"{peor_salto['periodo']} — vale la pena revisar qué pasó ese mes puntualmente (¿cambió algo "
            f"en la captura de datos, o es un mes con muestra chica?), no solo tratarlo como parte de una "
            f"tendencia."
        )
    else:
        st.info("No se detectaron saltos abruptos en las variables seleccionadas.")
        st.markdown("**Conclusión operativa:** el drift, donde existe, se mueve gradual — no hay un evento puntual que haya roto algo de un mes a otro.")

    # 4. Detección de tendencia (regresión lineal simple de PSI vs. número de período)
    st.subheader("Tendencia (pendiente de PSI a lo largo del tiempo)")
    tendencias = []
    for variable in variables_seleccionadas:
        serie = datos_psi_todas[datos_psi_todas["variable"] == variable].sort_values("periodo")
        if len(serie) < 2:
            continue
        x = np.arange(len(serie))
        pendiente, _ = np.polyfit(x, serie["valor"].to_numpy(), 1)
        tendencias.append({"variable": variable, "pendiente_psi_por_periodo": round(pendiente, 5)})

    if tendencias:
        tabla_tendencias = pd.DataFrame(tendencias).sort_values("pendiente_psi_por_periodo", ascending=False)
        st.dataframe(tabla_tendencias, hide_index=True, use_container_width=True)
        degradando = tabla_tendencias[tabla_tendencias["pendiente_psi_por_periodo"] > 0]["variable"].tolist()
        if degradando:
            st.markdown(
                f"**Conclusión operativa:** {', '.join(degradando)} tienen pendiente positiva — se están "
                f"degradando despacio, período a período. Son las que conviene volver a mirar en el próximo "
                f"ciclo de monitoreo aunque hoy no estén en rojo."
            )
        else:
            st.markdown("**Conclusión operativa:** ninguna de las variables seleccionadas muestra una tendencia sostenida a empeorar.")

    # 5. Tasa de aprobación vs. referencia
    st.subheader("Tasa de aprobación por período vs. referencia")
    tasa_aprobacion_referencia_pct = referencia_filas["prediccion"].mean() * 100
    datos_tasa = historico_drift[historico_drift["variable"] == "tasa_aprobacion"].sort_values("periodo")
    tasa_actual_pct = tasa_aprobacion_referencia_pct + datos_tasa["valor"]

    fig_tasa = go.Figure()
    fig_tasa.add_trace(go.Scatter(x=datos_tasa["periodo"], y=tasa_actual_pct, mode="lines+markers",
                                    name="Tasa de aprobación", line_color=COLOR_ACTUAL))
    fig_tasa.add_hline(y=tasa_aprobacion_referencia_pct, line_dash="dash", line_color=COLOR_REFERENCIA,
                        annotation_text="referencia")
    fig_tasa.update_layout(xaxis_title="Período", yaxis_title="Tasa de aprobación (%)", height=350)
    st.plotly_chart(fig_tasa, use_container_width=True)

    diferencia_maxima = datos_tasa["valor"].abs().max()
    st.markdown(
        f"**Conclusión operativa:** la tasa de aprobación se movió como máximo {diferencia_maxima:.1f} "
        f"puntos porcentuales respecto de la referencia ({tasa_aprobacion_referencia_pct:.1f}%). "
        + ("Es un movimiento grande — confirmar con el área de riesgo si fue un cambio de política deliberado."
           if diferencia_maxima > 10 else
           "No es un movimiento grande; no sugiere un cambio de política de riesgo por sí solo.")
    )

with tab_recomendaciones:
    mensajes_recomendaciones = generar_recomendaciones(historico_drift)

    tiene_critico = any(m.startswith("🔴") for m in mensajes_recomendaciones)
    tiene_moderado = any(m.startswith("🟡") or m.startswith("🟠") for m in mensajes_recomendaciones)

    if tiene_critico:
        veredicto = "El modelo requiere reentrenamiento."
    elif tiene_moderado:
        veredicto = "El modelo requiere revisión."
    else:
        veredicto = "El modelo está operando con normalidad."

    st.subheader("Diagnóstico general")
    if tiene_critico:
        st.error(f"🔴 {veredicto}")
    elif tiene_moderado:
        st.warning(f"🟡 {veredicto}")
    else:
        st.success(f"🟢 {veredicto}")

    st.subheader("Alertas y recomendaciones")
    for mensaje in mensajes_recomendaciones:
        if mensaje.startswith("🔴"):
            st.error(mensaje)
        elif mensaje.startswith("🟡") or mensaje.startswith("🟠"):
            st.warning(mensaje)
        else:
            st.success(mensaje)

    st.subheader("Qué hacer")
    if tiene_critico:
        st.markdown(
            "**Reentrenar.** Ventana de datos: incluir los períodos donde aparecieron las alertas en rojo "
            "en adelante — no hace falta descartar toda la historia, pero sí ponderar más los datos "
            "recientes. Por qué: si varias variables ya cambiaron de forma sostenida, un modelo entrenado "
            "solo con la referencia original ya no representa a la población actual."
        )
    elif tiene_moderado:
        st.markdown(
            "**Revisar variables.** Cuáles: las marcadas en amarillo o rojo en las alertas de arriba. Qué "
            "chequear en cada una: si el cambio viene de un fenómeno de negocio real (un producto nuevo, "
            "un segmento de clientes distinto) o de un problema de captura de datos (un campo mal cargado, "
            "un proceso upstream que cambió) — la respuesta cambia si conviene reentrenar o solo corregir "
            "el dato."
        )
    else:
        st.markdown(
            "**Sin acción por ahora.** Cuándo volver a mirar: en el próximo ciclo de monitoreo (el mes "
            "siguiente), sin urgencia — no hay ninguna señal que amerite adelantar la revisión."
        )

    st.divider()
    st.caption(
        "⚠️ **Limitación de este monitoreo:** detecta cambios en la población de entrada (data drift), pero "
        "**no puede medir si el modelo perdió precisión real** — para eso haría falta conocer el resultado "
        "real de los créditos recientes (si se pagaron a tiempo o no), y ese resultado todavía no existe: "
        "los créditos de los últimos períodos no maduraron. Hay que esperar a que la cartera madure para "
        "poder comparar la predicción contra el resultado real."
    )

    def _generar_informe_markdown():
        lineas = [
            f"# Informe de monitoreo - {periodo_seleccionado}",
            "",
            f"**Diagnóstico general:** {veredicto}",
            "",
            "## Métricas del período",
            "",
        ]
        for _, fila in datos_periodo.iterrows():
            if pd.notna(fila.get("severidad")):
                lineas.append(f"- {fila['variable']} ({fila['metrica']}): {fila['valor']:.4f} — {fila['severidad']}")
        lineas += ["", "## Alertas y recomendaciones", ""]
        lineas += [f"- {m}" for m in mensajes_recomendaciones]
        lineas += [
            "",
            "## Limitación",
            "",
            "Este monitoreo detecta cambios en la población de entrada, no pérdida de precisión real del "
            "modelo (eso requiere conocer el resultado real de los créditos recientes, que todavía no existe).",
        ]
        return "\n".join(lineas)

    st.download_button(
        "⬇️ Exportar informe del período (Markdown)",
        data=_generar_informe_markdown().encode("utf-8"),
        file_name=f"informe_monitoreo_{periodo_seleccionado}.md",
        mime="text/markdown",
    )

with tab_simulacion:
    st.warning(
        "⚠️ **SIMULACIÓN — datos sintéticos, no reales.** Esta pestaña genera un drift artificial "
        "(con `simular_drift`) para probar que el sistema de alertas responde de verdad ante un "
        "cambio. Nada de lo que se muestra debajo de esta línea representa la población real de "
        "créditos, y no debe usarse para ninguna decisión de negocio.",
        icon="⚠️",
    )

    tipo_variable_por_columna = historico_drift.drop_duplicates("variable").set_index("variable")["tipo_variable"]
    variables_numericas_sim = sorted(tipo_variable_por_columna[tipo_variable_por_columna == "numerica"].index)
    variables_categoricas_sim = sorted(tipo_variable_por_columna[tipo_variable_por_columna == "categorica"].index)

    col_tipo, col_variable, col_intensidad = st.columns(3)
    with col_tipo:
        tipo_simulacion = st.selectbox("Tipo de simulación", ["desplazamiento", "dispersion", "categorico"])
    with col_variable:
        opciones_variable = variables_categoricas_sim if tipo_simulacion == "categorico" else variables_numericas_sim
        variable_simulacion = st.selectbox("Variable a alterar", opciones_variable)
    with col_intensidad:
        if tipo_simulacion == "desplazamiento":
            intensidad_simulacion = st.slider("Intensidad (desvíos estándar)", -5.0, 5.0, 2.0, step=0.5)
        elif tipo_simulacion == "dispersion":
            intensidad_simulacion = st.slider("Intensidad (factor multiplicativo del desvío)", 0.1, 5.0, 2.0, step=0.1)
        else:
            intensidad_simulacion = st.slider(
                "Intensidad (proporción en la categoría más frecuente)", 0.0, 1.0, 0.9, step=0.05
            )

    datos_simulados = simular_drift(actual_filas, tipo_simulacion, variable_simulacion, intensidad_simulacion)

    st.subheader(f"Antes vs. después: {variable_simulacion}")

    if tipo_simulacion in ("desplazamiento", "dispersion"):
        psi_antes = calcular_psi(referencia_filas[variable_simulacion], actual_filas[variable_simulacion])
        psi_despues = calcular_psi(referencia_filas[variable_simulacion], datos_simulados[variable_simulacion])
        js_antes = calcular_jensen_shannon(referencia_filas[variable_simulacion], actual_filas[variable_simulacion])
        js_despues = calcular_jensen_shannon(referencia_filas[variable_simulacion], datos_simulados[variable_simulacion])
        ks_antes = calcular_ks(referencia_filas[variable_simulacion], actual_filas[variable_simulacion])
        ks_despues = calcular_ks(referencia_filas[variable_simulacion], datos_simulados[variable_simulacion])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("PSI", f"{psi_despues:.3f}", delta=f"{psi_despues - psi_antes:+.3f}")
        col_b.metric("Jensen-Shannon", f"{js_despues:.3f}", delta=f"{js_despues - js_antes:+.3f}")
        col_c.metric("KS (estadístico)", f"{ks_despues['estadistico']:.3f}",
                     delta=f"{ks_despues['estadistico'] - ks_antes['estadistico']:+.3f}")

        fig_simulacion = go.Figure()
        fig_simulacion.add_trace(go.Histogram(x=referencia_filas[variable_simulacion], name="Referencia",
                                                histnorm="probability density", marker_color=COLOR_REFERENCIA, opacity=0.5))
        fig_simulacion.add_trace(go.Histogram(x=actual_filas[variable_simulacion], name=f"{periodo_seleccionado} (real)",
                                                histnorm="probability density", marker_color=COLOR_ACTUAL, opacity=0.5))
        fig_simulacion.add_trace(go.Histogram(x=datos_simulados[variable_simulacion], name=f"{periodo_seleccionado} (SIMULADO)",
                                                histnorm="probability density", marker_color=COLOR_SIMULADO, opacity=0.5))
        fig_simulacion.update_layout(barmode="overlay", xaxis_title=variable_simulacion, yaxis_title="Densidad",
                                       title=f"{variable_simulacion}: referencia vs. real vs. simulado (SINTÉTICO)", height=420)
        st.plotly_chart(fig_simulacion, use_container_width=True)

        severidad_antes = clasificar_severidad("PSI", psi_antes, None, umbrales_usuario)
        severidad_despues = clasificar_severidad("PSI", psi_despues, None, umbrales_usuario)

    else:
        chi2_antes = calcular_chi2(referencia_filas[variable_simulacion], actual_filas[variable_simulacion])
        chi2_despues = calcular_chi2(referencia_filas[variable_simulacion], datos_simulados[variable_simulacion])

        col_a, col_b = st.columns(2)
        col_a.metric("Chi2 (estadístico)", f"{chi2_despues['estadistico']:.2f}",
                     delta=f"{chi2_despues['estadistico'] - chi2_antes['estadistico']:+.2f}")
        col_b.metric("p-valor", f"{chi2_despues['p_valor']:.4f}")

        pct_referencia = (referencia_filas[variable_simulacion].value_counts(normalize=True) * 100).rename("Referencia")
        pct_simulado = (datos_simulados[variable_simulacion].value_counts(normalize=True) * 100).rename("Simulado")
        tabla_pct_simulacion = pd.concat([pct_referencia, pct_simulado], axis=1).fillna(0).sort_index()

        fig_simulacion = go.Figure()
        fig_simulacion.add_trace(go.Bar(x=tabla_pct_simulacion.index.astype(str), y=tabla_pct_simulacion["Referencia"],
                                          name="Referencia", marker_color=COLOR_REFERENCIA))
        fig_simulacion.add_trace(go.Bar(x=tabla_pct_simulacion.index.astype(str), y=tabla_pct_simulacion["Simulado"],
                                          name="Simulado (SINTÉTICO)", marker_color=COLOR_SIMULADO))
        fig_simulacion.update_layout(barmode="group", xaxis_title="Categoría", yaxis_title="% de casos",
                                       title=f"{variable_simulacion}: referencia vs. simulado (SINTÉTICO)", height=420)
        st.plotly_chart(fig_simulacion, use_container_width=True)

        severidad_antes = clasificar_severidad("Chi2", None, chi2_antes["p_valor"], umbrales_usuario)
        severidad_despues = clasificar_severidad("Chi2", None, chi2_despues["p_valor"], umbrales_usuario)

    st.subheader("Alertas que se dispararían")
    etiqueta_severidad = {"verde": "🟢 estable", "amarillo": "🟡 moderado", "rojo": "🔴 significativo"}
    st.markdown(f"- **Sin simular** (datos reales de {periodo_seleccionado}): {etiqueta_severidad[severidad_antes]}")
    st.markdown(f"- **Con la simulación aplicada**: {etiqueta_severidad[severidad_despues]}")

    orden_severidad = {"verde": 0, "amarillo": 1, "rojo": 2}
    if orden_severidad[severidad_despues] > orden_severidad[severidad_antes]:
        st.success(
            f"✅ La simulación subió la severidad de {severidad_antes} a {severidad_despues} — el sistema "
            "de alertas responde correctamente ante un cambio real en los datos."
        )
    elif severidad_despues == severidad_antes:
        st.info("La severidad no cambió con esta intensidad — probá con un valor más alto.")
    else:
        st.warning("La severidad bajó con esta simulación (puede pasar con 'dispersion' e intensidad < 1).")
