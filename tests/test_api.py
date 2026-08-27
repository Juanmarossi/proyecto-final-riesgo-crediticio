"""Tests de la API de predicción (src/model_deploy.py)."""

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ft_engineering import crear_features
from src.model_deploy import SolicitudCredito, app

RAIZ = Path(__file__).resolve().parent.parent
EJEMPLO_VALIDO = SolicitudCredito.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _fila_a_payload(fila):
    """Convierte una fila del dataset a un dict apto para SolicitudCredito.

    pandas representa los nulos como NaN, que no es JSON válido — se
    convierten a None (el esquema acepta None en los 7 campos que
    legítimamente tienen nulos reales, ver model_deploy.py).
    """
    payload = fila.drop(labels=["puntaje", "Pago_atiempo"]).to_dict()
    payload = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in payload.items()}
    payload["fecha_prestamo"] = fila["fecha_prestamo"].isoformat()
    return payload


@pytest.fixture(scope="module")
def filas_reales():
    """10 filas reales del dataset, sin puntaje ni Pago_atiempo (semilla fija, reproducible).

    Se filtran las filas que no pasarían la validación de SolicitudCredito
    (por ejemplo, las ~150 con edad_cliente fuera de [18, 100] o las ~58
    con tendencia_ingresos fuera de dominio, ya documentadas en el EDA como
    errores de captura): esas ya se rechazan a propósito por validación
    (ver test de edad_cliente=15) y no son el objetivo de este test — acá
    interesa que la predicción coincida para solicitudes VÁLIDAS.
    """
    df = pd.read_excel(RAIZ / "data" / "raw" / "Base_de_datos.xlsx", engine="openpyxl")
    candidatas = df.sample(n=50, random_state=123).reset_index(drop=True)

    filas_validas = []
    for i in range(len(candidatas)):
        fila = candidatas.iloc[i]
        try:
            SolicitudCredito(**_fila_a_payload(fila))
            filas_validas.append(fila)
        except Exception:
            continue
        if len(filas_validas) == 10:
            break

    assert len(filas_validas) == 10, "no se juntaron 10 filas válidas en la muestra de 50"
    return pd.DataFrame(filas_validas).reset_index(drop=True)


# --- 1. /health -------------------------------------------------------
def test_health_devuelve_200_y_modelo_cargado(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["modelo_cargado"] is True
    assert datos["algoritmo"] == "GradientBoostingClassifier"


# --- 2. /predict válido -------------------------------------------------
def test_predict_registro_valido_devuelve_200_y_probabilidad_entre_0_y_1(client):
    respuesta = client.post("/predict", json=EJEMPLO_VALIDO)
    assert respuesta.status_code == 200
    probabilidad = respuesta.json()["probabilidad"]
    assert 0.0 <= probabilidad <= 1.0


# --- 3. El test que importa: coincide EXACTAMENTE con el modelo directo --
def test_predict_coincide_exactamente_con_modelo_cargado_directamente(client, filas_reales):
    modelo = joblib.load(RAIZ / "models" / "modelo_final.pkl")
    metadata = json.load(open(RAIZ / "models" / "modelo_final_metadata.json", encoding="utf-8"))
    columnas_esperadas = metadata["columnas_entrada_esperadas"]

    df_features = crear_features(filas_reales)
    probabilidades_directas = modelo.predict_proba(df_features[columnas_esperadas])[:, 1]

    for i in range(len(filas_reales)):
        payload = _fila_a_payload(filas_reales.iloc[i])
        respuesta = client.post("/predict", json=payload)
        assert respuesta.status_code == 200, respuesta.text
        probabilidad_api = respuesta.json()["probabilidad"]
        assert probabilidad_api == pytest.approx(float(probabilidades_directas[i]), abs=1e-9)


# --- 4. edad_cliente inválida -------------------------------------------
def test_predict_edad_cliente_15_da_422(client):
    payload = {**EJEMPLO_VALIDO, "edad_cliente": 15}
    respuesta = client.post("/predict", json=payload)
    assert respuesta.status_code == 422


# --- 5. Campo faltante ---------------------------------------------------
def test_predict_campo_faltante_da_422_y_nombra_el_campo(client):
    payload = {k: v for k, v in EJEMPLO_VALIDO.items() if k != "salario_cliente"}
    respuesta = client.post("/predict", json=payload)
    assert respuesta.status_code == 422
    detalle = respuesta.json()["detail"]
    ubicaciones = [str(err.get("loc")) for err in detalle]
    assert any("salario_cliente" in u for u in ubicaciones)


# --- 6. tipo_laboral inventado -------------------------------------------
def test_predict_tipo_laboral_inventado_da_422(client):
    payload = {**EJEMPLO_VALIDO, "tipo_laboral": "Freelancer"}
    respuesta = client.post("/predict", json=payload)
    assert respuesta.status_code == 422


# --- 7. /predict/batch mantiene el orden ---------------------------------
def test_predict_batch_50_registros_devuelve_50_en_el_mismo_orden(client):
    lote = [EJEMPLO_VALIDO] * 50
    respuesta = client.post("/predict/batch", json=lote)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert len(datos["resultados"]) == 50
    indices = [r["indice"] for r in datos["resultados"]]
    assert indices == list(range(50))
    assert datos["cantidad_procesada"] == 50
    assert datos["cantidad_con_error"] == 0


# --- 8. Un registro inválido en el medio no tira el lote -----------------
def test_predict_batch_registro_invalido_en_el_medio_procesa_el_resto(client):
    lote = [EJEMPLO_VALIDO] * 49 + [{**EJEMPLO_VALIDO, "edad_cliente": 999}]
    respuesta = client.post("/predict/batch", json=lote)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert len(datos["resultados"]) == 50
    assert datos["cantidad_procesada"] == 49
    assert datos["cantidad_con_error"] == 1

    resultado_invalido = next(r for r in datos["resultados"] if r["indice"] == 49)
    assert resultado_invalido["exitoso"] is False
    assert resultado_invalido["error"] is not None

    for r in datos["resultados"][:49]:
        assert r["exitoso"] is True


# --- 9. /predict/csv válido -----------------------------------------------
def test_predict_csv_valido_devuelve_csv_con_columnas_agregadas(client, filas_reales):
    payloads = [_fila_a_payload(filas_reales.iloc[i]) for i in range(len(filas_reales))]
    df_csv = pd.DataFrame(payloads)
    buffer = io.StringIO()
    df_csv.to_csv(buffer, index=False)
    buffer.seek(0)

    respuesta = client.post(
        "/predict/csv",
        files={"archivo": ("solicitudes.csv", buffer.getvalue(), "text/csv")},
    )
    assert respuesta.status_code == 200
    df_resultado = pd.read_csv(io.StringIO(respuesta.text))
    assert "probabilidad" in df_resultado.columns
    assert "prediccion" in df_resultado.columns
    assert len(df_resultado) == len(filas_reales)


# --- 10. /predict/csv con columnas faltantes ------------------------------
def test_predict_csv_columnas_faltantes_da_422(client, filas_reales):
    payloads = [_fila_a_payload(filas_reales.iloc[i]) for i in range(3)]
    df_csv = pd.DataFrame(payloads).drop(columns=["salario_cliente"])
    buffer = io.StringIO()
    df_csv.to_csv(buffer, index=False)
    buffer.seek(0)

    respuesta = client.post(
        "/predict/csv",
        files={"archivo": ("solicitudes.csv", buffer.getvalue(), "text/csv")},
    )
    assert respuesta.status_code == 422
    assert "salario_cliente" in respuesta.json()["detail"]
