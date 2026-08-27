"""API de predicción del modelo de riesgo crediticio.

Expone `models/modelo_final.pkl` como servicio HTTP con FastAPI. Nunca
reimplementa la lógica de features (las importa de `src.ft_engineering`) y
nunca predice sobre el JSON crudo: primero valida, después crea las
features, recién ahí predice.
"""

import io
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from src.ft_engineering import crear_features

logger = logging.getLogger("model_deploy")
logging.basicConfig(level=logging.INFO)

VERSION_API = "1.3.0"
LIMITE_LOTE = 1000

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "modelo_final.pkl"
RUTA_METADATA = RAIZ / "models" / "modelo_final_metadata.json"


@lru_cache(maxsize=1)
def _cargar_metadata():
    """Lee models/modelo_final_metadata.json, cacheado (se lee una sola vez)."""
    with open(RUTA_METADATA, encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------
# Dominios válidos, tomados del entrenamiento (comprension_eda.ipynb /
# ft_engineering.ipynb) — no inventados acá.
# ----------------------------------------------------------------------
TipoLaboral = Literal["Empleado", "Independiente"]
TendenciaIngresos = Literal["Creciente", "Estable", "Decreciente"]


class SolicitudCredito(BaseModel):
    """Datos disponibles al momento de evaluar una solicitud de crédito.

    Son los campos que el modelo espera como entrada (`columnas_entrada_
    esperadas` de `models/modelo_final_metadata.json`) menos las 12 features
    derivadas (`crear_features` las calcula) y menos `puntaje`, excluida por
    fuga en el Avance 2 — si esa columna no existía al momento de otorgar el
    crédito, pedírsela al cliente acá sería incoherente con esa decisión.

    Mezcla datos que declara el solicitante (capital, plazo, salario, tipo
    laboral) con datos que se consultan al buró DataCrédito en el momento de
    la solicitud (`puntaje_datacredito`, `saldo_*`, `huella_consulta`,
    `tendencia_ingresos`, etc.) — ambos disponibles antes de decidir, que es
    el único criterio que importa acá.
    """

    tipo_credito: int = Field(..., ge=0, description="Código del tipo de crédito solicitado")
    capital_prestado: float = Field(..., ge=0, description="Monto de capital solicitado, en COP")
    plazo_meses: int = Field(..., gt=0, description="Plazo del crédito, en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del cliente, en años")
    tipo_laboral: TipoLaboral = Field(..., description="Situación laboral del cliente")
    salario_cliente: float = Field(..., ge=0, description="Ingreso mensual declarado del cliente, en COP")
    total_otros_prestamos: float = Field(..., ge=0, description="Saldo de otras deudas del cliente, en COP")
    cuota_pactada: float = Field(..., ge=0, description="Cuota mensual pactada para este crédito, en COP")
    # Estos 7 campos son Optional a propósito: en el dataset de entrenamiento
    # tienen nulos reales (puntaje_datacredito 0.06%, saldo_mora/saldo_total
    # 1.45%, saldo_principal 3.76%, saldo_mora_codeudor 5.48%,
    # promedio_ingresos_datacredito y tendencia_ingresos ~27.2% cada una,
    # correlacionadas entre sí — ver comprension_eda.ipynb) que representan
    # consultas al buró sin resultado, no errores de carga. El
    # SimpleImputer del preprocesador del modelo está diseñado justamente
    # para imputarlos — forzarlos como obligatorios acá rechazaría de
    # entrada una porción real de las solicitudes.
    puntaje_datacredito: Optional[float] = Field(None, ge=0, description="Score del buró DataCrédito (None si no hay dato)")
    cant_creditosvigentes: int = Field(..., ge=0, description="Créditos vigentes del cliente en el sistema")
    huella_consulta: int = Field(..., ge=0, description="Consultas de crédito recientes sobre el cliente")
    saldo_mora: Optional[float] = Field(None, ge=0, description="Saldo en mora del cliente en otros créditos, en COP (None si no hay dato)")
    saldo_total: Optional[float] = Field(None, ge=0, description="Saldo total de deuda del cliente, en COP (None si no hay dato)")
    saldo_principal: Optional[float] = Field(None, ge=0, description="Saldo de capital de la deuda del cliente, en COP (None si no hay dato)")
    saldo_mora_codeudor: Optional[float] = Field(None, ge=0, description="Saldo en mora del codeudor, en COP (None si no hay dato)")
    creditos_sectorFinanciero: int = Field(..., ge=0, description="Créditos vigentes en el sector financiero")
    creditos_sectorCooperativo: int = Field(..., ge=0, description="Créditos vigentes en el sector cooperativo")
    creditos_sectorReal: int = Field(..., ge=0, description="Créditos vigentes en el sector real")
    promedio_ingresos_datacredito: Optional[float] = Field(None, ge=0, description="Ingreso promedio reportado por DataCrédito, en COP (None si no hay dato)")
    tendencia_ingresos: Optional[TendenciaIngresos] = Field(None, description="Tendencia de ingresos del cliente según DataCrédito (None si no hay dato)")
    fecha_prestamo: datetime = Field(
        default_factory=datetime.now,
        description="Fecha de la solicitud. Si no se envía, se usa la fecha de hoy (las features temporales la necesitan).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tipo_credito": 7,
                "capital_prestado": 3692160.0,
                "plazo_meses": 10,
                "edad_cliente": 42,
                "tipo_laboral": "Independiente",
                "salario_cliente": 8000000,
                "total_otros_prestamos": 2500000,
                "cuota_pactada": 341296,
                "puntaje_datacredito": 695.0,
                "cant_creditosvigentes": 10,
                "huella_consulta": 5,
                "saldo_mora": 0.0,
                "saldo_total": 51258.0,
                "saldo_principal": 51258.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 5,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 908526.0,
                "tendencia_ingresos": "Estable",
                "fecha_prestamo": "2024-12-21T11:31:35",
            }
        }
    }


class RespuestaPrediccion(BaseModel):
    """Resultado de evaluar una solicitud de crédito."""

    probabilidad: float = Field(..., description="Probabilidad predicha de que Pago_atiempo=1 (pagó a tiempo)")
    prediccion: int = Field(..., description="Clase predicha según el umbral usado: 1 = pagó a tiempo, 0 = no")
    umbral_usado: float = Field(..., description="Umbral de decisión aplicado para esta predicción")
    etiqueta_riesgo: Literal["bajo", "medio", "alto"] = Field(..., description="Etiqueta de riesgo legible")


class ResultadoRegistro(BaseModel):
    """Resultado de un registro dentro de una predicción por lotes."""

    indice: int = Field(..., description="Posición del registro dentro del lote enviado (empieza en 0)")
    exitoso: bool = Field(..., description="False si este registro particular falló al procesarse")
    probabilidad: Optional[float] = None
    prediccion: Optional[int] = None
    etiqueta_riesgo: Optional[str] = None
    error: Optional[str] = Field(None, description="Motivo del fallo, solo presente si exitoso=False")


class RespuestaLote(BaseModel):
    """Resultado de procesar un lote de solicitudes."""

    resultados: list[ResultadoRegistro]
    cantidad_procesada: int = Field(..., description="Registros procesados exitosamente")
    cantidad_con_error: int = Field(..., description="Registros que fallaron y no se pudieron predecir")
    tiempo_proceso_segundos: float = Field(..., description="Tiempo total de procesamiento del lote")


def _clasificar_riesgo(probabilidad):
    """Etiqueta de riesgo legible a partir de la probabilidad de pago a tiempo.

    Bandas fijas, independientes del umbral de decisión (el umbral decide
    la clase 0/1; esta etiqueta es una lectura más granular para humanos).
    """
    if probabilidad >= 0.8:
        return "bajo"
    if probabilidad >= 0.5:
        return "medio"
    return "alto"


def predecir(registros, modelo, umbral=0.5):
    """Predice sobre una lista de diccionarios con campos crudos.

    Orden fijo: valida (ya se asume que `registros` viene de instancias de
    `SolicitudCredito`, o de datos igualmente válidos), crea las features
    con `crear_features` (importada de `src.ft_engineering`, nunca
    reimplementada acá) y recién ahí predice — nunca se predice sobre el
    diccionario crudo.

    El umbral es un parámetro, no un `0.5` fijo en el medio del código: en
    un despliegue real ese umbral se calibra según el costo relativo de
    aprobar un mal crédito (falso negativo — el error caro, ver README)
    contra rechazar uno bueno (falso positivo, más barato), no se deja en
    el valor por defecto de por vida.

    Si un registro individual falla (por ejemplo, si le faltó una columna
    que el modelo necesita), no tira abajo el lote entero: ese índice
    queda en `errores` con su motivo, y el resto se sigue procesando.

    Parameters
    ----------
    registros : list[dict]
        Cada diccionario con los campos crudos de `SolicitudCredito`
        (incluyendo `fecha_prestamo`).
    modelo : sklearn.pipeline.Pipeline
        El pipeline completo (preprocesador + estimador), ya cargado.
    umbral : float, optional
        Umbral de decisión sobre la probabilidad de `Pago_atiempo=1`.
        Por defecto 0.5.

    Returns
    -------
    tuple[list[dict], list[dict]]
        `(resultados, errores)`. `resultados`: uno por registro exitoso,
        con `indice`, `probabilidad`, `prediccion`, `etiqueta_riesgo`.
        `errores`: uno por registro fallido, con `indice` y `error`.
    """
    columnas_esperadas = _cargar_metadata()["columnas_entrada_esperadas"]

    resultados = []
    errores = []

    for indice, registro in enumerate(registros):
        try:
            df_fila = pd.DataFrame([registro])
            df_features = crear_features(df_fila)

            faltantes = [c for c in columnas_esperadas if c not in df_features.columns]
            if faltantes:
                raise ValueError(f"faltan columnas requeridas por el modelo: {faltantes}")

            X = df_features[columnas_esperadas]
            probabilidad = float(modelo.predict_proba(X)[0, 1])
            prediccion = int(probabilidad >= umbral)

            resultados.append({
                "indice": indice,
                "probabilidad": probabilidad,
                "prediccion": prediccion,
                "etiqueta_riesgo": _clasificar_riesgo(probabilidad),
            })
        except Exception as exc:
            errores.append({"indice": indice, "error": str(exc)})

    return resultados, errores


class RespuestaHealth(BaseModel):
    """Estado de salud del servicio, para el HEALTHCHECK de Docker."""

    estado: str = Field(..., description="'ok' si el modelo está cargado, 'sin_modelo' si no")
    modelo_cargado: bool
    algoritmo: Optional[str] = None
    version_api: str


class RespuestaModelInfo(BaseModel):
    """Metadata completa del modelo entrenado."""

    nombre_algoritmo: str
    hiperparametros: dict
    fecha_entrenamiento: str
    metricas_test: dict
    columnas_entrada_esperadas: list[str]
    columnas_excluidas_por_fuga: list[str]


# ----------------------------------------------------------------------
# Aplicación FastAPI
# ----------------------------------------------------------------------
estado_app = {"modelo": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y su metadata UNA sola vez al arrancar (no por request).

    Si falta el .pkl o el metadata, la aplicación no llega a levantar y el
    error dice exactamente qué archivo falta — mejor fallar rápido acá que
    fallar silenciosamente en el primer request.
    """
    if not RUTA_MODELO.exists():
        raise RuntimeError(
            f"No se encontró el modelo en '{RUTA_MODELO}'. Verificar que "
            "models/modelo_final.pkl esté presente antes de levantar la API."
        )
    if not RUTA_METADATA.exists():
        raise RuntimeError(
            f"No se encontró el metadata en '{RUTA_METADATA}'. Verificar que "
            "models/modelo_final_metadata.json esté presente antes de levantar la API."
        )

    logger.info("Cargando modelo desde %s ...", RUTA_MODELO)
    estado_app["modelo"] = joblib.load(RUTA_MODELO)
    estado_app["metadata"] = _cargar_metadata()
    logger.info("Modelo cargado: %s", estado_app["metadata"]["nombre_algoritmo"])

    yield

    estado_app.clear()


app = FastAPI(
    title="API de Riesgo Crediticio",
    description=(
        "Expone el modelo de clasificación de riesgo crediticio (entrenado en los "
        "Avances 2 y 3 del proyecto) como servicio HTTP. Ver /docs para probar cada "
        "endpoint interactivamente."
    ),
    version=VERSION_API,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def medir_tiempo_request(request: Request, call_next):
    """Mide y registra en el log el tiempo de cada request."""
    inicio = time.perf_counter()
    response = await call_next(request)
    duracion = time.perf_counter() - inicio
    logger.info("%s %s -> %s (%.4fs)", request.method, request.url.path, response.status_code, duracion)
    response.headers["X-Tiempo-Proceso-Segundos"] = f"{duracion:.4f}"
    return response


@app.exception_handler(Exception)
async def manejador_excepciones_global(request: Request, exc: Exception):
    """Registra el traceback completo en el log del servidor, pero al cliente
    solo le devuelve un mensaje claro y un identificador para poder buscar
    el error en los logs — nunca el traceback."""
    id_error = str(uuid.uuid4())
    logger.error("Error id=%s en %s %s:\n%s", id_error, request.method, request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detalle": "Ocurrió un error interno inesperado.", "id_error": id_error},
    )


@app.get(
    "/health",
    response_model=RespuestaHealth,
    tags=["Monitoreo"],
    summary="Estado de salud del servicio",
    description="Indica si el modelo está cargado y qué algoritmo es. Es el endpoint que usa el HEALTHCHECK de Docker para saber si el contenedor está sano.",
)
def health():
    modelo = estado_app.get("modelo")
    metadata = estado_app.get("metadata")
    return RespuestaHealth(
        estado="ok" if modelo is not None else "sin_modelo",
        modelo_cargado=modelo is not None,
        algoritmo=metadata["nombre_algoritmo"] if metadata else None,
        version_api=VERSION_API,
    )


@app.get(
    "/model/info",
    response_model=RespuestaModelInfo,
    tags=["Modelo"],
    summary="Metadata del modelo entrenado",
    description="Devuelve algoritmo, hiperparámetros, fecha de entrenamiento, métricas de test, columnas de entrada esperadas y columnas excluidas por fuga.",
)
def model_info():
    metadata = estado_app.get("metadata")
    if metadata is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")
    return RespuestaModelInfo(**{campo: metadata[campo] for campo in RespuestaModelInfo.model_fields})


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    tags=["Predicción"],
    summary="Predice sobre una solicitud de crédito",
    description="Recibe una solicitud de crédito y devuelve la probabilidad de pago a tiempo, la clase predicha según el umbral, y una etiqueta de riesgo legible.",
)
def predict(
    solicitud: SolicitudCredito,
    umbral: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de decisión sobre la probabilidad de pago a tiempo"),
):
    modelo = estado_app.get("modelo")
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    resultados, errores = predecir([solicitud.model_dump()], modelo, umbral=umbral)
    if errores:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la solicitud: {errores[0]['error']}")

    resultado = resultados[0]
    return RespuestaPrediccion(
        probabilidad=resultado["probabilidad"],
        prediccion=resultado["prediccion"],
        umbral_usado=umbral,
        etiqueta_riesgo=resultado["etiqueta_riesgo"],
    )


@app.post(
    "/predict/batch",
    response_model=RespuestaLote,
    tags=["Predicción"],
    summary="Predicción por lotes",
    description=(
        f"Recibe una lista de hasta {LIMITE_LOTE} solicitudes (cada una con los mismos campos que "
        "SolicitudCredito, ver /predict) y devuelve una predicción por cada una, en el mismo orden. "
        "Si un registro es inválido (le falta un campo, tiene un valor fuera de rango, etc.) o falla "
        "al predecirse, no tira abajo el lote entero: ese índice queda marcado con su motivo y el "
        "resto se procesa igual."
    ),
)
def predict_batch(
    solicitudes: list[dict],
    umbral: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de decisión sobre la probabilidad de pago a tiempo"),
):
    # A propósito NO se tipa el parámetro como list[SolicitudCredito]: si se
    # hiciera, FastAPI validaría la lista entera antes de que este código
    # corra, y un solo registro inválido tiraría un 422 para todo el lote.
    # Acá se valida registro por registro, a mano, para poder aislar los que
    # fallan sin perder los que sí son válidos.
    if len(solicitudes) > LIMITE_LOTE:
        raise HTTPException(
            status_code=413,
            detail=f"El lote tiene {len(solicitudes)} registros; el máximo permitido es {LIMITE_LOTE}.",
        )

    modelo = estado_app.get("modelo")
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    inicio = time.perf_counter()

    items = []
    registros_validos = []  # [(indice_original, dict_validado), ...]
    for indice, dato_crudo in enumerate(solicitudes):
        try:
            solicitud = SolicitudCredito(**dato_crudo)
            registros_validos.append((indice, solicitud.model_dump()))
        except PydanticValidationError as exc:
            items.append(ResultadoRegistro(indice=indice, exitoso=False, error=str(exc)))

    if registros_validos:
        registros = [dato for _, dato in registros_validos]
        resultados, errores_prediccion = predecir(registros, modelo, umbral=umbral)
        for r in resultados:
            indice_original = registros_validos[r["indice"]][0]
            items.append(ResultadoRegistro(
                indice=indice_original, exitoso=True, probabilidad=r["probabilidad"],
                prediccion=r["prediccion"], etiqueta_riesgo=r["etiqueta_riesgo"],
            ))
        for e in errores_prediccion:
            indice_original = registros_validos[e["indice"]][0]
            items.append(ResultadoRegistro(indice=indice_original, exitoso=False, error=e["error"]))

    items.sort(key=lambda item: item.indice)
    duracion = time.perf_counter() - inicio

    return RespuestaLote(
        resultados=items,
        cantidad_procesada=sum(1 for item in items if item.exitoso),
        cantidad_con_error=sum(1 for item in items if not item.exitoso),
        tiempo_proceso_segundos=duracion,
    )


@app.post(
    "/predict/csv",
    tags=["Predicción"],
    summary="Predicción por lotes desde un CSV",
    description="Recibe un CSV con las columnas de SolicitudCredito, valida que estén todas presentes, predice fila por fila y devuelve el mismo CSV con dos columnas agregadas (probabilidad, prediccion), como descarga.",
)
async def predict_csv(
    archivo: UploadFile,
    umbral: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de decisión sobre la probabilidad de pago a tiempo"),
):
    modelo = estado_app.get("modelo")
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el CSV: {exc}")

    campos_requeridos = [campo for campo in SolicitudCredito.model_fields if campo != "fecha_prestamo"]
    faltantes = [campo for campo in campos_requeridos if campo not in df.columns]
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan columnas en el CSV: {faltantes}")

    if "fecha_prestamo" not in df.columns:
        df["fecha_prestamo"] = datetime.now().isoformat()

    registros = df.to_dict(orient="records")
    resultados, errores = predecir(registros, modelo, umbral=umbral)

    df["probabilidad"] = pd.NA
    df["prediccion"] = pd.NA
    for r in resultados:
        df.loc[r["indice"], "probabilidad"] = r["probabilidad"]
        df.loc[r["indice"], "prediccion"] = r["prediccion"]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicciones.csv"},
    )
