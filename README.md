# Proyecto Final - Riesgo Crediticio

Modelo de clasificación de riesgo crediticio, con su ingeniería de características,
comparación de modelos, monitoreo de data drift y una app de Streamlit para
visualizar ese monitoreo.

## 1. Caso de negocio

El modelo predice, para un crédito que se está por otorgar, si el cliente **va a
pagar a tiempo o no** (`Pago_atiempo`). Lo usaría un analista o un área de riesgo
crediticio, en el momento de evaluar una solicitud: la predicción habilita la
decisión de **otorgar el crédito, rechazarlo, o pedir una revisión manual** antes
de desembolsar el dinero.

El costo de equivocarse no es simétrico:

- **Falso negativo** (el modelo predice que va a pagar y en realidad no paga):
  el error caro. Es plata real que sale y no vuelve — capital prestado que se
  convierte en cartera incobrable.
- **Falso positivo** (el modelo predice que no va a pagar y en realidad sí
  pagaba): el error barato. Es una oportunidad de negocio perdida (un cliente
  bueno al que no se le prestó), no una pérdida de capital.

Esta asimetría es la razón por la que, en la etapa de modelamiento, no se eligió
el modelo solo por accuracy: con el desbalance de clases de este dataset (ver
sección 4), el accuracy no distingue estos dos tipos de error.

## 2. Los datos

- **Origen:** `data/raw/Base_de_datos.xlsx` — dataset de ejemplo no productivo
  (no son datos reales de una entidad financiera), entregado para la cursada.
- **Volumen:** 10.763 créditos otorgados, 23 columnas.
- **Período cubierto:** `fecha_prestamo` va del 2024-11-26 al 2026-04-26.
- **Variable objetivo:** `Pago_atiempo` (`1` = pagó a tiempo, `0` = no pagó a
  tiempo).
- **Balance de clases:** 95.25% clase `1`, 4.75% clase `0` — fuertemente
  desbalanceado.
- Los montos están en pesos colombianos (COP); `puntaje_datacredito` es el score
  del buró de crédito colombiano DataCrédito.

## 3. Proceso

### 3.1 Análisis exploratorio (`notebooks/cargar_datos.ipynb`, `notebooks/comprension_eda.ipynb`)

Hallazgos principales:

- Fuerte desbalance de clases (95.25% / 4.75%), que condiciona toda la elección
  de métrica más adelante.
- `puntaje` mostró una correlación punto-biserial de ≈0.92 con el objetivo y un
  valor de relleno en 87.40% de las filas — primera señal de sospecha de fuga,
  confirmada después en `ft_engineering.ipynb`.
- Valores imposibles en varias columnas: `edad_cliente` hasta 123 años,
  `salario_cliente` hasta 22.000 millones de COP, `puntaje_datacredito` con un
  mínimo negativo.
- `saldo_mora` y `saldo_mora_codeudor` casi constantes en 0 (99.48% y 99.97%
  respectivamente) y duplicadas entre sí (99.69% de coincidencia) — candidatas a
  redundancia.
- Numéricas de monto fuertemente asimétricas a la derecha; `saldo_total` y
  `saldo_principal` con correlación de Pearson 0.735 pero de Spearman 0.946 (la
  diferencia la explican los mismos outliers extremos).

### 3.2 Ingeniería de características (`notebooks/ft_engineering.ipynb`, `src/ft_engineering.py`)

**Columnas excluidas por fuga:** `puntaje` — con evidencia definitiva, no solo
sospecha. Su AUC univariado contra `Pago_atiempo` da **1.0000**: existe un umbral
exacto (entre 62.67 y 63.81) que separa las dos clases con **100% de exactitud**.
Ninguna variable de negocio real se comporta así. `saldo_mora`, `saldo_total`,
`saldo_principal` y `saldo_mora_codeudor` quedaron marcadas como **dudosas mas no
excluidas**: su AUC individual es bajo (0.51-0.52), pero un "saldo" describe el
estado de una deuda — si es de otros créditos previos es información legítima, si
es de este crédito post-desembolso sería fuga temporal, y esa duda no se puede
resolver solo con estadística.

**Features derivadas** (`crear_features`, 12 en total, cada una con su hipótesis
de negocio documentada en el código):

| Feature | Fórmula | Hipótesis de negocio |
|---|---|---|
| `ratio_cuota_salario` | cuota_pactada / salario_cliente | Capacidad de pago |
| `ratio_endeudamiento` | total_otros_prestamos / salario_cliente | Sobreendeudamiento previo |
| `ratio_capital_salario` | capital_prestado / salario_cliente | Tamaño del crédito relativo al ingreso |
| `carga_total` | (cuota_pactada + total_otros_prestamos) / salario_cliente | Presión financiera total |
| `total_creditos_sectores` | suma de créditos por sector | Exposición crediticia total |
| `diferencia_ingresos` | salario_cliente - promedio_ingresos_datacredito | Confiabilidad del ingreso declarado |
| `ratio_ingresos_declarados` | salario_cliente / promedio_ingresos_datacredito | Igual, en términos relativos |
| `sin_historial_ingresos` | 1 si promedio_ingresos_datacredito es nulo | Falta de huella en el buró |
| `mes_prestamo`, `trimestre_prestamo`, `anio_prestamo` | de fecha_prestamo | Estacionalidad y efecto cosecha |
| `dia_semana_prestamo` | de fecha_prestamo | Perfil según día de originación |

Toda transformación (imputación, escalado, encoding) vive en el
`ColumnTransformer` de `construir_preprocesador`, ajustado **solo sobre train**
dentro del `Pipeline` de cada modelo — nunca sobre el dataset completo antes del
split.

### 3.3 Modelamiento (`notebooks/modelamiento.ipynb`)

9 variantes entrenadas (6 algoritmos + 3 con `class_weight="balanced"`),
comparadas por ROC-AUC de test (la métrica elegida porque accuracy/precision/
recall/f1 se inflan solas con el desbalance de clases):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC (test) | CV AUC (media ± std) |
|---|---|---|---|---|---|---|
| **GradientBoosting** | 0.9536 | 0.9544 | 0.9990 | 0.9762 | **0.6890** | 0.7068 ± 0.0208 |
| XGBoost | 0.9526 | 0.9552 | 0.9971 | 0.9757 | 0.6733 | 0.6561 ± 0.0190 |
| LogisticRegression | 0.9531 | 0.9531 | 1.0000 | 0.9760 | 0.6652 | 0.6692 ± 0.0298 |
| LogisticRegression (balanceado) | 0.6484 | 0.9661 | 0.6538 | 0.7799 | 0.6637 | — |
| RandomForest | 0.9526 | 0.9526 | 1.0000 | 0.9757 | 0.6594 | — |
| RandomForest (balanceado) | 0.9545 | 0.9548 | 0.9995 | 0.9767 | 0.6561 | — |
| DecisionTree (balanceado) | 0.6289 | 0.9637 | 0.6343 | 0.7651 | 0.5981 | — |
| DecisionTree | 0.9526 | 0.9539 | 0.9985 | 0.9757 | 0.5852 | — |
| KNN | 0.9522 | 0.9526 | 0.9995 | 0.9755 | 0.5575 | — |

**Modelo elegido: `GradientBoostingClassifier`.** Gana en test por poco (0.0157
sobre XGBoost, una diferencia menor al desvío de CV de ambos — no concluyente
sola), pero gana claro en cross-validation (0.7068 vs. 0.6561, una diferencia
mayor al desvío de los dos). Con el umbral por defecto (0.5), acierta solo 4 de
102 créditos que realmente no se pagaron en test — comete casi exclusivamente el
error caro (ver sección 1). El umbral operativo no se optimizó en este avance
(ver sección 7).

### 3.4 Monitoreo (`notebooks/monitoreo.ipynb`, `src/model_monitoring.py`)

- **Referencia:** créditos de `2024-11` a `2025-01` (3.319 registros) — la
  ventana más antigua con volumen suficiente para ser una base estable. Se fija
  una sola vez y no cambia entre períodos.
- **Períodos de monitoreo:** 12, uno por mes, de `2025-02` a `2026-01`. Se
  descartan los períodos con menos de 100 registros (`2026-02`, `2026-03`,
  `2026-04`).
- **Métricas:** KS y PSI y Jensen-Shannon para variables numéricas, chi-cuadrado
  para categóricas — todas con los bins/categorías fijados **sobre la
  referencia** y aplicados tal cual a cada período. También se monitorea la
  distribución de `probabilidad_predicha` (la salida del modelo) y la tasa de
  aprobación agregada.
- **Umbrales** (un único diccionario `UMBRALES` en `src/model_monitoring.py`,
  ajustables desde la app): PSI < 0.10 estable, 0.10-0.25 moderado, > 0.25
  significativo; Jensen-Shannon < 0.10 / 0.10-0.20 / > 0.20; KS y chi-cuadrado
  por p-valor < 0.05.
- La app de Streamlit (`app/app.py`) **no calcula nada de esto**: solo lee
  `data/processed/historico_drift.csv` y `data/processed/tabla_scoring.parquet`.

## 4. Principales hallazgos

- El desbalance de clases (95.25% clase `1` / 4.75% clase `0`) hace que el
  accuracy no sirva como métrica: se usó ROC-AUC para comparar modelos.
- `puntaje` tiene fuga de información confirmada: AUC univariado = **1.0000**,
  con un umbral exacto que separa las clases con 100% de exactitud — se excluyó
  del modelo.
- El mejor modelo (`GradientBoostingClassifier`) alcanza un ROC-AUC de test de
  **0.6890** (CV: 0.7068 ± 0.0208) — un desempeño modesto, sin ninguna variable
  que domine tras sacar la fuga.
- `saldo_mora` y `saldo_mora_codeudor` coinciden en el **99.69%** de las filas
  comparables — candidatas a redundancia, pendientes de una decisión de negocio.
- En el monitoreo, `promedio_ingresos_datacredito` llega a un **PSI de 1.087**
  en el último período (2026-01) contra la referencia — el drift más marcado
  detectado, muy por encima del umbral de "significativo" (0.25).

## 5. Estructura del repositorio

```
proyecto-final-riesgo-crediticio/
├── app/
│   └── app.py                  # App de Streamlit (monitoreo + simulador)
├── data/
│   ├── raw/                    # Base_de_datos.xlsx (crudo) y su CSV materializado
│   └── processed/               # tabla_scoring.parquet, historico_drift.csv
├── models/
│   ├── modelo_final.pkl         # Pipeline completo (preprocesador + modelo)
│   └── modelo_final_metadata.json
├── notebooks/
│   ├── cargar_datos.ipynb       # Ingesta: Excel -> CSV materializado
│   ├── comprension_eda.ipynb    # EDA completo (univariable/bivariable/multivariable)
│   ├── ft_engineering.ipynb     # Descarte de fuga + features + preprocesador
│   ├── modelamiento.ipynb       # Batería de modelos, evaluación, selección
│   └── monitoreo.ipynb          # Partición temporal + prueba del job de monitoreo
├── reports/
│   ├── figuras/                 # Gráficos comparativos de modelos (PNG)
│   └── tabla_resumen_modelos.csv
├── src/
│   ├── config.json
│   ├── ft_engineering.py        # crear_features, construir_preprocesador, preparar_datos
│   └── model_monitoring.py      # scoring, métricas de drift, recomendaciones, simulador
├── requirements.txt
├── set_up.bat                   # Setup del entorno en Windows
├── set_up.sh                    # Setup del entorno en macOS/Linux
└── CLAUDE.md                    # Convenciones del proyecto
```

## 6. Cómo reproducirlo

### macOS / Linux

```bash
git clone git@github.com:Juanmarossi/proyecto-final-riesgo-crediticio.git
cd proyecto-final-riesgo-crediticio
./set_up.sh
source pf-riesgo-venv/bin/activate
python -m src.model_monitoring
streamlit run app/app.py
```

> **Nota macOS sin Homebrew (XGBoost necesita `libomp`):** si `python -m
> src.model_monitoring` o la app fallan al importar `xgboost` con un error de
> `libomp.dylib`, lo normal es resolverlo con `brew install libomp`. Si no
> tenés Homebrew instalado, una alternativa es apuntar a la `libomp.dylib` que
> ya trae `scikit-learn` dentro del propio venv:
> ```bash
> export DYLD_LIBRARY_PATH="$(pwd)/pf-riesgo-venv/lib/python3.14/site-packages/sklearn/.dylibs"
> ```
> (ajustar `python3.14` a la versión real del intérprete del venv) antes de
> correr los comandos de arriba.

### Windows

```cmd
git clone git@github.com:Juanmarossi/proyecto-final-riesgo-crediticio.git
cd proyecto-final-riesgo-crediticio
set_up.bat
pf-riesgo-venv\Scripts\activate
python -m src.model_monitoring
streamlit run app\app.py
```

Los dos scripts de setup (`set_up.sh` / `set_up.bat`) leen `project_code` de
`src/config.json`, crean el entorno virtual `pf-riesgo-venv`, instalan
`requirements.txt` y registran el kernel de Jupyter. `python -m
src.model_monitoring` regenera `data/processed/tabla_scoring.parquet` y
`data/processed/historico_drift.csv` (ya están commiteados, así que este paso es
opcional salvo que se quiera recalcular todo). `streamlit run app/app.py` levanta
la app en `http://localhost:8501`.

## 7. Limitaciones y próximos pasos

- **No es un modelo de producción real:** dataset de ejemplo no productivo, sin
  validación con datos de una entidad financiera real.
- **ROC-AUC modesto (0.689)** una vez removida la fuga — no se hizo búsqueda de
  hiperparámetros en este avance; hay margen de mejora ahí.
- **`saldo_mora`/`saldo_total`/`saldo_principal`/`saldo_mora_codeudor` sin
  resolver:** falta confirmar con negocio en qué momento se calculan, antes de
  decidir si son fuga o no.
- **Umbral de decisión sin optimizar:** se usó 0.5 por defecto; con el costo de
  error tan asimétrico de este problema (sección 1), el umbral debería fijarse
  en base a ese costo, no dejarse en el valor por defecto.
- **El monitoreo no mide precisión real:** detecta cambios en la población de
  entrada (data drift), pero no puede saber si el modelo sigue siendo preciso
  porque el resultado real de los créditos recientes todavía no se conoce (la
  cartera no maduró).
- **Sin infraestructura de producción:** no hay CI/CD, tests automatizados, API
  de scoring, autenticación en la app, ni un job programado que corra
  `python -m src.model_monitoring` periódicamente — todo se ejecuta a mano hoy.

Próximos pasos razonables: tuning de hiperparámetros del modelo elegido, definir
el umbral de decisión con el área de riesgo, resolver la duda de `saldo_*`,
agregar tests automatizados y un pipeline de CI, y programar el job de monitoreo
para que corra solo (por ejemplo, mensualmente) en vez de correrlo a mano.
