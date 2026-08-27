## Proyecto
Modelo de clasificación de riesgo crediticio. El dataset es `Base_de_datos.xlsx`: 23 columnas, cada fila es un crédito otorgado. La variable objetivo es `Pago_atiempo` (1 = pagó a tiempo, 0 = no). Los montos están en pesos colombianos (COP) y `puntaje_datacredito` es el score del buró de crédito colombiano DataCrédito.

## Convenciones
- Todo en español: código, comentarios, celdas markdown y mensajes de commit.
- Los notebooks llevan una celda markdown antes de cada bloque de código explicando el porqué, no solo el qué.
- Nunca borres celdas ni análisis previos sin avisarme.
- No afirmes ningún número que no salga de una celda efectivamente ejecutada.
- Rutas siempre relativas a la raíz del proyecto con pathlib, nunca absolutas.

## Versionado
- `main`: solo recibe merges por Pull Request aprobado.
- `developer`: rama de trabajo.
- v1.0.0 = estructura de carpetas idéntica en todas las ramas.
- v1.0.1 = `cargar_datos.ipynb` y `comprension_eda.ipynb` completos y mergeados a main.

## Reglas de trabajo
- Antes de crear un archivo, verificá si ya existe.
- No corras `git commit`, `git push` ni `git merge` salvo que te lo pida explícitamente.
- Una vez creada la estructura de carpetas, no la modifiques.

## Avance 2 - modelamiento
- El módulo src/ft_engineering.py es la única fuente de features. Los notebooks lo importan, no reimplementan la lógica.
- Toda transformación va dentro de un Pipeline o ColumnTransformer de sklearn. Nada de transformar el DataFrame completo antes del split: eso filtra información del test al train.
- El split es estratificado por Pago_atiempo, con random_state=42 en todo lo que acepte semilla.
- Ninguna métrica se reporta sin decir sobre qué conjunto se calculó.
- Antes de agregar una feature nueva, escribí en una celda markdown qué hipótesis de negocio la justifica.

## Avance 3 - monitoreo
- La ventana de referencia se define una sola vez y no cambia entre períodos.
- Los bins de una métrica se calculan sobre la referencia y se aplican tal cual a la ventana actual. Nunca se recalculan por período: si no, comparás dos escalas distintas.
- Toda métrica de drift se reporta junto a la cantidad de observaciones del período. Con muestras chicas, cualquier métrica es ruido.
- Los umbrales de alerta viven en un solo diccionario de configuración, no repartidos por el código.
- La app de Streamlit no calcula drift: solo lee lo que model_monitoring.py dejó escrito.
