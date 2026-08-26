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
