# Instrumentación con MLflow — Fase 3 (Tracking y Model Registry)

**Proyecto MLOps — Detección de Fraude con Tarjetas de Crédito**
**Fase 3 (Tracking / Registry) · Responsable: Integrante 3**

Este documento describe la instrumentación del flujo de la Fase 2 con MLflow,
implementada en la sección "Fase 3" del notebook
[`deteccion_fraude_mlops.ipynb`](../deteccion_fraude_mlops.ipynb).

## 1. Configuración del seguimiento

Se emplea MLflow con un backend local en **SQLite** (archivo `mlflow.db`), que es
el requisito para poder utilizar el Model Registry. El experimento se nombra
`deteccion_fraude_v1`, siguiendo una convención versionable.

```python
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('deteccion_fraude_v1')
```

## 2. Qué se registra y por qué

Cada uno de los tres modelos de la Fase 2 se registra como un *run*. Se registran
únicamente los elementos relevantes para la comparación, no todo lo posible:

- **Parámetros:** el nombre del modelo, el tratamiento del desbalance
  (`class_weight`) y, cuando aplica, el número de árboles (`n_estimators`).
- **Métricas:** F1 y AUPRC de validación cruzada, y F1, precisión, recall y
  AUPRC sobre el conjunto de prueba. Son las métricas decididas en la Fase 0.
- **Artefactos:** el modelo entrenado y, para el candidato final, la matriz de
  confusión y la curva precisión-recall.

## 3. Criterio de promoción

El criterio se define por escrito y se aplica por código: un modelo se promueve
a producción únicamente si, sobre el conjunto de prueba y en su umbral de
operación, cumple **recall > 0.75 y precisión > 0.85**.

El Random Forest, evaluado en su umbral de operación (0.25), obtiene precisión
0.910 y recall 0.771, por lo que **cumple el criterio**.

## 4. Ciclo de vida en el Model Registry

A partir de MLflow 2.9 los *stages* del Model Registry (`Staging`, `Production`)
quedaron deprecados y se sustituyen por **aliases** y **tags**. En este trabajo se
reproduce el ciclo de vida completo mediante dos aliases sobre el modelo
registrado `deteccion_fraude_rf`:

- **`staging`** — entorno de validación (dev). Toda versión candidata entra aquí
  con el tag `validation_status=testing`.
- **`champion`** — entorno de producción (prod). Una versión se promueve a este
  alias solo si cumple el criterio, y se marca con `validation_status=approved`.

```python
# Entra a validación
client.set_registered_model_alias('deteccion_fraude_rf', 'staging', version)
# Si cumple el criterio, se promueve a producción
client.set_registered_model_alias('deteccion_fraude_rf', 'champion', version)
```

Este esquema es más flexible que los stages: una misma versión puede tener varios
aliases y permite escenarios como pruebas A/B o despliegues graduales.

## 5. Interfaz de MLflow

La interfaz web se levanta apuntando al mismo backend SQLite. Desde la carpeta del
proyecto, con el entorno virtual activado:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Se abre en `http://localhost:5000`. Permite comparar los *runs* de los tres
modelos, revisar métricas y artefactos, y consultar el modelo registrado con sus
aliases en la sección *Models*.

## 6. Entrega a la Fase 4

El modelo a desplegar es la versión con alias `champion`, que se carga de forma
estable con:

```python
mlflow.pyfunc.load_model('models:/deteccion_fraude_rf@champion')
```

Al usarse el alias en lugar de un número de versión fijo, la Fase 4 siempre
tomará la versión vigente en producción, incluso si en el futuro se promueve una
versión distinta.
