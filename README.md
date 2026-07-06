# Detección de Fraude con Tarjetas de Crédito — Proyecto MLOps

Proyecto de la **Semana 14 – Software Inteligente (UNMSM, Ciclo 9)**.
Caso de estudio: detección de fraude con tarjetas de crédito, aplicando el
ciclo de vida de MLOps con **MLflow**.

Este README documenta las **Decisiones de Diseño Iniciales (Fase 0 – Kickoff)**:
qué decidió el equipo *antes* de escribir una sola línea de modelado, y **por
qué**. La idea es demostrar criterio, no copiar un tutorial.

---

## 1. El dataset (verificado con código, no supuesto)

Antes de decidir nada, el equipo inspeccionó el archivo real `creditcard.csv`
(dataset público de Kaggle, transacciones de tarjetas europeas de 2013). Los
números que fundamentan todas las decisiones de abajo son:

| Característica | Valor |
|---|---|
| Total de transacciones | **284,807** |
| Transacciones legítimas (Clase 0) | 284,315 — **99.827 %** |
| Transacciones fraudulentas (Clase 1) | **492 — 0.173 %** |
| Desbalance | **1 fraude por cada ~577 transacciones legítimas** |
| Valores nulos | 0 |
| Filas duplicadas | **1,081** |
| Columnas | 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Monto (`Amount`) fraude | mediana **$9.25** |
| Monto (`Amount`) legítimas | mediana **$22.00** |

Las variables `V1`–`V28` son componentes ya anonimizados mediante PCA (por
confidencialidad de los datos bancarios); `Time` y `Amount` son las únicas
variables en su escala original.

> **Conclusión del equipo:** el problema es de **clasificación binaria
> altamente desbalanceada**. Todo lo demás se decide en función de ese hecho.

---

## 2. Decisiones de diseño y su justificación

### Decisión 1 — Métrica de éxito: F1 y AUPRC (NO accuracy)

**Qué decidimos:** las métricas principales serán **F1-score** y **AUPRC**
(Area Under the Precision-Recall Curve). Reportaremos además **precision** y
**recall** por separado.

**Por qué:**
- Con un 99.827 % de transacciones legítimas, un modelo que prediga *"todo es
  legítimo"* obtendría **99.83 % de accuracy** sin detectar un solo fraude. El
  accuracy es **engañoso** en datasets desbalanceados y no mide lo que nos
  importa.
- **AUPRC** en lugar de AUC-ROC: cuando la clase positiva es tan rara, la curva
  ROC luce optimista de forma artificial. La curva **Precision-Recall** refleja
  de manera honesta el desempeño sobre la clase minoritaria (fraude).
- **F1** equilibra los dos errores que sí cuestan dinero:
  - *Falsos negativos* (fraude no detectado) → pérdida directa.
  - *Falsos positivos* (transacción legítima bloqueada) → fricción y molestia
    al cliente.
- Reportamos precision y recall por separado porque el **criterio de promoción
  de modelos (Fase 3)** los evalúa individualmente.

### Decisión 2 — División de datos: 75/25 estratificado (con validación cruzada dentro del train)

**Qué decidimos:** dividir en **dos conjuntos físicos** — **75 % entrenamiento
/ 25 % prueba** — con **estratificación** por la columna `Class` y
`random_state` fijo. **No** apartamos un tercer conjunto de validación fijo; en
su lugar, la comparación entre modelos se hace con **validación cruzada
estratificada (StratifiedKFold) dentro del 75 % de train**.

```
100 % de los datos (sin duplicados)
│
├── 75 %  → TRAIN  (~213,000 filas)  → el modelo aprende aquí.
│                                       Comparación de modelos e hipótesis
│                                       mediante cross-validation interna.
│
└── 25 %  → TEST   (~71,000 filas)   → evaluación final, UNA sola vez.
                                        Intacto hasta el final.
```

**Por qué 75/25 estratificado:**
- Con solo **492 fraudes**, una división aleatoria simple podría dejar una
  cantidad insuficiente o desbalanceada en el conjunto de prueba. La
  **estratificación garantiza** que la proporción 0.173 % de fraude se mantenga
  en *ambos* conjuntos (~369 fraudes en train, ~123 en test).
- Fijar `random_state` asegura **reproducibilidad**, requisito central de MLOps:
  cualquier integrante debe poder reproducir exactamente el mismo split.

**Por qué validación cruzada y NO un tercer conjunto de validación fijo:**
- Comparar modelos (baseline vs Random Forest, etc.) directamente sobre el test
  lo **contaminaría**: terminaríamos eligiendo el modelo que mejor le va a *ese*
  test específico, perdiendo la imparcialidad de la evaluación final. Por eso la
  comparación necesita un mecanismo aparte.
- Apartar un tercer bloque fijo de validación nos dejaría con **aún menos
  fraudes** en cada conjunto. La **validación cruzada aprovecha los 369 fraudes
  del train rotándolos** por los *folds*, en lugar de sacrificar una porción.
- `StratifiedKFold` mantiene la proporción de fraude en cada *fold*, igual que
  en el split principal.
- Resultado: el **25 % de test permanece sin tocar** hasta la evaluación final,
  que alimenta el criterio de promoción de la Fase 3.

### Decisión 3 — No balancear artificialmente (usar `class_weight`)

**Qué decidimos:** **no** aplicar sobremuestreo sintético (SMOTE) ni
submuestreo. En su lugar, manejar el desbalance con `class_weight='balanced'`
dentro de los propios modelos.

**Por qué:**
- Generar datos sintéticos o descartar datos reales puede **inflar las métricas**
  y alejar la evaluación de la realidad de producción.
- Es clave que el **conjunto de prueba permanezca intacto y real**: queremos
  medir el modelo sobre la distribución verdadera de fraude, no sobre una
  artificial.
- `class_weight='balanced'` penaliza más los errores sobre la clase minoritaria
  **sin alterar los datos**, logrando el efecto deseado de forma más limpia.
- Es una **decisión consciente y documentada**, no una omisión. (Queda abierta a
  revisión por el Integrante 1 en la Fase 1 si el EDA sugiere lo contrario.)

### Decisión 4 — Eliminar las 1,081 filas duplicadas antes del split

**Qué decidimos:** eliminar los duplicados exactos **antes** de dividir en
train/test.

**Por qué:**
- Si una misma transacción duplicada quedara una copia en train y otra en test,
  se produciría **fuga de datos (data leakage)**: el modelo "vería" en
  entrenamiento un registro idéntico al de la evaluación, inflando el
  desempeño.
- Eliminarlos primero mantiene la integridad de la evaluación.

---

## 3. Estructura del repositorio

```
deteccion_fraude_mlops.ipynb  -> notebook del proyecto (crece por fases)
README.md                      -> este archivo (decisiones + cómo correr)
reports/                       -> documentación por fase + gráficos
   ├── hallazgos_fase1.md      -> análisis del EDA (Fase 1)
   └── *.png                   -> gráficos generados por el notebook
requirements.txt
.gitignore
```

> El archivo `creditcard.csv` pesa ~144 MB, por lo que **no se versiona en Git**
> (queda listado en `.gitignore`). Se descarga aparte desde Kaggle o se coloca
> junto al notebook / en una carpeta `data/` local.

---

## 4. Reparto de fases

El proyecto sigue el ciclo de vida de MLOps, con una fase por integrante y
**revisiones cruzadas** entre ellos (para evitar el "muro de la confusión"
dentro del propio equipo).

| Fase | Responsable | Entrega | Revisa |
|---|---|---|---|
| **0 – Kickoff** | Todo el equipo | Estas decisiones de diseño + repo Git | — |
| **1 – Data** | Integrante 1 | EDA + hallazgos clave del dataset | Integrante 4 |
| **2 – Modeling** | Integrante 2 | Script de entrenamiento **sin MLflow** + hipótesis | Integrante 3 |
| **3 – Tracking / Registry** | Integrante 3 | Instrumentación MLflow + criterio de promoción | — |
| **4 – Deployment** | Integrante 4 | Modelo servido + script de validación + monitoreo | Revisa el informe completo |

**Flujo de dependencias:**
`Fase 0 (todos) → Fase 1 → Fase 2 → Fase 3 → Fase 4 → Entrega final`

---

## 5. Criterio de promoción de modelos (se define en Fase 3)

A definir por el Integrante 3, pero el equipo acordó el marco desde ya: un
modelo pasa a **Staging** solo si cumple umbrales **explícitos y escritos** de
recall y precision sobre el conjunto de prueba (ej.: *recall > 0.75 Y
precision > 0.85*). Los valores finales se ajustarán con los resultados reales.

---

## 6. Cómo correr el proyecto

> *(Se completará a medida que avancen las fases 2–4. Placeholder inicial.)*

```bash
# 1. Crear entorno e instalar dependencias
pip install -r requirements.txt

# 2. Colocar creditcard.csv dentro de /data
# 3. Ejecutar el EDA (Fase 1)
# 4. Ejecutar el entrenamiento (Fase 2)
# 5. Levantar MLflow UI (Fase 3)
# 6. Servir y validar el modelo (Fase 4)
```

---

*Documento de Decisiones de Diseño Iniciales — Fase 0. Redactado por el equipo
antes de comenzar el modelado, como evidencia de que las decisiones se tomaron
con criterio y no por inercia de un tutorial.*
