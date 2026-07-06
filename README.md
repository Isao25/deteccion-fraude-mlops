# Detección de Fraude con Tarjetas de Crédito — Proyecto MLOps

Proyecto de la asignatura Software Inteligente (UNMSM, Ciclo 9), Semana 14. El
caso de estudio consiste en la detección de fraude con tarjetas de crédito
aplicando el ciclo de vida de MLOps con la herramienta MLflow.

El presente documento reúne las decisiones de diseño iniciales del equipo
(Fase 0, Kickoff), es decir, los acuerdos adoptados antes de iniciar la
implementación del modelado, junto con su justificación técnica.

## 1. Descripción del dataset

De forma previa a cualquier decisión, se inspeccionó el archivo `creditcard.csv`
(conjunto de datos público de Kaggle, correspondiente a transacciones de
tarjetas europeas realizadas en 2013). Las cifras que sustentan las decisiones
posteriores son las siguientes:

| Característica | Valor |
|---|---|
| Total de transacciones | 284,807 |
| Transacciones legítimas (Clase 0) | 284,315 (99.827 %) |
| Transacciones fraudulentas (Clase 1) | 492 (0.173 %) |
| Desbalance | 1 fraude por cada ~577 transacciones legítimas |
| Valores nulos | 0 |
| Filas duplicadas | 1,081 |
| Columnas | 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Monto (`Amount`) en fraude | mediana de $9.25 |
| Monto (`Amount`) en legítimas | mediana de $22.00 |

Las variables `V1`–`V28` corresponden a componentes obtenidos mediante Análisis
de Componentes Principales (PCA), anonimizados por confidencialidad de los datos
bancarios. Únicamente `Time` y `Amount` conservan su escala original.

Se concluye que el problema es de clasificación binaria con un fuerte desbalance
de clases, condición que determina el resto de las decisiones metodológicas.

## 2. Decisiones de diseño y justificación

### 2.1 Métrica de éxito: F1 y AUPRC

Se adoptan como métricas principales el F1-score y el AUPRC (Area Under the
Precision-Recall Curve). Adicionalmente se reportan la precisión y el recall de
manera independiente.

La justificación es la siguiente. Dado que el 99.827 % de las transacciones son
legítimas, un clasificador que prediga siempre la clase mayoritaria alcanzaría
una exactitud (accuracy) cercana al 99.83 % sin detectar ningún fraude. Por
ello, la exactitud resulta engañosa en conjuntos desbalanceados y no se emplea
como métrica principal. El AUPRC se prefiere sobre el AUC-ROC porque, cuando la
clase positiva es poco frecuente, la curva ROC tiende a mostrar resultados
artificialmente optimistas, mientras que la curva de precisión-recall describe
con mayor fidelidad el desempeño sobre la clase minoritaria. El F1-score, por su
parte, equilibra los dos tipos de error relevantes: los falsos negativos (fraude
no detectado, con pérdida económica directa) y los falsos positivos
(transacción legítima rechazada, con perjuicio para el cliente). La precisión y
el recall se reportan por separado porque el criterio de promoción de modelos
(Fase 3) los evalúa de forma individual.

### 2.2 División de datos: 75/25 estratificado

Los datos se dividen en dos subconjuntos: 75 % para entrenamiento y 25 % para
prueba, aplicando estratificación según la variable `Class` y fijando el
parámetro `random_state`. No se reserva un tercer subconjunto de validación
fijo; en su lugar, la comparación entre modelos se realiza mediante validación
cruzada estratificada (StratifiedKFold) sobre el conjunto de entrenamiento.

```
100 % de los datos (sin duplicados)
├── 75 %  ->  Entrenamiento (~213,000 filas). El modelo se ajusta aquí.
│             La comparación entre modelos se hace por validación cruzada.
└── 25 %  ->  Prueba (~71,000 filas). Evaluación final, una sola vez.
```

La estratificación es necesaria porque, al existir únicamente 492 fraudes, una
división aleatoria simple podría asignar una cantidad insuficiente o
desproporcionada de casos positivos al conjunto de prueba. La estratificación
mantiene la proporción de fraude (0.173 %) en ambos subconjuntos, con
aproximadamente 369 fraudes en entrenamiento y 123 en prueba. Fijar
`random_state` garantiza la reproducibilidad del experimento, requisito central
en MLOps.

Se descarta un subconjunto de validación fijo por dos razones. En primer lugar,
comparar modelos directamente sobre el conjunto de prueba comprometería la
imparcialidad de la evaluación final, ya que se terminaría seleccionando el
modelo que mejor se ajusta a ese conjunto en particular. En segundo lugar,
reservar un tercer bloque reduciría aún más el número de fraudes disponibles en
cada subconjunto; la validación cruzada, en cambio, aprovecha la totalidad de
los fraudes del entrenamiento rotándolos entre los distintos folds. De esta
manera, el conjunto de prueba permanece intacto hasta la evaluación final, la
cual alimenta el criterio de promoción de la Fase 3.

### 2.3 Tratamiento del desbalance sin remuestreo

Se decide no aplicar sobremuestreo sintético (SMOTE) ni submuestreo. El
desbalance se gestiona mediante el parámetro `class_weight='balanced'` en los
modelos que lo admiten.

La generación de datos sintéticos o la eliminación de datos reales puede alterar
las métricas y distanciar la evaluación de las condiciones reales de producción.
Resulta prioritario que el conjunto de prueba conserve la distribución original
del fenómeno. El uso de `class_weight` penaliza en mayor medida los errores
sobre la clase minoritaria sin modificar los datos. Esta decisión queda sujeta a
revisión durante la Fase 1 en caso de que el análisis exploratorio aporte
evidencia en contra.

### 2.4 Eliminación de duplicados previa a la división

Las 1,081 filas duplicadas se eliminan antes de dividir los datos en
entrenamiento y prueba. De no hacerlo, una misma transacción podría quedar
repartida entre ambos subconjuntos, lo que produciría fuga de información (data
leakage) y una estimación optimista del desempeño. Eliminar los duplicados en
primer lugar preserva la integridad de la evaluación.

## 3. Estructura del repositorio

```
deteccion_fraude_mlops.ipynb   Notebook del proyecto (se amplía por fases)
README.md                       Este documento (decisiones y ejecución)
reports/                        Documentación por fase y gráficos
   ├── hallazgos_fase1.md       Análisis del EDA (Fase 1)
   └── *.png                    Gráficos generados por el notebook
requirements.txt
.gitignore
```

El archivo `creditcard.csv` ocupa aproximadamente 144 MB, por lo que no se
versiona en Git (figura en `.gitignore`). Debe descargarse de Kaggle y ubicarse
junto al notebook o en una carpeta local `data/`.

## 4. Organización por fases

El proyecto sigue el ciclo de vida de MLOps, asignando una fase a cada
integrante e incorporando revisiones cruzadas entre ellos.

| Fase | Responsable | Entregable | Revisión |
|---|---|---|---|
| 0 – Kickoff | Todo el equipo | Decisiones de diseño y repositorio | — |
| 1 – Data | Integrante 1 | EDA y hallazgos del dataset | Integrante 4 |
| 2 – Modeling | Integrante 2 | Entrenamiento (sin MLflow) e hipótesis | Integrante 3 |
| 3 – Tracking / Registry | Integrante 3 | Instrumentación MLflow y criterio de promoción | — |
| 4 – Deployment | Integrante 4 | Modelo servido, validación y monitoreo | Informe completo |

Secuencia de dependencias: Fase 0 (todos) → Fase 1 → Fase 2 → Fase 3 → Fase 4 →
Entrega final.

## 5. Criterio de promoción de modelos

El criterio definitivo se establece en la Fase 3. El equipo acuerda desde el
inicio que un modelo solo se promueve al estado de Staging si cumple umbrales
explícitos de recall y precisión sobre el conjunto de prueba (por ejemplo,
recall superior a 0.75 y precisión superior a 0.85). Los valores definitivos se
ajustarán a partir de los resultados obtenidos.

## 6. Ejecución del proyecto

Esta sección se completará conforme avancen las fases 2 a 4.

```bash
# 1. Crear el entorno e instalar dependencias
pip install -r requirements.txt

# 2. Ubicar creditcard.csv junto al notebook o en una carpeta data/
# 3. Ejecutar el EDA (Fase 1)
# 4. Ejecutar el entrenamiento (Fase 2)
# 5. Levantar la interfaz de MLflow (Fase 3)
# 6. Servir y validar el modelo (Fase 4)
```
