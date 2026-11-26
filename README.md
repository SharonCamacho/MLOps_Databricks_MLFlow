# MLOps en Databricks con MLflow

Ciclo de vida de un modelo de predicción de churn en Databricks: carga de datos en Unity Catalog, entrenamiento con MLflow, registro y promoción por alias, inferencia batch, monitoreo de drift y reentrenamiento.

El proyecto está pensado para ejecutarse en [**Databricks Free Edition**](https://www.databricks.com/learn/free-edition) (compute serverless) y usa el dataset [*Telco Customer Churn* de IBM](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).

## Caso de uso

Una empresa de telecomunicaciones quiere estimar qué clientes tienen mayor probabilidad de cancelar el servicio, para dirigir acciones de retención. El modelo clasifica a cada cliente según su riesgo de churn.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) (datos de ejemplo de IBM, publicados en Kaggle): 7,043 clientes y 33 columnas con demografía, servicios contratados, contrato y facturación. La variable objetivo es `Churn Value` (1 = el cliente canceló). La tasa de churn es 26.5 %.

El archivo está en [`data/Telco_customer_churn.csv`](data/Telco_customer_churn.csv).

Columnas excluidas del entrenamiento:

| Columna | Motivo |
|---|---|
| `Churn Label`, `Churn Value` | Variable objetivo |
| `Churn Score` | Puntuación calculada a partir del resultado. Con esta columna sola el AUC es 0.94 |
| `Churn Reason` | Solo existe para clientes que ya cancelaron |
| `CLTV` | Se calcula con información posterior al evento |
| `CustomerID`, `Count`, ubicación | Identificadores y datos geográficos |

El dataset es una captura sin dimensión temporal. Para simular producción, el notebook 01 reserva el 50 % de los clientes y los reparte en 6 semanas. En las semanas 4 a 6 se aplica un evento **simulado**:

- Aumento del 25 % en `monthly_charges`.
- Clientes de fibra óptica con más de 24 meses de antigüedad cancelan con probabilidad 0.6 (competidor).
- Clientes con contrato mensual y hasta 12 meses de antigüedad que habían cancelado se mantienen con probabilidad 0.6 (promoción de retención).

## Estructura

```
.
├── data/
│   └── Telco_customer_churn.csv
├── img/                      # capturas de Databricks y logos usados en la presentación
├── notebooks/
│   ├── 01_carga_datos_telco.py
│   └── 02_mlops_telco.py
├── presentacion/
│   └── charla_mlops.html
├── LICENSE
└── README.md
```

Los notebooks están en formato fuente de Databricks (`.py` con celdas `# COMMAND ----------`) y se importan desde *Workspace → Import*.

## Presentación

[`presentacion/charla_mlops.html`](presentacion/charla_mlops.html) es una presentación en un solo archivo HTML (se abre en el navegador, sin servidor). Teclas: `←` `→` para navegar, `N` para ver las notas del orador, `T` para el cronómetro, `D` para ir a la slide de demo, `B` para las slides de respaldo y `F` para pantalla completa. Usa los logos y capturas de la carpeta `img/` y carga la tipografía DM Sans desde Google Fonts si hay conexión.

## Requisitos

- Cuenta de [Databricks Free Edition](https://www.databricks.com/learn/free-edition) (o un workspace con Unity Catalog).
- Compute **serverless**. Las librerías usadas (`mlflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`) vienen preinstaladas.
- Un catálogo y un schema con permisos de escritura. Por defecto: `workspace.default`. Se cambian en la celda de setup de cada notebook (`CATALOG`, `SCHEMA`).

## Ejecución

1. Importa los dos notebooks a tu workspace.
2. Abre `01_carga_datos_telco` y ejecuta la celda de setup. Crea el Volume `workspace.default.telco` y se detiene si no encuentra el CSV.
3. Sube `data/Telco_customer_churn.csv` al Volume: *Catalog → workspace → default → telco → Upload to this volume*.
4. Ejecuta el notebook 01 completo.
5. Ejecuta el notebook 02 completo.

## Notebooks

### 01 · Carga de datos

| Etapa | Resultado |
|---|---|
| Ingesta del CSV desde el Volume | Tabla `telco_raw` |
| Revisión de fuga de datos | AUC de `churn_score` y presencia de `churn_reason` |
| Limpieza y selección de variables con SQL | Tabla `telco_clean` |
| Asignación de clientes a entrenamiento o semana de producción | Tabla `telco_split_keys` |
| Datos de entrenamiento | Tabla `telco_train` |
| Datos de producción con el evento simulado | Tabla `telco_prod`, con liquid clustering por `week` |

Las transformaciones se hacen con Spark SQL, de modo que Unity Catalog registra el linaje `telco_raw → telco_clean → telco_train / telco_prod`.

La asignación de clientes usa semillas fijas y se guarda en `telco_split_keys`, junto con dos valores aleatorios por cliente que usa el evento simulado. Esto hace que el reparto sea reproducible.

### 02 · MLOps

| Paso | Contenido |
|---|---|
| 1 | Entrenamiento de un `RandomForestClassifier` dentro de un `Pipeline` (one-hot encoding + modelo) con `mlflow.sklearn.autolog`. El dataset se registra con `mlflow.log_input` |
| 2 | Entrenamiento de variantes (`rf_shallow`, `logreg`, `tree_stump`) y comparación en MLflow Experiments |
| 3 | Registro del modelo en Unity Catalog (`workspace.default.telco_churn_model`) y asignación del alias `champion` |
| 4 | Registro de una versión de menor rendimiento como `challenger`, promoción y rollback reasignando el alias |
| 5 | Inferencia batch cargando `models:/...@champion` y escritura de las predicciones en `telco_predictions` (liquid clustering por `week`) |
| 6 | Cálculo semanal de AUC y PSI por variable, registro en MLflow, alerta si el AUC cae bajo 0.80, reentrenamiento con las semanas 4-5 y promoción si supera al champion por más de 0.02 de AUC |

## Tablas y objetos que se crean

Todos en `workspace.default` (configurable).

| Objeto | Tipo | Creado por |
|---|---|---|
| `telco` | Volume | 01 |
| `telco_raw`, `telco_clean`, `telco_split_keys`, `telco_train` | Tablas | 01 |
| `telco_prod` | Tabla (cluster by `week`) | 01 |
| `telco_predictions` | Tabla (cluster by `week`) | 02 |
| `telco_recent` | Tabla (cluster by `week`) | 02 |
| `telco_churn_model` | Modelo en Unity Catalog | 02 |
| `/Users/<usuario>/mlops_telco_churn` | Experimento de MLflow | 02 |

Cada notebook incluye al final una celda comentada para eliminar lo que creó.

## Resultados de referencia

Obtenidos ejecutando el código de ambos notebooks en un entorno local con MLflow 3.18 y scikit-learn 1.9. En ese entorno las sentencias SQL se sustituyeron por equivalentes en pandas. Los valores pueden variar unas centésimas en Databricks.

| Modelo | AUC en test |
|---|---|
| Regresión logística | 0.869 |
| Random forest (profundidad 8) | 0.862 |
| Random forest (profundidad 3) | 0.858 |
| Árbol de profundidad 1 | 0.718 |

| Semana | AUC | PSI máximo |
|---|---|---|
| 1 | 0.854 | 0.037 |
| 2 | 0.868 | 0.023 |
| 3 | 0.850 | 0.025 |
| 4 | 0.663 | 1.186 |
| 5 | 0.691 | 1.066 |
| 6 | 0.679 | 1.212 |

Tras el reentrenamiento con las semanas 4 y 5, el AUC sobre la semana 6 pasa de 0.679 (champion anterior) a 0.862.

El PSI solo señala `monthly_charges` (1.21). El cambio de comportamiento de los clientes de fibra, que reduce el AUC, no aparece en ninguna distribución de variables, por lo que el monitoreo combina drift de datos y métrica de rendimiento.

## Notas sobre MLflow

- Los modelos se guardan con `serialization_format="cloudpickle"`. Con el formato por defecto de MLflow 3.x (`skops`) el registro de modelos basados en árboles falla por tipos no confiables.
- El registro usa `mlflow.set_registry_uri("databricks-uc")` y la firma del modelo se infiere con `infer_signature`, requisito de Unity Catalog.
- Los consumidores cargan `models:/<modelo>@champion`; promover o revertir una versión consiste en reasignar el alias.

## Autoras

- Ana Maria Lopez: [LinkedIn](https://www.linkedin.com/in/amlopez81/)
- Sharon Camacho: [LinkedIn](https://www.linkedin.com/in/sharoncamachog/)

## Licencia

Código bajo licencia MIT (ver [LICENSE](LICENSE)). El dataset pertenece a IBM; revisa las condiciones de uso de la fuente antes de redistribuirlo.
