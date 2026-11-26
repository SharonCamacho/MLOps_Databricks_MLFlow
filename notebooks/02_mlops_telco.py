# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · MLOps con MLflow y Unity Catalog: ciclo de vida del modelo
# MAGIC
# MAGIC Databricks Free Edition (serverless) · MLflow + Unity Catalog
# MAGIC
# MAGIC **Caso de negocio:** una telco quiere saber qué clientes se van a ir para ofrecerles retención a tiempo.
# MAGIC
# MAGIC **Requisito:** haber ejecutado antes el notebook **01_carga_datos_telco** (crea las tablas `telco_train` y `telco_prod`).
# MAGIC
# MAGIC | Paso | Contenido |
# MAGIC |---|---|
# MAGIC | 1 | Entrenamiento con `autolog` y registro del dataset con `mlflow.log_input` |
# MAGIC | 2 | Comparación de runs en MLflow Experiments |
# MAGIC | 3 | Registro del modelo en Unity Catalog y asignación del alias `champion` |
# MAGIC | 4 | Promoción de una versión y rollback reasignando el alias |
# MAGIC | 5 | Inferencia batch consumiendo `@champion` |
# MAGIC | 6 | Monitoreo (AUC y PSI), alerta, reentrenamiento y promoción |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Free Edition: catálogo "workspace", schema "default" (cámbialos si usas otros)
CATALOG, SCHEMA = "workspace", "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.telco_churn_model"
T_TRAIN = f"{CATALOG}.{SCHEMA}.telco_train"
T_PROD = f"{CATALOG}.{SCHEMA}.telco_prod"
T_RECENT = f"{CATALOG}.{SCHEMA}.telco_recent"
T_PRED = f"{CATALOG}.{SCHEMA}.telco_predictions"

NUM = ["tenure_months", "monthly_charges", "total_charges"]
CAT = ["gender", "senior_citizen", "partner", "dependents", "phone_service", "multiple_lines",
       "internet_service", "online_security", "online_backup", "device_protection", "tech_support",
       "streaming_tv", "streaming_movies", "contract", "paperless_billing", "payment_method"]
FEATURES = NUM + CAT

for t in (T_TRAIN, T_PROD):
    if not spark.catalog.tableExists(t):
        raise RuntimeError(f"Falta la tabla {t}. Ejecuta primero el notebook 01_carga_datos_telco.")

mlflow.set_registry_uri("databricks-uc")          # el registro de modelos vive en Unity Catalog
user = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/mlops_telco_churn")
client = MlflowClient()
print("Modelo:", MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones auxiliares

# COMMAND ----------

def make_pipeline(estimator, scale=False):
    """Preprocesamiento + modelo en un solo objeto: lo que se registra es el pipeline completo."""
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", num, NUM)])
    return Pipeline([("pre", pre), ("model", estimator)])


def uc_dataset(table, name):
    """Dataset de MLflow apuntando a una tabla de UC (habilita el lineage tabla → modelo)."""
    return mlflow.data.from_spark(spark.table(table), table_name=table, name=name)


def log_sklearn(model, X, name="model"):
    """Loguea el modelo con firma y ejemplo (UC exige firma). Compatible MLflow 2.x y 3.x."""
    sig = infer_signature(X, model.predict(X))
    try:
        return mlflow.sklearn.log_model(model, name=name, signature=sig, input_example=X.head(3),
                                        serialization_format="cloudpickle")
    except TypeError:
        return mlflow.sklearn.log_model(model, artifact_path=name, signature=sig, input_example=X.head(3),
                                        serialization_format="cloudpickle")


def train_and_log(model, run_name, Xtr, ytr, Xte, yte, data_version="v1", dataset=None):
    """Entrena, mide y registra todo lo necesario para reproducir el run."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"data_version": data_version, "owner": user, "stage": "experiment"})
        if dataset is not None:
            mlflow.log_input(dataset, context="training")        # qué tabla se usó para entrenar
        mlflow.log_param("train_rows", len(Xtr))
        mlflow.log_param("n_features", Xtr.shape[1])
        model.fit(Xtr, ytr)                                      # autolog registra params + métricas de train
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        mlflow.log_metric("test_auc", auc)
        info = log_sklearn(model, Xtr)
        return run.info.run_id, auc, info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · Entrenamiento
# MAGIC `autolog` guarda parámetros y métricas **sin escribir una sola línea de tracking**. Además registramos el dataset con `mlflow.log_input`.
# MAGIC
# MAGIC Después: *Catalog → workspace → default → telco_train → **Lineage***.

# COMMAND ----------

mlflow.sklearn.autolog(log_models=False, log_datasets=False)   # modelos y datasets los logueamos nosotros

train = spark.table(T_TRAIN).toPandas()                        # leemos de la tabla gobernada
train_ds = uc_dataset(T_TRAIN, "telco_train_v1")
X, y = train[FEATURES], train["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

run_id, auc, info = train_and_log(
    make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=-1)),
    "rf_baseline", Xtr, ytr, Xte, yte, dataset=train_ds)
print(f"Modelo entrenado · AUC test = {auc:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · Comparar runs
# MAGIC Abre **Experiments** en la barra lateral → selecciona los runs → **Compare**.
# MAGIC
# MAGIC El tracking permite responder qué datos, parámetros y código produjeron el modelo que está en producción.

# COMMAND ----------

candidates = {
    "rf_shallow": make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, n_jobs=-1)),
    "logreg":     make_pipeline(LogisticRegression(max_iter=1000), scale=True),
    "tree_stump": make_pipeline(DecisionTreeClassifier(max_depth=1, random_state=0)),   # modelo de menor rendimiento, usado en el paso 4
}
results = {"rf_baseline": (run_id, auc, info)}
for name, mdl in candidates.items():
    results[name] = train_and_log(mdl, name, Xtr, ytr, Xte, yte, dataset=train_ds)

pd.DataFrame({k: {"test_auc": round(v[1], 3), "run_id": v[0][:8]} for k, v in results.items()}).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · Registrar en Unity Catalog
# MAGIC De "un notebook con un modelo" a **un activo versionado, con dueño, permisos y lineage**.

# COMMAND ----------

mv1 = mlflow.register_model(results["rf_baseline"][2].model_uri, MODEL_NAME)
print(f"Registrado: {MODEL_NAME} versión {mv1.version}")

client.set_registered_model_alias(MODEL_NAME, "champion", mv1.version)     # el puntero que usan los consumidores
client.set_model_version_tag(MODEL_NAME, mv1.version, "validated_by", "manual-review")
print("alias @champion →", client.get_model_version_by_alias(MODEL_NAME, "champion").version)

# Muestra en la UI: Catalog → workspace → default → Models → telco_churn_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · Promoción y rollback
# MAGIC Alguien sube un modelo peor (v2) y lo promueve. Volvemos atrás **sin tocar ni redeployar código**.

# COMMAND ----------

mv2 = mlflow.register_model(results["tree_stump"][2].model_uri, MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", mv2.version)


def auc_of(alias):
    m = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


print(f"champion  (v{mv1.version}): AUC {auc_of('champion'):.3f}")
print(f"challenger(v{mv2.version}): AUC {auc_of('challenger'):.3f}")

# COMMAND ----------

# Se promueve el challenger (simula una promoción incorrecta)
client.set_registered_model_alias(MODEL_NAME, "champion", mv2.version)
print("champion ahora apunta a v", client.get_model_version_by_alias(MODEL_NAME, "champion").version, "→ AUC", round(auc_of("champion"), 3))

# COMMAND ----------

# Rollback: se reasigna el alias, sin redeploy
client.set_registered_model_alias(MODEL_NAME, "champion", mv1.version)
print("champion de vuelta en v", client.get_model_version_by_alias(MODEL_NAME, "champion").version, "→ AUC", round(auc_of("champion"), 3))
client.delete_registered_model_alias(MODEL_NAME, "challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · Consumir el modelo (inferencia batch)
# MAGIC El consumidor **solo conoce `@champion`**. Cuando cambia el modelo, este código no cambia.
# MAGIC
# MAGIC > En un entorno productivo se ejecutaría como un **Job** programado (*Workflows → Create job* apuntando a este notebook) o un serving endpoint.

# COMMAND ----------

def batch_score(df_in, week):
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    out = df_in.copy()
    out["churn_score"] = model.predict_proba(df_in[FEATURES])[:, 1]
    out["week"] = week
    out["model_version"] = int(version)
    return out


prod = spark.table(T_PROD).toPandas()
scored = pd.concat([batch_score(prod[prod.week == w], w) for w in range(1, 7)], ignore_index=True)

# Tabla de predicciones con liquid clustering por semana
spark.createDataFrame(scored).createOrReplaceTempView("scored_v")
spark.sql(f"CREATE OR REPLACE TABLE {T_PRED} CLUSTER BY (week) AS SELECT * FROM scored_v")
print(f"Guardado en {T_PRED}: {len(scored):,} filas")
display(spark.table(T_PRED).select("week", "contract", "monthly_charges", "churn", "churn_score", "model_version").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6 · Monitoreo: el modelo se degrada y nadie avisa
# MAGIC Dos señales:
# MAGIC 1. **Rendimiento** (AUC por semana, cuando llegan las etiquetas observadas).
# MAGIC 2. **Drift de datos** (PSI por variable; > 0.2 = alerta). El PSI avisa *antes* de que lleguen las etiquetas.

# COMMAND ----------

def psi(expected, actual, bins=10):
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    else:
        cats = sorted(set(expected) | set(actual))
        e = expected.value_counts(normalize=True).reindex(cats).fillna(0).values
        a = actual.value_counts(normalize=True).reindex(cats).fillna(0).values
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


AUC_THRESHOLD = 0.80
PSI_THRESHOLD = 0.20

weekly = []
for w in range(1, 7):
    d = scored[scored.week == w]
    weekly.append({"week": w,
                   "auc": roc_auc_score(d.churn, d.churn_score),
                   "max_psi": max(psi(Xtr[f], d[f]) for f in FEATURES)})
weekly = pd.DataFrame(weekly)

with mlflow.start_run(run_name="monitoring_weekly"):             # la salud del modelo también queda trazada
    for _, r in weekly.iterrows():
        mlflow.log_metric("prod_auc", r.auc, step=int(r.week))
        mlflow.log_metric("max_psi", r.max_psi, step=int(r.week))

last = scored[scored.week == 6]
psis = pd.Series({f: psi(Xtr[f], last[f]) for f in FEATURES}).sort_values(ascending=False).head(8)[::-1]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(weekly.week, weekly.auc, marker="o")
ax[0].axhline(AUC_THRESHOLD, color="red", ls="--", label=f"umbral {AUC_THRESHOLD}")
ax[0].set_title("AUC en producción por semana"); ax[0].set_xlabel("semana"); ax[0].legend()
ax[1].barh(psis.index, psis.values, color=["#d62728" if v > PSI_THRESHOLD else "#1f77b4" for v in psis.values])
ax[1].axvline(PSI_THRESHOLD, color="red", ls="--"); ax[1].set_title("PSI por variable (semana 6, top 8)")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cerrar el ciclo: alerta → reentrenar → comparar → promover
# MAGIC En producción esto sería un Job programado. Aquí lo hacemos a mano para verlo.

# COMMAND ----------

if weekly.auc.iloc[-1] < AUC_THRESHOLD:
    print(f"ALERTA: AUC semana 6 = {weekly.auc.iloc[-1]:.3f} < {AUC_THRESHOLD}. Reentrenando con datos recientes…")

    # Tabla de datos recientes (semanas 4-5, ya con etiquetas) derivada por SQL → nuevo linaje para la v3
    spark.sql(f"CREATE OR REPLACE TABLE {T_RECENT} CLUSTER BY (week) AS SELECT * FROM {T_PROD} WHERE week IN (4, 5)")
    recent = spark.table(T_RECENT).toPandas()
    holdout = prod[prod.week == 6]                   # semana 6, no vista

    r_id, r_auc, r_info = train_and_log(
        make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=-1)),
        "rf_retrain_recent", recent[FEATURES], recent.churn, holdout[FEATURES], holdout.churn,
        data_version="v2-weeks4-5", dataset=uc_dataset(T_RECENT, "telco_recent"))

    champ = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    champ_auc = roc_auc_score(holdout.churn, champ.predict_proba(holdout[FEATURES])[:, 1])
    print(f"champion actual: {champ_auc:.3f}  |  reentrenado: {r_auc:.3f}")

    if r_auc > champ_auc + 0.02:                     # regla de promoción explícita
        mv3 = mlflow.register_model(r_info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", mv3.version)
        print(f"Promovido: champion → v{mv3.version}")
else:
    print("Modelo saludable ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Epílogo: el consumidor sigue igual
# MAGIC Misma línea de código que en el paso 5, **distinto modelo detrás**.

# COMMAND ----------

new = batch_score(prod[prod.week == 6], 6)
print("versión usada:", new.model_version.iloc[0], "· AUC semana 6:", round(roc_auc_score(new.churn, new.churn_score), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reset (para repetir la ejecución)
# MAGIC Descomenta y ejecuta para dejar el registro limpio. (No borra las tablas del notebook 01 ni el CSV.)

# COMMAND ----------

# for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
#     client.delete_model_version(MODEL_NAME, v.version)
# client.delete_registered_model(MODEL_NAME)
# for t in (T_RECENT, T_PRED):
#     spark.sql(f"DROP TABLE IF EXISTS {t}")
