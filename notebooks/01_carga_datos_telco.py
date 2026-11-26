# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Carga de datos: Telco Customer Churn (IBM)
# MAGIC
# MAGIC Este notebook prepara los datos; el notebook 02 contiene el flujo de MLOps. Se ejecuta una vez, antes del 02.
# MAGIC
# MAGIC **Qué hace**
# MAGIC 1. Lee el CSV desde un **Volume** de Unity Catalog.
# MAGIC 2. Lo guarda como tabla **bronze** (`telco_raw`) y lo limpia a **silver** (`telco_clean`) con SQL.
# MAGIC 3. Muestra la **fuga de datos** (columnas que ya contienen la respuesta) y las excluye.
# MAGIC 4. Reparte los clientes en **histórico** (entrenamiento) y **"futuro"** (producción en 6 semanas) y simula un evento de mercado en las semanas 4-6.
# MAGIC 5. Guarda las tablas de producción con **liquid clustering** por `week`.
# MAGIC
# MAGIC > **Nota:** el dataset es una captura estática sin fechas. Las semanas de producción se construyen repartiendo los clientes; el evento de las semanas 4-6 (precio +25 %, competidor que se lleva clientes de fibra leales, promo que retiene a clientes mensuales nuevos) es **simulado**.
# MAGIC
# MAGIC **Antes:** sube `Telco_customer_churn.csv` al Volume `workspace.default.telco` (*Catalog → workspace → default → telco → Upload to this volume*). La celda de setup crea el Volume si no existe.
# MAGIC
# MAGIC **Linaje:** todas las transformaciones son SQL/Spark, así que Unity Catalog puede dibujar `telco_raw → telco_clean → telco_train / telco_prod`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import os
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Free Edition: catálogo "workspace", schema "default" (cámbialos si usas otros)
CATALOG, SCHEMA = "workspace", "default"
T_RAW = f"{CATALOG}.{SCHEMA}.telco_raw"
T_CLEAN = f"{CATALOG}.{SCHEMA}.telco_clean"
T_KEYS = f"{CATALOG}.{SCHEMA}.telco_split_keys"
T_TRAIN = f"{CATALOG}.{SCHEMA}.telco_train"
T_PROD = f"{CATALOG}.{SCHEMA}.telco_prod"
CSV_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/telco/Telco_customer_churn.csv"

NUM = ["tenure_months", "monthly_charges", "total_charges"]
CAT = ["gender", "senior_citizen", "partner", "dependents", "phone_service", "multiple_lines",
       "internet_service", "online_security", "online_backup", "device_protection", "tech_support",
       "streaming_tv", "streaming_movies", "contract", "paperless_billing", "payment_method"]

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.telco")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"No encuentro {CSV_PATH}. Sube Telco_customer_churn.csv: Catalog → {CATALOG} → {SCHEMA} → telco → Upload to this volume")
print("CSV encontrado:", CSV_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Ingesta → tabla bronze (`telco_raw`)

# COMMAND ----------

raw = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
raw = raw.toDF(*[c.strip().lower().replace(" ", "_") for c in raw.columns])     # nombres válidos para Delta
raw.write.mode("overwrite").saveAsTable(T_RAW)

n_raw = spark.table(T_RAW).count()
print(f"{T_RAW}: {n_raw:,} filas · {len(raw.columns)} columnas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Fuga de datos (data leakage)
# MAGIC El dataset incluye columnas que contienen la respuesta o se calculan a partir de ella. Usarlas para entrenar produce métricas que no se reproducen en producción, donde esa información no está disponible.

# COMMAND ----------

leak = spark.table(T_RAW).select("churn_value", "churn_score", "churn_reason").toPandas()
print(f"AUC usando SOLO 'churn_score': {roc_auc_score(leak.churn_value, leak.churn_score):.3f}   (columna no disponible en producción)")
print(f"'churn_reason' solo existe cuando el cliente YA se fue: {leak.churn_reason.notna().sum()} filas con motivo = {int(leak.churn_value.sum())} churners")
print("→ Las excluimos: churn_label, churn_score, churn_reason, cltv (y IDs / ubicación).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Limpieza → tabla silver (`telco_clean`)
# MAGIC Solo variables disponibles **antes** de que el cliente se vaya. `total_charges` trae 11 valores en blanco (clientes nuevos, tenure 0): los dejamos en 0.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T_CLEAN} AS
SELECT
  customerid,
  CAST(tenure_months AS DOUBLE)    AS tenure_months,
  CAST(monthly_charges AS DOUBLE)  AS monthly_charges,
  COALESCE(TRY_CAST(TRIM(CAST(total_charges AS STRING)) AS DOUBLE), 0.0) AS total_charges,
  {", ".join(CAT)},
  churn_value AS churn
FROM {T_RAW}
""")

clean = spark.table(T_CLEAN)
n_clean, churn_rate = clean.count(), clean.agg(F.avg("churn")).first()[0]

# Chequeos de calidad mínimos (en producción: Lakeflow expectations o similares)
assert clean.dropna().count() == n_clean, "hay nulos en features"
assert 0.15 < churn_rate < 0.40, "balance de clases inesperado"
print(f"Calidad OK · {n_clean:,} filas · churn = {churn_rate:.1%} · {len(NUM) + len(CAT)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Histórico vs. "futuro" (producción en 6 semanas)
# MAGIC Con semilla fija, asignamos a cada cliente: **train** (50 %, estratificado) o una **semana** de producción (1-6). La asignación se guarda en una tabla de claves (`telco_split_keys`) junto con dos números aleatorios por cliente que usará el evento simulado. Así el resultado es **reproducible** y la transformación sigue siendo SQL (linaje).

# COMMAND ----------

ids = clean.select("customerid", "churn").toPandas().sort_values("customerid").reset_index(drop=True)
train_ids, pool = train_test_split(ids, test_size=0.5, random_state=7, stratify=ids.churn)
pool = pool.sample(frac=1, random_state=1).reset_index(drop=True)

parts = [pd.DataFrame({"customerid": train_ids.customerid.values, "split": "train", "week": 0, "u1": 1.0, "u2": 1.0})]
for i in range(6):
    c = pool.iloc[i::6]
    r = np.random.default_rng(i)
    parts.append(pd.DataFrame({"customerid": c.customerid.values, "split": "prod", "week": i + 1,
                               "u1": r.random(len(c)), "u2": r.random(len(c))}))
keys = pd.concat(parts, ignore_index=True)

spark.createDataFrame(keys).write.mode("overwrite").saveAsTable(T_KEYS)
print(keys.groupby("split").size().to_dict(), "· clientes por semana:", keys[keys.split == "prod"].groupby("week").size().tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabla de entrenamiento (`telco_train`)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T_TRAIN} AS
SELECT c.* FROM {T_CLEAN} c JOIN {T_KEYS} k ON c.customerid = k.customerid
WHERE k.split = 'train'
""")
print(f"{T_TRAIN}: {spark.table(T_TRAIN).count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabla de producción (`telco_prod`) con **liquid clustering** por `week`
# MAGIC Semanas 1-3: datos tal cual. **Semanas 4-6: evento simulado** (precio +25 %, competidor se lleva clientes de fibra leales, promo retiene a clientes mensuales nuevos).
# MAGIC
# MAGIC `CLUSTER BY (week)` activa **liquid clustering**: Delta organiza los archivos por `week` y puedes cambiar la clave después sin reescribir la tabla. Es el reemplazo moderno de `PARTITIONED BY`.

# COMMAND ----------

cat_c = ", ".join(f"c.{x}" for x in CAT)
spark.sql(f"""
CREATE OR REPLACE TABLE {T_PROD}
CLUSTER BY (week)
AS
SELECT
  c.customerid,
  k.week,
  c.tenure_months,
  CASE WHEN k.week >= 4 THEN ROUND(c.monthly_charges * 1.25, 2) ELSE c.monthly_charges END AS monthly_charges,
  CASE WHEN k.week >= 4 THEN c.total_charges + 0.25 * c.monthly_charges * c.tenure_months * 0.3
       ELSE c.total_charges END AS total_charges,
  {cat_c},
  CASE
    WHEN k.week >= 4 AND c.internet_service = 'Fiber optic' AND c.tenure_months > 24 AND k.u1 < 0.6 THEN 1
    WHEN k.week >= 4 AND c.contract = 'Month-to-month' AND c.tenure_months <= 12 AND c.churn = 1 AND k.u2 < 0.6 THEN 0
    ELSE c.churn
  END AS churn
FROM {T_CLEAN} c JOIN {T_KEYS} k ON c.customerid = k.customerid
WHERE k.split = 'prod'
""")
spark.sql(f"OPTIMIZE {T_PROD}")          # aplica el clustering a los archivos existentes

display(spark.sql(f"DESCRIBE DETAIL {T_PROD}").select("name", "clusteringColumns", "numFiles", "sizeInBytes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Resumen y siguiente paso

# COMMAND ----------

display(spark.sql(f"""
SELECT week, COUNT(*) AS clientes, ROUND(AVG(churn), 3) AS churn_rate, ROUND(AVG(monthly_charges), 1) AS cargo_mensual
FROM {T_PROD} GROUP BY week ORDER BY week
"""))
print("Datos listos. Siguiente: notebook 02_mlops_telco.")
print("Linaje: Catalog → workspace → default → telco_train (o telco_prod) → pestaña Lineage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reset (para repetir la carga)
# MAGIC Descomenta y ejecuta. (No borra el CSV del Volume.)

# COMMAND ----------

# for t in (T_RAW, T_CLEAN, T_KEYS, T_TRAIN, T_PROD):
#     spark.sql(f"DROP TABLE IF EXISTS {t}")
