import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa: F401

import tensorflow as tf
from tensorflow import keras

# -----------------------------
# 0) Repro + (optional) CPU thread sanity
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# If you see thread chaos / slowdowns, uncomment:
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(6)

# -----------------------------
# 1) Load
# -----------------------------
train_gdf = gpd.read_file("data/train.geojson")
test_gdf  = gpd.read_file("data/test.geojson")

change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
y = train_gdf["change_type"].map(change_type_map).astype(int).values  # numpy for TF

# -----------------------------
# 2) Feature engineering
# -----------------------------
def add_geometry_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:3857")
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]

    geom = gdf.geometry
    out = pd.DataFrame(index=gdf.index)

    out["area"] = geom.area
    out["perimeter"] = geom.length

    bounds = geom.bounds
    out["bbox_w"] = bounds["maxx"] - bounds["minx"]
    out["bbox_h"] = bounds["maxy"] - bounds["miny"]
    out["bbox_aspect"] = out["bbox_w"] / (out["bbox_h"] + 1e-9)

    out["compactness"] = 4 * np.pi * out["area"] / ((out["perimeter"] ** 2) + 1e-9)
    hull_area = geom.convex_hull.area
    out["convexity_area_ratio"] = out["area"] / (hull_area + 1e-9)

    def n_vertices(g):
        try:
            return len(g.exterior.coords)
        except Exception:
            return 0

    out["n_vertices"] = geom.apply(n_vertices)
    return out

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for k in range(1, 6):
        r = out[f"img_red_mean_date{k}"]
        g = out[f"img_green_mean_date{k}"]
        b = out[f"img_blue_mean_date{k}"]
        s = r + g + b + 1e-9
        out[f"bright_mean_date{k}"] = s / 3.0
        out[f"r_ratio_date{k}"] = r / s
        out[f"g_ratio_date{k}"] = g / s
        out[f"b_ratio_date{k}"] = b / s

    for c in ["red", "green", "blue"]:
        out[f"{c}_mean_delta_1_5"] = out[f"img_{c}_mean_date5"] - out[f"img_{c}_mean_date1"]
        out[f"{c}_std_delta_1_5"]  = out[f"img_{c}_std_date5"]  - out[f"img_{c}_std_date1"]

    for c in ["red", "green", "blue"]:
        means = np.vstack([out[f"img_{c}_mean_date{k}"].values for k in range(1, 6)]).T
        out[f"{c}_mean_volatility"] = means.std(axis=1)

    status_cols = [f"change_status_date{i}" for i in range(0, 5)]
    statuses = out[status_cols].astype(str)

    out["status_n_unique"] = statuses.nunique(axis=1)

    def count_transitions(row):
        seq = row.values
        return sum(seq[i] != seq[i-1] for i in range(1, len(seq)))

    out["status_transitions"] = statuses.apply(count_transitions, axis=1)
    return out

# Build feature frames with aligned indices
train_X = train_gdf.drop(columns=["change_type"])
test_X  = test_gdf.copy()

train_geom = add_geometry_features(train_gdf.loc[train_X.index])
test_geom  = add_geometry_features(test_gdf.loc[test_X.index])

train_feat = pd.concat([train_X.drop(columns=["geometry"]), train_geom], axis=1)
test_feat  = pd.concat([test_X.drop(columns=["geometry"]),  test_geom], axis=1)

train_feat = add_temporal_features(train_feat)
test_feat  = add_temporal_features(test_feat)

# Date parsing -> y/m/d
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = []
    for col in df.columns:
        v = df[col].iloc[0]
        if isinstance(v, str) and ("-" in v or "/" in v):
            date_cols.append(col)

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_day"] = parsed.dt.day
    return df

train_feat = extract_date_features(train_feat)
test_feat  = extract_date_features(test_feat)

def is_date_col(df: pd.DataFrame, col: str) -> bool:
    v = df[col].iloc[0]
    return isinstance(v, str) and ("-" in v or "/" in v)

# -----------------------------
# 3) Preprocess (sparse) then convert to dense for MLP
# -----------------------------
cat_cols = ["urban_type", "geography_type"] + [f"change_status_date{i}" for i in range(0, 5)]
cat_cols = [c for c in cat_cols if c in train_feat.columns]
num_cols = [c for c in train_feat.columns if c not in cat_cols and not is_date_col(train_feat, c)]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ],
    remainder="drop",
)

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# Fit preprocess once and materialize dense matrices (MLP needs dense)
X_train = to_dense(preprocess.fit_transform(train_feat))
X_test  = to_dense(preprocess.transform(test_feat))



f1_weighted = make_scorer(f1_score, average="weighted")
loaded_model = keras.models.load_model("model/model_mlp.keras")

# Prédire les probabilités sur le jeu de test
test_prob = loaded_model.predict(X_test, verbose=0)  # shape: (n_samples, n_classes)

# Récupérer le nombre de classes
n_classes = test_prob.shape[1]

# Générer les labels de classe (0,1,2,... si pas de noms spécifiques)
class_labels = [f"class_{i}" for i in range(n_classes)]

# Créer le DataFrame avec toutes les probabilités
PROBA_MLP = pd.DataFrame(test_prob, columns=[f"proba_class_{c}" for c in class_labels])
PROBA_MLP.insert(0, "Id", np.arange(len(PROBA_MLP)))

# Sauvegarder en CSV
PROBA_MLP.to_csv("Proba/PROBA_MLP.csv", index=False)
print("Probabilities saved: PROBA_MLP.csv")
