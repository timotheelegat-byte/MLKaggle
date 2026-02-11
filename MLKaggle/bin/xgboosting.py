## Trop long voir script avec Optuna pour XGBoosting

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

import xgboost as xgb

# ---------- 1) Load ----------
train_gdf = gpd.read_file("train.geojson")
test_gdf  = gpd.read_file("test.geojson")

change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
y = train_gdf["change_type"].map(change_type_map).astype(int)

# ---------- 2) Feature engineering ----------
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
    # safety: replace inf/nan with 0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
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

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = [col for col in df.columns if isinstance(df[col].iloc[0], str) and ("-" in df[col].iloc[0] or "/" in df[col].iloc[0])]
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_day"] = parsed.dt.day
    return df

def is_date_col(df, col): 
    v = df[col].iloc[0]
    return isinstance(v, str) and ("-" in v or "/" in v)

# ---------- 3) Build feature tables ----------
train_X = train_gdf.drop(columns=["change_type"])
test_X  = test_gdf.copy()

train_geom = add_geometry_features(train_gdf.loc[train_X.index])
test_geom  = add_geometry_features(test_gdf.loc[test_X.index])

train_feat = pd.concat([train_X.drop(columns=["geometry"]), train_geom], axis=1)
test_feat  = pd.concat([test_X.drop(columns=["geometry"]),  test_geom], axis=1)

train_feat = add_temporal_features(train_feat)
test_feat  = add_temporal_features(test_feat)

train_feat = extract_date_features(train_feat)
test_feat  = extract_date_features(test_feat)

# ---------- 4) Preprocess ----------
cat_cols = ["urban_type", "geography_type"] + [f"change_status_date{i}" for i in range(0, 5)]
cat_cols = [c for c in cat_cols if c in train_feat.columns]
num_cols = [c for c in train_feat.columns if c not in cat_cols and not is_date_col(train_feat, c)]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# ---------- 5) XGBoost model + HalvingGridSearchCV ----------
f1_weighted = make_scorer(f1_score, average="weighted")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(change_type_map),
        random_state=42,
        learning_rate=0.01,
        n_jobs=-1,
        tree_method="hist",
        n_estimators=2500,
        #max_depth=14,
        min_child_weight=30,
        subsample=0.7,
        max_depth=14,
    )),
])

param_grid = {
    #"clf__n_estimators": [2500,3000],  ##overfit train 100% vers 3000 estim, depth 14, col 0.3 et sub 0.7
    #"clf__max_depth": [14, None],
    #"clf__subsample": [0.7, 1.0],
    "clf__colsample_bytree": [0.3,0.6],
    #"clf__min_child_weight": [5, 30],
    "clf__lambda": [0.01, 0.1, 1],
    "clf__alpha": [0.01, 0.2],
}

search = HalvingGridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    factor=5,
    verbose=3,
)

train_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
mask_valid = train_feat.notna().all(axis=1)
train_feat = train_feat[mask_valid].reset_index(drop=True)
y = y[mask_valid].reset_index(drop=True)

print("Tuning XGBoost with HalvingGridSearchCV...")
search.fit(train_feat, y)
print("Best CV weighted-F1:", search.best_score_)
print("Best params:", search.best_params_)

# ---------- 6) Predict test & save ----------
best_model = search.best_estimator_
pred = best_model.predict(test_feat)

sub = pd.DataFrame({"Id": np.arange(0, len(pred)), "change_type": pred})
sub.to_csv("sample_submission_xgb.csv", index=False)
joblib.dump(best_model, "model_xgb.joblib")
print("Saved submission and model.")
