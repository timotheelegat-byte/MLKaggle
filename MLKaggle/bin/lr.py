## Trop peu précis

import numpy as np
import pandas as pd
import joblib
import geopandas as gpd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

# ---------- 1) Load ----------
train_gdf = gpd.read_file("train.geojson")
test_gdf  = gpd.read_file("test.geojson")

# target mapping (matches competition)
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2,
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}
y = train_gdf["change_type"].map(change_type_map).astype(int)

# ---------- 2) Feature engineering ----------
def add_geometry_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # Reproject to a projected CRS for accurate area/length
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:3857")

    # Keep valid geometries only
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

    # brightness + channel ratios per date from means
    for k in range(1, 6):
        r = out[f"img_red_mean_date{k}"]
        g = out[f"img_green_mean_date{k}"]
        b = out[f"img_blue_mean_date{k}"]
        s = r + g + b + 1e-9
        out[f"bright_mean_date{k}"] = s / 3.0
        out[f"r_ratio_date{k}"] = r / s
        out[f"g_ratio_date{k}"] = g / s
        out[f"b_ratio_date{k}"] = b / s

    # deltas: date5 - date1
    for c in ["red", "green", "blue"]:
        out[f"{c}_mean_delta_1_5"] = out[f"img_{c}_mean_date5"] - out[f"img_{c}_mean_date1"]
        out[f"{c}_std_delta_1_5"]  = out[f"img_{c}_std_date5"]  - out[f"img_{c}_std_date1"]

    # volatility across dates for each channel mean
    for c in ["red", "green", "blue"]:
        means = np.vstack([out[f"img_{c}_mean_date{k}"].values for k in range(1, 6)]).T
        out[f"{c}_mean_volatility"] = means.std(axis=1)

    # change_status sequence features
    status_cols = [f"change_status_date{i}" for i in range(0, 5)]
    statuses = out[status_cols].astype(str)

    out["status_n_unique"] = statuses.nunique(axis=1)

    def count_transitions(row):
        seq = row.values
        return sum(seq[i] != seq[i-1] for i in range(1, len(seq)))

    out["status_transitions"] = statuses.apply(count_transitions, axis=1)
    return out

# ---------- 3) Build feature tables (keep indices aligned!) ----------
train_X = train_gdf.drop(columns=["change_type"])
test_X  = test_gdf.copy()

# IMPORTANT: geometry features must be computed on the same rows (same index) as train_X/test_X
train_geom = add_geometry_features(train_gdf.loc[train_X.index])
test_geom  = add_geometry_features(test_gdf.loc[test_X.index])

train_feat = pd.concat([train_X.drop(columns=["geometry"]), train_geom], axis=1)
test_feat  = pd.concat([test_X.drop(columns=["geometry"]),  test_geom], axis=1)

train_feat = add_temporal_features(train_feat)
test_feat  = add_temporal_features(test_feat)

# ---------- 4) Date parsing (extract y/m/d) ----------
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = []
    for col in df.columns:
        val = df[col].iloc[0]
        if isinstance(val, str) and ("-" in val or "/" in val):
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

# ---------- 5) Preprocess ----------
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

# ---------- 6) Model + tuning (Logistic Regression only) ----------
f1_weighted = make_scorer(f1_score, average="weighted")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        solver="saga",
        class_weight="balanced",
        tol = 1e-3,
        max_iter=1000,
        n_jobs=-1,         # IMPORTANT
        random_state=42
    )),
])

param_grid = [
    {
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1, 5, 10]
    },
    {
        "clf__penalty": ["l1"],
        "clf__C": [0.1, 1, 5]
    },
    {
        "clf__penalty": ["elasticnet"],
        "clf__C": [0.1, 1, 5],
        "clf__l1_ratio": [0.2, 0.5, 0.8]
    }
]


search = HalvingGridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring=f1_weighted,
    n_jobs=-1,
    factor=3,
    verbose=3
)


train_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
mask_valid = train_feat.notna().all(axis=1)
train_feat = train_feat[mask_valid].reset_index(drop=True)
y = y[mask_valid].reset_index(drop=True)
print(y.value_counts())
n_valid = mask_valid.sum()
n_invalid = (~mask_valid).sum()

print(f"Lignes valides : {n_valid}")
print(f"Lignes invalides : {n_invalid}")
print(f"Total lignes dans mask : {len(mask_valid)}")

print("Tuning Logistic Regression with HalvingGridSearchCV...")
search.fit(train_feat, y)
print("Best CV weighted-F1:", search.best_score_)
print("Best params:", search.best_params_)

# ---------- 7) Train best model on full train, predict test, write submission ----------
best_model = search.best_estimator_
pred = best_model.predict(test_feat)

sub = pd.DataFrame({"Id": np.arange(1, len(pred) + 1), "change_type": pred})
sub.to_csv("sample_submission_lr_1.csv", index=False)
joblib.dump(best_model, "model_lr.joblib")
print("Wrote sample_submission_lr_1.csv")
