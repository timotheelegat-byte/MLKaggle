import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

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

# ---------- 3) Build feature tables (keep indices aligned!) ----------
train_X = train_gdf.drop(columns=["change_type"])
test_X  = test_gdf.copy()

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

# RandomForest in sklearn needs dense input -> convert AFTER preprocessing
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# ---------- 6) Model + tuning (RandomForest only) ----------
f1_macro = make_scorer(f1_score, average="macro")

# Toggle for faster iterative tuning vs. final run
FAST_MODE = True
REFINE_STAGE = True  # run a second, narrower search around the best fast params
cv = StratifiedKFold(n_splits=3 if FAST_MODE else 5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("prep", preprocess),
    ("dense", FunctionTransformer(to_dense, validate=False)),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )),
])

if FAST_MODE:
    param_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 20],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": ["sqrt", 0.5],
    }
else:
    param_grid = {
        "clf__n_estimators": [300, 800, 1400],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 3, 10],
        "clf__max_features": ["sqrt", 0.5, 0.2],
    }

search = HalvingGridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring=f1_macro,
    n_jobs=-1,
    factor=5 if FAST_MODE else 3,
    aggressive_elimination=FAST_MODE,
    verbose=2
)

print("Tuning RandomForest with HalvingGridSearchCV...")
search.fit(train_feat, y)
print("Best CV macro-F1:", search.best_score_)
print("Best params:", search.best_params_)

best_search = search

if FAST_MODE and REFINE_STAGE:
    best = search.best_params_

    def unique_sorted(vals):
        return sorted(set(vals))

    n_est = best.get("clf__n_estimators", 300)
    n_est_grid = unique_sorted([max(100, n_est // 2), n_est, int(n_est * 1.5)])

    md = best.get("clf__max_depth", None)
    if md is None:
        md_grid = [None, 20, 40]
    else:
        md_grid = unique_sorted([max(5, md // 2), md, int(md * 1.5)])

    msl = best.get("clf__min_samples_leaf", 1)
    msl_grid = unique_sorted([1, msl, max(2, int(msl * 2))])

    mf = best.get("clf__max_features", "sqrt")
    mf_grid = unique_sorted([mf, "sqrt", 0.5])

    refine_grid = {
        "clf__n_estimators": n_est_grid,
        "clf__max_depth": md_grid,
        "clf__min_samples_leaf": msl_grid,
        "clf__max_features": mf_grid,
    }

    refine_search = HalvingGridSearchCV(
        estimator=pipe,
        param_grid=refine_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=f1_macro,
        n_jobs=-1,
        factor=3,
        verbose=2
    )

    print("Refining around best fast params...")
    refine_search.fit(train_feat, y)
    print("Refine best CV macro-F1:", refine_search.best_score_)
    print("Refine best params:", refine_search.best_params_)
    best_search = refine_search

# ---------- 7) Train best model on full train, predict test, write submission ----------
best_model = best_search.best_estimator_
pred = best_model.predict(test_feat)

sub = pd.DataFrame({"Id": np.arange(1, len(pred) + 1), "change_type": pred})
sub.to_csv("sample_submission_rf_1.csv", index=False)
print("Wrote sample_submission_rf_1.csv")
