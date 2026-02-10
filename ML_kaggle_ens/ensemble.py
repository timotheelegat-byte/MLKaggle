import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import StratifiedKFold, cross_val_score, ParameterGrid
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin

class DenseKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y, **fit_kwargs):
        X = X.toarray() if hasattr(X, 'toarray') else X
        self.model_ = KerasClassifier(build_fn=self.build_fn, **self.kwargs)
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        X = X.toarray() if hasattr(X, 'toarray') else X
        return self.model_.predict(X)

    def predict_proba(self, X):
        X = X.toarray() if hasattr(X, 'toarray') else X
        return self.model_.predict_proba(X)

# ---------- 1) Load ----------
train_gdf = gpd.read_file("train.geojson")
test_gdf  = gpd.read_file("test.geojson")

# target mapping (matches competition)
change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2,
                   'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}

y = train_gdf["change_type"].map(change_type_map).astype(int)

# ---------- 2) Feature engineering ----------
def add_geometry_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # Reproject to a projected CRS for accurate area/length
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:3857")
    # Remove invalid geometries
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]
    geom = gdf.geometry
    out = pd.DataFrame(index=gdf.index)

    # Safely calculate area and perimeter
    try:
        out["area"] = geom.area
    except Exception:
        out["area"] = np.nan
    try:
        out["perimeter"] = geom.length
    except Exception:
        out["perimeter"] = np.nan

    bounds = geom.bounds  # minx, miny, maxx, maxy
    out["bbox_w"] = bounds["maxx"] - bounds["minx"]
    out["bbox_h"] = bounds["maxy"] - bounds["miny"]
    out["bbox_aspect"] = out["bbox_w"] / (out["bbox_h"] + 1e-9)

    out["compactness"] = 4 * np.pi * out["area"] / ((out["perimeter"] ** 2) + 1e-9)
    try:
        hull_area = geom.convex_hull.area
    except Exception:
        hull_area = np.nan
    out["convexity_area_ratio"] = out["area"] / (hull_area + 1e-9)

    # vertices (rough proxy for complexity)
    def n_vertices(g):
        if g is None:
            return 0
        try:
            return len(g.exterior.coords)
        except Exception:
            return 0

    out["n_vertices"] = geom.apply(n_vertices)
    return out

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # brightness per date from means
    for k in range(1, 6):
        r = out[f"img_red_mean_date{k}"]
        g = out[f"img_green_mean_date{k}"]
        b = out[f"img_blue_mean_date{k}"]
        s = r + g + b + 1e-9
        out[f"bright_mean_date{k}"] = s / 3.0
        out[f"r_ratio_date{k}"] = r / s
        out[f"g_ratio_date{k}"] = g / s
        out[f"b_ratio_date{k}"] = b / s

    # deltas: date5 - date1 (big story)
    for c in ["red", "green", "blue"]:
        out[f"{c}_mean_delta_1_5"] = out[f"img_{c}_mean_date5"] - out[f"img_{c}_mean_date1"]
        out[f"{c}_std_delta_1_5"]  = out[f"img_{c}_std_date5"]  - out[f"img_{c}_std_date1"]

    # volatility across dates for each channel mean
    for c in ["red", "green", "blue"]:
        means = np.vstack([out[f"img_{c}_mean_date{k}"].values for k in range(1, 6)]).T
        out[f"{c}_mean_volatility"] = means.std(axis=1)

    # change_status features: count changes
    status_cols = [f"change_status_date{i}" for i in range(0, 5)]
    statuses = out[status_cols].astype(str)

    out["status_n_unique"] = statuses.nunique(axis=1)

    def count_transitions(row):
        seq = row.values
        return sum(seq[i] != seq[i-1] for i in range(1, len(seq)))
    out["status_transitions"] = statuses.apply(count_transitions, axis=1)

    return out

# base columns (drop target from train)
train_X = train_gdf.drop(columns=["change_type"])
test_X  = test_gdf.copy()

# engineer
train_feat = pd.concat(
    [train_X.drop(columns=["geometry"]),
     add_geometry_features(train_gdf),
     ],
    axis=1
)
test_feat = pd.concat(
    [test_X.drop(columns=["geometry"]),
     add_geometry_features(test_gdf),
     ],
    axis=1
)

train_feat = add_temporal_features(train_feat)
test_feat  = add_temporal_features(test_feat)

# ---------- 3) Preprocess ----------

# Exclude columns containing date strings from numeric features
cat_cols = ["urban_type", "geography_type"] + [f"change_status_date{i}" for i in range(0, 5)]
cat_cols = [c for c in cat_cols if c in train_feat.columns]


# Extract year, month, day from date columns and add as numeric features
def extract_date_features(df):
    date_cols = []
    for col in df.columns:
        sample = str(df[col].iloc[0])
        if isinstance(sample, str) and ("-" in sample or "/" in sample):
            date_cols.append(col)
    for col in date_cols:
        # Try parsing date, fallback to NaT if fails, use dayfirst=True
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_day"] = parsed.dt.day
    return df

train_feat = extract_date_features(train_feat)
test_feat = extract_date_features(test_feat)

# Now, exclude original date columns, but keep extracted features
def is_date_col(col):
    sample = str(train_feat[col].iloc[0])
    return isinstance(sample, str) and ("-" in sample or "/" in sample)

num_cols = [c for c in train_feat.columns if c not in cat_cols and not is_date_col(c)]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ],
    remainder="drop"
)

# ---------- 4) Preprocess and transform ----------

X_train = preprocess.fit_transform(train_feat)
X_test = preprocess.transform(test_feat)




def to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X


# Define MLP model builder
def build_mlp(hidden_dim=64, hidden_layers=2, input_dim=None, n_classes=6):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(hidden_dim, activation="relu"))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

f1_macro = make_scorer(f1_score, average="macro")
mlp = None  # we'll tune the MLP separately using the pipeline's preprocess to get input dims

# Build pipelines for each sklearn model (no prep, since we preprocess upfront)
pipelines = {
    "LogisticRegression": Pipeline([("clf", LogisticRegression( solver="saga", penalty="elasticnet", l1_ratio=0.5, C=0.3,max_iter=3000, class_weight="balanced", multi_class="multinomial"))]),
    "SVM": Pipeline([("clf", SVC(kernel="linear", probability=True, class_weight="balanced"))]),
    "HistGradientBoosting": Pipeline([("to_dense", FunctionTransformer(to_dense, validate=False)), ("clf", HistGradientBoostingClassifier(
        early_stopping=True,
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=50
    ))]),
    "RandomForest": Pipeline([("clf", RandomForestClassifier(class_weight="balanced", random_state=42))]),
    "KNN": Pipeline([("clf", KNeighborsClassifier())])
}

# Param grids for each pipeline
param_grids = {
    "LogisticRegression": [
        {"clf__penalty": ["l2"], "clf__C": [10]},
    ],
    "SVM": {"clf__C": [0.1, 1, 10]},
    "HistGradientBoosting": {"clf__max_iter": [400], "clf__n_iter_no_change": [5, 10]},
    "RandomForest": {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 10]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9, 11, 13]}
}

results = {}
# best_score = -np.inf
# best_model = None
# best_name = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run successive halving (HalvingGridSearchCV) per pipeline (faster than exhaustive grid)
for name, pipe_model in pipelines.items():
    print(f"Tuning {name} with HalvingGridSearchCV...")
    halving = HalvingGridSearchCV(pipe_model, param_grids[name], cv=cv, scoring=f1_macro, n_jobs=-1, verbose=2, factor=3)
    halving.fit(X_train, y)
    results[name] = halving
    print(f"{name} best CV macro-F1: {halving.best_score_}")
    # if halving.best_score_ > best_score:
    #     best_score = halving.best_score_
    #     best_model = halving.best_estimator_
    #     best_name = name

# Tune MLP separately using ParameterGrid and manual CV
mlp_param_grid = {"hidden_dim": [32, 64], "hidden_layers": [1, 2], "epochs": [10, 20]}
mlp_best_score = -np.inf
mlp_best_params = None
print("Tuning MLP separately...")
for params in list(ParameterGrid(mlp_param_grid)):
    fold_scores = []
    for tr_idx, val_idx in cv.split(X_train, y):
        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        # Convert to dense for Keras
        X_tr = X_tr.toarray() if hasattr(X_tr, 'toarray') else X_tr
        X_val = X_val.toarray() if hasattr(X_val, 'toarray') else X_val

        # build and train mlp for this fold
        mlp_model = build_mlp(hidden_dim=params["hidden_dim"], hidden_layers=params["hidden_layers"], input_dim=X_train.shape[1], n_classes=len(change_type_map))
        mlp_model.fit(X_tr, y_tr, epochs=params["epochs"], batch_size=32, verbose=0)
        prob = mlp_model.predict(X_val)
        preds = np.argmax(prob, axis=1)
        fold_scores.append(f1_score(y_val, preds, average="macro"))

    mean_score = np.mean(fold_scores)
    print(f"MLP params {params} -> CV macro-F1: {mean_score}")
    if mean_score > mlp_best_score:
        mlp_best_score = mean_score
        mlp_best_params = params

print(f"MLP best CV macro-F1: {mlp_best_score} params: {mlp_best_params}")

# Add MLP to results for ensemble consideration
results["MLP"] = type('MockHalving', (), {'best_score_': mlp_best_score, 'best_estimator_': DenseKerasClassifier(build_fn=lambda: build_mlp(hidden_dim=mlp_best_params["hidden_dim"], hidden_layers=mlp_best_params["hidden_layers"], input_dim=X_train.shape[1], n_classes=len(change_type_map)), epochs=mlp_best_params["epochs"])})()

# Build weighted soft-voting ensemble from top-K models (now includes MLP if it's among top)
K = 3
sorted_models = sorted(results.items(), key=lambda kv: kv[1].best_score_, reverse=True)
topk = sorted_models[:K]
estimators = [(name, res.best_estimator_) for name, res in topk]
scores = np.array([res.best_score_ for _, res in topk])
weights = scores / scores.sum()
print("Ensembling top models:", [name for name, _ in topk], "weights:", weights)

voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=-1)
voting.fit(X_train, y)
pred = voting.predict(X_test)

sub = pd.DataFrame({"Id": np.arange(1, len(pred)+1), "change_type": pred})
sub.to_csv("sample_submission_ensemble_1.csv", index=False)
print("Wrote sample_submission_ensemble.csv")
