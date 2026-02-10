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
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# If you see thread chaos / slowdowns, uncomment:
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(6)

# -----------------------------
# 1) Load
# -----------------------------
train_gdf = gpd.read_file("train.geojson")
test_gdf  = gpd.read_file("test.geojson")

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

# -----------------------------
# 4) MLP model builder
# -----------------------------
def build_mlp(input_dim: int,
              hidden_units=(256, 128),
              dropout=0.2,
              lr=1e-3,
              l2=1e-5,
              n_classes=6):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for u in hidden_units:
        x = keras.layers.Dense(
            u, activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2)
        )(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# 5) Manual CV tuning (small grid, early stopping)
# -----------------------------
f1_macro = make_scorer(f1_score, average="macro")

# Toggle for faster iterative tuning vs. final run
FAST_MODE = True
REFINE_STAGE = True  # run a second, narrower search around the best fast params

cv = StratifiedKFold(n_splits=3 if FAST_MODE else 5, shuffle=True, random_state=SEED)

if FAST_MODE:
    param_grid = [
        {"hidden_units": (128, 64), "dropout": 0.2, "lr": 1e-3, "l2": 1e-5},
        {"hidden_units": (256, 128), "dropout": 0.3, "lr": 1e-3, "l2": 1e-5},
    ]
else:
    param_grid = [
        {"hidden_units": (256, 128), "dropout": 0.2, "lr": 1e-3, "l2": 1e-5},
        {"hidden_units": (256, 128), "dropout": 0.3, "lr": 1e-3, "l2": 1e-5},
        {"hidden_units": (512, 256), "dropout": 0.3, "lr": 5e-4, "l2": 1e-5},
    ]

EPOCHS = 30 if FAST_MODE else 60
BATCH = 512 if FAST_MODE else 256  # larger batch is faster on CPU
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5 if FAST_MODE else 8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3 if FAST_MODE else 4, factor=0.5, min_lr=1e-5),
]

best_score = -np.inf
best_params = None

print("Tuning MLP (manual CV)...")
for params in param_grid:
    fold_scores = []
    for tr_idx, va_idx in cv.split(X_train, y):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = build_mlp(
            input_dim=X_train.shape[1],
            hidden_units=params["hidden_units"],
            dropout=params["dropout"],
            lr=params["lr"],
            l2=params["l2"],
            n_classes=6
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS,
            batch_size=BATCH,
            verbose=0,
            callbacks=callbacks
        )

        prob = model.predict(X_va, verbose=0)
        pred = prob.argmax(axis=1)
        fold_scores.append(f1_score(y_va, pred, average="macro"))

    mean_f1 = float(np.mean(fold_scores))
    print(f"params={params} -> CV macro-F1={mean_f1:.4f}")
    if mean_f1 > best_score:
        best_score = mean_f1
        best_params = params

print(f"\nBest MLP CV macro-F1={best_score:.4f} with params={best_params}")

if FAST_MODE and REFINE_STAGE:
    # Build a narrower grid around the best fast params
    bu = best_params["hidden_units"]
    du = best_params["dropout"]
    lr = best_params["lr"]
    l2 = best_params["l2"]

    refine_grid = [
        {"hidden_units": bu, "dropout": du, "lr": lr, "l2": l2},
        {"hidden_units": bu, "dropout": min(0.5, du + 0.1), "lr": lr, "l2": l2},
        {"hidden_units": bu, "dropout": max(0.0, du - 0.1), "lr": lr, "l2": l2},
    ]

    best_score = -np.inf
    best_params = None
    print("\nRefining around best fast params...")
    for params in refine_grid:
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_train, y):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = build_mlp(
                input_dim=X_train.shape[1],
                hidden_units=params["hidden_units"],
                dropout=params["dropout"],
                lr=params["lr"],
                l2=params["l2"],
                n_classes=6
            )

            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=EPOCHS,
                batch_size=BATCH,
                verbose=0,
                callbacks=callbacks
            )

            prob = model.predict(X_va, verbose=0)
            pred = prob.argmax(axis=1)
            fold_scores.append(f1_score(y_va, pred, average="macro"))

        mean_f1 = float(np.mean(fold_scores))
        print(f"refine params={params} -> CV macro-F1={mean_f1:.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    print(f"\nRefine best MLP CV macro-F1={best_score:.4f} with params={best_params}")

# -----------------------------
# 6) Train best on full data + predict test + submission
# -----------------------------
final_model = build_mlp(
    input_dim=X_train.shape[1],
    hidden_units=best_params["hidden_units"],
    dropout=best_params["dropout"],
    lr=best_params["lr"],
    l2=best_params["l2"],
    n_classes=6
)

final_model.fit(
    X_train, y,
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ],
)

test_prob = final_model.predict(X_test, verbose=0)
test_pred = test_prob.argmax(axis=1)

sub = pd.DataFrame({"Id": np.arange(1, len(test_pred) + 1), "change_type": test_pred})
sub.to_csv("sample_submission.csv", index=False)
print("Wrote sample_submission.csv")
