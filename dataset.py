from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TARGET_COL = "label"
LEAK_COLS = ["attack_cat"]
DROP_COLS = ["id"]

def load_data(train_path: str | Path, test_path: str | Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def split_features_target(df: pd.DataFrame):
    drop_cols = [TARGET_COL] + [c for c in LEAK_COLS + DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df[TARGET_COL].astype(int).copy() if TARGET_COL in df.columns else None
    return X, y

def build_preprocessor(train_df: pd.DataFrame):
    X_train, _ = split_features_target(train_df)
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols

def prepare_binary_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train_df)

    X_train_df, y_train = split_features_target(train_df)
    X_test_df, y_test = split_features_target(test_df)

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_count_after_transform": int(X_train.shape[1]),
        "dropped_columns": [TARGET_COL] + LEAK_COLS + DROP_COLS,
    }
    return X_train, y_train.to_numpy(), X_test, y_test.to_numpy(), preprocessor, meta

def transform_for_inference(df: pd.DataFrame, preprocessor):
    drop_cols = [TARGET_COL] + LEAK_COLS + DROP_COLS
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    return preprocessor.transform(X)
