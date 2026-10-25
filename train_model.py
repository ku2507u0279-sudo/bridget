from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PATH = Path("credit_card_fraud_custom.csv")
MODEL_PATH = Path("fraud_model.joblib")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the raw transaction columns."""
    X = df.drop(columns=["IsFraud", "TransactionID"], errors="ignore").copy()

    # Derive useful time features, then drop the original timestamp.
    if "TransactionDate" in X.columns:
        dt = pd.to_datetime(X["TransactionDate"], errors="coerce")
        X["TransactionHour"] = dt.dt.hour
        X["TransactionDayOfWeek"] = dt.dt.dayofweek
        X["TransactionMonth"] = dt.dt.month
        X = X.drop(columns=["TransactionDate"])

    # Ensure numeric values are parsed consistently.
    for col in ["Amount", "MerchantID"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put the CSV in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH)
    if "IsFraud" not in df.columns:
        raise ValueError("Dataset must contain the target column 'IsFraud'.")

    y = pd.to_numeric(df["IsFraud"], errors="raise").astype(int)
    X = prepare_features(df)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
