import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler


def split_data(df, feature_cols, target, test_size=0.2, random_state=42):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print("Train class balance:")
    print(y_train.value_counts(normalize=True).round(3), "\n")
    return X_train, X_test, y_train, y_test


def tune_random_forest(X_train, y_train, cv_folds=5, random_state=42):
    """GridSearchCV over RandomForest. Returns best params dict."""
    param_grid = {
        "n_estimators":      [100, 200, 500],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV f1_macro: {grid.best_score_:.4f}\n")
    return grid.best_params_


def tune_decision_tree(X_train, y_train, cv_folds=5, random_state=42):
    """GridSearchCV over DecisionTreeClassifier. Returns best params dict."""
    param_grid = {
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "criterion":         ["gini", "entropy"],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV f1_macro: {grid.best_score_:.4f}\n")
    return grid.best_params_


def tune_logistic_regression(X_train, y_train, cv_folds=5, random_state=42):
    """GridSearchCV over LogisticRegression. Returns best params dict."""
    param_grid = {
        "C":       [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver":  ["lbfgs", "saga"],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=random_state),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV f1_macro: {grid.best_score_:.4f}\n")
    return grid.best_params_


def tune_knn(X_train, y_train, cv_folds=5, random_state=42):
    """GridSearchCV over KNeighborsClassifier. Returns best params dict."""
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights":     ["uniform", "distance"],
        "p":           [1, 2],          # 1 = Manhattan, 2 = Euclidean
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print(f"Best CV f1_macro: {grid.best_score_:.4f}\n")
    return grid.best_params_

def show_results(model, X_test, y_test, title=None):
    """Confusion matrix + metrics, all inside the same figure."""
    y_pred = model.predict(X_test)

    # metrics first
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # figure sized taller so there's room under the matrix
    fig, ax = plt.subplots(figsize=(6, 6.8))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    ax.set_title(title or f"Confusion Matrix — {type(model).__name__}")

    # metrics text box under the confusion matrix
    report_text = (
        f"Accuracy  : {acc:.4f}\n"
        f"Precision : {prec:.4f}\n"
        f"Recall    : {rec:.4f}\n"
        f"F1-score  : {f1:.4f}"
    )
    fig.text(
        0.5, 0.02, report_text,
        ha="center", va="bottom",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"),
    )

    # leave room at the bottom for the text box
    plt.subplots_adjust(bottom=0.24)

    plt.show()

    # still print in console in case you're redirecting output
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

def RandomForestTest(X_train, y_train, X_test, y_test):
    # RandomForest
    best_params_rf = tune_random_forest(X_train, y_train)
    model_rf = RandomForestClassifier(random_state=42, **best_params_rf)
    model_rf.fit(X_train, y_train)
    show_results(model_rf, X_test, y_test, title="Matriz de Confusão — RandomForest (tuned)")

def DecisionTreeTest(X_train, y_train, X_test, y_test):
    # DecisionTree
    best_params_dt = tune_decision_tree(X_train, y_train)
    model_dt = DecisionTreeClassifier(random_state=42, **best_params_dt)
    model_dt.fit(X_train, y_train)
    show_results(model_dt, X_test, y_test, title="Matriz de Confusão — DecisionTree (tuned)")


def LogisticRegressionTest(X_train_s, y_train, X_test_s, y_test):
    # LogisticRegression
    best_params_lr = tune_logistic_regression(X_train_s, y_train)
    model_lr = LogisticRegression(max_iter=2000, random_state=42, **best_params_lr)
    model_lr.fit(X_train_s, y_train)
    show_results(model_lr, X_test_s, y_test, title="Matriz de Confusão — LogisticRegression (tuned)")

def KNNTest(X_train_s, y_train, X_test_s, y_test):
    # KNeighborsClassifier
    best_params_knn = tune_knn(X_train_s, y_train)
    model_knn = KNeighborsClassifier(**best_params_knn)
    model_knn.fit(X_train_s, y_train)
    show_results(model_knn, X_test_s, y_test, title="Matriz de Confusão — KNN (tuned)")



if __name__ == "__main__":
    df = pd.read_csv("data\\geometric_features.csv")

    target = "label"
    feature_cols = [
        c for c in df.columns
        if c not in {"image_path", "label", "face_found", "subject_id",
                    "source", "n_faces", "iod_px", "error"}
    ]
    
    print(df.isna().sum().sort_values(ascending=False))

    # total NaNs in the feature matrix you are about to train on
    X_full = df[feature_cols]
    print("Total NaNs in X:", X_full.isna().sum().sum())
    print(X_full.isna().sum()[X_full.isna().sum() > 0])

    print("Rows dropped:", df[feature_cols].isna().any(axis=1).sum())
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # 1) split
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target)
    
    scaler = StandardScaler().fit(X_train)   # fit on train only!
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    KNNTest(X_train_s, y_train, X_test_s, y_test)

    """RandomForestTest(X_train, y_train, X_test, y_test)
    DecisionTreeTest(X_train, y_train, X_test, y_test)

    # LR and KNN use scaled data, trees don't care
    scaler = StandardScaler().fit(X_train)   # fit on train only!
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    LogisticRegressionTest(X_train_s, y_train, X_test_s, y_test)
    KNNTest(X_train_s, y_train, X_test_s, y_test)"""