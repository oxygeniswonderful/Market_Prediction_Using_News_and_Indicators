import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def ml_class_train_template(
    X,
    y,
    model_class,
    args: dict,
    k_folds:int = 5,
    test_size:float = 0.33,
    feature_importances: bool = False,
    method: str = "choice"
) -> tuple:
    models = {
        f"{i+1}_fold" : None for i in range(k_folds)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y
    )

    if method == "choice":
        y_pred = np.empty((k_folds, y_test.shape[0]))

    elif method == "proba":
        y_pred = np.empty((k_folds, y_test.shape[0], 2))

    cv = StratifiedKFold(n_splits=k_folds)
    if feature_importances:
        feature_importances_array = np.empty((k_folds, X_train.shape[-1]))

    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = model_class(**args)
        model.fit(X_train[train], y_train[train])
        models[f'{i+1}_fold'] = model
        if method == "choice":
            y_pred[i] = model.predict(X_test).reshape(-1)

        elif method == "proba":
            y_pred[i] = model.predict_proba(X_test)

        if feature_importances:
            feature_importances_array[i] = model.feature_importances_

    if method == "choice":
        y_pred_result = np.empty_like(y_test)
        for index in range(y_pred_result.shape[0]):
            unique_values, counts = np.unique(y_pred[:, index], return_counts=True)
            y_pred_result[index] = unique_values[counts.argmax()]

    elif method == "proba":
        y_pred = y_pred.sum(axis=0)
        y_pred_result = np.argmax(y_pred, axis=1)
        y_pred_result[y_pred_result == 0] = -1

    precision = precision_score(y_test, y_pred_result)
    recall = recall_score(y_test, y_pred_result)
    accuracy = accuracy_score(y_test, y_pred_result)
    f1 = f1_score(y_test, y_pred_result)

    metrics = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1
    }

    if feature_importances:
        metrics["feature_importances"] = feature_importances_array.mean(axis=0)

    return metrics, models