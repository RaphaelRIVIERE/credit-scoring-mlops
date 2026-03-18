import time
import numpy as np
import mlflow
from mlflow.sklearn import log_model as mlflow_log_model
from sklearn.model_selection import cross_validate


def cout_metier(y_true, y_proba, seuil=0.5, cout_fn=10, cout_fp=1):
    """Calcule le coût métier total selon le seuil de décision.

    FN = mauvais client prédit bon → crédit accordé → perte en capital (coût × 10)
    FP = bon client prédit mauvais → refus crédit → manque à gagner (coût × 1)
    """
    predictions = (y_proba >= seuil).astype(int)
    fn = ((y_true == 1) & (predictions == 0)).sum()
    fp = ((y_true == 0) & (predictions == 1)).sum()
    return cout_fn * fn + cout_fp * fp


def trouver_seuil_optimal(y_true, y_proba):
    """Balaye les seuils de 0.05 à 0.95 et retourne celui qui minimise le coût métier."""
    seuils = np.arange(0.05, 0.95, 0.01)
    couts = [cout_metier(y_true, y_proba, s) for s in seuils]
    idx_opt = np.argmin(couts)
    return round(float(seuils[idx_opt]), 2), int(couts[idx_opt])


def cross_validate_model(pipeline, X_train, y_train, cv, scoring):
    """
    Lance une validation croisée et agrège toutes les métriques (train et test).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline contenant le preprocessor et le modèle.
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Cible d'entraînement.
    cv : cross-validation splitter
        Stratégie de découpage (ex: StratifiedKFold).
    scoring : str ou dict
        Métrique(s) d'évaluation.

    Returns
    -------
    dict avec les clés :
        - 'cv_results'       : dict brut retourné par sklearn cross_validate
        - 'training_time_sec': temps d'exécution en secondes
        - 'metrics_summary'  : dict {metric_name: {test_mean, test_std, train_mean, train_std}}
    """
    start = time.time()

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,  # les modèles parallelisent déjà en interne via n_jobs=-1
    )

    training_time = time.time() - start

    # Extraction des noms de métriques
    metric_names = [scoring] if isinstance(scoring, str) else list(scoring.keys())

    # Agrégation train/test pour chaque métrique
    metrics_summary = {}
    for metric in metric_names:
        test_key  = 'test_score'  if isinstance(scoring, str) else f'test_{metric}'
        train_key = 'train_score' if isinstance(scoring, str) else f'train_{metric}'

        metrics_summary[metric] = {
            'test_mean':  cv_results[test_key].mean(),
            'test_std':   cv_results[test_key].std(),
            'train_mean': cv_results[train_key].mean(),
            'train_std':  cv_results[train_key].std(),
        }

    return {
        'cv_results':        cv_results,
        'training_time_sec': training_time,
        'metrics_summary':   metrics_summary,
    }


def evaluate_model(name, pipeline, params, X_train, y_train, X_val, cv, scoring, dataset_source=None):
    """
    Entraîne un modèle et logue les métriques génériques dans le run MLflow actif.

    Doit être appelé dans un contexte mlflow.start_run().

    Parameters
    ----------
    name : str
        Nom affiché dans les logs.
    pipeline : sklearn.pipeline.Pipeline
        Pipeline contenant le preprocessor et le modèle.
    params : dict
        Paramètres à logger dans MLflow.
    X_train, y_train : features et cible d'entraînement.
    X_val : features de validation (pour predict_proba).
    cv : cross-validation splitter
    scoring : str ou dict
        Métrique(s) de validation croisée.
    dataset_source : str, optional
        Chemin vers le fichier source du dataset (pour MLflow Inputs).

    Returns
    -------
    dict avec :
        - 'pipeline'    : pipeline entraîné sur X_train complet
        - 'y_proba_val' : probabilités prédites sur X_val
        - 'summary'     : résumé des métriques de CV
    """
    print(f"\n[{name}] Entraînement en cours...\n")

    # Enregistrement optionnel du dataset source
    if dataset_source is not None:
        dataset = mlflow.data.from_pandas(X_train, source=dataset_source, name="train_processed")
        mlflow.log_input(dataset, context="training")

    # Paramètres
    mlflow.log_param('cv_folds', cv.n_splits)
    mlflow.log_param('preprocessor', 'median_imputer+scaler | mode_imputer+ohe')
    for k, v in params.items():
        mlflow.log_param(k, v)

    # Validation croisée
    result = cross_validate_model(pipeline, X_train, y_train, cv, scoring)
    summary = result['metrics_summary']

    for metric, vals in summary.items():
        mlflow.log_metric(f'{metric}_test_mean',  round(vals['test_mean'],  4))
        mlflow.log_metric(f'{metric}_test_std',   round(vals['test_std'],   4))
        mlflow.log_metric(f'{metric}_train_mean', round(vals['train_mean'], 4))

    # Entraînement final sur X_train complet
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time_final = round(time.time() - t0, 3)

    # Prédiction sur X_val
    t0 = time.time()
    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    predict_time_val = round(time.time() - t0, 3)

    mlflow.log_metric('fit_time_cv',      round(result['training_time_sec'], 3))
    mlflow.log_metric('fit_time_final',   fit_time_final)
    mlflow.log_metric('predict_time_val', predict_time_val)

    mlflow_log_model(pipeline, name="model")

    return {
        'pipeline':    pipeline,
        'y_proba_val': y_proba_val,
        'summary':     summary,
    }
