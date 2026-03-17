import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Fonctions publiques — appelables step-by-step dans le notebook
# ─────────────────────────────────────────────────────────────

def supprimer_colonnes_nan(
    df: pd.DataFrame,
    seuil: float = 0.5,
    exceptions: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Supprime les colonnes dont le taux de NaN dépasse le seuil.
    Les statistiques sont calculées sur le df fourni — toujours appeler sur le train.

    Returns
    -------
    df_nettoye : pd.DataFrame
    cols_supprimees : list — à réutiliser pour appliquer le même filtre sur le test
    """
    if exceptions is None:
        exceptions = ['EXT_SOURCE_1']
    nan_ratio = df.isnull().mean()
    cols_a_supprimer = nan_ratio.loc[nan_ratio > seuil].index.tolist()
    cols_a_supprimer = [c for c in cols_a_supprimer if c not in exceptions]
    return df.drop(columns=cols_a_supprimer), cols_a_supprimer


def corriger_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les anomalies connues et crée les flags associés.
    - DAYS_EMPLOYED = 365243 → NaN + flag DAYS_EMPLOYED_ANOM
    - CODE_GENDER = 'XNA'    → 'F' (mode)
    """
    df = df.copy()
    # Flag créé avant la correction pour ne pas perdre l'information
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER']        = df['CODE_GENDER'].replace('XNA', 'F')
    return df


def agregger_bureau(data_raw: str) -> pd.DataFrame:
    """Agrège bureau.csv par SK_ID_CURR (global + split actif/clôturé)."""
    bureau = pd.read_csv(data_raw + 'bureau.csv')

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        bureau_count        = ('SK_ID_BUREAU', 'count'),
        bureau_debt_mean    = ('AMT_CREDIT_SUM_DEBT', 'mean'),
        bureau_overdue_mean = ('AMT_CREDIT_SUM_OVERDUE', 'mean'),
        bureau_active_count = ('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    ).reset_index()

    actif   = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    cloture = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']

    actif_agg = actif.groupby('SK_ID_CURR').agg(
        actif_debt_mean = ('AMT_CREDIT_SUM_DEBT', 'mean'),
        actif_count     = ('SK_ID_BUREAU', 'count'),
    ).reset_index()

    cloture_agg = cloture.groupby('SK_ID_CURR').agg(
        cloture_debt_mean = ('AMT_CREDIT_SUM_DEBT', 'mean'),
        cloture_count     = ('SK_ID_BUREAU', 'count'),
    ).reset_index()

    bureau_agg = bureau_agg.merge(actif_agg,   on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(cloture_agg, on='SK_ID_CURR', how='left')
    return bureau_agg


def agregger_previous(data_raw: str) -> pd.DataFrame:
    """Agrège previous_application.csv par SK_ID_CURR (global + split approuvé/refusé)."""
    prev = pd.read_csv(data_raw + 'previous_application.csv')

    prev_agg = prev.groupby('SK_ID_CURR').agg(
        prev_count         = ('SK_ID_PREV', 'count'),
        prev_refused_count = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        prev_credit_mean   = ('AMT_CREDIT', 'mean'),
    ).reset_index()
    prev_agg['taux_refus'] = prev_agg['prev_refused_count'] / prev_agg['prev_count']

    approve = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    refuse  = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']

    approve_agg = approve.groupby('SK_ID_CURR').agg(
        approve_credit_mean = ('AMT_CREDIT', 'mean'),
        approve_count       = ('SK_ID_PREV', 'count'),
    ).reset_index()

    refuse_agg = refuse.groupby('SK_ID_CURR').agg(
        refuse_credit_mean = ('AMT_CREDIT', 'mean'),
        refuse_count       = ('SK_ID_PREV', 'count'),
    ).reset_index()

    prev_agg = prev_agg.merge(approve_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(refuse_agg,  on='SK_ID_CURR', how='left')
    return prev_agg


def agregger_pos(data_raw: str) -> pd.DataFrame:
    """Agrège POS_CASH_balance.csv par SK_ID_CURR."""
    pos = pd.read_csv(data_raw + 'POS_CASH_balance.csv')
    return pos.groupby('SK_ID_CURR').agg(
        pos_dpd_mean = ('SK_DPD', 'mean'),
        pos_dpd_max  = ('SK_DPD', 'max'),
    ).reset_index()


def agregger_installments(data_raw: str) -> pd.DataFrame:
    """Agrège installments_payments.csv par SK_ID_CURR."""
    inst = pd.read_csv(data_raw + 'installments_payments.csv')
    inst['retard_jours'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['diff_montant'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    return inst.groupby('SK_ID_CURR').agg(
        inst_retard_mean = ('retard_jours', 'mean'),
        inst_retard_max  = ('retard_jours', 'max'),
        inst_diff_mean   = ('diff_montant', 'mean'),
        a_eu_retard      = ('retard_jours', lambda x: int((x > 0).any())),
    ).reset_index()


def agregger_credit_card(data_raw: str) -> pd.DataFrame:
    """Agrège credit_card_balance.csv par SK_ID_CURR."""
    cc = pd.read_csv(data_raw + 'credit_card_balance.csv')
    cc['utilisation'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    return cc.groupby('SK_ID_CURR').agg(
        cc_utilisation_mean = ('utilisation', 'mean'),
        cc_dpd_mean         = ('SK_DPD', 'mean'),
        cc_balance_mean     = ('AMT_BALANCE', 'mean'),
    ).reset_index()


def fusionner(base: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Fusionne une liste de tables sur SK_ID_CURR par left join."""
    df = base.copy()
    for table in tables:
        df = df.merge(table, on='SK_ID_CURR', how='left')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées (ratios, âge) — appeler après fusion."""
    df = df.copy()
    df['AGE_YEARS']            = -df['DAYS_BIRTH'] / 365
    df['DAYS_EMPLOYED_PERC']   = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['RATIO_ANNUITE_REVENU'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE']         = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['RATIO_CREDIT_REVENU']  = df['AMT_CREDIT']  / df['AMT_INCOME_TOTAL']
    return df


def regrouper_modalites_rares(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    seuil: float = 0.01,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regroupe les modalités rares (< seuil) en 'Autre'.
    Les fréquences sont calculées sur le train uniquement.

    Parameters
    ----------
    train : pd.DataFrame
    test  : pd.DataFrame, optional — si fourni, le même filtre est appliqué
    seuil : float — fréquence minimale pour conserver une modalité (défaut : 1%)

    Returns
    -------
    train (et test si fourni)
    """
    cat_cols = train.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        frequences = train[col].value_counts(normalize=True)
        modalites_rares = frequences.loc[frequences < seuil].index.tolist()
        train[col] = train[col].replace(modalites_rares, 'Autre')
        if test is not None:
            test[col] = test[col].replace(modalites_rares, 'Autre')
    if test is None:
        return train
    return train, test


# ─────────────────────────────────────────────────────────────
# Fonction principale — composition de toutes les étapes
# ─────────────────────────────────────────────────────────────

def charger_et_fusionner(
    data_raw_path: str,
    seuil_nan: float = 0.5,
    exceptions_nan: list[str] | None = None,
    seuil_rare: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge, nettoie et fusionne toutes les tables Home Credit.

    Composition de toutes les fonctions publiques du module.
    Retourne train et test SANS imputation ni encodage one-hot — ces étapes
    sont déléguées à la Pipeline sklearn (SimpleImputer, OneHotEncoder).

    Parameters
    ----------
    data_raw_path : str
        Chemin vers le dossier data/raw/ (avec slash final).
    seuil_nan : float
        Seuil de suppression des colonnes à fort taux de NaN (défaut : 0.5).
    exceptions_nan : list, optional
        Colonnes à conserver malgré un taux NaN élevé (défaut : ['EXT_SOURCE_1']).
    seuil_rare : float
        Seuil de regroupement des modalités rares (défaut : 0.01 = 1%).

    Returns
    -------
    train, test : pd.DataFrame
    """
    if exceptions_nan is None:
        exceptions_nan = ['EXT_SOURCE_1']

    # Chargement
    app_train = pd.read_csv(data_raw_path + 'application_train.csv')
    app_test  = pd.read_csv(data_raw_path + 'application_test.csv')

    # Suppression colonnes NaN (calculé sur train, appliqué aux deux)
    app_train, cols_supprimes = supprimer_colonnes_nan(app_train, seuil_nan, exceptions_nan)
    app_test = app_test.drop(columns=[c for c in cols_supprimes if c in app_test.columns])
    print(f'{len(cols_supprimes)} colonnes supprimées (>{seuil_nan*100:.0f}% NaN)')

    # Correction anomalies
    app_train = corriger_anomalies(app_train)
    app_test  = corriger_anomalies(app_test)

    # Agrégation des tables secondaires
    print('Agrégation des tables secondaires...')
    tables = [
        agregger_bureau(data_raw_path),
        agregger_previous(data_raw_path),
        agregger_pos(data_raw_path),
        agregger_installments(data_raw_path),
        agregger_credit_card(data_raw_path),
    ]

    # Fusion
    train = fusionner(app_train, tables)
    test  = fusionner(app_test,  tables)

    # Feature engineering
    train = feature_engineering(train)
    test  = feature_engineering(test)

    # Regroupement des modalités rares
    resultat = regrouper_modalites_rares(train, test, seuil=seuil_rare)
    assert isinstance(resultat, tuple)
    train, test = resultat

    print(f'Train : {train.shape} | Test : {test.shape}')
    return train, test
