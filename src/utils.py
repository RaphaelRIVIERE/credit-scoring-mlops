import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Dict, List, Optional, Union
import src.visualizer as vis

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes dans un DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
    
    Returns:
        pd.DataFrame: DataFrame avec les statistiques de valeurs manquantes par colonne.
    """
    # Pourcentage global de cellules vides
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    pct_missing_global = (missing_cells / total_cells) * 100
    print(f"\n🌐 Pourcentage de cellules vides sur tout le DataFrame : {pct_missing_global:.2f}%")
    
    
    # Pourcentage par colonne
    missing_by_column = df.isna().sum()
    pct_by_column = (missing_by_column / len(df)) * 100
    nb_cols_with_missing = (missing_by_column > 0).sum()
    print(f"\n📊 Nombre de colonne avec des cellules vides : {nb_cols_with_missing}")
    
    # Créer un DataFrame pour les statistiques
    missing_df = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs manquantes': missing_by_column.values,
        'Pourcentage (%)': pct_by_column.values
    })
    
    # Trier par pourcentage décroissant
    missing_df = missing_df.sort_values('Pourcentage (%)', ascending=False)

    return missing_df


def classify_columns(
    df: pd.DataFrame,
    force_qualitative=None,
    force_ordinal=None,
    low_card_threshold=15
):
    """
    Classifie les colonnes d'un DataFrame par type statistique : quantitatif, qualitatif, ordinal, datetime.

    Parameters:
    -----------
    force_qualitative : list, optional
        Colonnes à forcer en catégoriel nominal
    force_ordinal : list, optional
        Colonnes à forcer en ordinal (ex: notes, satisfactions)
    low_card_threshold : int
        Seuil de cardinalité pour suggérer un type ordinal
    """

    force_qualitative = force_qualitative or []
    force_ordinal = force_ordinal or []

    # Numériques
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Datetime
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Catégorielles naturelles (nominal)
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numériques à faible cardinalité → ORDINAL potentiel
    low_card_numeric = [
        col for col in numeric_cols
        if df[col].nunique() <= low_card_threshold
        and col not in force_qualitative
    ]

    # Classification finale
    ordinal = sorted(set(low_card_numeric + force_ordinal))

    qualitative = sorted(
        set(categorical_cols + force_qualitative) - set(ordinal)
    )

    quantitative = sorted([
        col for col in numeric_cols
        if col not in ordinal
        and col not in qualitative
    ])

    return {
        "quantitative": quantitative,
        "qualitative": qualitative,
        "ordinal": ordinal,
        "datetime": sorted(datetime_cols),
    }

def check_duplicates(df: pd.DataFrame, subset=None):
    """
    Vérifie les doublons dans un DataFrame.
    """

    total_rows = len(df)
    duplicate_rows = df.duplicated(subset=subset).sum()

    return {
        "subset": subset if subset is not None else "all_columns",
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": duplicate_rows / total_rows if total_rows > 0 else 0
    }




def explore_dataframe(df: pd.DataFrame, show_missing: bool=True):
    """
    Affiche les informations principales d'un DataFrame :
    - shape
    - head
    - info
    - describe
    - statistiques de valeurs manquantes

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à analyser
    show_missing : bool, optional
        Affiche l'analyse des valeurs manquantes (default=True)
    """
    print("📋 INFORMATIONS GÉNÉRALES")
    print(f"• Lignes    : {df.shape[0]}")
    print(f"• Colonnes : {df.shape[1]}")

    print("\n--- INFO ---")
    df.info()
    
    print("\n--- DESCRIBE ---")
    display(df.describe())

    print("\n--- MISSING VALUES ---")
    missing_stats = analyze_missing_values(df)
    if show_missing:
        display(missing_stats.head().round(2))

    col_types = classify_columns(df)
    print("\n=== CLASSIFICATION DES VARIABLES ===")
    for k, v in col_types.items():
        print(f"{k.upper():<12} ({len(v)}): {v}")

    dup_info = check_duplicates(df)
    print("\n=== DUPLICATES ===")
    print(f"Lignes totales  : {dup_info['total_rows']}")
    print(f"Lignes dupliquées: {dup_info['duplicate_rows']}")
    print(f"Taux            : {dup_info['duplicate_ratio']:.2%}")



def distribution_column(
    df: pd.DataFrame,
    column: str,
    show_title: bool = True,
    max_rows: int = 20,
    target: Optional[str] = None,
    target_label: str = 'Taux de classe positive (%)'
) -> None:
    """
    Affiche la distribution des valeurs d'une colonne.
    Si target est fourni, ajoute le taux de la classe positive par modalité.

    Args:
        df: DataFrame pandas
        column: Nom de la colonne
        show_title: Afficher le titre (défaut: True)
        max_rows: Nombre maximum de lignes à afficher (défaut: 20)
        target: Nom de la colonne cible binaire (0/1) (défaut: None)
        target_label: Libellé de la colonne de taux (défaut: 'Taux de classe positive (%)')
    """
    if show_title:
        print(f"\n📊 Distribution de la colonne '{column}'")
        print("-" * 100)

    value_counts = df[column].value_counts(dropna=False)
    value_pct = (value_counts / len(df)) * 100

    distribution_summary = pd.DataFrame({
        'Effectif': value_counts,
        'Pourcentage (%)': value_pct.round(2)
    })

    if target is not None:
        taux = df.groupby(column, dropna=False)[target].mean() * 100
        distribution_summary[target_label] = taux.round(3)

    if len(distribution_summary) > max_rows:
        print(f"│  ℹ️  Affichage des {max_rows} valeurs les plus fréquentes (total: {len(distribution_summary)})")
        display(distribution_summary.head(max_rows))
    else:
        display(distribution_summary)


def display_single_column_info(
    df: pd.DataFrame, 
    col: str, 
    show_distribution: bool = False,
    max_distribution_rows: int = 10
) -> None:
    """Affiche un résumé descriptif et visuel d'une seule colonne.
    
    Args:
        df: DataFrame pandas
        col: Nom de la colonne à analyser
        show_distribution: Afficher la distribution détaillée (défaut: False)
        max_distribution_rows: Limite pour l'affichage de distribution (défaut: 10)
    """
    
    total_rows = len(df)
    
    if col not in df.columns:
        print(f"┌─ {col}")
        print("│  ❌ Colonne inexistante")
        print("└" + "─" * 78)
        print()
        return

    series = df[col]
    n_unique = series.nunique(dropna=True)
    n_missing = series.isna().sum()
    pct_unique = n_unique / total_rows * 100
    pct_missing = n_missing / total_rows * 100

    # En-tête
    print(f"┌─ {col}")
    print("│")

    # Type
    if pd.api.types.is_numeric_dtype(series):
        type_emoji = "🔢"
    elif pd.api.types.is_datetime64_any_dtype(series):
        type_emoji = "📅"
    else:
        type_emoji = "🔤"

    print(f"│  {type_emoji} Type: {series.dtype}")
    print(f"│  🎯 Uniques: {n_unique:,} ({pct_unique:.1f}%)")

    # Valeurs manquantes
    if n_missing > 0:
        print(f"│  ⚠️ Manquantes: {n_missing:,} ({pct_missing:.1f}%)")
    else:
        print("│  ✅ Manquantes: 0 (0.0%)")

    # Valeurs explicites si peu nombreuses
    if 0 < n_unique <= 10:
        values = series.dropna().unique()
        values_str = ", ".join(map(str, values))
        if len(values_str) > 60:
            values_str = values_str[:60] + "..."
        print(f"│  📋 Valeurs: {values_str}")

    # Statistiques numériques
    if pd.api.types.is_numeric_dtype(series) and n_unique > 10:
        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()
        mean_str = f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A"
        print(f"│  📈 Min: {min_val:.2f} | Max: {max_val:.2f} | Moyenne: {mean_str}")
    
    # Distribution détaillée (optionnelle et conditionnelle)
    if show_distribution and n_unique <= max_distribution_rows:
        print("│")
        distribution_column(df, col, show_title=False, max_rows=max_distribution_rows)
    
    print("└" + "─" * 78)
    print()

def remove_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    verbose: bool = True,
    strict: bool = False
) -> pd.DataFrame:
    """
    Supprime les colonnes spécifiées du DataFrame.

    Args:
        df: Le DataFrame d'origine
        columns: Liste des noms de colonnes à supprimer
        verbose: Afficher les messages de progression (défaut: True)
        strict: Si True, lève une erreur si une colonne n'existe pas (défaut: False)

    Returns:
        pd.DataFrame: Le DataFrame sans les colonnes supprimées
        
    Raises:
        KeyError: Si strict=True et qu'une colonne n'existe pas
    """
    if not columns:
        if verbose:
            print("⚠️ Aucune colonne à supprimer")
        return df
    
    if verbose:
        print(f"🗂️ Suppression de colonnes | shape initiale : {df.shape}")
    
    df = df.copy()
    
    # Colonnes réellement présentes
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    
    # Mode strict : lever une erreur si colonne manquante
    if strict and missing_cols:
        raise KeyError(f"Colonnes inexistantes : {missing_cols}")
    
    # Supprimer les colonnes existantes
    if existing_cols:
        df = df.drop(columns=existing_cols)
    
    # Affichage des résultats
    if verbose:
        if missing_cols:
            print(f"⚠️ Colonnes inexistantes (ignorées) : {missing_cols}")
        
        nb_supprimees = len(existing_cols)
        nb_ignorees = len(missing_cols)
        
        colonne_txt = "colonne" + ("s" if nb_supprimees > 1 else "")
        supprimee_txt = "supprimée" + ("s" if nb_supprimees > 1 else "")
        
        print(
            f"✅ {nb_supprimees} {colonne_txt} {supprimee_txt} | "
            f"{nb_ignorees} inexistante{'s' if nb_ignorees > 1 else ''} | "
            f"shape finale : {df.shape}"
        )
    
    return df


def compare_group_means(
	df: pd.DataFrame,
	target_col: str,
	quanti_cols: List[str],
	group_labels: Optional[Dict[Union[int, str], str]] = None,
	sort_by_gap: bool = True,
	decimals: int = 2
) -> pd.DataFrame:
    """
    Compare les moyennes de variables quantitatives entre groupes définis par une variable cible.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données
    target_col : str
        Nom de la colonne cible (variable de groupement)
    quanti_cols : list
        Liste des colonnes quantitatives à comparer
    group_labels : dict, optional
        Dictionnaire pour renommer les groupes {valeur_originale: nouveau_label}
        Ex: {0: 'Restés (Non)', 1: 'Partis (Oui)'}
    sort_by_gap : bool, default=True
        Si True, trie les résultats par écart absolu décroissant
    decimales : int, default=2
        Nombre de décimales pour l'arrondi final
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les moyennes par groupe et l'écart en %
    """
    # Calculer les moyennes par groupe
    comparison = df.groupby(target_col)[quanti_cols].mean().T
    if len(comparison.columns) != 2:
        raise ValueError(f"compare_group_means attend exactement 2 groupes, {len(comparison.columns)} trouvés dans '{target_col}'.")
    
    # Renommer les colonnes si labels fournis
    if group_labels:
        comparison.columns = [group_labels.get(col, col) for col in comparison.columns]
    
    # Calculer l'écart en % entre les deux groupes (suppose 2 groupes)
    cols = comparison.columns
    comparison['Écart (%)'] = (
        (comparison[cols[1]] - comparison[cols[0]]) / comparison[cols[0]] * 100
    ).round(1)
    
    # Trier par écart absolu si demandé
    if sort_by_gap:
        comparison = comparison.sort_values('Écart (%)', key=abs, ascending=False)
    
    # Arrondir le résultat final
    return comparison.round(decimals)



def show_outliers(df: pd.DataFrame, colonne: str, plot: bool = False):
    """Affiche simplement les outliers d'une colonne"""
    # Calculer les bornes
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    # Trouver les outliers
    outliers = df[(df[colonne] < borne_inf) | (df[colonne] > borne_sup)]
    
    # Afficher
    print(f"\n{'='*60}")
    print(f"OUTLIERS : {colonne}")
    print(f"{'='*60}")
    print(f"Borne inférieure : {borne_inf:.2f}")
    print(f"Borne supérieure : {borne_sup:.2f}")
    print(f"\nNombre d'outliers : {len(outliers)} sur {len(df)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # Visualisation optionnelle
    if plot:
        import seaborn as sns
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Boxplot via visualizer
        vis.create_boxplot(
            df, axes[0],
            y=colonne,
            title=f'Boxplot - {colonne}',
            ylabel=colonne
        )
        axes[0].axhline(borne_sup, color='red', linestyle='--', alpha=0.7)
        axes[0].axhline(borne_inf, color='red', linestyle='--', alpha=0.7)
        
        # Histogramme avec style cohérent
        sns.histplot(df[colonne], bins=30, ax=axes[1], alpha=0.7)
        axes[1].axvline(borne_sup, color='red', linestyle='--', alpha=0.7, label='Bornes IQR')
        axes[1].axvline(borne_inf, color='red', linestyle='--', alpha=0.7)
        vis._apply_formatting(
            axes[1],
            title=f'Distribution - {colonne}',
            xlabel=colonne,
            ylabel='Fréquence'
        )
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return outliers
