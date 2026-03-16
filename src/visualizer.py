import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import Patch, Rectangle
from typing import Literal
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


def _apply_formatting(
    ax: Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xrotation: int = 0,
    yrotation: int = 0,
    grid: bool = False,
    legend_title: str | None = None,
    show_legend: bool = False,
    xticklabels=None,
    subtitle: str | None = None,
    suptitle_y: float = 0.92,
):
    """Applique le formatage de base à un axe matplotlib."""
    if title:
        if subtitle:
            ax.set_title(title, fontsize=14, fontweight='bold', y=1.02)
        else:
            ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if subtitle:
        ax.figure.suptitle(subtitle, fontsize=11, y=suptitle_y, style='italic')

    ax.tick_params(axis='x', rotation=xrotation)
    ax.tick_params(axis='y', rotation=yrotation)

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(title=legend_title, loc='best', fontsize=10)
    elif (legend := ax.get_legend()) is not None:
        legend.remove()

    if xticklabels is not None:
        ax.set_xticks(ax.get_xticks(), labels=xticklabels, fontsize=11)


def _annotate_bars(ax: Axes, fmt: str = '{:.2f}', ylim_margin: float = 1.15):
    """Ajoute les valeurs au-dessus des barres."""
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=10)

    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(ymin, ymax * ylim_margin)


def create_countplot(
    df: pd.DataFrame, ax: Axes, x: str,
    hue: str | None = None,
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    xrotation: int = 45, palette=None,
    legend_title: str | None = None, show_legend: bool = False,
    annot: bool = True, fmt: str | None = None,
    alpha: float = 1.0, edgecolor=None, linewidth: float = 0.0,
    normalize: bool = False, xticklabels=None,
    grid: bool = False, ylim_margin: float = 1.15,
):
    """Crée un countplot (distribution de fréquences d'une variable catégorielle)."""
    stat = 'percent' if normalize else 'count'
    sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax,
                  alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
                  stat=stat)

    _apply_formatting(
        ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation, grid=grid, legend_title=legend_title,
        show_legend=show_legend, xticklabels=xticklabels,
    )

    if annot:
        if fmt is None:
            fmt = '{:.1f}%' if normalize else '{:.0f}'
        _annotate_bars(ax, fmt=fmt, ylim_margin=ylim_margin)


def create_barplot(
    df: pd.DataFrame, ax: Axes, x: str, y: str,
    hue: str | None = None,
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    xrotation: int = 45, palette=None,
    legend_title: str | None = None, show_legend: bool = False,
    annot: bool = True, fmt: str = '{:.2f}',
    alpha: float = 1.0, edgecolor=None, linewidth: float = 0.0,
    width: float = 0.8, errorbar=None, estimator=np.mean,
    ylim_margin: float = 1.15, xticklabels=None, grid: bool = False,
):
    """Crée un barplot (valeurs agrégées par catégorie)."""
    sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax,
                alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
                width=width, errorbar=errorbar, estimator=estimator)

    _apply_formatting(
        ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation, grid=grid, legend_title=legend_title,
        show_legend=show_legend, xticklabels=xticklabels,
    )

    if annot:
        _annotate_bars(ax, fmt=fmt, ylim_margin=ylim_margin)


def create_histplot(
    df: pd.DataFrame, ax: Axes, x: str,
    hue: str | None = None,
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    xrotation: int = 0, palette=None,
    legend_title: str | None = None, show_legend: bool = True,
    bins: int | str = 50, alpha: float = 0.6,
    stat: str = 'count', kde: bool = False,
    multiple: str = 'layer',
    annot: bool = False, fmt: str = '{:.0f}',
    grid: bool = False, xticklabels=None,
):
    """Crée un histogramme (distribution d'une variable numérique)."""
    sns.histplot(
        data=df, x=x, hue=hue, palette=palette, ax=ax,
        bins=bins, alpha=alpha, stat=stat, kde=kde, multiple=multiple,
    )

    auto_legend = show_legend if show_legend is not None else hue is not None
    _apply_formatting(
        ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation, grid=grid, legend_title=legend_title,
        show_legend=auto_legend, xticklabels=xticklabels,
    )

    if annot:
        _annotate_bars(ax, fmt=fmt)


def create_boxplot(
    df: pd.DataFrame, ax: Axes,
    x: str | None = None, y: str | None = None, hue: str | None = None,
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    xrotation: int | None = None, palette=None,
    legend_title: str | None = None, show_legend: bool = False,
    log_scale: bool = False, showfliers: bool = True,
    grid: bool = False, width: float = 0.8, linewidth: float = 1.5,
    xticklabels=None,
):
    """Crée un boxplot."""
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette,
                showfliers=showfliers, width=width, linewidth=linewidth)

    if log_scale:
        ax.set_yscale('log')
        if title:
            title = f'{title} (log scale)'

    _apply_formatting(
        ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation or 0, grid=grid, legend_title=legend_title,
        # Affiche la légende uniquement si explicitement demandé
        show_legend=show_legend, xticklabels=xticklabels,
    )


def create_scatterplot(
    df: pd.DataFrame, ax: Axes, x: str, y: str,
    hue: str | None = None, size: str | None = None, style: str | None = None,
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    palette=None, color=None,
    alpha: float | None = None, s: int | None = None,
    edgecolor=None, linewidth: float = 0,
    marker: str = 'o',
    regression: bool = False,
    annotate_stats: bool = False,
    grid: bool = False,
    legend_title: str | None = None,
    show_legend: bool | None = None,
    xrotation: int = 0,
    xticklabels=None,
):
    """
    Scatterplot Seaborn avec régression linéaire optionnelle.

    Parameters
    ----------
    alpha : float, optional
        Opacité des points. Si None, calculé automatiquement : min(0.6, 500/n)
        pour réduire le surplotting sur les grands jeux de données.
    regression : bool
        Trace une droite de régression (par groupe si hue est défini).
    annotate_stats : bool
        Affiche le coefficient de corrélation r sur le graphique.
    show_legend : bool, optional
        Si None, affichage automatique (True si hue/size/style/regression).
    """
    if s is None:
        s = 25

    # Alpha automatique : réduit le surplotting sur les grands datasets
    if alpha is None:
        alpha = min(0.6, 500 / max(len(df), 1))

    sns.scatterplot(
        data=df, x=x, y=y,
        hue=hue, size=size, style=style,
        ax=ax,
        palette=palette if hue else None,
        color=color if not hue else None,
        alpha=alpha, s=s,
        edgecolor=edgecolor, linewidth=linewidth,
        marker=marker,
    )

    if regression:
        groups = [(None, df)] if not hue else df.groupby(hue)

        for name, group in groups:
            x_clean = group[x].dropna()
            y_clean = group.loc[x_clean.index, y].dropna()

            if len(x_clean) < 2:
                continue

            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            x_vals = np.linspace(x_clean.min(), x_clean.max(), 200)
            y_vals = slope * x_vals + intercept

            ax.plot(
                x_vals, y_vals,
                linestyle='--', linewidth=2,
                label=f'Régression ({name})' if name is not None else 'Régression',
            )

            if annotate_stats:
                r = np.corrcoef(x_clean, y_clean)[0, 1]
                ax.text(
                    0.02, 0.95 if name is None else 0.90,
                    f'{name}: r = {r:.2f}' if name else f'r = {r:.2f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                )

    auto_legend = hue is not None or size is not None or style is not None or regression
    _apply_formatting(
        ax,
        title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation, grid=grid, legend_title=legend_title,
        show_legend=show_legend if show_legend is not None else auto_legend,
        xticklabels=xticklabels,
    )


def create_heatmap(
    df: pd.DataFrame, ax: Axes,
    annot: bool = True,
    # fmt suit la syntaxe Python format spec (ex: '.2f', 'd') — pas '{:.2f}'
    fmt: str = '.2f',
    cmap: str = 'coolwarm',
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    xrotation: int = 45, yrotation: int = 0,
    linewidths: float = 0.5, linecolor: str = 'white',
    vmin=None, vmax=None, center=None,
    cbar: bool = True, square: bool = False, mask=None,
    grid: bool = False, show_legend: bool = False, legend_title: str | None = None,
    xticklabels=None,
):
    """Crée une heatmap."""
    sns.heatmap(data=df, ax=ax, annot=annot, fmt=fmt, cmap=cmap,
                linewidths=linewidths, linecolor=linecolor,
                vmin=vmin, vmax=vmax, center=center,
                cbar=cbar, square=square, mask=mask)

    _apply_formatting(
        ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
        xrotation=xrotation, yrotation=yrotation,
        grid=grid, show_legend=show_legend, legend_title=legend_title,
        xticklabels=xticklabels,
    )


def plot_contingency_analysis(
    df: pd.DataFrame,
    *,
    rows: str,
    cols: str,
    axes: Sequence[Axes],
    normalize: Literal['index', 'columns', 'all', 0, 1] = 'index',
    heatmap_title: str | None = None,
    barplot_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_title: str | None = None,
):
    """
    Analyse de contingence avec heatmap (brute) et barplot (normalisé).

    Parameters
    ----------
    df : DataFrame
    rows : str — Variable en lignes
    cols : str — Variable en colonnes
    axes : Sequence de 2 axes matplotlib
    normalize : str — Type de normalisation ('index', 'columns', 'all')
    """
    if len(axes) != 2:
        raise ValueError('axes doit contenir exactement 2 sous-graphiques')

    df_clean = df[[rows, cols]].dropna()

    contingency = pd.crosstab(df_clean[rows], df_clean[cols])
    ct_norm = pd.crosstab(df_clean[cols], df_clean[rows], normalize=normalize) * 100
    ct_long = ct_norm.reset_index().melt(
        id_vars=cols,
        var_name=rows,
        value_name='pourcentage',
    )

    create_heatmap(
        contingency, axes[0],
        title=heatmap_title,
        xlabel=xlabel or cols,
        ylabel=ylabel or rows,
    )

    create_barplot(
        ct_long, axes[1],
        x=cols, y='pourcentage', hue=rows,
        title=barplot_title,
        legend_title=legend_title or rows,
        xlabel=xlabel or cols,
        ylabel='Pourcentage',
        show_legend=True,
    )


def create_pairplot(
    df: pd.DataFrame,
    columns: list[str],
    hue: str | None = None,
    title: str | None = None,
    palette=None,
    legend_title: str | None = None,
    xrotation: int = 0, yrotation: int = 0,
    grid: bool = False,
    diag_kind: Literal['auto', 'hist', 'kde'] = 'kde',
    alpha: float = 0.4, s: int = 15, corner: bool = True,
) -> sns.PairGrid:
    """Crée un pairplot Seaborn.

    Retourne l'objet PairGrid pour permettre des ajustements ultérieurs.
    """
    # Inclure hue dans les données si elle n'est pas déjà dans columns
    plot_cols = list(columns) + ([hue] if hue and hue not in columns else [])

    g = sns.pairplot(
        df[plot_cols],
        hue=hue,
        palette=palette,
        diag_kind=diag_kind,
        plot_kws={'alpha': alpha, 's': s},
        corner=corner,
    )

    for ax in g.axes.flat:
        if ax is not None:
            _apply_formatting(
                ax,
                xrotation=xrotation,
                yrotation=yrotation,
                grid=grid,
                show_legend=False,
            )

    if title:
        g.figure.suptitle(title, y=1.02, fontsize=14, fontweight='bold')

    if hue is not None and g.legend is not None and legend_title:
        g.legend.set_title(legend_title)

    return g


def create_barh(
    df: pd.DataFrame, ax: Axes,
    x: str, y: str,
    color: str = '#3498db',
    title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
    subtitle: str | None = None,
    grid: bool = True, show_legend: bool = False, legend_title: str | None = None,
    annot: bool = True, fmt: str = '{:.3f}', sort: bool = True,
):
    """Crée un barplot horizontal trié et annoté.

    Utile pour visualiser des importances de features ou tout ranking de valeurs.
    Gère correctement les valeurs négatives et mixtes.

    Parameters
    ----------
    x : str — Colonne des valeurs numériques (longueur des barres)
    y : str — Colonne des labels (axe vertical)
    """
    data = df.sort_values(x) if sort else df
    values = data[x].tolist()
    labels = data[y].tolist()

    ax.barh(labels, values, color=color)

    if annot:
        span = max(abs(v) for v in values) or 1
        for bar, val in zip(ax.patches, values):
            if not isinstance(bar, Rectangle):
                continue
            offset = span * 0.01 if val >= 0 else -span * 0.01
            ha = 'left' if val >= 0 else 'right'
            ax.text(
                bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va='center', ha=ha, fontsize=9,
            )
        # Étendre les limites pour laisser de la place aux annotations
        padding = span * 0.15
        xmin, xmax = ax.get_xlim()
        if min(values) < 0:
            ax.set_xlim(left=xmin - padding)
        if max(values) > 0:
            ax.set_xlim(right=xmax + padding)

    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle,
                      grid=grid, show_legend=show_legend, legend_title=legend_title)


def plot_metrics_comparison(
    df_results: pd.DataFrame,
    ax: Axes,
    metric_cols: dict | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fmt: str = '{:.2f}',
):
    """
    Barplot groupé comparant plusieurs métriques pour chaque modèle.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame indexé par le nom du modèle, avec des colonnes de métriques.
    ax : Axes
        Axe sur lequel dessiner.
    metric_cols : dict | None
        Mapping {nom_colonne: label_affiché}.
        Par défaut : ROC-AUC, PR-AUC, Recall, F1, Precision.
    """
    if metric_cols is None:
        metric_cols = {
            'test_roc_auc': 'ROC-AUC',
            'test_pr_auc': 'PR-AUC',
            'test_recall': 'Recall',
            'test_f1': 'F1',
            'test_precision': 'Precision',
        }

    df_long = (
        df_results[list(metric_cols.keys())]
        .rename(columns=metric_cols)
        .rename_axis('model')
        .reset_index()
        .melt(id_vars='model', var_name='metric', value_name='score')
    )

    create_barplot(
        df_long, ax, x='model', y='score', hue='metric',
        title=title, xlabel=xlabel, ylabel=ylabel,
        show_legend=True, legend_title='Métrique',
        fmt=fmt, xrotation=0,
    )


def plot_cv_results(
    all_cv_results: dict,
    df_all_results: pd.DataFrame,
    axes: Sequence[Axes],
    metric: str = 'f1',
    suffix: str | None = None,
):
    """
    Visualise les résultats de cross-validation avec deux graphiques :
    - Distribution des scores par fold
    - Comparaison train vs test pour détecter l'overfitting

    Parameters
    ----------
    all_cv_results : dict
        {model_name: {'cv_results': {f'test_{metric}': [scores]}}}
    df_all_results : pd.DataFrame
        DataFrame avec les colonnes f'cv_train_{metric}' et f'cv_test_{metric}'
        et un index 'model'.
    axes : Sequence de 2 axes matplotlib
    metric : str, default='f1'
        Métrique à visualiser (f1, accuracy, precision, recall, etc.)
    suffix : str, optional
        Suffixe ajouté aux titres (ex: "balanced", "SMOTE").
    """
    label = metric.upper().replace('_', '-')
    suffix_str = f' — {suffix}' if suffix else ''

    # Distribution des scores par fold
    rows = []
    for name, res in all_cv_results.items():
        test_key = f'test_{metric}'
        if test_key in res['cv_results']:
            for score in res['cv_results'][test_key]:
                rows.append({'model': name, f'{metric}_score': score})

    df_cv_folds = pd.DataFrame(rows)

    # Comparaison train vs test
    train_col = f'cv_train_{metric}'
    test_col = f'cv_test_{metric}'

    df_overfit = df_all_results[[train_col, test_col]].reset_index().melt(
        id_vars='model',
        var_name='set',
        value_name=f'{metric}_score',
    )

    create_boxplot(
        df_cv_folds, axes[0],
        x='model', y=f'{metric}_score',
        title=f'Distribution {label} par fold (CV){suffix_str}',
        ylabel=label,
    )

    create_barplot(
        df_overfit, axes[1],
        x='model', y=f'{metric}_score', hue='set',
        title=f'{label} Train vs Test (diagnostic overfitting){suffix_str}',
        show_legend=True, legend_title='Jeu',
        fmt='{:.2f}',
    )


def create_pr_curves(
    all_eval_results: dict,
    y_test,
    ax: Axes,
    title: str = 'Courbes Précision–Rappel',
):
    """Trace les courbes Précision–Rappel."""
    for name, res in all_eval_results.items():
        y_proba = res.get('y_proba')
        if y_proba is None:
            continue
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba,
            name=f"{name} (AP={res['pr_auc']:.3f})",
            ax=ax,
        )

    prevalence = y_test.mean()
    ax.axhline(
        y=prevalence, linestyle='--', color='grey', linewidth=1,
        label=f'Prévalence ({prevalence:.1%})',
    )
    ax.set_xlabel('Recall')
    ax.set_ylabel('Précision')
    ax.set_title(title)
    ax.legend(loc='upper right')


def create_roc_curves(
    all_eval_results: dict,
    y_test,
    ax: Axes,
    title: str = 'Courbes ROC',
):
    """Trace les courbes ROC."""
    for name, res in all_eval_results.items():
        y_proba = res.get('y_proba')
        if y_proba is None:
            continue
        RocCurveDisplay.from_predictions(
            y_test, y_proba,
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Classifieur aléatoire')
    ax.set_xlabel('Taux de faux positifs (FPR)')
    ax.set_ylabel('Taux de vrais positifs (Recall)')
    ax.set_title(title)
    ax.legend(loc='lower right')


def plot_roc_pr_curves(
    all_eval_results: dict,
    y_test,
    axes: Sequence[Axes],
    suptitle: str = 'ROC vs PR',
):
    """
    Trace les courbes ROC et Précision-Rappel côte à côte pour tous les modèles.

    Parameters
    ----------
    all_eval_results : dict
        {model_name: {'y_proba': array, 'roc_auc': float, 'pr_auc': float}}
    y_test : array-like
        Vraies étiquettes du jeu de test.
    axes : Sequence de 2 axes matplotlib
    suptitle : str
        Titre principal de la figure.
    """
    create_roc_curves(all_eval_results, y_test, ax=axes[0])
    create_pr_curves(all_eval_results, y_test, ax=axes[1])
    axes[0].figure.suptitle(suptitle, fontsize=14, fontweight='bold')


def plot_confusion_matrices(
    all_eval_results: dict,
    class_names: list[str],
    axes: Sequence[Axes],
    ncols: int | None = None,
):
    """
    Trace les matrices de confusion pour tous les modèles évalués.

    Parameters
    ----------
    all_eval_results : dict
        {model_name: {'confusion_matrix': array}}
    class_names : list[str]
        Noms des classes (ex: ['Non-défaut', 'Défaut']).
    axes : Sequence d'axes matplotlib
    ncols : int, optional
        Nombre de colonnes (utilisé uniquement pour valider la forme des axes).
    """
    axes_flat = np.array(axes).flatten()

    for ax, (name, res) in zip(axes_flat, all_eval_results.items()):
        cm_df = pd.DataFrame(res['confusion_matrix'],
                             index=class_names, columns=class_names)
        create_heatmap(cm_df, ax, fmt='d', cmap='Blues',
                       title=f'Matrice de confusion — {name}',
                       xlabel='Prédit', ylabel='Réel')

    for ax in axes_flat[len(all_eval_results):]:
        ax.set_visible(False)


def plot_model_versions(
    results: dict[str, pd.DataFrame],
    models: dict[str, str],
    axes: Sequence[Axes],
    metrics: list[str] | None = None,
    name_pattern: str = '{model}_{version}',
):
    """
    Compare les performances de modèles entre différentes versions.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        {version_name: df_results} — chaque DF indexé par nom du modèle.
    models : dict[str, str]
        Mapping court → long (ex: {'LR': 'LogisticRegression'}).
    axes : Sequence d'axes matplotlib (un axe par métrique)
    metrics : list[str], optional
        Colonnes à comparer. Défaut: ['test_roc_auc', 'test_pr_auc', 'test_recall'].
    name_pattern : str
        Pattern pour retrouver l'index. Défaut: '{model}_{version}'.
    """
    if metrics is None:
        metrics = ['test_roc_auc', 'test_pr_auc', 'test_recall']

    versions = list(results.keys())
    df_combined = pd.concat(results.values(), verify_integrity=True)

    rows = []
    for model, model_label in models.items():
        for version in versions:
            index_key = name_pattern.format(model=model, version=version)
            for metric in metrics:
                label = metric.replace('test_', '').upper().replace('_', '-')
                rows.append({
                    'Modèle': model_label,
                    'Version': version,
                    'Métrique': label,
                    'Score': df_combined.loc[index_key, metric],
                })

    df_comp = pd.DataFrame(rows)
    metric_labels = [m.replace('test_', '').upper().replace('_', '-') for m in metrics]

    title_suffix = ' vs '.join(versions)
    for ax, metric_label in zip(np.array(axes).flatten(), metric_labels):
        subset = df_comp[df_comp['Métrique'] == metric_label]
        create_barplot(
            subset, ax, x='Modèle', y='Score', hue='Version',
            title=f'{metric_label} : {title_suffix}',
            show_legend=True, legend_title='Version',
            fmt='{:.2f}',
        )


def plot_distributions(
    series: dict[str, pd.DataFrame],
    col: str, ax: Axes,
    title: str | None = None, xlabel: str | None = None,
    fill: bool = True, alpha: float = 0.4,
    grid: bool = True, show_legend: bool = True,
):
    """Compare la distribution d'une variable entre plusieurs ensembles de données."""
    for label, df in series.items():
        sns.kdeplot(df[col].dropna(), ax=ax, label=label, fill=fill, alpha=alpha)

    _apply_formatting(ax, title=title or col, xlabel=xlabel or col,
                      grid=grid, show_legend=show_legend)


def plot_missing_values(missing_df: pd.DataFrame, top_n: int = 15, min_threshold: float = 0.1):
    """
    Visualise les valeurs manquantes sous forme de graphique à barres horizontales.
    
    Args:
        missing_df (pd.DataFrame): DataFrame retourné par analyze_missing_values().
        top_n (int): Nombre maximum de colonnes à afficher (par défaut: 15).
        min_threshold (float): Pourcentage minimum pour afficher une colonne (par défaut: 0.1%).
    """
    # Filtrer les colonnes selon le seuil
    missing_cols = missing_df[missing_df['Pourcentage (%)'] >= min_threshold].head(top_n).copy()
    
    if len(missing_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, max(6, len(missing_cols) * 0.4)))
        
        # Créer le graphique horizontal
        bars = ax.barh(range(len(missing_cols)), missing_cols['Pourcentage (%)'])
        ax.set_yticks(range(len(missing_cols)))
        ax.set_yticklabels(missing_cols['Colonne'], fontsize=10)
        ax.set_xlabel('Pourcentage de valeurs manquantes (%)', fontsize=12, fontweight='bold')
        ax.set_title('Principales colonnes avec valeurs manquantes', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Colorer les barres selon le niveau de gravité
        colors = ['#2ecc71' if x < 1 else '#f39c12' if x < 5 else '#e74c3c' 
                  for x in missing_cols['Pourcentage (%)']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Ajouter les valeurs sur les barres
        for i, (idx, row) in enumerate(missing_cols.iterrows()):
            ax.text(row['Pourcentage (%)'] + 0.5, i, f"{row['Pourcentage (%)']:.2f}%", 
                    va='center', fontsize=9, fontweight='bold')
        
        legend_elements = [
            Patch(facecolor='#2ecc71', label='< 1% manquant (excellente couverture)'),
            Patch(facecolor='#f39c12', label='1-5% manquant (bonne couverture)'),
            Patch(facecolor='#e74c3c', label='> 5% manquant (attention requise)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Ajuster les marges pour éviter les warnings
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
        plt.show()
    else:
        print(f"✅ Aucune colonne avec ≥ {min_threshold}% de valeurs manquantes !")