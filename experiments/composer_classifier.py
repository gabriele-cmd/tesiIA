"""
experiments/composer_classifier.py
=====================================
Classificatore compositore su feature PCH/PCTM.
Fornisce una misura quantitativa della capacità discriminativa delle feature,
complementare alla visualizzazione UMAP.

Modalità:
  classify   — addestra classificatore, salva confusion/similarity matrix
  dendrogram — carica le similarity matrix salvate, calcola consensus, plotta dendrogramma

Uso:
    # Classifica con RF
    python experiments/composer_classifier.py \
        --feature_type pctm_w3.0_h1.5_concat --classifier randomforest

    # Classifica con SVM
    python experiments/composer_classifier.py \
        --feature_type pctm_w3.0_h1.5_concat --classifier svm

    # Classifica con Ensemble RF+SVM
    python experiments/composer_classifier.py \
        --feature_type pctm_w3.0_h1.5_concat --classifier ensemble

    # Dendrogramma (dopo aver lanciato almeno un classificatore)
    python experiments/composer_classifier.py \
        --feature_type pctm_w3.0_h1.5_concat --mode dendrogram
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble         import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics          import (confusion_matrix, classification_report,
                                      ConfusionMatrixDisplay)
from sklearn.pipeline         import Pipeline


COMPOSERS_DEFAULT = [
    'chopin', 'schubert', 'bach', 'liszt', 'beethoven',
    'schumann', 'rachmaninoff', 'debussy', 'mozart', 'haydn',
]


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento feature
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(features_dir: Path, composers: list,
                 feature_type: str, n_samples: int,
                 seed: int = 42) -> tuple:
    X_list, y_list = [], []
    rng = np.random.default_rng(seed)

    print(f"\nCaricamento feature '{feature_type}':")
    for i, composer in enumerate(composers):
        path = features_dir / f"{composer}_{feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Feature non trovate: {path}")
        data = np.load(path).astype(np.float32)
        n    = min(n_samples, len(data))
        idx  = rng.choice(len(data), n, replace=False)
        data = data[idx]
        X_list.append(data)
        y_list.extend([i] * len(data))
        print(f"  {composer:15s}: {len(data):5d} chunk")

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"\n  Totale: {X.shape[0]} chunk, {X.shape[1]} dim, {len(composers)} classi")
    return X, y, composers


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(classifier: str = 'randomforest',
                   n_estimators: int = 200,
                   max_depth: int = None,
                   seed: int = 42) -> Pipeline:
    from sklearn.svm import SVC

    rf = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        n_jobs       = -1,
        random_state = seed,
        class_weight = 'balanced',
    )
    svm = SVC(
        kernel       = 'rbf',
        C            = 10,
        gamma        = 'scale',
        class_weight = 'balanced',
        random_state = seed,
        probability  = True,   # necessario per soft voting
    )

    if classifier == 'svm':
        return Pipeline([('scaler', StandardScaler()), ('clf', svm)])

    if classifier == 'ensemble':
        # Ensemble RF + SVM con soft voting
        # Nota: VotingClassifier non supporta Pipeline come estimatore diretto,
        # quindi scaliamo i dati prima fuori dal VotingClassifier
        ensemble = VotingClassifier(
            estimators = [('rf', rf), ('svm', svm)],
            voting     = 'soft',
            n_jobs     = -1,
        )
        return Pipeline([('scaler', StandardScaler()), ('clf', ensemble)])

    # default: randomforest
    return Pipeline([('scaler', StandardScaler()), ('clf', rf)])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def run_crossval(X: np.ndarray, y: np.ndarray,
                 classifier: str = 'randomforest',
                 n_folds: int = 5, n_estimators: int = 200,
                 max_depth: int = None, seed: int = 42) -> dict:

    pipeline = build_pipeline(classifier, n_estimators, max_depth, seed)
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print(f"\nCross-validation ({n_folds} fold, stratificato) — {classifier.upper()}...")
    results = cross_validate(
        pipeline, X, y,
        cv                 = cv,
        scoring            = 'accuracy',
        return_train_score = True,
        n_jobs             = -1,
    )

    test_acc  = results['test_score']
    train_acc = results['train_score']
    print(f"  Train accuracy: {train_acc.mean():.3f} ± {train_acc.std():.3f}")
    print(f"  Test  accuracy: {test_acc.mean():.3f} ± {test_acc.std():.3f}")
    print(f"  Per fold:       {[f'{a:.3f}' for a in test_acc]}")

    return {
        'test_mean':  float(test_acc.mean()),
        'test_std':   float(test_acc.std()),
        'train_mean': float(train_acc.mean()),
        'train_std':  float(train_acc.std()),
        'folds':      test_acc.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fit finale + salvataggio matrici numeriche
# ─────────────────────────────────────────────────────────────────────────────
def run_final_fit(X: np.ndarray, y: np.ndarray, composers: list,
                  classifier: str = 'randomforest',
                  n_estimators: int = 200, max_depth: int = None,
                  seed: int = 42, output_dir: Path = None,
                  feature_type: str = '') -> dict:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    pipeline = build_pipeline(classifier, n_estimators, max_depth, seed)
    pipeline.fit(X_train, y_train)
    y_pred   = pipeline.predict(X_test)

    acc    = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred,
                                   target_names=composers, digits=3)
    print(f"\n  Test accuracy (80/20 split): {acc:.3f}")
    print(f"\n{report}")

    cm  = confusion_matrix(y_test, y_pred, normalize='true')
    sim = (cm + cm.T) / 2

    if output_dir is not None:
        comp_labels = [c.capitalize() for c in composers]
        clf_label   = classifier.upper()
        base        = f"{feature_type}_{classifier}"

        # ── Salva matrici numeriche ────────────────────────────────────────
        np.save(output_dir / f"cm_{base}.npy",  cm)
        np.save(output_dir / f"sim_{base}.npy", sim)
        print(f"  Matrici salvate: cm_{base}.npy, sim_{base}.npy")

        # ── Confusion matrix ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(cm, display_labels=comp_labels).plot(
            ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
        ax.set_title(f'Confusion Matrix — {feature_type.upper()}\n'
                     f'{clf_label}  |  Accuracy = {acc:.3f}', fontsize=11)
        plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{base}.png",
                    dpi=130, bbox_inches='tight')
        plt.close()

        # ── Similarity matrix ─────────────────────────────────────────────
        _plot_similarity(sim, composers,
                         title=f'Similarità stilistica — {feature_type.upper()}\n'
                               f'{clf_label} (confusion simmetrizzata, diag=0)',
                         output_path=output_dir / f"similarity_matrix_{base}.png")

        # ── Feature importance (solo RF) ──────────────────────────────────
        clf_step = pipeline.named_steps['clf']
        if hasattr(clf_step, 'feature_importances_'):
            top = np.argsort(clf_step.feature_importances_)[::-1][:10]
            print(f"  Top 10 importance: "
                  f"{clf_step.feature_importances_[top].round(4).tolist()}")

    return {'accuracy': acc, 'report': report, 'cm': cm, 'sim': sim}


# ─────────────────────────────────────────────────────────────────────────────
# Plot similarity (riusabile)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_similarity(sim: np.ndarray, composers: list,
                     title: str, output_path: Path):
    sim_plot = sim.copy()
    np.fill_diagonal(sim_plot, 0)
    comp_labels = [c.capitalize() for c in composers]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_plot, cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(composers)))
    ax.set_yticks(range(len(composers)))
    ax.set_xticklabels(comp_labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(comp_labels, fontsize=9)
    for i in range(len(composers)):
        for j in range(len(composers)):
            ax.text(j, i, f'{sim_plot[i,j]:.2f}', ha='center', va='center',
                    fontsize=7,
                    color='white' if sim_plot[i,j] > sim_plot.max()*0.6 else 'black')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Similarity matrix salvata: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dendrogramma gerarchico
# ─────────────────────────────────────────────────────────────────────────────
def plot_dendrogram(output_dir: Path, feature_type: str,
                    composers: list, classifiers: list = None):
    """
    Carica le similarity matrix salvate, calcola consensus e plotta dendrogramma.
    classifiers: lista di classificatori da combinare (default: tutti disponibili)
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance  import squareform

    if classifiers is None:
        classifiers = ['randomforest', 'svm', 'ensemble']

    # Carica matrici disponibili
    sim_matrices = []
    loaded = []
    for clf in classifiers:
        path = output_dir / f"sim_{feature_type}_{clf}.npy"
        if path.exists():
            sim_matrices.append(np.load(path))
            loaded.append(clf)
            print(f"  Caricata: {path.name}")

    if not sim_matrices:
        raise FileNotFoundError(
            "Nessuna similarity matrix trovata. "
            "Lancia prima --mode classify con almeno un classificatore."
        )

    print(f"\n  Consensus su: {loaded}")

    # Consensus = media delle similarity matrix disponibili
    consensus = np.mean(sim_matrices, axis=0)
    np.fill_diagonal(consensus, 0)

    # Salva consensus
    np.save(output_dir / f"sim_{feature_type}_consensus.npy", consensus)

    # Plotta consensus similarity
    _plot_similarity(
        consensus, composers,
        title=f'Similarità stilistica — Consensus ({"+".join(loaded)})\n'
              f'{feature_type.upper()}',
        output_path=output_dir / f"similarity_matrix_{feature_type}_consensus.png",
    )

    # Converti similarità in distanza: dist = 1 - sim (già normalizzata 0-1)
    # Forza simmetria numerica e diagonale zero
    dist_matrix = 1.0 - consensus
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # forza simmetria

    # Clustering gerarchico con linkage average (UPGMA — standard in musicologia)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    # Plot dendrogramma
    comp_labels = [c.capitalize() for c in composers]
    fig, ax = plt.subplots(figsize=(10, 6))

    dendrogram(
        Z,
        labels      = comp_labels,
        ax          = ax,
        orientation = 'top',
        color_threshold = 0.65 * max(Z[:, 2]),  # soglia colore automatica
        leaf_font_size  = 12,
    )

    ax.set_ylim(bottom=0.8)

    ax.set_title(
        f'Albero stilistico — {feature_type.upper()}\n'
        f'Clustering gerarchico (UPGMA) su similarità da classificatore\n'
        f'Consensus: {" + ".join(c.upper() for c in loaded)}',
        fontsize=11,
    )
    ax.set_ylabel('Distanza stilistica', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"dendrogram_{feature_type}_consensus.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Dendrogramma salvato: {out_path}")

    # Stampa coppie più simili (distanza minima)
    print("\n  Coppie più simili (consensus):")
    pairs = []
    for i in range(len(composers)):
        for j in range(i+1, len(composers)):
            pairs.append((consensus[i,j], composers[i], composers[j]))
    for val, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    {a:15s} ↔ {b:15s}  sim={val:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir',  type=str, default='data/features')
    parser.add_argument('--feature_type',  type=str, default='pctm_w3.0_h1.5_concat')
    parser.add_argument('--composers',     nargs='+', default=COMPOSERS_DEFAULT)
    parser.add_argument('--n_samples',     type=int,  default=1500)
    parser.add_argument('--n_folds',       type=int,  default=5)
    parser.add_argument('--n_estimators',  type=int,  default=200)
    parser.add_argument('--max_depth',     type=int,  default=None)
    parser.add_argument('--classifier',    type=str,  default='randomforest',
                        choices=['randomforest', 'svm', 'ensemble'])
    parser.add_argument('--mode',          type=str,  default='classify',
                        choices=['classify', 'dendrogram'],
                        help='classify: addestra e salva | dendrogram: carica e plotta')
    parser.add_argument('--output_dir',    type=str,  default='results/classifier')
    parser.add_argument('--seed',          type=int,  default=42)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Modalità dendrogramma ─────────────────────────────────────────────
    if args.mode == 'dendrogram':
        print('=' * 60)
        print(f'Dendrogramma stilistico — {args.feature_type.upper()}')
        print('=' * 60)
        plot_dendrogram(output_dir, args.feature_type, args.composers)
        exit(0)

    # ── Modalità classify ─────────────────────────────────────────────────
    print('=' * 60)
    print(f'Classificatore — {args.classifier.upper()}')
    print(f'Feature:     {args.feature_type}')
    print(f'Compositori: {len(args.composers)}')
    print(f'n_samples:   {args.n_samples} per classe')
    print(f'max_depth:   {args.max_depth}')
    print('=' * 60)

    X, y, composers = load_dataset(
        features_dir, args.composers,
        args.feature_type, args.n_samples, args.seed,
    )

    cv_results = run_crossval(
        X, y,
        classifier   = args.classifier,
        n_folds      = args.n_folds,
        n_estimators = args.n_estimators,
        max_depth    = args.max_depth,
        seed         = args.seed,
    )

    final_results = run_final_fit(
        X, y, composers,
        classifier   = args.classifier,
        n_estimators = args.n_estimators,
        max_depth    = args.max_depth,
        seed         = args.seed,
        output_dir   = output_dir,
        feature_type = args.feature_type,
    )

    # Salva risultati testuali
    out_txt = output_dir / f"results_{args.feature_type}_{args.classifier}.txt"
    with open(out_txt, 'w') as f:
        f.write(f"Feature:        {args.feature_type}\n")
        f.write(f"Classifier:     {args.classifier}\n")
        f.write(f"Compositori:    {args.composers}\n")
        f.write(f"n_samples/cl:   {args.n_samples}\n")
        f.write(f"max_depth:      {args.max_depth}\n\n")
        f.write(f"CV accuracy:    {cv_results['test_mean']:.3f} "
                f"± {cv_results['test_std']:.3f}\n")
        f.write(f"CV per fold:    {cv_results['folds']}\n\n")
        f.write(f"Final accuracy: {final_results['accuracy']:.3f}\n\n")
        f.write(final_results['report'])
    print(f"\n  Risultati salvati: {out_txt}")

    print('\n' + '=' * 60)
    print(f"RISULTATO — {args.feature_type.upper()} — {args.classifier.upper()}")
    print(f"CV accuracy: {cv_results['test_mean']:.3f} ± {cv_results['test_std']:.3f}")
    print('=' * 60)