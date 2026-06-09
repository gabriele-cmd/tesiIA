"""
experiments/composer_classifier.py
=====================================
Classificatore compositore su feature PCH/PCTM.
Fornisce una misura quantitativa della capacità discriminativa delle feature,
complementare alla visualizzazione UMAP.

Uso:
    python experiments/composer_classifier.py \
        --features_dir data/features \
        --feature_type pctm_w3.0_h1.5_concat \
        --composers chopin schubert bach liszt beethoven \
                    schumann rachmaninoff debussy mozart haydn \
        --n_samples 1500 \
        --classifier randomforest \
        --output_dir results/classifier
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble         import RandomForestClassifier
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

        n   = min(n_samples, len(data))
        idx = rng.choice(len(data), n, replace=False)
        data = data[idx]

        X_list.append(data)
        y_list.extend([i] * len(data))
        print(f"  {composer:15s}: {len(data):5d} chunk")

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"\n  Totale: {X.shape[0]} chunk, {X.shape[1]} dim, {len(composers)} classi")
    return X, y, composers


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — tutti i parametri PRIMA del seed
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(classifier: str = 'randomforest',
                   n_estimators: int = 200,
                   max_depth: int = None,
                   seed: int = 42) -> Pipeline:
    if classifier == 'svm':
        from sklearn.svm import SVC
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(kernel='rbf', C=10, gamma='scale',
                           class_weight='balanced', random_state=seed)),
        ])
    # default: Random Forest
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            n_jobs       = -1,
            random_state = seed,
            class_weight = 'balanced',
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def run_crossval(X: np.ndarray, y: np.ndarray, composers: list,
                 classifier: str = 'randomforest',
                 n_folds: int = 5, n_estimators: int = 200,
                 max_depth: int = None, seed: int = 42) -> dict:
    pipeline = build_pipeline(classifier, n_estimators, max_depth, seed)
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print(f"\nCross-validation ({n_folds} fold, stratificato)...")
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
# Fit finale + confusion matrix + similarity matrix
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
    y_pred = pipeline.predict(X_test)

    acc = (y_pred == y_test).mean()
    print(f"\n  Test accuracy (80/20 split): {acc:.3f}")

    report = classification_report(
        y_test, y_pred, target_names=composers, digits=3,
    )
    print(f"\n{report}")

    if output_dir is not None:
        clf_label  = classifier.upper()
        comp_labels = [c.capitalize() for c in composers]
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        # ── Confusion matrix ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix = cm,
            display_labels   = comp_labels,
        )
        disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
        ax.set_title(
            f'Confusion Matrix — {feature_type.upper()}\n'
            f'{clf_label}  |  Accuracy = {acc:.3f}',
            fontsize=11,
        )
        plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        out_cm = output_dir / f"confusion_matrix_{feature_type}_{classifier}.png"
        plt.savefig(out_cm, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix salvata: {out_cm}")

        # ── Similarity matrix (simmetrica) ────────────────────────────────
        sim = (cm + cm.T) / 2
        np.fill_diagonal(sim, 0)   # togli diagonale per vedere meglio le confusioni

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim, cmap='Reds', vmin=0)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(composers)))
        ax.set_yticks(range(len(composers)))
        ax.set_xticklabels(comp_labels, rotation=35, ha='right', fontsize=9)
        ax.set_yticklabels(comp_labels, fontsize=9)
        for i in range(len(composers)):
            for j in range(len(composers)):
                ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if sim[i,j] < 0.15 else 'white')
        ax.set_title(
            f'Similarità stilistica — {feature_type.upper()}\n'
            f'(confusion matrix simmetrizzata, diagonale=0)',
            fontsize=11,
        )
        plt.tight_layout()
        out_sim = output_dir / f"similarity_matrix_{feature_type}_{classifier}.png"
        plt.savefig(out_sim, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Similarity matrix salvata: {out_sim}")

        # ── Feature importance (solo Random Forest) ───────────────────────
        clf_step = pipeline.named_steps['clf']
        if hasattr(clf_step, 'feature_importances_'):
            importances = clf_step.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print(f"  Top 10 feature importance: "
                  f"{importances[top_idx].round(4).tolist()}")

    return {'accuracy': acc, 'report': report}


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
    parser.add_argument('--max_depth',     type=int,  default=None,
                        help='Max depth Random Forest (None=illimitato, es. 20 riduce overfitting)')
    parser.add_argument('--classifier',    type=str,  default='randomforest',
                        choices=['randomforest', 'svm'])
    parser.add_argument('--output_dir',    type=str,  default='results/classifier')
    parser.add_argument('--seed',          type=int,  default=42)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Classificatore compositore — {args.classifier.upper()}')
    print(f'Feature:     {args.feature_type}')
    print(f'Compositori: {len(args.composers)}')
    print(f'n_samples:   {args.n_samples} per classe')
    print(f'n_folds:     {args.n_folds}')
    print(f'max_depth:   {args.max_depth}')
    print('=' * 60)

    X, y, composers = load_dataset(
        features_dir, args.composers,
        args.feature_type, args.n_samples, args.seed,
    )

    cv_results = run_crossval(
        X, y, composers,
        classifier    = args.classifier,
        n_folds       = args.n_folds,
        n_estimators  = args.n_estimators,
        max_depth     = args.max_depth,
        seed          = args.seed,
    )

    final_results = run_final_fit(
        X, y, composers,
        classifier    = args.classifier,
        n_estimators  = args.n_estimators,
        max_depth     = args.max_depth,
        seed          = args.seed,
        output_dir    = output_dir,
        feature_type  = args.feature_type,
    )

    # Salva risultati
    out_txt = output_dir / f"results_{args.feature_type}_{args.classifier}.txt"
    with open(out_txt, 'w') as f:
        f.write(f"Feature:        {args.feature_type}\n")
        f.write(f"Classifier:     {args.classifier}\n")
        f.write(f"Compositori:    {args.composers}\n")
        f.write(f"n_samples/cl:   {args.n_samples}\n")
        f.write(f"max_depth:      {args.max_depth}\n\n")
        f.write(f"CV accuracy:    {cv_results['test_mean']:.3f} ± {cv_results['test_std']:.3f}\n")
        f.write(f"CV per fold:    {cv_results['folds']}\n\n")
        f.write(f"Final accuracy: {final_results['accuracy']:.3f}\n\n")
        f.write(final_results['report'])
    print(f"\n  Risultati salvati: {out_txt}")

    print('\n' + '=' * 60)
    print(f"RISULTATO FINALE — {args.feature_type.upper()} — {args.classifier.upper()}")
    print(f"CV accuracy: {cv_results['test_mean']:.3f} ± {cv_results['test_std']:.3f}")
    print('=' * 60)