"""
Classificatore compositore su feature PCH/PCTM.
Fornisce una misura quantitativa della capacità discriminativa delle feature,
complementare alla visualizzazione UMAP.

Uso:
    python experiments/composer_classifier.py \
        --features_dir data/features \
        --feature_type pctm \
        --composers chopin schubert bach liszt beethoven \
                    schumann rachmaninoff debussy mozart haydn \
        --n_samples 1500 \
        --output_dir results/classifier
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline


COMPOSERS_DEFAULT = [
    'chopin', 'schubert', 'bach', 'liszt', 'beethoven',
    'schumann', 'rachmaninoff', 'debussy', 'mozart', 'haydn',
]

# Caricamento feature
def load_dataset(features_dir: Path,
                 composers: list,
                 feature_type: str,
                 n_samples: int,
                 seed: int = 42) -> tuple:
    """
    Carica e bilancia le feature di tutti i compositori.
    Restituisce X (N, d), y (N,) con label intere, e lista nomi classi.
    """
    X_list, y_list = [], []
    rng = np.random.default_rng(seed)

    print(f"\nCaricamento feature '{feature_type}':")
    for i, composer in enumerate(composers):
        path = features_dir / f"{composer}_{feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Feature non trovate: {path}")
        data = np.load(path).astype(np.float32)

        # Subsample bilanciato
        n = min(n_samples, len(data))
        idx = rng.choice(len(data), n, replace=False)
        data = data[idx]

        X_list.append(data)
        y_list.extend([i] * len(data))
        print(f"  {composer:15s}: {len(data):5d} chunk")

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"\n  Totale: {X.shape[0]} chunk, {X.shape[1]} dim, {len(composers)} classi")
    return X, y, composers


#Classificatore
def build_pipeline(n_estimators: int = 200, seed: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth    = None,
            n_jobs       = -1,
            random_state = seed,
            class_weight = 'balanced',
        )),
    ])


def run_crossval(X: np.ndarray,
                 y: np.ndarray,
                 composers: list,
                 n_folds: int = 5,
                 n_estimators: int = 200,
                 seed: int = 42) -> dict:
    """
    Stratified K-Fold cross-validation.
    Restituisce dizionario con accuracy per fold e media±std.
    """
    pipeline = build_pipeline(n_estimators, seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print(f"\nCross-validation ({n_folds} fold, stratificato)...")
    results = cross_validate(
        pipeline, X, y,
        cv      = cv,
        scoring = 'accuracy',
        return_train_score = True,
        verbose = 0,
        n_jobs  = -1,
    )

    test_acc  = results['test_score']
    train_acc = results['train_score']

    print(f"\n  Train accuracy: {train_acc.mean():.3f} ± {train_acc.std():.3f}")
    print(f"  Test  accuracy: {test_acc.mean():.3f} ± {test_acc.std():.3f}")
    print(f"  Per fold:       {[f'{a:.3f}' for a in test_acc]}")

    return {
        'test_mean':  float(test_acc.mean()),
        'test_std':   float(test_acc.std()),
        'train_mean': float(train_acc.mean()),
        'train_std':  float(train_acc.std()),
        'folds':      test_acc.tolist(),
    }


def run_final_fit(X: np.ndarray,
                  y: np.ndarray,
                  composers: list,
                  n_estimators: int = 200,
                  seed: int = 42,
                  output_dir: Path = None,
                  feature_type: str = '') -> dict:
    """
    Fit finale su 80% dei dati, valutazione sul 20%.
    Produce confusion matrix e report per classe.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    pipeline = build_pipeline(n_estimators, seed)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = (y_pred == y_test).mean()
    print(f"\n  Test accuracy (80/20 split): {acc:.3f}")

    # Report per classe
    report = classification_report(
        y_test, y_pred,
        target_names=composers,
        digits=3,
    )
    print(f"\n{report}")

    # Confusion matrix
    if output_dir is not None:
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[c.capitalize() for c in composers],
        )
        disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
        ax.set_title(
            f'Confusion Matrix — {feature_type.upper()}\n'
            f'Random Forest (n_estimators={n_estimators})\n'
            f'Accuracy = {acc:.3f}',
            fontsize=11,
        )
        plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()

        out_path = output_dir / f"confusion_matrix_{feature_type}.png"
        plt.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"\n  Confusion matrix salvata: {out_path}")

    # Feature importance (top 20)
    rf = pipeline.named_steps['clf']
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print(f"\n  Top 10 feature importance: "
          f"{importances[top_idx[:10]].round(4).tolist()}")

    return {'accuracy': acc, 'report': report}

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir',  type=str, default='data/features')
    parser.add_argument('--feature_type',  type=str, default='pctm',
                        help='Feature da usare: pch, pctm, pctm_w3.0_h1.5_concat, ...')
    parser.add_argument('--composers',     nargs='+', default=COMPOSERS_DEFAULT)
    parser.add_argument('--n_samples',     type=int,  default=1500,
                        help='Chunk per compositore (bilanciamento)')
    parser.add_argument('--n_folds',       type=int,  default=5)
    parser.add_argument('--n_estimators',  type=int,  default=200)
    parser.add_argument('--output_dir',    type=str,  default='results/classifier')
    parser.add_argument('--seed',          type=int,  default=42)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Classificatore compositore — Random Forest')
    print(f'Feature:     {args.feature_type}')
    print(f'Compositori: {len(args.composers)}')
    print(f'n_samples:   {args.n_samples} per classe')
    print(f'n_folds:     {args.n_folds}')
    print('=' * 60)

    # Carica dataset
    X, y, composers = load_dataset(
        features_dir, args.composers,
        args.feature_type, args.n_samples, args.seed,
    )

    # Cross-validation
    cv_results = run_crossval(
        X, y, composers,
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    # Fit finale + confusion matrix
    final_results = run_final_fit(
        X, y, composers,
        n_estimators=args.n_estimators,
        seed=args.seed,
        output_dir=output_dir,
        feature_type=args.feature_type,
    )

    # Salva risultati numerici
    out_txt = output_dir / f"results_{args.feature_type}.txt"
    with open(out_txt, 'w') as f:
        f.write(f"Feature:        {args.feature_type}\n")
        f.write(f"Compositori:    {args.composers}\n")
        f.write(f"n_samples/cl:   {args.n_samples}\n\n")
        f.write(f"CV accuracy:    {cv_results['test_mean']:.3f} ± {cv_results['test_std']:.3f}\n")
        f.write(f"CV per fold:    {cv_results['folds']}\n\n")
        f.write(f"Final accuracy: {final_results['accuracy']:.3f}\n\n")
        f.write(final_results['report'])
    print(f"\n  Risultati salvati: {out_txt}")

    print('\n' + '=' * 60)
    print(f"RISULTATO FINALE — {args.feature_type.upper()}")
    print(f"CV accuracy: {cv_results['test_mean']:.3f} ± {cv_results['test_std']:.3f}")
    print('=' * 60)