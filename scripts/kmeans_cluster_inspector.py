"""
Trova i chunk più rappresentativi di ogni cluster K-means
(i più vicini al centroide) e stampa i loro path MIDI.

Uso:
    python scripts/kmeans_cluster_inspector.py \
        --model_path    models/kmeans_key_final.pkl \
        --features_dir  data/features \
        --manifest_dir  data/features \
        --composer      mozart \
        --feature_type  pch \
        --n_per_cluster 3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',    type=str,
                        default='models/kmeans_key_final.pkl')
    parser.add_argument('--features_dir',  type=str,
                        default='data/features')
    parser.add_argument('--manifest_dir',  type=str, default=None)
    parser.add_argument('--composer',      type=str, default='mozart')
    parser.add_argument('--feature_type',  type=str, default='pch')
    parser.add_argument('--n_per_cluster', type=int, default=3)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir) if args.manifest_dir \
                   else Path(args.features_dir)

    # Carica modello K-means
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    kmeans                = model_data['kmeans']
    normalizer            = model_data.get('normalizer', 'l2')
    cluster_to_key_argmax = model_data.get('cluster_to_key_argmax', {})
    cluster_to_key_ks     = model_data.get('cluster_to_key_ks', {})

    # Carica feature PCH
    feat_path = Path(args.features_dir) / \
                f'{args.composer}_{args.feature_type}.npy'
    if not feat_path.exists():
        raise FileNotFoundError(f'Feature non trovate: {feat_path}')
    feats = np.load(feat_path).astype(np.float32)
    print(f'Feature caricate: {feats.shape}')

    # Carica manifest — prova prima con feature_type, poi senza prefisso
    manifest_path = manifest_dir / f'{args.composer}_{args.feature_type}_manifest.txt'
    if not manifest_path.exists():
        # fallback: manifest senza prefisso tipo feature
        manifest_path = manifest_dir / f'{args.composer}_manifest.txt'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Manifest non trovato. '
            f'Riesegui estrazione con salvataggio manifest.'
        )
    with open(manifest_path) as f:
        paths = [line.strip() for line in f if line.strip()]
    assert len(paths) == len(feats), \
        f'Manifest e feature hanno lunghezze diverse: {len(paths)} vs {len(feats)}'

    # Normalizza e predici cluster
    if normalizer == 'l2':
        feats_norm = normalize(feats, norm='l2')
    else:
        feats_norm = feats

    cluster_labels = kmeans.predict(feats_norm)

    print(f'\n{"="*60}')
    print(f'ISPEZIONE CLUSTER K-MEANS — {args.composer.upper()}')
    print(f'Modello: {args.model_path}  |  n_init=20')
    print(f'{"="*60}')

    for cluster_idx in range(kmeans.n_clusters):
        mask    = cluster_labels == cluster_idx
        n_chunk = mask.sum()
        key_argmax = cluster_to_key_argmax.get(cluster_idx, -1)
        key_ks     = cluster_to_key_ks.get(cluster_idx, -1)
        key_name   = NOTE_NAMES[key_argmax] if key_argmax >= 0 else '?'
        key_ks_name = NOTE_NAMES[key_ks] if key_ks >= 0 else '?'
        match = '✓' if key_argmax == key_ks else '✗'
        centroid = kmeans.cluster_centers_[cluster_idx]

        print(f'\nCluster {cluster_idx:2d} → {key_name} maj '
              f'(KS={key_ks_name} {match})  n={n_chunk}')

        if n_chunk == 0:
            print('  [cluster vuoto per questo compositore]')
            continue

        # Distanza dal centroide
        chunk_feats = feats_norm[mask]
        dists       = np.linalg.norm(chunk_feats - centroid, axis=1)
        sorted_idx  = np.argsort(dists)

        original_indices = np.where(mask)[0]
        n_show = min(args.n_per_cluster, n_chunk)

        for rank, i in enumerate(sorted_idx[:n_show]):
            orig_idx = original_indices[i]
            print(f'  {rank+1}. dist={dists[i]:.4f}  {paths[orig_idx]}')

    print(f'\n{"="*60}')
    print('Apri i file .mid in Reaper per ascoltare i cluster.')
    print(f'{"="*60}')