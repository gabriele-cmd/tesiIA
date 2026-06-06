"""
Estrae la feature KITM (Key-Indexed Transition Matrix) da chunk MIDI.

La KITM è una matrice 12x12 dove:
  - righe  = tonalità del chunk (0=C, 1=C#, ..., 11=B) — indice FISSO
  - colonne = istogramma degli intervalli melodici in semitoni (0-11)

Tre metodi di assegnazione della tonalità:

  kmeans:     K-means individua i cluster, argmax del centroide
              assegna la tonalità a ciascun cluster (C=0...B=11).
              Per un nuovo chunk: K-means predict → riga = tonalità
              del cluster tramite argmax. Biiezione 100% garantita
              sul full dataset.

  kmeans_ks:  K-means individua i cluster, KS sul centroide assegna
              la tonalità a ciascun cluster. Per un nuovo chunk:
              K-means predict → riga = tonalità del cluster tramite KS.
              Musicalmente più corretto, biiezione non garantita al 100%
              su subsampling (96% su 30k, 82% su 20k).

  ks:         KS direttamente sul PCH del singolo chunk → riga = tonalità.
              Nessun K-means. Musicalmente più corretto, ma più sensibile
              a frammenti ambigui o con modulazioni.

In tutti e tre i metodi le righe della matrice corrispondono a tonalità
musicali fisse (riga 0 = C, riga 1 = C#, ..., riga 11 = B).

Uso:
    python scripts/extract_kit_features.py \
        --chunks_dir data/chunks_real \
        --output     data/features_real \
        --composers  maestro_test \
        --method     kmeans \
        --model_path models/kmeans_key_final.pkl
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pickle
import pretty_midi
from pathlib import Path
from sklearn.preprocessing import normalize
from tqdm import tqdm

N_PITCH_CLASSES = 12
N_INTERVALS     = 12

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profili Krumhansl-Schmuckler per tonalità maggiori (Krumhansl 1990)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])


# ─────────────────────────────────────────────────────────────────────────────
# Calcolo PCH
# ─────────────────────────────────────────────────────────────────────────────
def compute_pch(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    counts = np.zeros(N_PITCH_CLASSES, dtype=np.float32)
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            counts[note.pitch % N_PITCH_CLASSES] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Assegnazione tonalità — i tre metodi
# ─────────────────────────────────────────────────────────────────────────────
def assign_key_kmeans(pch: np.ndarray, kmeans,
                      normalizer: str,
                      cluster_to_key: dict) -> int:
    """
    K-means predict → cluster_idx → tonalità via argmax.
    La mappa cluster_to_key_argmax garantisce biiezione sul full dataset.
    """
    if normalizer == 'l2':
        pch_norm = normalize(pch.reshape(1, -1), norm='l2')[0]
    else:
        pch_norm = pch
    cluster_idx = int(kmeans.predict(pch_norm.reshape(1, -1))[0])
    return cluster_to_key[cluster_idx]


def assign_key_kmeans_ks(pch: np.ndarray, kmeans,
                         normalizer: str,
                         cluster_to_key: dict) -> int:
    """
    K-means predict → cluster_idx → tonalità via KS sui centroidi.
    La mappa cluster_to_key_ks è musicalmente più corretta.
    """
    if normalizer == 'l2':
        pch_norm = normalize(pch.reshape(1, -1), norm='l2')[0]
    else:
        pch_norm = pch
    cluster_idx = int(kmeans.predict(pch_norm.reshape(1, -1))[0])
    return cluster_to_key[cluster_idx]


def assign_key_ks(pch: np.ndarray) -> int:
    """
    KS direttamente sul PCH del chunk — nessun K-means.
    Restituisce l'indice 0-11 della tonalità con massima correlazione KS.
    """
    best_r, best_key = -np.inf, 0
    for k in range(12):
        profile = np.roll(KS_MAJOR, k)
        if pch.std() == 0:
            continue
        r = np.corrcoef(pch, profile)[0, 1]
        if r > best_r:
            best_r, best_key = r, k
    return best_key


# ─────────────────────────────────────────────────────────────────────────────
# Istogramma intervalli melodici
# ─────────────────────────────────────────────────────────────────────────────
def compute_interval_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    all_notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            all_notes.append((note.start, note.pitch))
    all_notes.sort(key=lambda x: x[0])

    counts = np.zeros(N_INTERVALS, dtype=np.float32)
    for i in range(len(all_notes) - 1):
        interval = abs(all_notes[i + 1][1] - all_notes[i][1]) % N_INTERVALS
        counts[interval] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Feature KITM per un singolo chunk
# ─────────────────────────────────────────────────────────────────────────────
def compute_kit(midi_path: str, method: str,
                kmeans=None, normalizer: str = 'l2',
                cluster_to_key: dict = None) -> np.ndarray | None:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        if midi.get_end_time() < 2.0:
            return None

        pch      = compute_pch(midi)
        interval = compute_interval_histogram(midi)
        if interval.sum() == 0:
            return None

        # Assegna tonalità → indice riga (0=C, 1=C#, ..., 11=B)
        if method == 'kmeans':
            key_idx = assign_key_kmeans(pch, kmeans, normalizer, cluster_to_key)
        elif method == 'kmeans_ks':
            key_idx = assign_key_kmeans_ks(pch, kmeans, normalizer, cluster_to_key)
        else:  # ks
            key_idx = assign_key_ks(pch)

        # Matrice 12×12: solo la riga della tonalità assegnata è non-zero
        matrix = np.zeros((N_PITCH_CLASSES, N_INTERVALS), dtype=np.float32)
        matrix[key_idx] = interval
        return matrix.flatten()

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Elaborazione cartella
# ─────────────────────────────────────────────────────────────────────────────
def process_composer(chunks_dir: Path, composer: str, method: str,
                     kmeans=None, normalizer: str = 'l2',
                     cluster_to_key: dict = None) -> np.ndarray:
    folder = chunks_dir / composer
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")

    midi_files = sorted(folder.glob("*.mid")) + sorted(folder.glob("*.midi"))
    if not midi_files:
        raise ValueError(f"Nessun file MIDI in {folder}")

    print(f"\n  {composer}: {len(midi_files)} chunk trovati")
    features, skipped = [], 0

    for midi_path in tqdm(midi_files, desc=f"   KITM {composer}"):
        feat = compute_kit(str(midi_path), method, kmeans, normalizer, cluster_to_key)
        if feat is not None:
            features.append(feat)
        else:
            skipped += 1

    print(f"    -> {len(features)} chunk validi, {skipped} scartati")
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir",  type=str, required=True)
    parser.add_argument("--output",      type=str, required=True)
    parser.add_argument("--composers",   nargs="+",
                        default=['mozart', 'chopin', 'debussy'])
    parser.add_argument("--method",      type=str, default="kmeans",
                        choices=["kmeans", "kmeans_ks", "ks"],
                        help=(
                            "kmeans:    K-means + argmax sui centroidi (biiezione 100%)\n"
                            "kmeans_ks: K-means + KS sui centroidi (musicalmente più corretto)\n"
                            "ks:        KS diretto sul chunk (nessun K-means)"
                        ))
    parser.add_argument("--model_path",  type=str,
                        default="models/kmeans_key_final.pkl",
                        help="Percorso modello K-means (non usato per --method ks)")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"Estrazione feature KITM — metodo: {args.method.upper()}")
    print("=" * 55)

    kmeans, normalizer, cluster_to_key = None, 'l2', {}

    if args.method in ('kmeans', 'kmeans_ks'):
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Modello non trovato: {args.model_path}")
        with open(args.model_path, 'rb') as f:
            model_data = pickle.load(f)
        kmeans     = model_data['kmeans']
        normalizer = model_data.get('normalizer', 'l2')

        if args.method == 'kmeans':
            cluster_to_key = model_data.get('cluster_to_key_argmax', {})
            if not cluster_to_key:
                # Retrocompatibilità: modello vecchio senza mappa salvata
                print("  ATTENZIONE: mappa argmax non trovata nel modello.")
                print("  Riesegui train_kmeans_key.py per aggiornare il modello.")
                cluster_to_key = {i: int(np.argmax(c))
                                  for i, c in enumerate(kmeans.cluster_centers_)}
        else:  # kmeans_ks
            cluster_to_key = model_data.get('cluster_to_key_ks', {})
            if not cluster_to_key:
                print("  ATTENZIONE: mappa KS non trovata nel modello.")
                print("  Riesegui train_kmeans_key.py per aggiornare il modello.")
                KS = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                               2.52,5.19,2.39,3.66,2.29,2.88])
                for i, c in enumerate(kmeans.cluster_centers_):
                    best_r, best_k = -np.inf, 0
                    for k in range(12):
                        r = np.corrcoef(c, np.roll(KS, k))[0, 1]
                        if r > best_r:
                            best_r, best_k = r, k
                    cluster_to_key[i] = best_k

        print(f"Modello caricato: {args.model_path}")
        print(f"  k={model_data['n_clusters']}, normalizer={normalizer}")
        print(f"  Mappa cluster→tonalità ({args.method}):")
        NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        for ci, ki in sorted(cluster_to_key.items()):
            print(f"    Cluster {ci:2d} → riga {ki:2d} ({NOTE_NAMES[ki]})")

    suffix = f"_kitm_{args.method}"

    for composer in args.composers:
        features = process_composer(
            chunks_dir, composer, args.method, kmeans, normalizer, cluster_to_key
        )
        out_path = output_dir / f"{composer}{suffix}.npy"
        np.save(out_path, features)
        print(f" Salvato: {out_path} shape={features.shape}")

    print("\n" + "=" * 55)
    print("Estrazione completata.")
    print("=" * 55)