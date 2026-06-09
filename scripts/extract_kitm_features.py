"""
Estrae la feature KITM (Key-Indexed Transition Matrix) da chunk MIDI.

La KITM è una matrice 12x12 dove:
  - righe  = cluster tonale assegnato al chunk (0-11)
  - colonne = istogramma degli intervalli melodici in semitoni (0-11)

Per ogni chunk:
  1. Si assegna il cluster tonale tramite K-means sul PCH (con classificazione argmax)
     oppure tramite Krumhansl-Schmuckler (KS)
  2. Si calcolano gli intervalli tra note consecutive (mod 12)
  3. Si riempie la riga corrispondente al cluster con l'istogramma

Il vettore feature finale è la matrice 12x12 appiattita = 144 dimensioni.
Solo la riga del cluster assegnato è non-zero per chunk.

Uso:
    # K-means + classificazione argmax
    python scripts/extract_kit_features.py \
        --chunks_dir    data/chunks \
        --output        data/features \
        --composers     mozart chopin debussy \
        --model_path    models/kmeans_key_final.pkl \
        --method        kmeans

    # algoritmo KS diretto (senza K-means)
    python scripts/extract_kit_features.py \
        --chunks_dir    data/chunks \
        --output        data/features \
        --composers     mozart chopin debussy \
        --method        ks
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

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

#Profili Krumhansl-Schmuckler per tonalità maggiori
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

#Calcolo PCH di frammenti MIDI
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

#Assegnazione cluster con K-means + argmax
def assign_cluster_kmeans(pch: np.ndarray, kmeans, normalizer: str) -> int:
    if normalizer == 'l2':
        pch_norm = normalize(pch.reshape(1, -1), norm='l2')[0]
    else:
        pch_norm = pch
    return int(kmeans.predict(pch_norm.reshape(1, -1))[0])

#Assegnazione tonalità con KS diretto
def assign_cluster_ks(pch: np.ndarray) -> int:
    best_r, best_key = -np.inf, 0
    for k in range(12):
        profile = np.roll(KS_MAJOR, k) #ruota il profilo per ogni tonalità
        if pch.std() == 0:
            continue
        r = np.corrcoef(pch, profile)[0, 1] #calcola correlazione con quel profilo
        if r > best_r:
            best_r, best_key = r, k #trova la correlazione migliore tra quel profilo e la tonalità corrente
    return best_key #indice 0-11 della tonalità più simile

#Calcolo istrogramma intervalli melodici
def compute_interval_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Conta gli intervalli in semitoni (mod 12) tra note consecutive.
    Restituisce un vettore normalizzato di 12 dimensioni.
    """
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

#Calcolo feature KIT per un singolo chunk
def compute_kit(midi_path: str, method: str,
                kmeans=None, normalizer: str = 'l2') -> np.ndarray | None:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)

        if midi.get_end_time() < 2.0:
            return None

        pch = compute_pch(midi)
        interval = compute_interval_histogram(midi)

        if interval.sum() == 0:
            return None

        #Assegna cluster
        if method == 'kmeans':
            cluster = assign_cluster_kmeans(pch, kmeans, normalizer)
        else:
            cluster = assign_cluster_ks(pch)

        #Costruisce matrice 12x12 — solo la riga del cluster è non-zero
        matrix = np.zeros((N_PITCH_CLASSES, N_INTERVALS), dtype=np.float32)
        matrix[cluster] = interval

        return matrix.flatten()

    except Exception as e:
        return None

#Elaborazione di una cartella di frammenti
def process_composer(chunks_dir: Path, composer: str, method: str,
                     kmeans=None, normalizer: str = 'l2') -> tuple:
    folder = chunks_dir / composer
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")

    midi_files = sorted(folder.glob("*.mid")) + sorted(folder.glob("*.midi"))
    if not midi_files:
        raise ValueError(f"Nessun file MIDI in {folder}")

    print(f"\n  {composer}: {len(midi_files)} chunk trovati")

    features = []
    skipped = 0
    valid_paths = []

    for midi_path in tqdm(midi_files, desc=f"   KITM {composer}"):
        feat = compute_kit(str(midi_path), method, kmeans, normalizer)
        if feat is not None:
            features.append(feat)
            valid_paths.append(str(midi_path))
        else:
            skipped += 1

    print(f"    -> {len(features)} chunk validi, {skipped} scartati")
    return np.array(features, dtype=np.float32), valid_paths

#MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--composers", nargs="+",
                        default=['mozart', 'chopin', 'debussy'])
    parser.add_argument("--method", type=str, default="kmeans",
                        choices=["kmeans", "ks"],
                        help="kmeans: K-means + argmax | ks: Krumhansl-Schmuckler diretto")
    parser.add_argument("--model_path", type=str,
                        default="models/kmeans_key_final.pkl",
                        help="Percorso modello K-means (solo per --method kmeans)")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"Estrazione feature KITM — metodo: {args.method.upper()}")
    print("=" * 55)

    #Carica modello K-means se necessario
    kmeans = None
    normalizer = 'l2'
    if args.method == 'kmeans':
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Modello non trovato: {args.model_path}")
        with open(args.model_path, 'rb') as f:
            model_data = pickle.load(f)
        kmeans = model_data['kmeans']
        normalizer = model_data.get('normalizer', 'l2')
        print(f"Modello caricato: {args.model_path}")
        print(f"  k={model_data['n_clusters']}, "
              f"normalizer={normalizer}, "
              f"compositori training={model_data['composers']}")

    #Estrae la nuova feature per ogni compositore
    suffix = f"_kitm_{args.method}"

    for composer in args.composers:
        features, valid_paths = process_composer(
            chunks_dir, composer, args.method, kmeans, normalizer
        )
        out_path = output_dir / f"{composer}{suffix}.npy"
        np.save(out_path, features)
        print(f" Salvato: {out_path} shape={features.shape}")

        np.save(out_path, features)
        manifest_path = output_dir / f"{composer}{suffix}_manifest.txt"
        with open(manifest_path, 'w') as f:
            f.write('\n'.join(valid_paths))
        print(f" Manifest: {manifest_path} ({len(valid_paths)} chunk)")

    print("\n" + "=" * 55)
    print("Estrazione completata.")
    print("=" * 55)