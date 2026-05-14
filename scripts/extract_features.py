"""
scripts/extract_features.py
============================
Estrae le feature Pitch Class Histogram (PCH) e Pitch Class Transition Matrix (PCTM) da ogni chunk MIDI.
Esecuzione:
    python3 scripts/extract_features.py --chunks_dir data/chunks --output data/features
"""
import argparse
import numpy as np
import pretty_midi
from pathlib import Path

from numpy.f2py.crackfortran import skipemptyends
from tqdm import tqdm

#12 Class Pitch
N_PITCH_CLASSES = 12
COMPOSERS = ['mozart', 'chopin', 'debussy', 'bach', 'beethoven']

#FEATURE 1 - Pitch Class Histogram (PCH)
#Conta quante volte appare ciascuna delle 12 note, catturando il profilo armonico del compositore
def pitch_class_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    counts = np.zeros(N_PITCH_CLASSES)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pc = note.pitch % N_PITCH_CLASSES
            counts[pc] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts

#FEATURE 2 - Pitch Class Transition Matrix (PCTM)
#Conta quante volte una nota è seguito da un'altra nota, cattura il profilo melodico del compositore
def pitch_class_transition_matrix(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    matrix = np.zeros((N_PITCH_CLASSES, N_PITCH_CLASSES))

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        #ordina le note per tempo di inizio
        notes = sorted(instrument.notes, key=lambda n: n.start)
        for i in range(len(notes) - 1):
            pc_curr = notes[i].pitch % N_PITCH_CLASSES
            pc_next = notes[i + 1].pitch % N_PITCH_CLASSES
            matrix[pc_curr, pc_next] += 1

    #Normalizza per riga
    row_sums = matrix.sum(axis=1, keepdims=True)
    #Evita divisione per zero per righe vuote
    row_sums[row_sums == 0] = 1
    matrix /= row_sums

    return matrix

#Estrazione feature da un singolo chunk
def extract_features(midi_path: str) -> tuple | None:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        #Verifica che ci siano note
        total_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
        if total_notes == 0:
            return None

        pch = pitch_class_histogram(midi) #[12]
        pctm = pitch_class_transition_matrix(midi) #[12, 12]

        return pch, pctm.flatten()

    except Exception:
        return None

#Elaborazione di una cartella di chunk
def process_composer(chunks_dir: Path, composer: str) -> tuple:
    folder = chunks_dir / composer
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")

    midi_files = sorted(folder.glob("*.mid")) + sorted(folder.glob("*.midi"))
    if len(midi_files) == 0:
        raise ValueError(f"Nessun file MIDI in {folder}")
    print(f"\n{composer}: {len(midi_files)} chunk trovati")

    pch_list = []
    pctm_list = []
    skipped = 0

    for midi_path in tqdm(midi_files, desc=f"   Estrazione {composer}"):
        feat = extract_features(str(midi_path))
        if feat is not None:
            pch, pctm = feat
            pch_list.append(pch)
            pctm_list.append(pctm)
        else:
            skipped += 1
    print(f"    -> {len(pch_list)} chunk validi, {skipped} scartati (vuoti o corrotti)")

    return np.array(pch_list, dtype=np.float32), np.array(pctm_list, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", required=True, help="Cartella con le sottocartelle dei chunk per compositore")
    parser.add_argument("--output", required=True, help="Cartella dove salvare i file .npy delle feature")
    parser.add_argument("--composers", nargs="+", default=['mozart', 'chopin', 'debussy'], help="Lista compositori da processare")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Estrazione feature PCH + PCTM da chunk MIDI")
    print("=" * 55)

    for composer in args.composers:
        pch_features, pctm_features = process_composer(chunks_dir, composer)
        out_pch = output_dir / f"{composer}_pch.npy"
        out_pctm = output_dir / f"{composer}_pctm.npy"

        np.save(out_pch, pch_features)
        np.save(out_pctm, pctm_features)

        print(f" Salvato: {out_pch} shape={pch_features.shape}")
        print(f" Salvato: {out_pctm} shape={pctm_features.shape}")

    print("\n" + "=" * 55)
    print("Estrazione completata.")
    print("=" * 55)