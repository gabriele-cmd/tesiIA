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

#FEATURE 3 - Average Pitch
#media delle altezze MIDI (0-127) di tutte le note del chunk. Cattura il registro medio usato dal compositore
def average_pitch(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    pitches = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitches.append(note.pitch)
    if len(pitches) == 0:
        return np.array([0.0])
    return np.array([np.mean(pitches) /127.0]) #normalizzato in [0,1]

#FEATURE 4 - Pitch Range
#differenza tra nota più alta e più bassa nel chunk. Cattura l'ampiezza del registro usato
def pitch_range(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    pitches = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitches.append(note.pitch)
    if len(pitches) == 0:
        return np.array([0.0])
    return np.array([(max(pitches) - min(pitches)) / 127.0]) #normalizzato in [0,1]

#FEATURE 5 - Average Pitch Interval
#intervallo medio tra note consecutive. Cattura il movimento melodico
def average_pitch_interval(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    intervals = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)
        for i in range(len(notes) - 1):
            intervals.append(abs(notes[i+1].pitch - notes[i].pitch))
    if len(intervals) == 0:
        return np.array([0.0])
    return np.array([np.mean(intervals) / 127.0]) #normalizzato in [0,1]

# FEATURE 6 — Interval Histogram
#distribuzione degli intervalli melodici. Cattura quanto spesso la melodia si muove per gradi vs salti.
def interval_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    counts = np.zeros(13)
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)
        for i in range(len(notes) - 1):
            interval = abs(notes[i+1].pitch - notes[i].pitch)
            if interval <= 12:
                counts[interval] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts

#FEATURE 7 - Note Lenght Histogram (NLH)
#Categorie ritmiche come da Yang & Lerch
NOTE_LENGHTS = [
    4.0, #full note
    2.0, #half note
    1.0, #quarter note
    0.5, #8th note
    0.25, #16th note
    3.0, #dotted half
    1.5, #dotted quarter
    0.75, #dotted 8th
    0.375,#dotted 16th
    2/3, #half triplet
    1/3, #quarter triplet
    1/6, #8th triplet
]
N_NOTE_LENGTHS = len(NOTE_LENGHTS)

#Quantizza una durata in beats alla categoria ritmica più vicina
def quantize_duration(duration_beats: float) -> int:
    distances = [abs(duration_beats - l) for l in NOTE_LENGHTS]
    return int(np.argmin(distances))
#Calcola il NLH normalizzato. Ogni nota è quantizzata alla categoria ritmica a lei più vicina
def note_length_histogram(
        midi: pretty_midi.PrettyMIDI,
) -> np.ndarray:
    counts = np.zeros(N_NOTE_LENGTHS)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            #Durata in secondi
            duration_sec = note.end - note.start
            #Converte secondi in beats usando il tempo del brano
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[1]) > 0:
                bpm = tempo_changes[1][0]
            else:
                bpm = 120.0 #DEFAULT
            beats_per_sec = bpm / 60.0
            duration_beats = duration_sec * beats_per_sec

            idx = quantize_duration(duration_beats)
            counts[idx] += 1

    total = counts.max()
    if total > 0:
        counts /= total
    return counts

#FEATURE 8 - Note Length Transition Matrix (NLTM)
#Calcola la NLTM normalizzata per riga
def note_length_transition_matrix(
        midi: pretty_midi.PrettyMIDI,
) -> np.ndarray:
    matrix = np.zeros((N_NOTE_LENGTHS, N_NOTE_LENGTHS))

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)

        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            bpm = tempo_changes[1][0]
        else:
            bpm = 120.0
        beats_per_sec = bpm / 60.0

        for i in range(len(notes) - 1):
            dur_curr = (notes[i].end - notes[i].start) * beats_per_sec
            dur_next = (notes[i+1].end - notes[i+1].start) * beats_per_sec
            idx_curr = quantize_duration(dur_curr)
            idx_next = quantize_duration(dur_next)
            matrix[idx_curr, idx_next] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
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
        avg_p = average_pitch(midi) #[1]
        p_range = pitch_range(midi) #[1]
        avg_int = average_pitch_interval(midi) #[1]
        ih = interval_histogram(midi)  # [13]
        nlh = note_length_histogram(midi) # [12]
        nltm = note_length_transition_matrix(midi) #[12, 12]

        return pch, pctm.flatten(), avg_p, p_range, avg_int, ih, nlh, nltm.flatten()

    except Exception:
        return None

#Elaborazione di una cartella di chunk
def process_composer(chunks_dir: Path, composer: str) -> dict:
    folder = chunks_dir / composer
    if not folder.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder}")

    midi_files = sorted(folder.glob("*.mid")) + sorted(folder.glob("*.midi"))
    if len(midi_files) == 0:
        raise ValueError(f"Nessun file MIDI in {folder}")
    print(f"\n{composer}: {len(midi_files)} chunk trovati")

    lists = {
        'pch': [],
        'pctm': [],
        'avg_pitch': [],
        'pitch_range': [],
        'avg_interval': [],
        'interval_histogram': [],
        'nlh': [],
        'nltm': [],
    }
    skipped = 0

    for midi_path in tqdm(midi_files, desc=f"   Estrazione {composer}"):
        feat = extract_features(str(midi_path))
        if feat is not None:
            pch, pctm, avg_p, p_range, avg_int, ih, nltm, nlh = feat
            lists['pch'].append(pch)
            lists['pctm'].append(pctm)
            lists['avg_pitch'].append(avg_p)
            lists['pitch_range'].append(p_range)
            lists['avg_interval'].append(avg_int)
            lists['interval_histogram'].append(ih)
            lists['nlh'].append(nlh)
            lists['nltm'].append(nltm)
        else:
            skipped += 1
    n_valid = len(lists['pch'])
    print(f"    -> {n_valid} chunk validi, {skipped} scartati")

    return {k: np.array(v, dtype=np.float32) for k, v in lists.items()}


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
        feat_dict = process_composer(chunks_dir, composer)

        for feat_name, data in feat_dict.items():
            out_path = output_dir / f"{composer}_{feat_name}.npy"
            np.save(out_path, data)
            print(f" Salvato: {out_path} shape={data.shape}")

    print("\n" + "=" * 55)
    print("Estrazione completata.")
    print("=" * 55)