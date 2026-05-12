"""
scripts/prepare_composers.py

Legge il .csv di Maestro e copia i file MIDI dei compositori scelti in cartelle separate

Esecuzione:
    python3 scripts/prepare_composers.py --csv data/maestro-v3.0.0/maestro-v3.0.0.csv --midi data/maestro-v3.0.0 --output data/composers
"""
import os
import shutil
import argparse
import pandas as pd
from pathlib import Path

#Mappa nome cartella -> stringa da cercare nel .csv
COMPOSERS = {
    'mozart': 'Mozart',
    'chopin': 'Chopin',
    'debussy': 'Debussy',
}

def prepare_composers(csv_path: str, midi_root: str, output_dir: str):
    df = pd.read_csv(csv_path)

    print(f"Totale brani nel .csv: {len(df)}")
    print(f"Colonne: {list(df.columns)}\n")

    #Mostra i composiori disponibili
    print("Compositori nel dataset:")
    for c, count in df['canonical_composer'].value_counts().items():
        print(f" {c}: {count} brani")
    print()

    midi_path = Path(midi_root)
    output_dir = Path(output_dir)

    for folder_name, search_str in COMPOSERS.items():
        #Filtra per compositore
        mask = df['canonical_composer'].str.contains(search_str, case=False, na=False)
        subset = df[mask]

        if len(subset) == 0:
            print(f"ATTENZIONE: nessun brano trovato per '{search_str}'")
            continue

        #Crea cartella output
        dest = output_dir / folder_name
        dest.mkdir(parents=True, exist_ok=True)

        print(f"{folder_name} ({search_str}): {len(subset)} brani trovati")

        copied = 0
        for _, row in subset.iterrows():
            #Il .csv ha una colonna 'midi_filename' con path relativo
            src = midi_path / row['midi_filename']
            if src.exists():
                shutil.copy2(src, dest / src.name)
                copied += 1
            else:
                print(f" File non trovato: {src}")
        print(f" -> {copied} file copiati in {dest}\n")

    print("Preparazione completata.")
    print(f"\nCartelle create in {output_dir}:")
    for folder_name in COMPOSERS:
        folder = output_dir / folder_name
        if folder.exists():
            n = len(list(folder.glob("*.mid"))) + len(list(folder.glob("*.midi")))
            print(f" {folder_name}/: {n} file MIDI")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path al CSV di Maestro")
    parser.add_argument("--midi", required=True, help="Cartella root con i file MIDI di Maestro")
    parser.add_argument("--output", required=True, help="Cartella output per i compositori")
    args = parser.parse_args()

    prepare_composers(args.csv, args.midi, args.output)