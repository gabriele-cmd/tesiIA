"""
utils/manifest.py
==================
Salvataggio e caricamento dei manifest MIDI.

Un manifest è un file .txt con un path MIDI per riga,
in ordine parallelo ai corrispondenti vettori feature .npy.
"""

from pathlib import Path


def save_manifest(output_dir: str | Path,
                  composer: str,
                  suffix: str,
                  valid_paths: list[str]) -> Path:
    """
    Salva il manifest per un compositore e un tipo di feature.

    Args:
        output_dir:   cartella output (stessa dei .npy)
        composer:     nome compositore (es. 'maestro_test')
        suffix:       suffisso feature (es. '_pctm_w3.0_h1.5_concat' o '' per statiche)
        valid_paths:  lista di path MIDI validi, parallela ai vettori feature

    Returns:
        Path del file manifest salvato
    """
    path = Path(output_dir) / f"{composer}{suffix}_manifest.txt"
    with open(path, 'w') as f:
        f.write('\n'.join(str(p) for p in valid_paths))
    return path


def load_manifest(manifest_dir: str | Path,
                  composer: str,
                  feature_suffix: str) -> list[str] | None:
    """
    Carica il manifest per un compositore e tipo di feature.

    Args:
        manifest_dir:   cartella in cui cercare il manifest
        composer:       nome compositore
        feature_suffix: suffisso feature (es. '_pctm_w3.0_h1.5_concat')

    Returns:
        Lista di path MIDI, oppure None se il manifest non esiste
    """
    path = Path(manifest_dir) / f"{composer}{feature_suffix}_manifest.txt"
    if not path.exists():
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]