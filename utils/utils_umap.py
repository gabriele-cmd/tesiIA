"""
utils/umap_utils.py
====================
Funzioni di supporto per visualizzazioni UMAP interattive.

Uso tipico negli script UMAP:

    from utils import attach_interactive_picker

    # Costruisci il plot normalmente, raccogliendo scatter_to_global
    scatter_to_global = {}
    for composer in composers:
        mask = labels == composer
        sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                        picker=True, pickradius=5, ...)
        scatter_to_global[sc] = np.where(mask)[0]

    # Attacca picker se richiesto
    if args.interactive:
        attach_interactive_picker(fig, ax, embedding, scatter_to_global, all_paths)
        plt.show()
    else:
        plt.savefig(output_path, dpi=130, bbox_inches='tight')
        plt.close()
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def attach_interactive_picker(
        fig,
        ax,
        embedding: np.ndarray,
        scatter_to_global: dict,
        paths: list[str],
) -> None:
    """
    Attacca un pick event a un plot UMAP già costruito.
    Click su un punto → stampa il path MIDI a terminale e
    evidenzia il punto selezionato con una stella rossa.

    Args:
        fig:               figura matplotlib
        ax:                asse matplotlib con scatter objects
        embedding:         array (N, 2) delle coordinate UMAP
        scatter_to_global: dict {scatter_artist: np.array di indici globali}
                           costruito durante il plot, un entry per compositore
        paths:             lista di N path MIDI, parallela a embedding
    """
    # Marker per il punto selezionato
    selected_marker, = ax.plot([], [], 'r*', markersize=15, zorder=10,
                                label='_nolegend_')

    # Testo info nell'angolo
    info_text = ax.text(
        0.02, 0.02, '',
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        zorder=11,
    )

    def on_pick(event):
        sc = event.artist
        if sc not in scatter_to_global:
            return

        local_idx  = event.ind[0]
        global_idx = int(scatter_to_global[sc][local_idx])

        midi_path = paths[global_idx] if global_idx < len(paths) \
                    else '[path non disponibile]'

        print(f'\n{"=" * 60}')
        print(f'Indice globale : {global_idx}')
        print(f'File MIDI      : {midi_path}')
        print(f'{"=" * 60}')

        # Aggiorna marker e testo
        x, y = embedding[global_idx, 0], embedding[global_idx, 1]
        selected_marker.set_data([x], [y])
        info_text.set_text(Path(midi_path).name if midi_path else '')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    print('\n[UMAP interattiva] Clicca su un punto per stampare il path MIDI.')