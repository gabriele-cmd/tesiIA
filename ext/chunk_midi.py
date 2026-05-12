import os
from pathlib import Path
from note_seq import midi_io, sequences_lib
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def split_midi_by_duration(midi_file, output_dir, chunk_duration=10.24):
    """
    Splits a MIDI file into chunks of specified duration and saves them in the output directory.
    """
    try:
        note_sequence = midi_io.midi_file_to_note_sequence(midi_file)
        total_time = note_sequence.total_time
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_start = 0
        chunk_index = 0
        while chunk_start < total_time:
            chunk_end = min(chunk_start + chunk_duration, total_time)
            chunk_sequence = sequences_lib.extract_subsequence(note_sequence, chunk_start, chunk_end)
            chunk_filename = f"{Path(midi_file).stem}_chunk_{chunk_index}.mid"
            output_path = output_dir / chunk_filename
            midi_io.note_sequence_to_midi_file(chunk_sequence, str(output_path))
            chunk_index += 1
            chunk_start = chunk_end
        return midi_file
    except Exception as e:
        return f"Error processing {midi_file}: {e}"


def process_midi_directory(source_dir, output_dir, chunk_duration=10.24, num_workers=4):
    """
    Processes all MIDI files in a directory and splits them into chunks in parallel.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    midi_files = list(source_dir.rglob("*.mid")) + list(source_dir.rglob("*.midi"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(split_midi_by_duration, midi_file, output_dir, chunk_duration)
            for midi_file in midi_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MIDI files"):
            result = future.result()
            if isinstance(result, str) and result.startswith("Error"):
                print(result)

    print("Splitting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk MIDI files into specified lengths.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing the MIDI files to chunk.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the chunked MIDI files will be saved.")
    parser.add_argument("--chunk_length", type=float, default=10.24, help="Length to chunk the MIDI file to (s).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    process_midi_directory(args.input_path, args.output_dir, chunk_duration=args.chunk_length, num_workers=args.num_workers)
