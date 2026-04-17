#!/usr/bin/env python3
"""
Vocal separation module using Spleeter for audio source separation.
Separates vocals from background music in WAV/MP3 files.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from spleeter.separator import Separator


def load_audio(filepath: str, duration: float = 60.0):
    """
    Load audio file using librosa.

    Args:
        filepath: Path to the audio file (WAV or MP3)
        duration: Maximum duration to load in seconds

    Returns:
        Tuple of (audio_timeseries, sample_rate)

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    supported_extensions = {'.wav', '.mp3'}
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file format: {path.suffix}. "
                         f"Supported formats: {supported_extensions}")

    try:
        y, sr = librosa.load(filepath, sr=None, duration=duration)
        return y, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def save_audio(filepath: str, y: np.ndarray, sr: int):
    """
    Save audio file using soundfile.

    Args:
        filepath: Output file path
        y: Audio timeseries array
        sr: Sample rate
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    sf.write(filepath, y, sr)


def separate_vocals(input_path: str, output_vocal: str, output_accompaniment: str):
    """
    Separate vocals and accompaniment from an audio file using Spleeter.

    Args:
        input_path: Path to the input audio file (WAV or MP3)
        output_vocal: Path to save the separated vocals
        output_accompaniment: Path to save the accompaniment

    Returns:
        Tuple of (vocal_timeseries, accompaniment_timeseries, sample_rate)

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Initialize Spleeter separator with 2-stems model
    separator = Separator('spleeter:2stems')

    # Load audio with librosa
    y, sr = load_audio(input_path)

    # Convert mono to stereo if needed (Spleeter expects stereo input)
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=1)
    elif len(y.shape) == 2:
        # Ensure we have 2 channels
        if y.shape[1] == 1:
            y = np.concatenate([y, y], axis=1)
        elif y.shape[1] != 2:
            # Take first 2 channels if more than 2
            y = y[:, :2]

    # Perform separation
    result = separator.separate(y, sr)

    # Extract vocal and accompaniment
    vocal = result['vocals']
    accompaniment = result['accompaniment']

    # Save separated audio
    save_audio(output_vocal, vocal, sr)
    save_audio(output_accompaniment, accompaniment, sr)

    return vocal, accompaniment, sr


def main():
    """CLI entry point for vocal separation."""
    parser = argparse.ArgumentParser(
        description='Separate vocals from background music in audio files.'
    )
    parser.add_argument(
        'input',
        help='Input audio file path (WAV or MP3)'
    )
    parser.add_argument(
        '-v', '--vocal',
        required=True,
        help='Output path for separated vocals'
    )
    parser.add_argument(
        '-a', '--accompaniment',
        required=True,
        help='Output path for accompaniment'
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=60.0,
        help='Maximum duration to process in seconds (default: 60.0)'
    )

    args = parser.parse_args()

    try:
        vocal, accompaniment, sr = separate_vocals(
            args.input,
            args.vocal,
            args.accompaniment
        )
        print(f"Successfully separated vocals and accompaniment!")
        print(f"  Input: {args.input}")
        print(f"  Vocals: {args.vocal}")
        print(f"  Accompaniment: {args.accompaniment}")
        print(f"  Sample rate: {sr} Hz")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
