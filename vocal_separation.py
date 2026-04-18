#!/usr/bin/env python3
"""
Vocal separation module using Demucs for audio source separation.
Separates vocals from background music in WAV/MP3 files.

Requires: torch, torchaudio, demucs
GPU推荐: CUDA 12.1+ (RTX 3080 Ti 등)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import demucs
from demucs.apply import apply_model
from demucs.hdemucs import HDemucs
from demucs.utils import dispatch
import torch


# Module-level constant for model name
MODEL_NAME = 'htdemucs'


def load_audio(filepath: str):
    """
    Load audio file using soundfile.

    Args:
        filepath: Path to the audio file (WAV or MP3)

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor is (channels, time)

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
        y, sr = sf.read(filepath, dtype='float32')
        # Convert to (channels, time) format
        if y.ndim == 1:
            y = y[np.newaxis, :]
        else:
            y = y.T
        return torch.from_numpy(y), sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def save_audio(filepath: str, y: torch.Tensor, sr: int):
    """
    Save audio file using soundfile.

    Args:
        filepath: Output file path
        y: Audio tensor (channels, time)
        sr: Sample rate
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # Convert from (channels, time) to (time, channels) for soundfile
    y_np = y.numpy().T
    sf.write(filepath, y_np, sr)


def separate_vocals(input_path: str, output_vocal: str, output_accompaniment: str):
    """
    Separate vocals and accompaniment from an audio file using Demucs.

    Args:
        input_path: Path to the input audio file (WAV or MP3)
        output_vocal: Path to save the separated vocals
        output_accompaniment: Path to save the accompaniment

    Returns:
        Tuple of (vocal_tensor, accompaniment_tensor, sample_rate)

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load audio
    audio, sr = load_audio(input_path)

    # Load Demucs model
    model = HDemucs()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        demucs.__pretrained__.get(MODEL_NAME, demucs.__pretrained__[MODEL_NAME]),
        map_location='cpu'
    ))
    model.eval()

    # Dispatch to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    audio = audio.to(device)

    # Apply separation
    with torch.no_grad():
        separated = apply_model(model, audio[None, ...])[0]

    # Get vocals (first channel)
    vocal = separated[0]

    # Get accompaniment (sum of remaining stems)
    accompaniment = separated[1:].sum(dim=0)

    # Save outputs
    save_audio(output_vocal, vocal.cpu(), sr)
    save_audio(output_accompaniment, accompaniment.cpu(), sr)

    return vocal.cpu(), accompaniment.cpu(), sr


def main():
    """CLI entry point for vocal separation."""
    parser = argparse.ArgumentParser(
        description='Separate vocals from background music in audio files using Demucs.'
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
