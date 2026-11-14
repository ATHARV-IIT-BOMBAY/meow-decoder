"""Meow Decoder: guess what a cat's meow is about, with honest percentages.

from meow_decoder import Decoder
Decoder().predict("meow.wav")["results"]
"""

from importlib.metadata import PackageNotFoundError, version

from .core import CONTEXTS, ENCODERS, PRESETS, SIGNALS, Decoder, Head

try:
    __version__ = version("meow-decoder")  # single source: pyproject.toml
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "unknown"
__all__ = ["Decoder", "Head", "CONTEXTS", "PRESETS", "SIGNALS", "ENCODERS"]
