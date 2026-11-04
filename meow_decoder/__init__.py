"""Meow Decoder: guess what a cat's meow is about, with honest percentages.

    from meow_decoder import Decoder
    Decoder().predict("meow.wav")["results"]
"""
from .core import CONTEXTS, ENCODERS, Decoder, Head

__version__ = "0.2.0"
__all__ = ["Decoder", "Head", "CONTEXTS", "ENCODERS"]
