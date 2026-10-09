"""Result-decoding and normalization API for MATLAB-equivalent runs."""

from matlab_annotation.decode import (
    DecodedAnnotation,
    composition_dimensions,
    decode_annotation_code,
    lipid_class_from_template,
)
from matlab_annotation.isomer_resolver import resolve_positional_lyso_isomers

__all__ = [
    "DecodedAnnotation",
    "composition_dimensions",
    "decode_annotation_code",
    "lipid_class_from_template",
    "resolve_positional_lyso_isomers",
]
