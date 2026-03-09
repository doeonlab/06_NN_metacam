"""Compatibility shim for the canonical src-layout package.

The real implementation lives in `src/metacam`.
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_src_pkg = Path(__file__).resolve().parents[1] / "src" / "metacam"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

from .physics.meta_operator import MetaOperator
from .physics.propagation import FieldPropagator, asm, asm_master_alltorch, sasm

__all__ = [
    "MetaOperator",
    "FieldPropagator",
    "asm",
    "asm_master_alltorch",
    "sasm",
]
