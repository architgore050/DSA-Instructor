"""DSA Mentor — a RAG-grounded DSA instructor (see spec.md for the design).

This package is intentionally import-light: submodules (``config``, ``llm``, …)
are imported explicitly by consumers so that importing the package never pulls
in third-party dependencies.
"""

__version__ = "0.1.0"
