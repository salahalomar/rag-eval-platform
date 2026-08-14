"""The rag-eval-platform library.

Everything the system can do -- ingest, index, retrieve, generate -- lives here and
nowhere else. `apps/api` and `eval/` are both thin callers of this package, which is
what makes the claim "the evaluation harness measures the code the API serves"
checkable rather than aspirational.
"""

from rag.config import RetrievalConfig

__version__ = "0.1.0"

__all__ = ["RetrievalConfig", "__version__"]
