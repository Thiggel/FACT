import os
import sys

GRAPHSAINT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

sys.path.append(GRAPHSAINT_ROOT)

from .fairgraph_dataset import POKEC, NBA, ArtificialSensitiveGraphDataset

__all__ = [POKEC, NBA, ArtificialSensitiveGraphDataset]
