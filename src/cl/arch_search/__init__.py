"""
Architecture search for AWB (Adaptive Weight Basis).

Provides architecture search algorithms for finding optimal
architectures during continual learning.
"""

from .mlp_search import arch_search_MLP
from .cnn_search import arch_search_CNN, arch_search_CNN3D, prepABs, prepABs_CNN3D
from .gcn_search import arch_search_GCN, prepABs_GCN

__all__ = [
    'arch_search_MLP',
    'arch_search_CNN',
    'arch_search_CNN3D',
    'prepABs',
    'prepABs_CNN3D',
    # Added by Claude: GCN architecture search
    'arch_search_GCN',
    'prepABs_GCN',
]
