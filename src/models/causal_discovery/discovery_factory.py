from .pc_algorithm import run_pc_algorithm
from .grasp import run_grasp, threshold_grasp_matrix
import pandas as pd
from src.utils.graph_utils import remove_bidirectional_edges
from typing import Optional, Set, Tuple

class CausalDiscoveryFactory:
    def __init__(
        self,
        method: str = 'grasp',
        bidir_strategy: str = "random",
        bidir_strength: Optional[pd.DataFrame] = None,
        bidir_priority: Optional[Set[Tuple[str, str]]] = None,
        **kwargs
    ):
        self.method = method
        self.bidir_strategy = bidir_strategy
        self.bidir_strength = bidir_strength
        self.bidir_priority = bidir_priority
        self.params = kwargs
        
    def discover_graph(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'pc':
            adj = run_pc_algorithm(data, **self.params)
        elif self.method == 'grasp':
            adj = run_grasp(data, **self.params)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Remove bidirectional edges
        return remove_bidirectional_edges(
            adj,
            strategy=self.bidir_strategy,
            edge_strength=self.bidir_strength,
            priority_edges=self.bidir_priority
        )

    @staticmethod
    def compare_methods(data: pd.DataFrame) -> dict:
        """Compare PC vs GRaSP"""
        return {
            'pc': run_pc_algorithm(data),
            'grasp': threshold_grasp_matrix(run_grasp(data))
        }