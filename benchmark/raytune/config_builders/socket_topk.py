"""Configuration builder for SOCKET TopK attention."""

from typing import Dict, List, Optional, Tuple

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    SocketMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("socket_topk")
class SocketTopKConfigBuilder(BaseConfigBuilder):
    """Builder for Socket TopK sparse attention configurations."""

    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs,
    ) -> Tuple[
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
    ]:
        """Get all Socket TopK attention configurations.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
            memory_objectives: List[int] - List of memory objectives to build the configurations.
        Ignores:
            model_config: Dict[str, str] - Model configuration

        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []
        to_optimize_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []

        for sparsity_objective in sparsity_objectives:
            for memory_objective in memory_objectives:
                heavy_size: float = float(sparsity_objective) / 100.0
                aux_mem: int = memory_objective

                classes = [SinkMaskerConfig, LocalMaskerConfig, SocketMaskerConfig]
                name: str = get_masker_list_name(
                    classes,
                    other_params={
                        "builder": "socket_topk",
                        "sparsity_obj": sparsity_objective,
                        "memory_obj": memory_objective,
                    },
                )

                config = ResearchAttentionConfig(
                    masker_configs=[
                        SinkMaskerConfig(sink_size=128),
                        LocalMaskerConfig(window_size=128),
                        SocketMaskerConfig(
                            heavy_size=heavy_size - (256.0 / 32768), K=11, L=55, tau=0.5
                        ),
                    ]
                )

                config.masker_configs[2].search_space = {
                    "K": tune.grid_search([10, 11, 12]),
                    "L": tune.grid_search([40, 45, 50, 55, 60, 65, 70]),
                    "tau": tune.grid_search([0.3, 0.4, 0.5, 0.7]),
                }
                # Set validity constraint to use the correct memory_objective for comparison
                config.validity_constraint = lambda cfg: True
                # Set objective function
                config.objective = sparsity_objective

                to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs
