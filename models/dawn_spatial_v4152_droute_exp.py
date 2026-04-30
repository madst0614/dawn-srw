"""DAWN Spatial-R1 v4.1.5.2 paper-facing d_route-only module.

The active training configuration exposes only ``d_route``. The underlying
implementation is retained in ``models.legacy`` so existing checkpoints keep
their exact parameter structure.
"""

from .legacy.dawn_spatial_v4152_operator_route_free_emb_exp import (
    DAWN,
    make_sharded_srw,
    make_sharded_srw_paired,
)
