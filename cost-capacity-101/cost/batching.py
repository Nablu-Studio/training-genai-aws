"""Batches a list of inference requests into fixed-size contiguous chunks without external cloud dependencies.

Parent lab: cost-capacity-101.
AWS Services: none (local execution).
Generated artifacts: aucun (fonction utilitaire).
Mode : calcul local, aucun appel AWS.
"""


def chunk_requests(requests: list[str], batch_size: int) -> list[list[str]]:
    # Découpage par pas de `batch_size` via slicing contigu (memcpy).
    return [requests[index:index + batch_size] for index in range(0, len(requests), batch_size)]
