"""Defines the `create_ticket` tool (`ToolResult` dataclass + idempotent factory).

Parent lab: agents-tooluse-101.
AWS Services: none (local execution).
Generated artifacts: aucun (module de définition d'outil).
Mode : stub local, aucun appel AWS.
"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    # Format neutre consommé par le runner et l'agent Bedrock (status + payload).
    status: str
    payload: dict


def create_ticket(title: str, severity: str, idempotency_key: str) -> ToolResult:
    # Stub : on dérive un ID de ticket à partir de la clé d'idempotence.
    return ToolResult(
        status="created",
        payload={
            "ticketId": f"TCK-{idempotency_key[-4:]}",
            "title": title,
            "severity": severity,
        },
    )
