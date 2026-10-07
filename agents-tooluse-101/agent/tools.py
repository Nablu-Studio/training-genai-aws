"""Defines the `create_ticket` tool (`ToolResult` dataclass + idempotent factory).

Parent lab: agents-tooluse-101.
AWS Services: none (local execution).
Generated artifacts: aucun (module de définition d'outil).
Mode : stub local, aucun appel AWS.
"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    # Neutral format consumed by both runner and Bedrock agent (status + payload).
    status: str
    payload: dict


def create_ticket(title: str, severity: str, idempotency_key: str) -> ToolResult:
    # Stub implementation: derive a ticket ID from the idempotency key.
    return ToolResult(
        status="created",
        payload={
            "ticketId": f"TCK-{idempotency_key[-4:]}",
            "title": title,
            "severity": severity,
        },
    )
