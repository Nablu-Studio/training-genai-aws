"""Local entrypoint to test the `create_ticket` tool defined in `tools.py`.

Parent lab: agents-tooluse-101.
AWS Services: none (local execution).
Generated artifacts: aucun (sortie console uniquement).
Mode : stub local, aucun appel AWS.
"""

from tools import create_ticket


def main() -> None:
    # Direct tool invocation with a fixed idempotency key for reproducible demo.
    result = create_ticket(
        title="Investigate prompt regression",
        severity="medium",
        idempotency_key="idem-0001",
    )
    print(result)


if __name__ == "__main__":
    main()
