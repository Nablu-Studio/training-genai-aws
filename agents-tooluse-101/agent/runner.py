"""Local entrypoint to test the `create_ticket` tool defined in `tools.py`.

Parent lab: agents-tooluse-101.
AWS Services: none (local execution).
Generated artifacts: aucun (sortie console uniquement).
Mode : stub local, aucun appel AWS.
"""

from tools import create_ticket


def main() -> None:
    # Invocation directe du tool avec une clé d'idempotence fixe pour la démo.
    result = create_ticket(
        title="Investigate prompt regression",
        severity="medium",
        idempotency_key="idem-0001",
    )
    print(result)


if __name__ == "__main__":
    main()
