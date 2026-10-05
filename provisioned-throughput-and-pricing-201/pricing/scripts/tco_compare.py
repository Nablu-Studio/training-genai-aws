"""Compares Total Cost of Ownership (TCO) across Bedrock billing and inference models.

Parent lab: provisioned-throughput-and-pricing-201.
AWS Services: none (local execution).
Generated artifacts: pricing/audit/tco-comparison.json (TCO PT, on-demand, hybride par profil).
Mode : calcul hors-ligne, aucun appel AWS facturé.
"""
import json
from pathlib import Path


PRICING = json.loads(Path("pricing/data/model-pricing.json").read_text(encoding="utf-8"))
OUTPUT_PATH = Path("pricing/audit/tco-comparison.json")

HOURS_PER_MONTH = 24 * 30
PROFILES = {
    "low": {"tokensPerMin": 5000, "utilizationHours": 200},
    "medium": {"tokensPerMin": 30000, "utilizationHours": 500},
    "high": {"tokensPerMin": 90000, "utilizationHours": 700},
}


def mu_required(model: dict, tpm: int) -> int:
    # Une MU couvre `muCapacityTokensPerMin` tokens par minute, on arrondit au supérieur.
    return max(1, round(tpm / model["muCapacityTokensPerMin"]))


def tco_pt(model: dict, mu: int, hours: int) -> float:
    # Coût PT = MU * prix horaire * heures d'utilisation.
    return round(mu * model["ptPricePerMuHour"] * hours, 2)


def tco_on_demand(model: dict, tpm: int, hours: int) -> float:
    # Hypothèse 50/50 input/output pour le calcul on-demand.
    total_tokens = tpm * 60 * hours
    return round(
        (total_tokens * 0.5 * model["onDemandInputPer1k"] / 1000)
        + (total_tokens * 0.5 * model["onDemandOutputPer1k"] / 1000),
        2,
    )


def tco_hybrid(model: dict, tpm: int, hours: int) -> float:
    # Stratégie hybride : PT couvre la charge de base, on-demand absorbe les pics.
    base = min(tpm, model["muCapacityTokensPerMin"])
    peak = max(0, tpm - base)
    base_mu = mu_required(model, base)
    pt_cost = tco_pt(model, base_mu, hours)
    on_demand_cost = round(
        (peak * 60 * hours * 0.5 * model["onDemandInputPer1k"] / 1000)
        + (peak * 60 * hours * 0.5 * model["onDemandOutputPer1k"] / 1000),
        2,
    )
    return round(pt_cost + on_demand_cost, 2)


def main() -> None:
    # Sélection du modèle cible puis évaluation des trois stratégies par profil d'usage.
    model = next(item for item in PRICING if item["modelId"] == "amazon.nova-lite-v1:0")
    results: list[dict] = []
    for name, profile in PROFILES.items():
        mu = mu_required(model, profile["tokensPerMin"])
        pt = tco_pt(model, mu, profile["utilizationHours"])
        on_demand = tco_on_demand(model, profile["tokensPerMin"], profile["utilizationHours"])
        hybrid = tco_hybrid(model, profile["tokensPerMin"], profile["utilizationHours"])
        # Le "winner" est la stratégie la moins chère pour ce profil.
        winner = min(
            ("pt", pt),
            ("on_demand", on_demand),
            ("hybrid", hybrid),
            key=lambda pair: pair[1],
        )
        results.append(
            {
                "profile": name,
                "tokensPerMin": profile["tokensPerMin"],
                "hours": profile["utilizationHours"],
                "pt": pt,
                "on_demand": on_demand,
                "hybrid": hybrid,
                "winner": winner[0],
            }
        )
    audit = {"modelId": model["modelId"], "results": results, "hoursPerMonth": HOURS_PER_MONTH}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
