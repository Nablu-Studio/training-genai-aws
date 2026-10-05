"""Lexically re-ranks chunks based on query token presence and relevance score.

Parent lab: rag-production-201.
AWS Services: none (local execution).
Generated artifacts: aucun (fonction utilitaire).
Mode : calcul local, aucun appel AWS.
"""


def rerank(chunks: list[dict], query: str) -> list[dict]:
    # Score = nombre de tokens de la requête présents dans le contenu du chunk.
    query_tokens = set(query.lower().split())
    return sorted(
        chunks,
        key=lambda chunk: sum(token in chunk["content"].lower() for token in query_tokens),
        reverse=True,
    )
