"""
Merge near-duplicate entities in Neo4j after multiple ingestion runs.

Entities that refer to the same real-world thing but were extracted with slightly
different names (e.g. "Αναξίμανδρος" vs "Αναξίμανδρος ο Μιλήσιος") are grouped
by embedding similarity of both their **name** and their **summary**, then merged
into a single canonical node with all relationships preserved.

Two entities are considered the same if either:
  - their name embeddings are above the threshold, or
  - both have non-empty summaries whose embeddings are above the threshold

This catches both surface-form variants (nominative vs genitive) and
same-concept-different-wording (e.g. "ὕδωρ" and "το υγρό στοιχείο" described
with near-identical summaries).
"""

import os
import math
import argparse
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("CONSOLIDATE_SIMILARITY", "0.85"))
MAX_GROUP_SIZE = int(os.getenv("CONSOLIDATE_MAX_GROUP", "20"))


@dataclass
class EntityNode:
    """Minimal representation of a Neo4j Entity node."""
    name: str
    name_key: str
    type_: str
    summary: str
    name_embedding: List[float]
    summary_embedding: Optional[List[float]] = None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class GraphConsolidator:
    """Consolidate near-duplicate entities in the knowledge graph."""

    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD not set")

        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "ollama")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")

        self._driver = None
        self._embed_client: AsyncOpenAI = None

    async def initialize(self):
        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        self._embed_client = AsyncOpenAI(
            base_url=self.embedding_base_url, api_key=self.embedding_api_key
        )

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def _fetch_all_entities(self) -> List[EntityNode]:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.name_key AS name_key, "
                "e.type AS type, e.summary AS summary, e.name_embedding AS embedding"
            )
            rows = [r.data() async for r in result]

        entities = []
        for r in rows:
            emb = r.get("embedding")
            if emb is None:
                continue
            if isinstance(emb, list) and all(isinstance(v, (int, float)) for v in emb):
                entities.append(EntityNode(
                    name=r["name"],
                    name_key=r["name_key"],
                    type_=r.get("type", "other"),
                    summary=r.get("summary", ""),
                    name_embedding=[float(v) for v in emb],
                ))
        return entities

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = await self._embed_client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [d.embedding for d in resp.data]

    async def _compute_summary_embeddings(
        self, entities: List[EntityNode]
    ) -> None:
        """Compute and attach summary embeddings for entities with non-empty summaries."""
        to_embed = []
        indices = []
        for i, e in enumerate(entities):
            if e.summary and e.summary_embedding is None:
                to_embed.append(e.summary)
                indices.append(i)

        if not to_embed:
            return

        logger.info(f"Embedding {len(to_embed)} entity summaries...")
        embs = await self._embed(to_embed)
        for i, emb in zip(indices, embs):
            entities[i].summary_embedding = emb

    def _are_similar(
        self, a: EntityNode, b: EntityNode
    ) -> bool:
        """Return True if a and b refer to the same real-world entity.

        Checks two independent signals: name similarity and summary similarity.
        Either one above threshold is sufficient.
        """
        name_sim = cosine_similarity(a.name_embedding, b.name_embedding)
        if name_sim >= SIMILARITY_THRESHOLD:
            return True

        if a.summary_embedding and b.summary_embedding:
            summary_sim = cosine_similarity(a.summary_embedding, b.summary_embedding)
            if summary_sim >= SIMILARITY_THRESHOLD:
                return True

        return False

    def _group_similar(self, entities: List[EntityNode]) -> List[List[EntityNode]]:
        """Greedy grouping by name or summary embedding similarity."""
        n = len(entities)
        if n == 0:
            return []
        assigned = [False] * n
        groups: List[List[EntityNode]] = []

        for i in range(n):
            if assigned[i]:
                continue
            group = [entities[i]]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if self._are_similar(entities[i], entities[j]):
                    group.append(entities[j])
                    assigned[j] = True
                    if len(group) >= MAX_GROUP_SIZE:
                        break
            groups.append(group)

        return groups

    async def _merge_group(self, group: List[EntityNode], dry_run: bool = False):
        if len(group) < 2:
            return

        canonical = max(group, key=lambda e: len(e.name))
        duplicates = [e for e in group if e.name_key != canonical.name_key]

        if not duplicates:
            return

        dup_keys = [d.name_key for d in duplicates]

        if dry_run:
            logger.info(
                f"[DRY-RUN] Would merge {len(duplicates)} into '{canonical.name}' "
                f"(keys: {dup_keys})"
            )
            return

        async with self._driver.session() as session:
            all_summaries = [e.summary for e in group if e.summary]
            merged_summary = " | ".join(dict.fromkeys(all_summaries))

            # Bind canonical node first, then MATCH duplicates + their relationships,
            # then MERGE only the relationship (nodes are already bound).
            await session.run(
                """
                MATCH (c:Entity {name_key: $canonical_key})
                SET c.summary = $summary
                WITH c
                MATCH (d:Entity) WHERE d.name_key IN $dup_keys
                MATCH (s:Entity)-[r:RELATES_TO]->(d)
                MERGE (s)-[rel:RELATES_TO {
                    fact: r.fact,
                    predicate: r.predicate,
                    source_doc: r.source_doc,
                    chunk_id: r.chunk_id,
                    created_at: r.created_at
                }]->(c)
                """,
                canonical_key=canonical.name_key,
                summary=merged_summary or canonical.summary,
                dup_keys=dup_keys,
            )

            await session.run(
                """
                MATCH (c:Entity {name_key: $canonical_key})
                MATCH (d:Entity) WHERE d.name_key IN $dup_keys
                MATCH (d)-[r:RELATES_TO]->(o:Entity)
                MERGE (c)-[rel:RELATES_TO {
                    fact: r.fact,
                    predicate: r.predicate,
                    source_doc: r.source_doc,
                    chunk_id: r.chunk_id,
                    created_at: r.created_at
                }]->(o)
                """,
                canonical_key=canonical.name_key,
                dup_keys=dup_keys,
            )

            await session.run(
                "MATCH (d:Entity) WHERE d.name_key IN $dup_keys DETACH DELETE d",
                dup_keys=dup_keys,
            )

            logger.info(
                f"Merged {len(duplicates)} entities into '{canonical.name}' "
                f"(keys: {dup_keys})"
            )

    async def consolidate(self, dry_run: bool = False) -> Dict[str, Any]:
        if not self._driver:
            await self.initialize()

        entities = await self._fetch_all_entities()
        logger.info(f"Fetched {len(entities)} entities with embeddings")

        if not entities:
            return {"groups_found": 0, "entities_merged": 0, "duplicates_found": 0}

        missing_name = [e for e in entities if not e.name_embedding]
        if missing_name:
            texts = [e.name for e in missing_name]
            logger.info(f"Embedding {len(missing_name)} entities without name embeddings...")
            embs = await self._embed(texts)
            for e, emb in zip(missing_name, embs):
                e.name_embedding = emb

        await self._compute_summary_embeddings(entities)

        groups = self._group_similar(entities)
        groups = [g for g in groups if len(g) > 1]
        logger.info(f"Found {len(groups)} groups with similar entities")

        total_duplicates = sum(len(g) - 1 for g in groups)

        for group in groups:
            await self._merge_group(group, dry_run=dry_run)

        return {
            "groups_found": len(groups),
            "duplicates_found": total_duplicates,
            "entities_merged": total_duplicates if not dry_run else 0,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Consolidate near-duplicate entities in the knowledge graph"
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be merged without making changes")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold for entity merging (default: 0.85)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    global SIMILARITY_THRESHOLD

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.threshold is not None:
        SIMILARITY_THRESHOLD = args.threshold

    consolidator = GraphConsolidator()
    try:
        result = await consolidator.consolidate(dry_run=args.dry_run)
        print()
        print("=" * 50)
        print("CONSOLIDATION SUMMARY")
        print("=" * 50)
        print(f"Groups found:      {result['groups_found']}")
        print(f"Duplicates found:  {result['duplicates_found']}")
        print(f"Entities merged:   {result['entities_merged']}")
        if args.dry_run:
            print("\n(DRY RUN — no changes made)")
    finally:
        await consolidator.close()


if __name__ == "__main__":
    asyncio.run(main())
