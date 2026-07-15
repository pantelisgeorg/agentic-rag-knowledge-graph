"""
Knowledge graph utilities using Neo4j directly (no Graphiti).

A lightweight, custom graph client that extracts entities and relationships
from text via a single LLM call per chunk and stores them in Neo4j. This
replaces Graphiti to avoid its heavy multi-call-per-chunk design (which made
ingestion slow and expensive) and to keep full control of Unicode handling.

Public interface is kept compatible with the previous GraphitiClient so that
tools.py / api.py require minimal changes.
"""

import os
import json
import re
import logging
import uuid as uuid_mod
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _normalize_name(name: str) -> str:
    """Normalize an entity name for dedup (lowercase + stripped)."""
    return (name or "").strip().lower()


def _fix_garbled_unicode(s: str) -> str:
    """Repair entity names where Greek codepoints were mangled into
    control-char + literal hex (e.g. chr(3)+'94' instead of 'Δ').

    This is a safety net for models that escape Unicode despite instructions.
    The pattern: U+03XX got split into chr(3) + 'XX'; U+00XX into chr(0)+'XXX'.

    Args:
        s: Possibly garbled string.

    Returns:
        Repaired string with real Unicode characters.
    """
    if not s:
        return s
    # Only process if there are low control chars (the garble signature)
    if not any(ord(c) < 8 for c in s):
        return s
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        co = ord(c)
        if co < 8 and i + 1 < n:
            # Pattern A: chr(3) + 2 hex chars -> "03" + XX -> U+03XX (Greek)
            if co == 3 and i + 2 < n and s[i + 1] in _HEX_DIGITS and s[i + 2] in _HEX_DIGITS:
                out.append(chr(int("03" + s[i + 1:i + 3], 16)))
                i += 3
                continue
            # Pattern B: chr(0) + 3 hex chars -> "00" + XXX -> U+00XXX
            if co == 0 and i + 3 < n and all(s[i + k] in _HEX_DIGITS for k in (1, 2, 3)):
                out.append(chr(int("00" + s[i + 1:i + 4], 16)))
                i += 4
                continue
        out.append(c)
        i += 1
    return "".join(out)


# Max entities extracted per chunk (tune via env without code changes).
MAX_ENTITIES_PER_CHUNK = int(os.getenv("MAX_ENTITIES_PER_CHUNK", "8"))

EXTRACTION_SYSTEM_PROMPT = f"""You are an entity and relationship extraction system for ancient Greek philosophy texts.

Extract the entities and relationships from the user-provided text.

Return ONLY a JSON object (no markdown fences, no commentary) with this exact structure:
{{
  "entities": [
    {{"name": "exact name as written in the text", "type": "person|place|concept|school|work|other", "summary": "one short sentence describing the entity"}}
  ],
  "relationships": [
    {{"subject": "entity name", "predicate": "short relation in lowercase", "object": "entity name", "fact": "one sentence describing the relationship"}}
  ]
}}

Rules:
- Use the EXACT original characters from the text, including Greek letters, accents and breathings. Do NOT escape, transliterate or modify Unicode characters.
- Keep entity names short (1-4 words).
- Extract ONLY the most salient entities that are central to this passage: named philosophers, places, philosophical schools, named works, or key concepts that are actually discussed (not merely mentioned once). Limit to at most {MAX_ENTITIES_PER_CHUNK} entities.
- Do NOT extract: bibliographic citations, author surnames from references/footnotes, dates, centuries, numbers, single common words, adjectives, verbs, or entities only mentioned in passing.
- When in doubt whether something is a meaningful entity, leave it out.
- Every relationship subject and object must be one of the extracted entity names.
- If the text contains no clearly important entities, return {{"entities": [], "relationships": []}}."""


class GraphClient:
    """Manages knowledge graph operations in Neo4j (no Graphiti)."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize graph client configuration from environment.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8083/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY", "llamacpp")
        self.llm_choice = os.getenv("LLM_CHOICE", "gpt-4-turbo-preview")

        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "ollama")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")
        self.vector_dim = int(os.getenv("VECTOR_DIMENSION", "768"))

        self._driver = None
        self._llm_client: Optional[AsyncOpenAI] = None
        self._embed_client: Optional[AsyncOpenAI] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Neo4j driver, LLM/embedding clients, and schema."""
        if self._initialized:
            return

        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )
        self._llm_client = AsyncOpenAI(
            base_url=self.llm_base_url, api_key=self.llm_api_key
        )
        self._embed_client = AsyncOpenAI(
            base_url=self.embedding_base_url, api_key=self.embedding_api_key
        )

        await self._create_schema()
        self._initialized = True
        logger.info(
            f"GraphClient initialized (Neo4j direct, no Graphiti) | "
            f"LLM={self.llm_choice} | embedder={self.embedding_model} ({self.vector_dim}d)"
        )

    async def _create_schema(self) -> None:
        """Create constraints and vector index for the custom graph schema."""
        async with self._driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT entity_name_key_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name_key IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX entity_name_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            )
            try:
                await session.run(
                    f"CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS "
                    f"FOR (e:Entity) ON (e.name_embedding) "
                    f"OPTIONS {{ indexConfig: {{ "
                    f"`vector.dimensions`: {self.vector_dim}, "
                    f"`vector.similarity_function`: 'cosine' }} }}"
                )
            except Exception as e:
                # Reason: index may already exist with different config, or vector
                # indexes unavailable; fall back to Python-side similarity in search.
                logger.warning(f"Vector index creation skipped: {e}")

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._initialized = False
        logger.info("GraphClient closed")

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the configured embedding model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        resp = await self._embed_client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [d.embedding for d in resp.data]

    async def _extract(self, content: str) -> Dict[str, Any]:
        """Extract entities and relationships from text via one LLM call.

        Args:
            content: Text to extract from.

        Returns:
            Dict with 'entities' and 'relationships' lists.
        """
        resp = await self._llm_client.chat.completions.create(
            model=self.llm_choice,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content or ""
        raw = _fix_garbled_unicode(raw)
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        # Extract the JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}; raw[:200]={raw[:200]!r}")
            return {"entities": [], "relationships": []}

        # Sanitize and fix any garbled Unicode in names/facts
        for ent in data.get("entities", []):
            ent["name"] = _fix_garbled_unicode(str(ent.get("name", ""))).strip()
            ent["summary"] = _fix_garbled_unicode(str(ent.get("summary", ""))).strip()
            ent["type"] = str(ent.get("type", "other")).strip()
        for rel in data.get("relationships", []):
            for k in ("subject", "predicate", "object", "fact"):
                rel[k] = _fix_garbled_unicode(str(rel.get(k, ""))).strip()

        # Hard cap as a safety net (drop empty names, then limit count).
        ents = [e for e in data.get("entities", []) if e.get("name")]
        if len(ents) > MAX_ENTITIES_PER_CHUNK:
            logger.info(
                f"Trimming {len(ents)} -> {MAX_ENTITIES_PER_CHUNK} entities (cap)"
            )
            ents = ents[:MAX_ENTITIES_PER_CHUNK]
            kept_names = {e["name"] for e in ents}
            data["relationships"] = [
                r for r in data.get("relationships", [])
                if r.get("subject") in kept_names and r.get("object") in kept_names
            ]
        data["entities"] = ents
        return data

    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract entities/relationships from content and write to Neo4j.

        One LLM call is made for extraction; results are stored directly.

        Args:
            episode_id: Unique episode identifier.
            content: Text content to process.
            source: Source description (e.g. document title).
            timestamp: Episode timestamp.
            metadata: Additional metadata (document_title, document_source, chunk_index).

        Returns:
            Dict with entities_added, relationships_added, errors.
        """
        if not self._initialized:
            await self.initialize()

        metadata = metadata or {}
        source_doc = metadata.get("document_source", source)
        chunk_id = metadata.get("chunk_index", episode_id)
        now = (timestamp or datetime.now(timezone.utc)).isoformat()

        try:
            extracted = await self._extract(content)
        except Exception as e:
            logger.error(f"Extraction failed for {episode_id}: {e}")
            return {"entities_added": 0, "relationships_added": 0, "errors": [str(e)]}

        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        # Collect all unique entity names (from entities + relationship endpoints)
        all_names: Dict[str, None] = {}
        for ent in entities:
            if ent["name"]:
                all_names.setdefault(ent["name"], None)
        for rel in relationships:
            for k in ("subject", "object"):
                if rel.get(k):
                    all_names.setdefault(rel[k], None)
        name_list = list(all_names.keys())

        # Embed all names in one batch
        embeddings = await self._embed(name_list) if name_list else []
        name_to_emb = {name: emb for name, emb in zip(name_list, embeddings)}

        # Build entity records for MERGE
        ent_by_name = {ent["name"]: ent for ent in entities if ent["name"]}
        ent_records = []
        for name in name_list:
            info = ent_by_name.get(name, {})
            ent_records.append({
                "name_key": _normalize_name(name),
                "name": name,
                "type": info.get("type", "other"),
                "summary": info.get("summary", ""),
                "embedding": name_to_emb.get(name),
            })

        async with self._driver.session() as session:
            # Create/merge entities
            if ent_records:
                await session.run(
                    """
                    UNWIND $records AS rec
                    MERGE (e:Entity {name_key: rec.name_key})
                    ON CREATE SET e.name = rec.name, e.type = rec.type,
                                  e.summary = rec.summary, e.name_embedding = rec.embedding,
                                  e.created_at = $now, e.source_doc = $source_doc
                    ON MATCH SET e.name = COALESCE(e.name, rec.name),
                                 e.type = COALESCE(e.type, rec.type),
                                 e.summary = CASE WHEN e.summary IS NULL OR e.summary = ''
                                                  THEN rec.summary ELSE e.summary END,
                                 e.name_embedding = COALESCE(e.name_embedding, rec.embedding)
                    """,
                    records=ent_records,
                    now=now,
                    source_doc=source_doc,
                )

            # Create relationships
            rel_records = [
                {
                    "s_key": _normalize_name(rel["subject"]),
                    "s_name": rel["subject"],
                    "o_key": _normalize_name(rel["object"]),
                    "o_name": rel["object"],
                    "fact": rel.get("fact", ""),
                    "predicate": rel.get("predicate", "relates_to"),
                }
                for rel in relationships
                if rel.get("subject") and rel.get("object")
            ]
            if rel_records:
                await session.run(
                    """
                    UNWIND $records AS rec
                    MERGE (s:Entity {name_key: rec.s_key}) SET s.name = COALESCE(s.name, rec.s_name)
                    MERGE (o:Entity {name_key: rec.o_key}) SET o.name = COALESCE(o.name, rec.o_name)
                    CREATE (s)-[:RELATES_TO {fact: rec.fact, predicate: rec.predicate,
                                             source_doc: $source_doc, chunk_id: $chunk_id,
                                             created_at: $now}]->(o)
                    """,
                    records=rel_records,
                    source_doc=source_doc,
                    chunk_id=str(chunk_id),
                    now=now,
                )

        logger.info(
            f"Added episode {episode_id}: {len(ent_records)} entities, "
            f"{len(rel_records)} relationships"
        )
        return {
            "entities_added": len(ent_records),
            "relationships_added": len(rel_records),
            "errors": [],
        }

    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge graph via entity vector similarity + traversal.

        Args:
            query: Search query.
            center_node_distance: Unused (kept for compatibility).
            use_hybrid_search: Unused (kept for compatibility).

        Returns:
            List of result dicts with fact, uuid, entity, related_entity, source_doc.
        """
        if not self._initialized:
            await self.initialize()

        try:
            emb = (await self._embed([query]))[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        async with self._driver.session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes('entity_name_embedding', 10, $emb)
                    YIELD node, score
                    OPTIONAL MATCH (node)-[r:RELATES_TO]->(related:Entity)
                    RETURN node.name AS entity, node.summary AS summary, score,
                           collect({fact: r.fact, predicate: r.predicate,
                                    related: related.name,
                                    source_doc: r.source_doc}) AS rels
                    ORDER BY score DESC
                    """,
                    emb=emb,
                )
                records = [r.data() async for r in result]
            except Exception as e:
                logger.warning(f"Vector search failed ({e}); falling back to full scan")
                records = await self._fallback_search(session, emb)

        results: List[Dict[str, Any]] = []
        for rec in records:
            entity = rec.get("entity", "")
            summary = rec.get("summary", "")
            rels = rec.get("rels") or []
            if not rels and summary:
                # No edges: return the entity summary as a "fact"
                results.append({
                    "fact": f"{entity}: {summary}",
                    "uuid": str(uuid_mod.uuid4()),
                    "valid_at": None,
                    "invalid_at": None,
                    "source_node_uuid": entity,
                })
            for rel in rels:
                fact = rel.get("fact") or ""
                if not fact:
                    continue
                results.append({
                    "fact": fact,
                    "uuid": str(uuid_mod.uuid4()),
                    "valid_at": None,
                    "invalid_at": None,
                    "source_node_uuid": entity,
                })
        return results

    async def _fallback_search(
        self, session, emb: List[float]
    ) -> List[Dict[str, Any]]:
        """Python-side cosine similarity fallback when vector index is unavailable.

        Args:
            session: Neo4j session.
            emb: Query embedding.

        Returns:
            List of records matching the vector search format.
        """
        result = await session.run(
            "MATCH (e:Entity) WHERE e.name_embedding IS NOT NULL "
            "RETURN e.name AS entity, e.summary AS summary, e.name_embedding AS emb"
        )
        rows = [r.data() async for r in result]
        scored = []
        for row in rows:
            e = row.get("emb") or []
            if len(e) != len(emb):
                continue
            dot = sum(a * b for a, b in zip(emb, e))
            na = sum(a * a for a in emb) ** 0.5
            nb = sum(b * b for b in e) ** 0.5
            score = dot / (na * nb) if na and nb else 0
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:10]

        out = []
        for score, row in top:
            name = row.get("entity", "")
            res = await session.run(
                """
                MATCH (e:Entity {name: $name})-[r:RELATES_TO]->(related:Entity)
                RETURN collect({fact: r.fact, predicate: r.predicate,
                                related: related.name, source_doc: r.source_doc}) AS rels
                """,
                name=name,
            )
            recs = [r.data() async for r in res]
            rels = recs[0].get("rels") if recs else []
            out.append({"entity": name, "summary": row.get("summary"), "score": score, "rels": rels})
        return out

    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """Get entities related to a given entity by name.

        Args:
            entity_name: Name of the entity.
            relationship_types: Unused (kept for compatibility).
            depth: Traversal depth (1 = direct neighbors).

        Returns:
            Dict with central_entity, related_entities, relationships.
        """
        if not self._initialized:
            await self.initialize()

        name_key = _normalize_name(entity_name)
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {name_key: $name_key})
                OPTIONAL MATCH (e)-[r:RELATES_TO]->(o:Entity)
                OPTIONAL MATCH (s:Entity)-[r2:RELATES_TO]->(e)
                RETURN e.name AS entity, e.summary AS summary,
                       collect(DISTINCT {name: o.name, fact: r.fact, predicate: r.predicate}) AS outgoing,
                       collect(DISTINCT {name: s.name, fact: r2.fact, predicate: r2.predicate}) AS incoming
                """,
                name_key=name_key,
            )
            records = [r.data() async for r in result]

        if not records or not records[0].get("entity"):
            return {
                "central_entity": entity_name,
                "related_facts": [],
                "search_method": "neo4j_direct",
            }

        rec = records[0]
        facts = []
        related = set()
        for r in (rec.get("outgoing") or []):
            if r.get("name"):
                related.add(r["name"])
                facts.append({"fact": r.get("fact", ""), "uuid": str(uuid_mod.uuid4())})
        for r in (rec.get("incoming") or []):
            if r.get("name"):
                related.add(r["name"])
                facts.append({"fact": r.get("fact", ""), "uuid": str(uuid_mod.uuid4())})

        return {
            "central_entity": rec.get("entity", entity_name),
            "related_entities": list(related),
            "related_facts": facts,
            "search_method": "neo4j_direct",
        }

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get facts for an entity (temporal filtering not supported in custom graph).

        Args:
            entity_name: Name of the entity.
            start_date: Unused (kept for compatibility).
            end_date: Unused (kept for compatibility).

        Returns:
            List of fact dicts.
        """
        if not self._initialized:
            await self.initialize()

        name_key = _normalize_name(entity_name)
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {name_key: $name_key})-[r:RELATES_TO]->(o:Entity)
                RETURN e.name AS entity, r.fact AS fact, o.name AS related,
                       r.created_at AS created_at, r.source_doc AS source_doc
                """,
                name_key=name_key,
            )
            records = [r.data() async for r in result]

        return [
            {
                "fact": r.get("fact", ""),
                "uuid": str(uuid_mod.uuid4()),
                "valid_at": r.get("created_at"),
                "invalid_at": None,
            }
            for r in records
        ]

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get node and edge counts.

        Returns:
            Dict of graph statistics.
        """
        if not self._initialized:
            await self.initialize()

        async with self._driver.session() as session:
            ent_res = await session.run("MATCH (e:Entity) RETURN count(e) AS c")
            ent_records = [r async for r in ent_res]
            rel_res = await session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c"
            )
            rel_records = [r async for r in rel_res]

        return {
            "graphiti_initialized": self._initialized,
            "entities": ent_records[0]["c"] if ent_records else 0,
            "relationships": rel_records[0]["c"] if rel_records else 0,
            "method": "neo4j_direct",
        }

    async def clear_graph(self) -> None:
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()

        async with self._driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all data from knowledge graph")


# Global client instance
graph_client = GraphClient()


async def initialize_graph() -> None:
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph() -> None:
    """Close graph client."""
    await graph_client.close()


async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Add content to the knowledge graph.

    Args:
        content: Content to add.
        source: Source of the content.
        episode_id: Optional episode ID.
        metadata: Optional metadata.

    Returns:
        Episode ID.
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    await graph_client.add_episode(
        episode_id=episode_id,
        content=content,
        source=source,
        metadata=metadata,
    )
    return episode_id


async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """Search the knowledge graph.

    Args:
        query: Search query.

    Returns:
        Search results.
    """
    return await graph_client.search(query)


async def get_entity_relationships(
    entity: str, depth: int = 2
) -> Dict[str, Any]:
    """Get relationships for an entity.

    Args:
        entity: Entity name.
        depth: Maximum traversal depth.

    Returns:
        Entity relationships.
    """
    return await graph_client.get_related_entities(entity, depth=depth)


async def test_graph_connection() -> bool:
    """Test graph database connection.

    Returns:
        True if connection successful.
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False
