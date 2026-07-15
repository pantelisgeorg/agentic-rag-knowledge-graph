"""
Knowledge graph builder using the custom lightweight GraphClient.

Replaces the Graphiti-based builder. Extraction happens inside GraphClient
(one LLM call per chunk), so this module just loops over chunks and reports
results.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio

from dotenv import load_dotenv

from .chunker import DocumentChunk

try:
    from ..agent.graph_utils import GraphClient
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphClient

load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks via single-pass extraction."""

    def __init__(self) -> None:
        """Initialize graph builder."""
        self.graph_client = GraphClient()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3,
    ) -> Dict[str, Any]:
        """Add document chunks to the knowledge graph (one LLM call per chunk).

        Args:
            chunks: List of document chunks.
            document_title: Title of the document.
            document_source: Source of the document.
            document_metadata: Additional metadata.
            batch_size: Unused (kept for compatibility).

        Returns:
            Dict with entities_extracted, episodes_created, relationships_created, errors.
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return {"episodes_created": 0, "entities_extracted": 0,
                    "relationships_created": 0, "errors": []}

        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")

        total_entities = 0
        total_relationships = 0
        episodes_created = 0
        errors: List[str] = []

        for i, chunk in enumerate(chunks):
            try:
                episode_id = f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"
                source_description = f"Document: {document_title} (Chunk: {chunk.index})"

                result = await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=chunk.content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": document_title,
                        "document_source": document_source,
                        "chunk_index": chunk.index,
                    },
                )

                total_entities += result.get("entities_added", 0)
                total_relationships += result.get("relationships_added", 0)
                episodes_created += 1
                errors.extend(result.get("errors", []))

                logger.info(
                    f"✓ Added episode {episode_id} to knowledge graph "
                    f"({episodes_created}/{len(chunks)})"
                )

                if i < len(chunks) - 1:
                    await asyncio.sleep(0.3)

            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index} to graph: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        logger.info(
            f"Graph building complete: {episodes_created} episodes, "
            f"{total_entities} entities, {total_relationships} relationships, "
            f"{len(errors)} errors"
        )
        return {
            "episodes_created": episodes_created,
            "entities_extracted": total_entities,
            "relationships_created": total_relationships,
            "errors": errors,
        }

    async def extract_entities_from_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """No-op kept for compatibility; extraction happens during graph building.

        Args:
            chunks: List of document chunks.

        Returns:
            The same chunks, unchanged.
        """
        return chunks

    async def clear_graph(self) -> None:
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()
