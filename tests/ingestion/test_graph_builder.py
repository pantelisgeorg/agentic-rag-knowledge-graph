"""Tests for the lightweight GraphBuilder (single-pass extraction)."""

import pytest
from unittest.mock import AsyncMock, patch

from ingestion.graph_builder import GraphBuilder


class TestAddDocumentToGraph:
    """Tests for GraphBuilder.add_document_to_graph."""

    def _make_builder(self):
        """Build a GraphBuilder with a mocked graph client (no real connection)."""
        with patch('ingestion.graph_builder.GraphClient'):
            builder = GraphBuilder()
        builder.graph_client = AsyncMock()
        builder._initialized = True
        return builder

    @pytest.mark.asyncio
    async def test_processes_all_chunks(self, sample_chunks):
        """Expected use: all chunks processed, results aggregated."""
        builder = self._make_builder()
        builder.graph_client.add_episode = AsyncMock(
            return_value={"entities_added": 5, "relationships_added": 3, "errors": []}
        )

        result = await builder.add_document_to_graph(
            chunks=sample_chunks,
            document_title="aristotelis",
            document_source="aristotelis.md",
        )

        assert result["episodes_created"] == 3
        assert result["entities_extracted"] == 15  # 5 per chunk * 3
        assert result["relationships_created"] == 9  # 3 per chunk * 3
        assert result["errors"] == []
        assert builder.graph_client.add_episode.call_count == 3

    @pytest.mark.asyncio
    async def test_continues_on_chunk_error(self, sample_chunks):
        """Failure case: one chunk fails but the rest still process."""
        builder = self._make_builder()
        builder.graph_client.add_episode = AsyncMock(
            side_effect=[
                {"entities_added": 4, "relationships_added": 2, "errors": []},
                RuntimeError("LLM timed out"),
                {"entities_added": 6, "relationships_added": 3, "errors": []},
            ]
        )

        result = await builder.add_document_to_graph(
            chunks=sample_chunks,
            document_title="aristotelis",
            document_source="aristotelis.md",
        )

        assert result["episodes_created"] == 2  # only 2 succeeded
        assert result["entities_extracted"] == 10
        assert len(result["errors"]) == 1
        assert "LLM timed out" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zeros(self):
        """Edge case: no chunks produces zero results and no calls."""
        builder = self._make_builder()

        result = await builder.add_document_to_graph(
            chunks=[],
            document_title="empty",
            document_source="empty.md",
        )

        assert result["episodes_created"] == 0
        assert result["entities_extracted"] == 0
        assert result["relationships_created"] == 0
        assert result["errors"] == []
        builder.graph_client.add_episode.assert_not_called()


class TestExtractEntitiesFromChunks:
    """Tests for the compatibility no-op."""

    @pytest.mark.asyncio
    async def test_returns_chunks_unchanged(self, sample_chunks):
        """Expected use: no-op returns the same chunks (extraction is in graph build)."""
        with patch('ingestion.graph_builder.GraphClient'):
            builder = GraphBuilder()
        builder._initialized = True

        result = await builder.extract_entities_from_chunks(sample_chunks)

        assert result is sample_chunks
        assert len(result) == 3
