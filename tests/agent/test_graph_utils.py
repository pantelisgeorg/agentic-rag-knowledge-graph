"""Tests for the custom GraphClient (no Graphiti) in agent.graph_utils."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from agent.graph_utils import (
    GraphClient,
    _fix_garbled_unicode,
    _normalize_name,
)


class TestFixGarbledUnicode:
    """Tests for the garbled-Unicode repair safety net."""

    def test_repairs_greek_codepoints(self):
        """Expected use: chr(3)+'XX' patterns decode back to real Greek."""
        # 'Διογένης' where each U+03XX was split into chr(3)+XX
        garbled = '\x0394\x03b9\x03bf\x03b3\x03ad\x03bd\x03b7\x03c2'
        assert _fix_garbled_unicode(garbled) == 'Διογένης'

    def test_preserves_space_between_words(self):
        """A space inside a garbled multi-word name survives."""
        garbled = '\x03A3\x03c9\x03ba\x03c1\x03ac\x03c4\x03b7\x03c2 \x0394\x03b9'
        assert _fix_garbled_unicode(garbled) == 'Σωκράτης Δι'

    def test_clean_string_passes_through(self):
        """Edge case: an already-clean Greek string is returned unchanged."""
        clean = 'Σωκράτης'
        assert _fix_garbled_unicode(clean) == 'Σωκράτης'

    def test_empty_string(self):
        """Edge case: empty input returns empty."""
        assert _fix_garbled_unicode('') == ''

    def test_ascii_passes_through(self):
        """Edge case: plain ASCII is unchanged."""
        assert _fix_garbled_unicode('Brown, Janis 1982') == 'Brown, Janis 1982'


class TestNormalizeName:
    """Tests for entity-name normalization (dedup key)."""

    def test_lowercases_and_strips(self):
        """Expected use: strips whitespace and lowercases."""
        assert _normalize_name('  Σωκράτης ') == 'σωκράτης'

    def test_handles_none(self):
        """Failure case: None input does not crash."""
        assert _normalize_name(None) == ''


def _mock_llm_response(content: str):
    """Build a mock chat.completions.create response object."""
    msg = Mock()
    msg.content = content
    choice = Mock()
    choice.message = msg
    resp = Mock()
    resp.choices = [choice]
    return resp


class TestExtract:
    """Tests for GraphClient._extract (LLM JSON parsing)."""

    def _make_client(self):
        """Build a GraphClient with a mocked LLM client (no real connection)."""
        with patch('agent.graph_utils.AsyncGraphDatabase'):
            client = GraphClient()
        client._llm_client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_parses_clean_json(self):
        """Expected use: valid JSON entities/relationships are parsed."""
        client = self._make_client()
        client._llm_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(
                '{"entities": [{"name": "Αριστοτέλης", "type": "person", "summary": "philosopher"}], '
                '"relationships": [{"subject": "Αριστοτέλης", "predicate": "studied under", '
                '"object": "Πλάτων", "fact": "Αριστοτέλης was a student of Πλάτων"}]}'
            )
        )
        data = await client._extract("some text")
        assert len(data["entities"]) == 1
        assert data["entities"][0]["name"] == "Αριστοτέλης"
        assert len(data["relationships"]) == 1
        assert data["relationships"][0]["object"] == "Πλάτων"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        """Edge case: JSON wrapped in ```json fences is still parsed."""
        client = self._make_client()
        client._llm_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(
                '```json\n{"entities": [{"name": "Σωκράτης", "type": "person", "summary": "x"}], '
                '"relationships": []}\n```'
            )
        )
        data = await client._extract("text")
        assert data["entities"][0]["name"] == "Σωκράτης"

    @pytest.mark.asyncio
    async def test_repairs_garbled_in_response(self):
        """Edge case: garbled Greek in the LLM response is repaired."""
        client = self._make_client()
        garbled_name = '\x03A3\x03c9\x03ba\x03c1\x03ac\x03c4\x03b7\x03c2'
        client._llm_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(
                f'{{"entities": [{{"name": "{garbled_name}", "type": "person", "summary": "s"}}], '
                f'"relationships": []}}'
            )
        )
        data = await client._extract("text")
        assert data["entities"][0]["name"] == 'Σωκράτης'

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self):
        """Failure case: unparseable JSON returns empty entities/relationships."""
        client = self._make_client()
        client._llm_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("this is not json at all")
        )
        data = await client._extract("text")
        assert data["entities"] == []
        assert data["relationships"] == []

    @pytest.mark.asyncio
    async def test_caps_entity_count(self):
        """Edge case: more entities than MAX_ENTITIES_PER_CHUNK are trimmed, and
        relationships referencing dropped entities are removed too."""
        client = self._make_client()
        with patch("agent.graph_utils.MAX_ENTITIES_PER_CHUNK", 2):
            many = [{"name": f"E{i}", "type": "person", "summary": "s"} for i in range(5)]
            rels = [
                {"subject": "E0", "predicate": "knows", "object": "E1", "fact": "f"},
                {"subject": "E0", "predicate": "knows", "object": "E4", "fact": "f"},
            ]
            client._llm_client.chat.completions.create = AsyncMock(
                return_value=_mock_llm_response(
                    '{"entities": ' + str(many).replace("'", '"') +
                    ', "relationships": ' + str(rels).replace("'", '"') + '}'
                )
            )
            data = await client._extract("text")
        assert len(data["entities"]) == 2
        # E4 was dropped, so the relationship referencing it is removed
        assert len(data["relationships"]) == 1
        assert data["relationships"][0]["object"] == "E1"
