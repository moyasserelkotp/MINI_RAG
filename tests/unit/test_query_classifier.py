"""Unit tests for QueryClassifier — heuristics, LLM path, and fallback."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.classifier import QueryClassifier


@pytest.fixture
def classifier_no_llm():
    return QueryClassifier(llm_client=None)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_structured_output = MagicMock(return_value={"category": "DOCUMENT_QUESTION"})
    return llm


@pytest.fixture
def classifier_with_llm(mock_llm):
    return QueryClassifier(llm_client=mock_llm)


class TestHeuristics:
    @pytest.mark.asyncio
    async def test_complex_multi_step_compare(self, classifier_no_llm):
        result = await classifier_no_llm.classify("compare the two documents")
        assert result == "COMPLEX_MULTI_STEP"

    @pytest.mark.asyncio
    async def test_complex_multi_step_analyze_across(self, classifier_no_llm):
        result = await classifier_no_llm.classify("analyze across all sections")
        assert result == "COMPLEX_MULTI_STEP"

    @pytest.mark.asyncio
    async def test_complex_multi_step_vs(self, classifier_no_llm):
        result = await classifier_no_llm.classify("method A vs method B")
        assert result == "COMPLEX_MULTI_STEP"

    @pytest.mark.asyncio
    async def test_asset_metadata_files(self, classifier_no_llm):
        result = await classifier_no_llm.classify("what files have been uploaded?")
        assert result == "ASSET_METADATA"

    @pytest.mark.asyncio
    async def test_asset_metadata_documents(self, classifier_no_llm):
        result = await classifier_no_llm.classify("list the documents uploaded")
        assert result == "ASSET_METADATA"

    @pytest.mark.asyncio
    async def test_project_metadata(self, classifier_no_llm):
        result = await classifier_no_llm.classify("when was this project created?")
        assert result == "PROJECT_METADATA"

    @pytest.mark.asyncio
    async def test_default_fallback_no_llm(self, classifier_no_llm):
        """With no LLM and no heuristic match, should fall back to DOCUMENT_QUESTION."""
        result = await classifier_no_llm.classify("explain the main concept")
        assert result == "DOCUMENT_QUESTION"

    @pytest.mark.asyncio
    async def test_heuristic_case_insensitive(self, classifier_no_llm):
        result = await classifier_no_llm.classify("COMPARE the chapters")
        assert result == "COMPLEX_MULTI_STEP"


class TestLLMClassification:
    @pytest.mark.asyncio
    async def test_llm_called_for_ambiguous_query(self, classifier_with_llm, mock_llm):
        result = await classifier_with_llm.classify("tell me about chapter 3")
        mock_llm.generate_structured_output.assert_called_once()
        assert result == "DOCUMENT_QUESTION"

    @pytest.mark.asyncio
    async def test_llm_result_used_when_valid(self, classifier_with_llm, mock_llm):
        mock_llm.generate_structured_output.return_value = {"category": "CONVERSATION_REFERENCE"}
        result = await classifier_with_llm.classify("what did you just say?")
        assert result == "CONVERSATION_REFERENCE"

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_default(self, classifier_with_llm, mock_llm):
        mock_llm.generate_structured_output.side_effect = Exception("LLM timeout")
        result = await classifier_with_llm.classify("tell me about section 2")
        assert result == "DOCUMENT_QUESTION"

    @pytest.mark.asyncio
    async def test_heuristic_short_circuits_llm(self, classifier_with_llm, mock_llm):
        """If a heuristic matches, LLM should NOT be called."""
        result = await classifier_with_llm.classify("compare sections 1 and 2")
        mock_llm.generate_structured_output.assert_not_called()
        assert result == "COMPLEX_MULTI_STEP"
