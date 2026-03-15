import pytest
import unittest.mock as mock
from src.core.rag_engine import AzureRAGEngine, DocumentChunk
from src.agents.agentic_workflows import AutonomousAgenticWorkflow, AgentRequest, Tool

@pytest.fixture
def mock_rag_engine():
    """Fixture to provide a mocked AzureRAGEngine."""
    with mock.patch("src.core.rag_engine.SearchClient") as mock_search, \
         mock.patch("src.core.rag_engine.AzureOpenAIEmbeddings") as mock_embed:
        engine = AzureRAGEngine(index_name="test-index")
        yield engine, mock_search, mock_embed

@pytest.mark.asyncio
async def test_rag_engine_hybrid_search(mock_rag_engine):
    """Validates the hybrid search logic of the RAG engine."""
    engine, mock_search_client, mock_embed_client = mock_rag_engine
    
    # Mock embedding generation
    mock_embed_client.return_value.aembed_query.return_value = [0.1, 0.2, 0.3]
    
    # Mock search results
    mock_search_instance = mock_search_client.return_value
    mock_search_instance.__aenter__.return_value.search.return_value = mock.MagicMock()
    
    # Set up the async iterator mock
    mock_results = [
        {"id": "1", "content": "Test content", "@search.score": 0.9, "source": "docs", "category": "test"}
    ]
    
    async def mock_aiter():
        for res in mock_results:
            yield res
            
    mock_search_instance.__aenter__.return_value.search.return_value.__aiter__ = mock_aiter

    results = await engine.hybrid_search(query="test query")

    assert len(results) == 1
    assert isinstance(results[0], DocumentChunk)
    assert results[0].content == "Test content"
    assert results[0].score == 0.9

@pytest.mark.asyncio
async def test_agent_workflow_execution():
    """Validates the agentic workflow execution path."""
    with mock.patch("src.agents.agentic_workflows.AzureChatOpenAI") as mock_chat:
        # Mock the agent executor response
        mock_executor = mock.AsyncMock()
        mock_executor.ainvoke.return_value = {"output": "Agent response string"}
        
        with mock.patch("src.agents.agentic_workflows.AgentExecutor", return_value=mock_executor):
            tools = [Tool(name="TestTool", func=lambda x: x, description="test")]
            workflow = AutonomousAgenticWorkflow(tools=tools)
            
            request = AgentRequest(query="Hello agent", session_id="123")
            response = await workflow.execute_workflow(request)
            
            assert response["status"] == "success"
            assert response["output"] == "Agent response string"
            assert response["session_id"] == "123"

def test_pydantic_validation():
    """Ensures Pydantic models validate input correctly."""
    with pytest.raises(ValueError):
        # top_k must be >= 1
        from api.fastapi_app import QueryRequest
        QueryRequest(query="test", top_k=0)
