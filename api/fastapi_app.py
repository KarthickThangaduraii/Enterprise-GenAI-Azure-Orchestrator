import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any

from src.core.rag_engine import AzureRAGEngine
from src.agents.agentic_workflows import AutonomousAgenticWorkflow, AgentRequest, create_default_tools

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api.fastapi_app")

# Global instances (in a real app, use dependency injection properly)
rag_engine: AzureRAGEngine = None
agent_workflow: AutonomousAgenticWorkflow = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global rag_engine, agent_workflow
    logger.info("Initializing Enterprise GenAI Orchestrator...")
    
    # Initialize Core Components
    try:
        # The index name would typically come from configuration
        rag_engine = AzureRAGEngine(index_name="enterprise-docs-index")
        tools = create_default_tools()
        agent_workflow = AutonomousAgenticWorkflow(tools=tools)
        logger.info("All components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # In production, you might want to exit if critical components fail to load
        # raise

    yield 
    
    logger.info("Shutting down Enterprise GenAI Orchestrator...")

app = FastAPI(
    title="Enterprise GenAI Azure Orchestrator API",
    description="Production-ready REST API for RAG and Agentic Workflows.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    """Schema for a standard search query request."""
    query: str = Field(..., min_length=3, description="The search query string.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve.")

class RAGResponse(BaseModel):
    """Schema for the RAG endpoint response."""
    status: str
    results: list[Dict[str, Any]]

class HealthResponse(BaseModel):
    """Schema for the health check response."""
    status: str
    version: str

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Liveness probe endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")

@app.post("/api/v1/search", response_model=RAGResponse, tags=["Retrieval"])
async def search_documents(request: QueryRequest):
    """
    Endpoint to perform a hybrid search using the Azure RAG Engine.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not initialized.")
    
    try:
        results = await rag_engine.hybrid_search(query=request.query, top_k=request.top_k)
        # Convert Pydantic models to dicts for the JSON response
        formatted_results = [chunk.model_dump() for chunk in results]
        return RAGResponse(status="success", results=formatted_results)
    except Exception as e:
        logger.error(f"Search API error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during search execution.")

@app.post("/api/v1/agent/invoke", tags=["Agents"])
async def invoke_agent(request: AgentRequest):
    """
    Endpoint to invoke the autonomous agentic workflow.
    """
    if not agent_workflow:
        raise HTTPException(status_code=503, detail="Agent Workflow is not initialized.")
    
    result = await agent_workflow.execute_workflow(request)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
        
    return result
