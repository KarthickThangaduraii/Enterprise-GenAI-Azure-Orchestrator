import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    """Pydantic model representing a retrieved document chunk."""
    id: str = Field(..., description="Unique identifier for the document chunk.")
    content: str = Field(..., description="The text content of the chunk.")
    score: float = Field(..., description="Relevance score from the search index.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata.")


class AzureRAGEngine:
    """
    Advanced Retrieval-Augmented Generation (RAG) engine utilizing Azure AI Search 
    and Azure OpenAI embeddings for hybrid (keyword + semantic) search capabilities.
    """

    def __init__(self, index_name: str) -> None:
        """
        Initializes the RAG Engine with necessary Azure clients.

        Args:
            index_name (str): The name of the Azure AI Search index.
        """
        # Load environment variables (ideally via pydantic-settings in a real app)
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
        self.search_api_key = os.getenv("AZURE_SEARCH_KEY", "")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

        if not all([self.search_endpoint, self.search_api_key, self.openai_endpoint, self.openai_api_key]):
            logger.warning("Missing required Azure environment variables for RAG Engine.")

        # Initialize the Azure OpenAI Embeddings client via LangChain
        self.embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_api_key,
            azure_deployment=self.embedding_deployment,
            openai_api_version="2023-05-15"
        )

        # Initialize the Azure AI Search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.search_api_key)
        )

    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generates vector embeddings for a given text string using Azure OpenAI.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The generated vector embedding.
        """
        try:
            return await self.embeddings_client.aembed_query(text)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    async def hybrid_search(self, query: str, top_k: int = 5, filter_str: Optional[str] = None) -> List[DocumentChunk]:
        """
        Executes a hybrid search query (vector + full-text) against Azure AI Search.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.
            filter_str (Optional[str]): OData filter string to apply.

        Returns:
            List[DocumentChunk]: A list of retrieved document chunks.
        """
        logger.info(f"Executing hybrid search for query: '{query}' with top_k: {top_k}")
        
        # Generate the vector for the user query
        query_vector = await self.generate_embeddings(query)

        # Construct the vector query object
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="contentVector" # Ensure this matches the index definition
        )

        results: List[DocumentChunk] = []
        try:
            # Perform the search operation
            async with self.search_client:
                search_results = await self.search_client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    filter=filter_str,
                    top=top_k,
                    select=["id", "content", "source", "category"]
                )

                async for result in search_results:
                    chunk = DocumentChunk(
                        id=result.get("id", ""),
                        content=result.get("content", ""),
                        score=result.get("@search.score", 0.0),
                        metadata={
                            "source": result.get("source", "unknown"),
                            "category": result.get("category", "general")
                        }
                    )
                    results.append(chunk)

            logger.info(f"Retrieved {len(results)} chunks successfully.")
            return results

        except Exception as e:
            logger.error(f"Hybrid search execution failed: {e}")
            raise RuntimeError(f"Search execution failed: {e}")
