import os
import logging
from typing import Any, Dict, List
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AgentRequest(BaseModel):
    """Pydantic model for incoming agent requests."""
    query: str = Field(..., description="The user's input query.")
    session_id: str = Field(..., description="Unique identifier for the user session.")


class AutonomousAgenticWorkflow:
    """
    Implements multi-step reasoning and tool-calling capabilities using LangChain 
    and Azure OpenAI models. This orchestrator manages the lifecycle of the agent execution.
    """

    def __init__(self, tools: List[Tool]) -> None:
        """
        Initializes the agent workflow with a specific set of tools.

        Args:
            tools (List[Tool]): A list of LangChain Tools the agent can use.
        """
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4-turbo")
        
        if not all([self.openai_endpoint, self.openai_api_key]):
            logger.warning("Missing required Azure environment variables for Agent Workflow.")

        # Initialize the Azure OpenAI Chat model
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_api_key,
            azure_deployment=self.chat_deployment,
            openai_api_version="2023-12-01-preview",
            temperature=0.0
        )

        self.tools = tools
        self.agent_executor = self._initialize_agent()

    def _initialize_agent(self) -> AgentExecutor:
        """
        Constructs the LangChain agent with a robust prompt incorporating tool usage.

        Returns:
            AgentExecutor: The configured agent executor.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a Senior AI Assistant responsible for orchestrating complex enterprise tasks. "
             "You have access to a specific set of tools. Analyze the user's request carefully. "
             "If a tool is necessary to fulfill the request, use it. Ensure your reasoning is logical "
             "and you provide accurate, helpful answers based on the tool outputs. "
             "If you don't know the answer, state that clearly."
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the underlying agent utilizing OpenAI's specific tool-calling features
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # Wrap in an executor to handle tool invocation and error parsing
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True
        )

    async def execute_workflow(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Executes the agentic workflow for a given user query.

        Args:
            request (AgentRequest): The validated request payload.

        Returns:
            Dict[str, Any]: The final response from the agent.
        """
        logger.info(f"Executing workflow for session {request.session_id} with query: {request.query}")
        
        try:
            # Invoke the agent executor asynchronously
            response = await self.agent_executor.ainvoke({"input": request.query})
            
            logger.info(f"Workflow execution completed for session {request.session_id}")
            return {
                "status": "success",
                "output": response.get("output", ""),
                "session_id": request.session_id
            }
        except Exception as e:
            logger.error(f"Error executing agent workflow: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "session_id": request.session_id
            }

# Example factory function for tools (to be imported elsewhere)
def create_default_tools() -> List[Tool]:
    """Factory to create the default toolset for the agent."""
    def dummy_search(query: str) -> str:
        """A placeholder for a real search or RAG tool."""
        return f"Simulated search results for: {query}"

    return [
        Tool(
            name="EnterpriseSearch",
            func=dummy_search,
            description="Useful for searching internal enterprise documents."
        )
    ]
