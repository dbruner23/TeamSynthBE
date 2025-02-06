import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from enum import Enum
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Literal
from langgraph.types import Command

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WOLFRAM_ALPHA_APPID = os.getenv("WOLFRAM_ALPHA_APPID")

repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "Python code to execute."]):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str

@tool
def plot_data_tool(code: Annotated[str, "Python code to generate plots using matplotlib or seaborn"]):
    """Use this to create data visualizations. The code should use matplotlib or seaborn."""
    try:
        result = repl.run(code)
        return "Plot generated successfully"
    except Exception as e:
        return f"Failed to generate plot: {str(e)}"

@tool
def stem_expert_tool(question: Annotated[str, "Question to ask the STEM expert"]):
    """Use this to ask a question to a STEM expert."""
    wolfram = WolframAlphaAPIWrapper()
    response = wolfram.run(question)
    return response

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    WEB_RESEARCHER = "web_researcher"
    CODER = "coder"
    WRITER = "writer"
    DATA_SCIENTIST = "data_scientist"
    STEM_EXPERT = "stem_expert"

class RelationType(str, Enum):
    SUPERVISES = "supervises"
    REPORTS_TO = "reports_to"
    COLLABORATES = "collaborates_with"

class SupervisorResponse(BaseModel):
    next: str
    reasoning: str

@dataclass
class AgentRelation:
    from_agent: str
    to_agent: str
    relation_type: RelationType

@dataclass
class Agent:
    id: str
    agent_type: AgentType
    system_prompt: str
    relationships: List[AgentRelation]
    tools: List = None

    def __post_init__(self):
        self.tools = self._get_default_tools()

    def _get_default_tools(self) -> List:
        """Assign default tools based on agent type."""
        if self.agent_type == AgentType.WEB_RESEARCHER:
            return [TavilySearchResults(max_results=5)]
        elif self.agent_type == AgentType.CODER:
            return [python_repl_tool]  # Add coding-specific tools
        elif self.agent_type == AgentType.DATA_SCIENTIST:
            return [python_repl_tool, plot_data_tool]
        elif self.agent_type == AgentType.STEM_EXPERT:
            return [stem_expert_tool, TavilySearchResults(max_results=5), python_repl_tool]
        elif self.agent_type == AgentType.WRITER:
            return []  # Add writing-specific tools
        return []

class AgentGraphManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.graph = None
        self.llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-latest",
            api_key=ANTHROPIC_API_KEY,
            timeout=None,
            stop=None
        )

    def _get_supervisor(self) -> Optional[Agent]:
        """Find and return the supervisor agent if it exists."""
        return next(
            (agent for agent in self.agents.values() if agent.agent_type == AgentType.SUPERVISOR),
            None
        )

    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the graph. If adding a supervisor when one already exists,
        replace the old supervisor and transfer its relationships."""
        if agent.agent_type == AgentType.SUPERVISOR:
            # Find and remove existing supervisor if any
            existing_supervisor = self._get_supervisor()
            if existing_supervisor:
                # Transfer relationships from old supervisor to new one
                for other_agent in self.agents.values():
                    if other_agent.id != existing_supervisor.id:
                        # Update relationships that involved the old supervisor
                        other_agent.relationships = [
                            rel if rel.to_agent != existing_supervisor.id else
                            AgentRelation(rel.from_agent, agent.id, rel.relation_type)
                            for rel in other_agent.relationships
                        ]
                # Remove old supervisor
                del self.agents[existing_supervisor.id]
        else:
            # If this is not a supervisor and has no relationships, create default relationship
            if not agent.relationships:
                supervisor = self._get_supervisor()
                if supervisor:
                    agent.relationships = [
                        AgentRelation(agent.id, supervisor.id, RelationType.REPORTS_TO)
                    ]

        # Add reciprocal relationships
        for other_agent in self.agents.values():
            for rel in other_agent.relationships:
                if rel.to_agent == agent.id:
                    # Add reciprocal relationship
                    if rel.relation_type == RelationType.REPORTS_TO:
                        agent.relationships.append(
                            AgentRelation(agent.id, rel.from_agent, RelationType.SUPERVISES)
                        )
                    elif rel.relation_type == RelationType.SUPERVISES:
                        agent.relationships.append(
                            AgentRelation(agent.id, rel.from_agent, RelationType.REPORTS_TO)
                        )
                    elif rel.relation_type == RelationType.COLLABORATES:
                        agent.relationships.append(
                            AgentRelation(agent.id, rel.from_agent, RelationType.COLLABORATES)
                        )
        self.agents[agent.id] = agent
        self._update_supervisor_prompt()
        self._rebuild_graph()

    def _update_supervisor_prompt(self) -> None:
        """Update the supervisor's system prompt based on current agents."""
        agent_descriptions = []
        for agent_id, agent in self.agents.items():
            tools = [t.__class__.__name__ for t in (agent.tools or [])]
            desc = (f"- {agent_id} ({agent.agent_type.value}): {agent.system_prompt}"
                   f"\n  Tools: {', '.join(tools) if tools else 'None'}")
            agent_descriptions.append(desc)

        self.team_info = f"""Available agents:
        {chr(10).join(agent_descriptions)}

        For each interaction:
        1. Analyze the current state and messages
        2. Decide which agent should act next based on their capabilities and the task needs
        3. Return your decision as a structured response with:
        - 'next': The ID of the next agent to act (or 'END' if task is complete)
        - 'reasoning': Brief explanation of your choice

        Consider each agent's:
        - Specialization (agent_type)
        - Available tools
        - System prompt
        - Previous contributions to the task"""

    def _create_supervisor_node(self, agent: Agent):
        """Create the supervisor node function."""
        def supervisor_node(state: MessagesState) -> Command:
            try:
                combined_prompt = f"""{agent.system_prompt}

                Team Information:
                {self.team_info}"""

                messages = [
                    SystemMessage(content=combined_prompt)
                ] + state["messages"]

                response = self.llm.with_structured_output(SupervisorResponse).invoke(messages)

                next_agent = response.next
                reasoning = response.reasoning

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Supervisor decision: {reasoning}",
                                name="supervisor"
                            )
                        ]
                    },
                    goto=END if next_agent == "END" else next_agent
                )
            except Exception as e:
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Supervisor error: {str(e)}",
                                name="supervisor"
                            )
                        ]
                    },
                    goto=END
                )

        return supervisor_node

    def _create_agent_node(self, agent: Agent):
        """Create a node function for an agent."""
        if agent.agent_type == AgentType.SUPERVISOR:
            return self._create_supervisor_node(agent)

        base_agent = create_react_agent(
            self.llm,
            tools=agent.tools,
            state_modifier=agent.system_prompt
        )

        def node_func(state: MessagesState) -> Command:
            try:
                result = base_agent.invoke(state)
                next_node = self._get_next_node(agent.id)
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=result["messages"][-1].content,
                                name=agent.id
                            )
                        ]
                    },
                    goto=next_node
                )
            except Exception as e:
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Error in agent {agent.id}: {str(e)}",
                                name=agent.id
                            )
                        ]
                    },
                    goto=END
                )

        return node_func

    def _get_next_node(self, agent_id: str) -> str:
        """Determine the next node based on agent relationships."""
        agent = self.agents[agent_id]

        # Find supervisor relationship if it exists
        for rel in agent.relationships:
            if rel.relation_type == RelationType.REPORTS_TO:
                return rel.to_agent

        # If no supervisor found, end the chain
        return END

    def _rebuild_graph(self) -> None:
        """Rebuild the entire graph based on current agents and relationships."""
        builder = StateGraph(MessagesState)

        # Add initial edge from START to supervisor if it exists
        supervisor = self._get_supervisor()
        if supervisor:
            builder.add_edge(START, supervisor.id)

        # Add all agents as nodes
        for agent_id, agent in self.agents.items():
            builder.add_node(agent_id, self._create_agent_node(agent))

        # Add edges based on relationships
        for agent in self.agents.values():
            for rel in agent.relationships:
                builder.add_edge(rel.from_agent, rel.to_agent)

        self.graph = builder.compile()

    async def execute_task(self, task: str, session_id: str):
        """Execute a task using the current graph."""
        if not self.graph:
            raise ValueError("No agent graph has been built yet")

        print(f"\n[DEBUG] Starting new task execution: {task}\n")

        async for event in self.graph.astream(
            {"messages": [("user", task)]},
            stream_mode="values",
        ):
            # Get session manager instance
            from session_manager import SessionManager
            session_manager = SessionManager()

            # Check if task was cancelled
            # if not session_manager.is_task_active(session_id):
            #     break

            # Format the step data
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                step_data = {
                    "agent": last_message.name if hasattr(last_message, 'name') else "system",
                    "content": last_message.content,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                print(f"[DEBUG] Formatted step data: {step_data}\n")
                yield step_data