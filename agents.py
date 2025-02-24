import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Annotated, Literal
from typing_extensions import TypedDict
from enum import Enum
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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

from helpers import parse_ai_message
from prompts import get_data_scientist_prompt, get_coder_prompt  # Add import at top with other imports

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
        # Return both code and output as structured response
        return {
            "type": "code",
            "content": f"```python\n{code}\n```\nOutput:\n{result}"
        }
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

@tool
def plot_data_tool(code: Annotated[str, "Python code to generate plots using matplotlib or seaborn"]):
    """Use this to create data visualizations. The code should use matplotlib or seaborn."""
    try:
        # Return plot code as structured response
        return {
            "type": "plot",
            "content": code  # Frontend can re-execute to display
        }
    except Exception as e:
        return f"Failed to generate plot: {str(e)}"

@tool
def stem_expert_tool(question: Annotated[str, "Question to ask the STEM expert"]):
    """Use this to ask a question to a STEM expert."""
    wolfram = WolframAlphaAPIWrapper()
    response = wolfram.run(question)
    return response

@tool
def publish_code_tool(code: Annotated[str, "Python code to publish"]):
    """Use this to publish code that is ready to run."""
    return {
        "type": "code",
        "content": code
    }

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

class ArtifactState(TypedDict):
    type: Literal["code", "plot", "output"]
    content: str
    timestamp: str

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
            return [publish_code_tool]  # Add coding-specific tools
        elif self.agent_type == AgentType.DATA_SCIENTIST:
            return [plot_data_tool]
        elif self.agent_type == AgentType.STEM_EXPERT:
            return [stem_expert_tool, TavilySearchResults(max_results=5), plot_data_tool]
        elif self.agent_type == AgentType.WRITER:
            return []  # Add writing-specific tools
        return []

class AgentGraphManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.graph = None
        self.members: List[str] = []  # Track non-supervisor agents
        self.llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-latest",
            api_key=ANTHROPIC_API_KEY,
            timeout=None,
            stop=None
        )
        self.current_parsed_data = None  # Add this line to track latest parsed data

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
            # Add to members list if not supervisor
            if agent.id not in self.members:
                self.members.append(agent.id)

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

        self.team_info = f"""You are the supervisor agent. Your ONLY role is to evaluate task completion and agent selection.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You MUST return 'END' if ANY of these occur:
   - An agent returns an empty response
   - The same information has been repeated
   - An adequate answer has been provided
   - 3 or more exchanges occur without new information
   - Agents start asking open-ended follow-up questions

2. You MUST NOT:
   - Suggest next steps if the task is complete
   - Continue the conversation after a clear answer
   - Ignore empty responses from agents
   - Allow repetitive cycles of similar responses

Available Team Members:
{chr(10).join(agent_descriptions)}

For each interaction:
1. Check FIRST if any termination conditions are met
2. Only if no termination conditions are met, select the next most appropriate agent
3. Return your decision as:
   'next': Either 'END' or an agent ID"""



    def _create_supervisor_node(self, agent: Agent):
        """Create the supervisor node function."""
        def supervisor_node(state: MessagesState) -> Command:
            try:
                # Extract the last agent response if it exists
                # last_message = state["messages"][-1] if state["messages"] else None
                # last_agent = last_message.name if (isinstance(last_message, AIMessage) and hasattr(last_message, 'name')) else None

                # Add context about previous agent actions to prompt
                # context = ""
                # if last_agent:
                #     context = f"\nLast agent to respond: {last_agent}\nTheir response is in the message history."

                combined_prompt = f"""{agent.system_prompt}

                Team Information:
                {self.team_info}"""

                messages = [
                    SystemMessage(content=combined_prompt),
                ] + state["messages"]

                # print(f"\n[DEBUG] MESSAGES TO SUPERVISOR: {messages}\n")
                # print(f"\n[DEBUG] PROMPT TO SUPERVISOR: \n{combined_prompt}\n")
                # print(f"\n[DEBUG] MESSAGES TO SUPERVISOR: \n{messages}\n")
                options = self.members + ["END"]
                class SupervisorResponse(BaseModel):
                    next: Literal[*options]
                    # reasoning: str

                response = self.llm.with_structured_output(SupervisorResponse).invoke(messages)

                parsed_data = {}

                print(f"\n[DEBUG] SUPERVISOR RESPONSE: {response}\n")
                # reasoning = response.reasoning

                # If END, mark completed and don't schedule new tasks
                if response.next == "END":
                    parsed_data = { 'id': '', 'type': 'ai_message', 'content': [{"type": "text", "text": "Task complete"}], 'code_blocks': [], 'tool_calls': [], 'metadata': { 'model': '', 'stop_reason': '', 'usage': {} } }
                    self.current_parsed_data = [parsed_data]

                    return Command(
                        update={
                            "messages": [AIMessage(content="Task complete", name="supervisor")],
                        }
                    )

                # Otherwise route to next agent
                parsed_data = { 'id': '', 'type': 'ai_message', 'content': [{"type": "text", "text":f"Routing to: {response.next}"}], 'code_blocks': [], 'tool_calls': [], 'metadata': { 'model': '', 'stop_reason': '', 'usage': {} } }
                self.current_parsed_data = [parsed_data]
                return Command(

                    update={
                        "messages": [
                            AIMessage(content=f"Routing to: {response.next}", name="supervisor")
                        ]
                    },
                    goto=response.next
                )
            except Exception as e:
                return Command(
                    update={
                        "messages": [
                            AIMessage(
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

        # Add custom prompt for data scientist
        if agent.agent_type == AgentType.DATA_SCIENTIST:
            agent.system_prompt = get_data_scientist_prompt(agent.tools, agent.system_prompt)
        if agent.agent_type == AgentType.CODER:
            agent.system_prompt = get_coder_prompt(agent.tools, agent.system_prompt)


        # print(f"\n[DEBUG] {agent.agent_type} SYSTEM PROMPT: {agent.system_prompt}\n")

        base_agent = create_react_agent(
            self.llm,
            tools=agent.tools,
            state_modifier=agent.system_prompt
        )

        def node_func(state: MessagesState) -> Command:
            try:
                # print(f"\n[DEBUG] STATE PROMPT FOR AGENT: {state}\n")
                result = base_agent.invoke(state)
                print(f"\n[DEBUG] AGENT RESULT: {result}\n")

                # Find all messages since the last routing command
                all_messages = result["messages"]
                last_routing_idx = -1
                for i, msg in enumerate(reversed(all_messages)):
                    idx = len(all_messages) - 1 - i
                    if (isinstance(msg, AIMessage) and
                        hasattr(msg, 'content') and
                        isinstance(msg.content, str) and
                        msg.content.startswith('Routing to:')):
                        last_routing_idx = idx
                        break

                # Get all messages after the last routing command
                new_messages = all_messages[last_routing_idx + 1:]

                # Parse each new message
                parsed_messages = []
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        parsed_data = parse_ai_message(msg)
                        parsed_messages.append(parsed_data)

                # Store parsed message in manager instance
                self.current_parsed_data = parsed_messages
                print(f"\n[DEBUG] PARSED DATA: {self.current_parsed_data}\n")

                # Get the original content
                last_message = result["messages"][-1]
                content = last_message.content

                # Go through the parsed_data and get the first code_blocks entry if it exists to add to content
                for parsed_data in parsed_messages:
                    if parsed_data.get("code_blocks"):
                        content += f"Generated code:\n\n```python\n{parsed_data['code_blocks'][0]['code']}\n```"
                        break


                next_node = self._get_next_node(agent.id)

                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=content,
                                name=agent.agent_type.value
                            )
                        ]
                    },
                    goto=next_node
                )
            except Exception as e:
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=f"Error in agent {agent.id}: {str(e)}",
                                name=agent.agent_type.value
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

        # Add nodes for all agents
        for agent_id, agent in self.agents.items():
            builder.add_node(agent_id, self._create_agent_node(agent))

        supervisor = self._get_supervisor()
        if supervisor:
            # Start -> Supervisor
            builder.add_edge(START, supervisor.id)

        self.graph = builder.compile()

    async def execute_task(self, task: str, session_id: str):
        """Execute a task using the current graph."""
        if not self.graph:
            raise ValueError("No agent graph has been built yet")

        try:
            async for event in self.graph.astream(
                {"messages": [HumanMessage(content=task)]},
                stream_mode=["values", "debug"],
                debug=True
            ):
                print(f"\n[DEBUG] EVENT: {event}\n")

                # Handle tuple structure of events
                event_type, event_data = event

                if event_type == "values" and event_data.get("messages"):
                    messages = event_data["messages"]
                    if messages:
                        step_data = {
                            "agent": messages[-1].name if hasattr(messages[-1], 'name') else "system",
                            "content": messages[-1].content,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "parsed_data": self.current_parsed_data
                        }

                        print(f"[DEBUG] Yielding step data: {step_data}")
                        yield step_data
                        self.current_parsed_data = None  # Reset after yielding

        except Exception as e:
            error_data = {
                "agent": "system",
                "content": f"Error during task execution: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
                "parsed_data": None
            }
            yield error_data
            raise