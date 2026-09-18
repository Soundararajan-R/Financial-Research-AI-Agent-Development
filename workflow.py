from typing import TypedDict, Annotated, Literal
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from agents import FinancialAgent

# Define workflow state structure
class FinancialState(TypedDict):
    messages: Annotated[list, add_messages]

# Conditional edge function: check if LLM generated tool calls
def route_tools(state: FinancialState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

# Instantiate agent & graph builder
agent = FinancialAgent()
workflow_builder = StateGraph(FinancialState)

# Add nodes to graph
workflow_builder.add_node("agent", agent.agent_node)
workflow_builder.add_node("tools", agent.tool_node)

# Connect graph edges
workflow_builder.add_edge(START, "agent")
workflow_builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow_builder.add_edge("tools", "agent")

# Compile graph into executable runner
financial_graph = workflow_builder.compile()

if __name__ == "__main__":
    print("[OK] Financial StateGraph workflow compiled successfully!")
