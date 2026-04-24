from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .supervisor import get_supervisor_agent
from agents import get_calendar_agent, get_weather_agent, get_general_agent
from tools import schedule_meeting, check_availability, search_events, delete_event, reschedule_event, get_weather
from langchain_core.messages import HumanMessage, AIMessage

def build_workflow(llm):
    workflow = StateGraph(AgentState)
    
    # 1. Initialize our agents and supervisor
    supervisor_chain = get_supervisor_agent(llm)
    calendar_node = get_calendar_agent(llm)
    weather_node = get_weather_agent(llm)
    general_node = get_general_agent(llm)
    
    # 2. Tool nodes setup
    calendar_tools_node = ToolNode([schedule_meeting, check_availability, search_events, delete_event, reschedule_event])
    weather_tools_node = ToolNode([get_weather])
    
    # 3. Define the node functions
    def supervisor(state: AgentState):
        result = supervisor_chain.invoke(state)
        return {"next": result.next}
        
    def calendar(state: AgentState):
        result = calendar_node.invoke(state)
        return {"messages": [result]}
        
    def weather(state: AgentState):
        result = weather_node.invoke(state)
        return {"messages": [result]}
        
    def general(state: AgentState):
        result = general_node.invoke(state)
        return {"messages": [result]}

    # 4. Add Nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("calendar", calendar)
    workflow.add_node("weather", weather)
    workflow.add_node("general", general)
    workflow.add_node("calendar_tools", calendar_tools_node)
    workflow.add_node("weather_tools", weather_tools_node)
    
    # 5. Routing Logic
    
    # START -> Supervisor (route to the right agent)
    workflow.add_edge(START, "supervisor")
    
    # Supervisor -> Agent or END
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "calendar": "calendar",
            "weather": "weather",
            "general": "general",
            "FINISH": END
        }
    )
    
    # Calendar agent: if it wants to call a tool -> calendar_tools, otherwise -> END
    def route_calendar(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "calendar_tools"
        return "end"
        
    workflow.add_conditional_edges(
        "calendar",
        route_calendar,
        {"calendar_tools": "calendar_tools", "end": END}
    )
    
    # Weather agent: if it wants to call a tool -> weather_tools, otherwise -> END
    def route_weather(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "weather_tools"
        return "end"

    workflow.add_conditional_edges(
        "weather",
        route_weather,
        {"weather_tools": "weather_tools", "end": END}
    )
    
    # General agent has no tools, always finishes
    workflow.add_edge("general", END)
    
    # After tools execute, go back to the agent to formulate a final response
    workflow.add_edge("calendar_tools", "calendar")
    workflow.add_edge("weather_tools", "weather")
    
    return workflow
