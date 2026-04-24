from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# This model defines the structured output we want the supervisor to produce
class RouteChoice(BaseModel):
    next: str = Field(
        description="The next agent to route to. Choose from 'calendar', 'weather', 'general', or 'FINISH'."
    )

def get_supervisor_agent(llm):
    """
    Creates a supervisor node that uses structure output to pick the next step.
    """
    system_prompt = (
        "You are the Supervisor Agent in a multi-agent system. "
        "Your task is to route the user's request to the correct specialist.\n\n"
        "Options:\n"
        "- 'calendar': For ANYTHING related to the user's calendar, events, meetings, schedules, "
        "appointments, birthdays, reminders, event lookups, or checking what's on a specific date. "
        "This includes questions like 'when is X birthday', 'do I have any events', 'what's on my calendar'.\n"
        "- 'weather': For current weather conditions, forecasts, or weather-related queries.\n"
        "- 'general': ONLY for generic talk, knowledge questions, or if no other agent matches.\n"
        "- 'FINISH': If the user's request has been fully answered or the conversation has naturally concluded.\n\n"
        "When in doubt between calendar and general, prefer 'calendar' if the query mentions dates, events, or personal schedules.\n"
        "Review the conversation history and select the next appropriate agent."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    
    # We bind the LLM to output the RouteChoice struct
    supervisor_chain = prompt | llm.with_structured_output(RouteChoice)
    
    return supervisor_chain
