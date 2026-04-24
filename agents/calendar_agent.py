from langchain_core.prompts import ChatPromptTemplate
from tools import schedule_meeting, check_availability, search_events, delete_event, reschedule_event

def get_calendar_agent(llm):
    """
    Returns a calendar specialized agent/node.
    Binds the calendar tools to the provided LLM.
    """
    tools = [schedule_meeting, check_availability, search_events, delete_event, reschedule_event]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Calendar Assistant connected to the user's real Google Calendar.\n\n"
         "TOOLS AVAILABLE:\n"
         "1. schedule_meeting(title, datetime_start, duration_minutes, attendees) — Use this to CREATE a new event. "
         "The datetime_start MUST be in ISO format like '2026-04-24T13:00:00'. Convert user's natural language dates/times to ISO format yourself.\n"
         "2. check_availability(date) — Use this to CHECK what events exist on a date (format: YYYY-MM-DD).\n"
         "3. search_events(query) — Use this to SEARCH for events by keyword.\n"
         "4. delete_event(event_id) — Use this to CANCEL/DELETE an event. You must FIRST find the event ID using search_events or check_availability.\n"
         "5. reschedule_event(event_id, new_datetime_start, duration_minutes) — Use this to RESCHEDULE an event. FIRST find the event ID.\n\n"
         "RULES:\n"
         "- If the user asks to SCHEDULE/CREATE/BOOK a meeting, use schedule_meeting IMMEDIATELY. Do NOT call check_availability first.\n"
         "- If the user asks to CANCEL, DELETE, or RESCHEDULE, ALWAYS use search_events or check_availability first to find the exact event ID, then call delete_event or reschedule_event.\n"
         "- If the user provides a title, date, and time to schedule, you have everything needed — call schedule_meeting right away.\n"
         "- If the user says 'schedule X on April 24 at 1pm for 5 hours', convert to: title='X', datetime_start='2026-04-24T13:00:00', duration_minutes=300.\n"
         "- Only call check_availability if the user explicitly asks about their availability or free slots.\n"
         "- ALWAYS use your tools — never guess or say you don't have access."
        ),
        ("placeholder", "{messages}")
    ])
    
    model_with_tools = llm.bind_tools(tools)
    
    return prompt | model_with_tools
