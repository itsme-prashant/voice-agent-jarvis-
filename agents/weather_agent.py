from langchain_core.prompts import ChatPromptTemplate
from tools import get_weather

def get_weather_agent(llm):
    """
    Returns a weather specialized agent/node.
    Binds the weather tools to the provided LLM.
    """
    tools = [get_weather]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Weather Assistant. Your job is to answer weather-related queries by using the get_weather tool. Do not guess the weather; always use the tool. Convert locations to standard city formats if necessary."),
        ("placeholder", "{messages}")
    ])
    
    model_with_tools = llm.bind_tools(tools)
    
    return prompt | model_with_tools
