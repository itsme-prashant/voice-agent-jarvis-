from langchain_core.prompts import ChatPromptTemplate

def get_general_agent(llm):
    """
    Returns a general purpose agent/node without specific tools.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful General Assistant within a larger multi-agent system. "
                   "Address the user's queries directly and conversationally. "
                   "If they ask you to perform actions outside your scope, kindly inform them you cannot."),
        ("placeholder", "{messages}")
    ])
    
    return prompt | llm
