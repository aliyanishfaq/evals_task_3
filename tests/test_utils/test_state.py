from langchain_core.messages import HumanMessage

INITIAL_STATE = {
    "messages": [HumanMessage(content="Research Anthropic")],
    "company_name": "Anthropic",
    "notes": "Focus on AI safety research",
    "company_info": None,
    "search_queries_used": 0,
    "search_results": [],
    "reflection_count": 0,
    "is_complete": False
}

MINIMAL_STATE = {
    "messages": [HumanMessage(content="Research Anthropic")],
    "company_name": "Anthropic",
}

TEST_STATES = {
    "c_1": {
        "user_query": "how many artists are there?",
    },
    "c_2": {
        "user_query": "How many albums exist?",
    },
    "c_3": {
        "user_query": "What's the highest track price?",
    },
    "d_1": {
        "user_query": "What's the average invoice total per country, excluding countries with less than 5 invoices?",
    },
    "e_1": {
        "user_query": "Which customers bought tracks by Metallica? I want a list of all the customers.",
    },
    "e_2": {
        "user_query": "Which sales rep has the highest business revenue and how much?",
    },
    "f_1": {
        "user_query": "Which employees were hired between 2002 and 2003?",
    },
    "f_2": {
        "user_query": "Who won the world cup in 2022?",
    }
}