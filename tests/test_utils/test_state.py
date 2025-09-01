from langchain_core.messages import HumanMessage

INITIAL_STATE = {
    "messages": [HumanMessage(content="how many artists are there?")],
    "user_query": "how many artists are there?",
    "sql_query": "",
    "sql_result": None,
     "final_answer": ""
}

TEST_STATES = {
    "c_1": {
        "messages": [HumanMessage(content="how many artists are there?")],
        "user_query": "how many artists are there?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "c_2": {
        "messages": [HumanMessage(content="How many albums exist?")],
        "user_query": "How many albums exist?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "c_3": {
        "messages": [HumanMessage(content="What's the highest track price?")],
        "user_query": "What's the highest track price?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "d_1": {
        "messages": [HumanMessage(content="What's the average invoice total per country, excluding countries with less than 5 invoices?")],
        "user_query": "What's the average invoice total per country, excluding countries with less than 5 invoices?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "e_1": {
        "messages": [HumanMessage(content="Which customers bought tracks by Metallica? I want a list of all the customers.")],
        "user_query": "Which customers bought tracks by Metallica? I want a list of all the customers.",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "e_2": {
        "messages": [HumanMessage(content="Which sales rep has the highest business revenue and how much?")],
        "user_query": "Which sales rep has the highest business revenue and how much?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "f_1": {
        "messages": [HumanMessage(content="Which employees were hired between 2002 and 2003?")],
        "user_query": "Which employees were hired between 2002 and 2003?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    },
    "f_2": {
        "messages": [HumanMessage(content="Who won the world cup in 2022?")],
        "user_query": "Who won the world cup in 2022?",
        "sql_query": "",
        "sql_result": None,
        "final_answer": ""
    }
}