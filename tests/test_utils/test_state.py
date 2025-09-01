from langchain_core.messages import HumanMessage

INITIAL_STATE = {
    "messages": [HumanMessage(content="how many artists are there?")]
}

TEST_STATES = {
    "c_1": {
        "messages": [HumanMessage(content="how many artists are there?")]
    },
    "c_2": {
        "messages": [HumanMessage(content="How many albums exist?")]
    },
    "c_3": {
        "messages": [HumanMessage(content="What's the highest track price?")]
    },
    "d_1": {
        "messages": [HumanMessage(content="What's the average invoice total per country, excluding countries with less than 5 invoices?")]
    },
    "e_1": {
        "messages": [HumanMessage(content="Which customers bought tracks by Metallica? I want a list of all the customers.")]
    },
    "e_2": {
        "messages": [HumanMessage(content="Which sales rep has the highest business revenue and how much?")]
    },
    "f_1": {
            "messages": [HumanMessage(content="Which employees were hired between 2002 and 2003?")]
    },
    "f_2": {
        "messages": [HumanMessage(content="Who won the world cup in 2022?")]
    }
}