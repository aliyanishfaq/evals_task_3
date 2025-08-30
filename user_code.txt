import sqlite3
import requests
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
import json

# State definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sql_query: str
    query_result: str
    error: str

# Initialize LLM (lazy loading to avoid API key error on import)
llm = None

def get_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm

# Database setup and schema extraction
def setup_database():
    """Fetch Chinook database and set up in-memory SQLite database"""
    # Fetch the SQL script
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text
    
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Execute the SQL script
    cursor.executescript(sql_script)
    conn.commit()
    
    return conn

def get_database_schema(conn):
    """Extract database schema information"""
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        table_info = f"Table: {table_name}\n"
        table_info += "Columns:\n"
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            is_nullable = "NULL" if col[3] == 0 else "NOT NULL"
            is_pk = "PRIMARY KEY" if col[5] == 1 else ""
            table_info += f"  - {col_name}: {col_type} {is_nullable} {is_pk}\n"
        
        schema_info.append(table_info)
    
    return "\n".join(schema_info)

# Initialize database globally
conn = setup_database()
db_schema = get_database_schema(conn)

# Agent nodes
def generate_sql(state: State) -> State:
    """Generate SQL query from natural language"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Convert natural language questions to SQL queries for the Chinook database.
        
Database Schema:
{schema}

Instructions:
1. Generate ONLY the SQL query, nothing else
2. Use proper SQL syntax for SQLite
3. If the question cannot be answered using the database, respond with "CANNOT_ANSWER"
4. Make sure to use correct table and column names from the schema
5. For text comparisons, use LIKE with % wildcards when appropriate
6. Always limit results to reasonable numbers (max 20) unless specifically asked for more"""),
        ("human", "{question}")
    ])
    
    response = get_llm().invoke(prompt.format_messages(schema=db_schema, question=last_message))
    sql_query = response.content.strip()
    
    # Clean up the SQL query
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()
    
    state["sql_query"] = sql_query
    return state

def execute_sql(state: State) -> State:
    """Execute SQL query against the database"""
    sql_query = state.get("sql_query", "")
    
    if not sql_query or sql_query == "CANNOT_ANSWER":
        state["error"] = "Cannot answer this question using the database"
        return state
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Format results
        if results:
            formatted_results = []
            formatted_results.append(" | ".join(column_names))
            formatted_results.append("-" * 50)
            for row in results[:20]:  # Limit display to 20 rows
                formatted_results.append(" | ".join(str(val) for val in row))
            
            if len(results) > 20:
                formatted_results.append(f"... and {len(results) - 20} more rows")
            
            state["query_result"] = "\n".join(formatted_results)
        else:
            state["query_result"] = "Query executed successfully but returned no results"
        
        state["error"] = ""
    except Exception as e:
        state["error"] = f"SQL execution error: {str(e)}"
        state["query_result"] = ""
    
    return state

def generate_response(state: State) -> State:
    """Generate natural language response based on query results"""
    messages = state["messages"]
    original_question = messages[-1].content if messages else ""
    sql_query = state.get("sql_query", "")
    query_result = state.get("query_result", "")
    error = state.get("error", "")
    
    if error:
        if "Cannot answer" in error:
            response_text = "I don't know the answer to that question. I can only answer questions about the music store data in the Chinook database."
        else:
            response_text = f"I encountered an error while processing your request: {error}"
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about a music store database.
            
Given the original question, SQL query, and results, provide a clear and concise answer in natural language.
Be specific and include relevant details from the results.
If the results are empty, say so clearly."""),
            ("human", """Original question: {question}

SQL query executed:
{sql_query}

Results:
{results}

Please provide a natural language answer to the original question based on these results.""")
        ])
        
        response = get_llm().invoke(prompt.format_messages(
            question=original_question,
            sql_query=sql_query,
            results=query_result
        ))
        response_text = response.content
    
    # Add the response to messages
    state["messages"].append(AIMessage(content=response_text))
    return state

# Build the graph
def build_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()

# Export the compiled graph as 'app'
app = build_graph()

# Test function for local testing
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "How many customers are there?",
        "What are the top 5 best selling albums?",
        "Which artist has the most albums?",
        "What is the weather today?",  # This should trigger "don't know"
    ]
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("-" * 50)
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "sql_query": "",
            "query_result": "",
            "error": ""
        }
        
        result = app.invoke(initial_state)
        final_message = result["messages"][-1].content
        print(f"Answer: {final_message}")