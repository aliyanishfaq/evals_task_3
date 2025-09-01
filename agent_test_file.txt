import sqlite3
import requests
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


class SqlQuery(BaseModel):
    """Structured output for SQL query generation"""
    query: str = Field(description="The SQL query to execute")
    reasoning: str = Field(description="Explanation of why this query answers the question")


class DatabaseSchema:
    """Manages the Chinook database schema and connection"""
    
    def __init__(self):
        self.connection = None
        self.schema_info = self._get_schema_info()
        self._setup_database()
    
    def _setup_database(self):
        """Download and setup the Chinook database in memory"""
        try:
            # Fetch the SQL script
            response = requests.get("https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql")
            response.raise_for_status()
            sql_script = response.text
            
            # Create in-memory database
            self.connection = sqlite3.connect(":memory:")
            self.connection.executescript(sql_script)
            self.connection.commit()
            
        except Exception as e:
            raise Exception(f"Failed to setup Chinook database: {e}")
    
    def _get_schema_info(self) -> str:
        """Returns detailed schema information for the AI model"""
        return """
CHINOOK DATABASE SCHEMA:

Tables and Relationships:

1. Artist
   - ArtistId (INTEGER, PRIMARY KEY)
   - Name (NVARCHAR)

2. Album  
   - AlbumId (INTEGER, PRIMARY KEY)
   - Title (NVARCHAR)
   - ArtistId (INTEGER, FOREIGN KEY -> Artist.ArtistId)

3. Track
   - TrackId (INTEGER, PRIMARY KEY)
   - Name (NVARCHAR)
   - AlbumId (INTEGER, FOREIGN KEY -> Album.AlbumId)
   - MediaTypeId (INTEGER, FOREIGN KEY -> MediaType.MediaTypeId)
   - GenreId (INTEGER, FOREIGN KEY -> Genre.GenreId)
   - Composer (NVARCHAR)
   - Milliseconds (INTEGER)
   - Bytes (INTEGER) 
   - UnitPrice (NUMERIC)

4. Genre
   - GenreId (INTEGER, PRIMARY KEY)
   - Name (NVARCHAR)

5. MediaType
   - MediaTypeId (INTEGER, PRIMARY KEY)
   - Name (NVARCHAR)

6. Playlist
   - PlaylistId (INTEGER, PRIMARY KEY)
   - Name (NVARCHAR)

7. PlaylistTrack
   - PlaylistId (INTEGER, FOREIGN KEY -> Playlist.PlaylistId)
   - TrackId (INTEGER, FOREIGN KEY -> Track.TrackId)

8. Customer
   - CustomerId (INTEGER, PRIMARY KEY)
   - FirstName (NVARCHAR)
   - LastName (NVARCHAR)
   - Company (NVARCHAR)
   - Address (NVARCHAR)
   - City (NVARCHAR)
   - State (NVARCHAR)
   - Country (NVARCHAR)
   - PostalCode (NVARCHAR)
   - Phone (NVARCHAR)
   - Fax (NVARCHAR)
   - Email (NVARCHAR)
   - SupportRepId (INTEGER, FOREIGN KEY -> Employee.EmployeeId)

9. Employee
   - EmployeeId (INTEGER, PRIMARY KEY)
   - LastName (NVARCHAR)
   - FirstName (NVARCHAR)
   - Title (NVARCHAR)
   - ReportsTo (INTEGER, FOREIGN KEY -> Employee.EmployeeId)
   - BirthDate (DATETIME)
   - HireDate (DATETIME)
   - Address (NVARCHAR)
   - City (NVARCHAR)
   - State (NVARCHAR)
   - Country (NVARCHAR)
   - PostalCode (NVARCHAR)
   - Phone (NVARCHAR)
   - Fax (NVARCHAR)
   - Email (NVARCHAR)

10. Invoice
    - InvoiceId (INTEGER, PRIMARY KEY)
    - CustomerId (INTEGER, FOREIGN KEY -> Customer.CustomerId)
    - InvoiceDate (DATETIME)
    - BillingAddress (NVARCHAR)
    - BillingCity (NVARCHAR)
    - BillingState (NVARCHAR)
    - BillingCountry (NVARCHAR)
    - BillingPostalCode (NVARCHAR)
    - Total (NUMERIC)

11. InvoiceLine
    - InvoiceLineId (INTEGER, PRIMARY KEY)
    - InvoiceId (INTEGER, FOREIGN KEY -> Invoice.InvoiceId)
    - TrackId (INTEGER, FOREIGN KEY -> Track.TrackId)
    - UnitPrice (NUMERIC)
    - Quantity (INTEGER)

Key Business Relationships:
- Artists have Albums, Albums have Tracks
- Customers make Purchases (Invoices) containing InvoiceLines
- Tracks belong to Genres and MediaTypes
- Employees can be assigned as Support Representatives to Customers
- Playlists contain multiple Tracks through PlaylistTrack junction table
"""

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
        except Exception as e:
            raise Exception(f"SQL execution error: {e}")


# Initialize database
db = DatabaseSchema()

@tool
def execute_sql_query(query: str) -> str:
    """Execute a SQL query against the Chinook database and return results.
    
    Args:
        query: The SQL query to execute
        
    Returns:
        String representation of query results
    """
    try:
        results = db.execute_query(query)
        if not results:
            return "No results found."
        
        # Format results for display
        if len(results) == 1:
            return f"Result: {results[0]}"
        else:
            formatted_results = []
            for i, result in enumerate(results[:10]):  # Limit to first 10 results
                formatted_results.append(f"{i+1}. {result}")
            
            result_str = "\n".join(formatted_results)
            if len(results) > 10:
                result_str += f"\n... and {len(results) - 10} more results"
            
            return result_str
            
    except Exception as e:
        return f"Error executing query: {str(e)}"


def create_text_to_sql_agent():
    """Create a text-to-SQL agent using LangGraph"""
    
    # Initialize the model
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # Create system prompt with schema information
    system_prompt = f"""You are a text-to-SQL assistant for the Chinook music database. Your job is to:

1. Convert natural language questions into SQL queries
2. Execute those queries against the database  
3. Provide natural language responses based on the results

{db.schema_info}

IMPORTANT RULES:
- ONLY answer questions that can be answered using the Chinook database
- If a question cannot be answered with the available data, respond: "I don't know the answer to that question."
- Always use proper SQL syntax for SQLite
- Be precise with table and column names
- Use JOINs when querying across multiple tables
- Limit results to reasonable numbers (use LIMIT clause when appropriate)
- Provide clear, natural language explanations of your findings

When you receive a question:
1. Determine if it can be answered using the Chinook database
2. If yes, generate and execute the appropriate SQL query using the execute_sql_query tool
3. Interpret the results and provide a natural language response
4. If no, respond that you don't know the answer"""

    # Create the agent using prebuilt create_react_agent
    agent = create_react_agent(
        model=model,
        tools=[execute_sql_query],
        prompt=system_prompt
    )
    
    return agent


# Create and export the app
app = create_text_to_sql_agent()