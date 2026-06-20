from typing import TypedDict, Annotated
import sqlite3
import yfinance as yf

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

search = DuckDuckGoSearchRun(region="us-en")


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        result = search.invoke(query)
        return result if result else "No search results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform arithmetic operations: add, sub, mul, div."""
    try:
        operation = operation.lower()

        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not possible"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a ticker symbol."""
    try:
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period="1d")

        if data.empty:
            return f"No stock data found for {symbol.upper()}"

        price = data["Close"].iloc[-1]
        return f"Current price of {symbol.upper()} is ${price:.2f}"

    except Exception as e:
        return f"Stock lookup failed: {str(e)}"


tools = [web_search, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}


builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

chatbot = builder.compile(checkpointer=checkpointer)
