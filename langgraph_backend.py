from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict,Literal,Annotated

from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

#define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

#defien node
def chat_node(state: ChatState) -> ChatState:

    messages = state['messages']
    response=llm.invoke(messages)
    return {'messages':[response]}

#define graph
graph=StateGraph(ChatState)

#add node
graph.add_node('chat_node',chat_node)

#add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

#add checkpoint
checkpoint=InMemorySaver()

#compile graph
chatbot=graph.compile(checkpointer=checkpoint)



