
import uuid

import streamlit as st

from langchain_core.messages import HumanMessage,AIMessage

from langgraph_backend import chatbot,checkpointer

def retrieve_all_threads():
    threads = set()

    try:
        for cp in checkpointer.list(None):
            thread_id = cp.config.get(
                "configurable",
                {}
            ).get("thread_id")

            if thread_id:
                threads.add(thread_id)

    except Exception:
        pass

    return sorted(list(threads))

def get_messages(thread_id: str):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    if state is None:
        return []

    return state.values.get("messages", [])

def generate_thread_id():
    return str(uuid.uuid4())


def create_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state.thread_id = thread_id
    st.session_state.message_history = []

    create_thread(thread_id)


def load_chat(thread_id):
    messages = get_messages(thread_id)

    history = []

    for message in messages:
        role = (
            "user"
            if isinstance(message, HumanMessage)
            else "assistant"
        )

        history.append(
            {
                "role": role,
                "content": message.content
            }
        )

    st.session_state.thread_id = thread_id
    st.session_state.message_history = history


if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

create_thread(st.session_state.thread_id)

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("Conversations")

for thread_id in st.session_state.chat_threads:
    if st.sidebar.button(
        thread_id,
        key=f"thread_{thread_id}"
    ):
        load_chat(thread_id)
        st.rerun()

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Type your message..."
)

if user_input:
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        },
        'metadata':{
            "thread_id": st.session_state.thread_id
        },
        'run_name':'chat_turn'
    }

    with st.chat_message("assistant"):
        def only_ai_messages():
            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=config,
                stream_mode="messages"
            ):
                if isinstance(chunk,AIMessage):
                    yield chunk.content
        
        response=st.write_stream(only_ai_messages())

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )
