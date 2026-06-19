import uuid
import json
import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot


def generate_thread_id():
    return str(uuid.uuid4())


def initialize_session():
    defaults = {
        "thread_id": generate_thread_id(),
        "message_history": [],
        "chat_threads": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    create_thread(st.session_state["thread_id"])


def create_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def get_config():
    return {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


def get_graph_state(thread_id):
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )

        if state and hasattr(state, "values"):
            return state.values.get("messages", [])

    except Exception:
        pass

    return []


def convert_messages(messages):
    history = []

    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"

        history.append(
            {
                "role": role,
                "content": message.content
            }
        )

    return history


def load_chat(thread_id):
    messages = get_graph_state(thread_id)

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = convert_messages(messages)


def new_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    create_thread(thread_id)


def delete_chat(thread_id):
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)

    if st.session_state["thread_id"] == thread_id:
        new_chat()


def export_chat():
    return json.dumps(
        st.session_state["message_history"],
        indent=4
    )


def display_chat():
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def save_message(role, content):
    st.session_state["message_history"].append(
        {
            "role": role,
            "content": content
        }
    )


def stream_response(user_input):
    config = get_config()

    response = st.write_stream(
        chunk.content
        for chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        )
        if hasattr(chunk, "content") and chunk.content
    )

    return response


def sidebar():
    st.sidebar.title("LangGraph Chatbot")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("New Chat", use_container_width=True):
            new_chat()
            st.rerun()

    with col2:
        st.download_button(
            "Export",
            export_chat(),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

    st.sidebar.divider()

    st.sidebar.subheader("Conversations")

    for idx, thread_id in enumerate(
        reversed(st.session_state["chat_threads"]),
        start=1,
    ):
        col1, col2 = st.sidebar.columns([4, 1])

        with col1:
            if st.button(
                f"Chat {idx}",
                key=f"load_{thread_id}",
                use_container_width=True,
            ):
                load_chat(thread_id)
                st.rerun()

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_{thread_id}",
                use_container_width=True,
            ):
                delete_chat(thread_id)
                st.rerun()

    st.sidebar.divider()

    


def main():
    st.set_page_config(
        page_title="LangGraph Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    initialize_session()

    sidebar()

    st.title("🤖 LangGraph Chatbot")

    display_chat()

    user_input = st.chat_input(
        "Ask anything..."
    )

    if not user_input:
        return

    save_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = stream_response(user_input)

    save_message("assistant", response)


if __name__ == "__main__":
    main()