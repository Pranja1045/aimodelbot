import streamlit as st
from google import generativeai as genai
from pymongo import MongoClient
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
if not GEMINI_API_KEY and MONGODB_URI:
    GEMINI_API_KEY=st.secrets["api_key"]
    MONGODB_URI = st.secrets["MONGODB_URI"]


# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Use the latest available Gemini Flash model

try:
    client = MongoClient(MONGODB_URI)
    db = client["chatbot"]
    sessions = db["sessions"]
    messages = db["messages"]
except Exception as e:
    st.error(f"Cannot connect to MongoDB: {e}")
    st.stop()

st.set_page_config(page_title="AI Customer Support Bot", layout="wide")
st.title("AI Customer Support Bot")

# Define your bot greeting
bot_greeting = "Hello! Welcome to our support chat. How can I assist you today?"

with st.sidebar:
    st.header("User Settings")
    user_id = st.text_input("User ID", value="guest")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    if st.button("Reset Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        # Add bot greeting to new chat history
        st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]

if "session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "status": "active"
    })
    # Add bot greeting to new chat history and database
    messages.insert_one({
        "session_id": session_id,
        "sender": "assistant",
        "content": bot_greeting,
        "timestamp": datetime.utcnow()
    })
    st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]

def save_message(session_id, sender, content):
    messages.insert_one({
        "session_id": session_id,
        "sender": sender,
        "content": content,
        "timestamp": datetime.utcnow()
    })

def get_history(session_id, limit=10):
    return list(messages.find({"session_id": session_id}).sort("timestamp", -1).limit(limit))[::-1]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_history(st.session_state.session_id)
    # If chat history is empty, add the greeting
    if not st.session_state.chat_history:
        st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]

user_input = st.text_input("Type your message:")

if st.button("Send") and user_input:
    save_message(st.session_state.session_id, "user", user_input)
    st.session_state.chat_history.append({"sender": "user", "content": user_input})
    # Concatenate chat history for context
    context = "\n".join(
        [f"{m['sender'].capitalize()}: {m['content']}" for m in st.session_state.chat_history]
    )
    try:
        response = model.generate_content(context, generation_config={"temperature": temperature})
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Sorry, there was a problem with the AI service: {e}"

    save_message(st.session_state.session_id, "assistant", bot_reply)
    st.session_state.chat_history.append({"sender": "assistant", "content": bot_reply})
    st.rerun()

for msg in st.session_state.chat_history:
    st.markdown(f"**{msg['sender'].capitalize()}:** {msg['content']}")


