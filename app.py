import streamlit as st
from google import genai
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import uuid
import os
from dotenv import load_dotenv
import serpapi
import pandas as pd
import json
import requests

load_dotenv()

# --- UI config MUST be near top ---
st.set_page_config(page_title="AI Groundwater Agent", layout="wide")
st.title("AI Groundwater Agent")

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
MONGODB_URI    = os.getenv("MONGODB_URI")    or st.secrets.get("MONGODB_URI")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")    or st.secrets.get("SERPAPI_KEY")
WRIS_PROXY_URL = os.getenv("WRIS_PROXY")     or st.secrets.get("WRIS_PROXY")

# Global Model ID - "gemini-2.0-flash" is current (2.5 is not yet a standard version)
MODEL_ID = "gemini-1.5-flash"

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()

# Initialize the NEW Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- MongoDB Connection (Renamed to db_client to avoid conflict) ---
db_available = True
messages_col = None

if MONGODB_URI:
    try:
        # Changed 'client' to 'db_client' here
        db_client = MongoClient(MONGODB_URI) 
        db = db_client["chatbot"]
        messages_col = db["messages"]
    except Exception as e:
        st.warning(f"Cannot connect to MongoDB (logging disabled): {e}")
        db_available = False
else:
    db_available = False

if not SERPAPI_KEY:
    st.warning("SERPAPI_KEY not configured. Web search will be disabled.")
    search_client = None
else:
    search_client = serpapi.Client(api_key=SERPAPI_KEY)

# --- Helper Functions ---

def save_message(session_id, sender, content):
    if not db_available or messages_col is None:
        return
    messages_col.insert_one({
        "session_id": session_id,
        "sender": sender,
        "content": content,
        "timestamp": datetime.now(timezone.utc),
    })

def extract_params_from_llm(user_input: str):
    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = f"""
    Current Date: {today_str}
    User Input: "{user_input}"
    Task: Identify if the user wants groundwater data. 
    Return ONLY raw JSON in this format:
    {{
      "is_data_request": true,
      "locations": [
        {{ "district": "Name", "state": "State", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD" }}
      ]
    }}
    """
    try:
        # Use new client syntax
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=system_prompt
        )
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print("LLM Extraction Error:", e)
        return {"is_data_request": False}

def fetch_groundwater_api(state, district, start_date, end_date):
    url = f"{WRIS_PROXY_URL}/groundwater"
    params = {
        "stateName": state, "districtName": district, "agencyName": "CGWB",
        "startdate": start_date, "enddate": end_date, "download": "false",
        "page": "0", "size": "100"
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        return response.text if response.status_code == 200 else None
    except Exception as e:
        st.write(f"WRIS ERROR for {district}: {e}")
        return None

def process_groundwater_data(json_input, district_name):
    try:
        if not json_input: return None, False
        data = json.loads(json_input)
        for key in ["content", "data", "result"]:
            if key in data: data = data[key]
        df = pd.DataFrame(data)
        if "dataTime" not in df.columns or "dataValue" not in df.columns:
            return None, False
        df["timestamp"] = pd.to_datetime(df["dataTime"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["District"] = district_name
        return df, True
    except Exception as e:
        return None, False

# --- UI / Session Setup ---
bot_greeting = "Hello! I can compare groundwater trends for you."

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]
        if "groundwater_data" in st.session_state: del st.session_state.groundwater_data
        st.rerun()

# Display Chart
if "groundwater_data" in st.session_state:
    st.subheader("Groundwater Analysis")
    st.line_chart(st.session_state.groundwater_data, x="timestamp", y="dataValue", color="District")

# Chat UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Compare Bhopal and Raipur...")

if user_input:
    st.session_state.chat_history.append({"sender": "user", "content": user_input})
    save_message(st.session_state.session_id, "user", user_input)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.status("Thinking...") as status:
        params = extract_params_from_llm(user_input)
        
        if params.get("is_data_request"):
            locations = params.get("locations", [])
            combined_dfs = []
            valid_districts = []

            for loc in locations:
                d_name = loc["district"]
                status.write(f"Fetching {d_name}...")
                json_resp = fetch_groundwater_api(loc.get("state", ""), d_name, loc["start_date"], loc["end_date"])
                df, is_valid = process_groundwater_data(json_resp, d_name)
                if is_valid:
                    combined_dfs.append(df)
                    valid_districts.append(d_name)

            if combined_dfs:
                st.session_state.groundwater_data = pd.concat(combined_dfs)
                # Generate AI analysis
                analysis_prompt = f"Analyze these trends for {valid_districts}: {user_input}"
                response = client.models.generate_content(model=MODEL_ID, contents=analysis_prompt)
                bot_reply = response.text
            else:
                bot_reply = "No data found for those locations."
        else:
            # Simple AI chat
            response = client.models.generate_content(model=MODEL_ID, contents=user_input)
            bot_reply = response.text

    st.session_state.chat_history.append({"sender": "assistant", "content": bot_reply})
    save_message(st.session_state.session_id, "assistant", bot_reply)
    st.rerun()
