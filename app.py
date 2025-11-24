import streamlit as st
from google import generativeai as genai
from pymongo import MongoClient
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv
import serpapi
import pandas as pd
import json
import requests

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
SERPAPI_KEY = os.getenv("api_key")

if not GEMINI_API_KEY and "api_key" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if not MONGODB_URI and "MONGODB_URI" in st.secrets:
    MONGODB_URI = st.secrets["MONGODB_URI"]
if not SERPAPI_KEY and "api_key" in st.secrets:
    SERPAPI_KEY = st.secrets["api_key"]

search = serpapi.Client(api_key=SERPAPI_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# MongoDB Connection
try:
    client = MongoClient(MONGODB_URI)
    db = client["chatbot"]
    sessions = db["sessions"]
    messages = db["messages"]
except Exception as e:
    st.error(f"Cannot connect to MongoDB: {e}")
    st.stop()

# --- UI Setup ---
st.set_page_config(page_title="AI Groundwater Agent", layout="wide")
st.title("AI Groundwater Agent")

bot_greeting = "Hello! I can compare groundwater trends. Prepare robust chart to analyse the data points and estimate the ground water level"

# --- Session Management ---
if "session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat & Graphs"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = [{"sender": "assistant", "content": bot_greeting}]
        if "groundwater_data" in st.session_state:
            del st.session_state.groundwater_data
        st.rerun()

# --- Helper Functions ---

def save_message(session_id, sender, content):
    messages.insert_one({
        "session_id": session_id,
        "sender": sender,
        "content": content,
        "timestamp": datetime.utcnow()
    })

def extract_params_from_llm(user_input):
    """
    Updated to detect MULTIPLE locations for comparison.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # System prompt upgraded to handle lists
    system_prompt = f"""
    Current Date: {today_str}
    User Input: "{user_input}"
    
    Task:
    1. Analyze if the user wants groundwater data.
    2. If NO, return: {{"is_data_request": false}}
    3. If YES, identify ALL locations mentioned.
    
    Return JSON format:
    {{
      "is_data_request": true,
      "locations": [
        {{
          "district": "DistrictName1",
          "state": "StateName1 (Infer if missing)",
          "start_date": "YYYY-MM-DD (Default: 30 days ago)",
          "end_date": "YYYY-MM-DD (Default: Today)"
        }},
        {{
          "district": "DistrictName2",
          "state": "StateName2",
           ...
        }}
      ]
    }}
    
    Example: "Compare Raipur and Bhopal" -> returns 2 objects in "locations".
    Example: "Show Jaipur" -> returns 1 object in "locations".
    RETURN ONLY RAW JSON.
    """
    
    try:
        response = model.generate_content(system_prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        params = json.loads(clean_text)
        return params
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        return {"is_data_request": False}

def fetch_groundwater_api(state, district, start_date, end_date):
    url = "https://indiawris.gov.in/Dataset/Ground%20Water%20Level"
    params = {
        "stateName": state,
        "districtName": district,
        "agencyName": "CGWB",
        "startdate": start_date,
        "enddate": end_date,
        "download": "false",
        "page": "0",
        "size": "100"
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, params=params, headers=headers, verify=False)
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None

def process_groundwater_data(json_input, district_name):
    """
    Processes data and adds a 'District' column for comparison plotting.
    """
    try:
        if not json_input: return None, False
        data = json.loads(json_input)
        
        if isinstance(data, dict):
            for key in ["content", "data", "result"]:
                if key in data:
                    data = data[key]
                    break
        
        if not isinstance(data, list): return None, False
        df = pd.DataFrame(data)
        
        if 'dataTime' not in df.columns or 'dataValue' not in df.columns:
            return None, False

        df['timestamp'] = pd.to_datetime(df['dataTime'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if df.empty: return None, False
        
        df = df.sort_values('timestamp')
        
        df['District'] = district_name 
        
        return df, True
    except Exception:
        return None, False


if "groundwater_data" in st.session_state:
    districts = st.session_state.groundwater_data['District'].unique()
    title_text = f" Analysis: {' vs '.join(districts)}"
    
    st.subheader(title_text)
    
    tab1, tab2 = st.tabs(["Trend Graph", "Data Table"])
    with tab1:
        st.line_chart(st.session_state.groundwater_data, x='timestamp', y='dataValue', color='District')
    with tab2:
        st.dataframe(st.session_state.groundwater_data[['timestamp', 'dataValue', 'District', 'stationName']])
    
    st.divider()

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['sender'].capitalize()}:** {msg['content']}")

user_input = st.chat_input("Ask to compare cities ")

if user_input:
    save_message(st.session_state.session_id, "user", user_input)
    st.session_state.chat_history.append({"sender": "user", "content": user_input})
    
    with st.status("Processing Request...", expanded=True) as status:
        status.write("Analyzing locations...")
        params = extract_params_from_llm(user_input)
        
        if params.get("is_data_request"):
            locations = params.get("locations", [])
            
            if not locations:
                bot_reply = "I couldn't identify the district names. Please mention them clearly."
                status.update(label="Error", state="error")
            else:
                combined_dfs = []
                valid_districts = []
                
                for loc in locations:
                    d_name = loc['district']
                    status.write(f"Fetching data for {d_name}...")
                    
                    json_resp = fetch_groundwater_api(
                        loc.get("state", ""),
                        d_name, 
                        loc["start_date"], 
                        loc["end_date"]
                    )
                    
                    df, is_valid = process_groundwater_data(json_resp, d_name)
                    if is_valid and not df.empty:
                        combined_dfs.append(df)
                        valid_districts.append(d_name)
                    else:
                        status.write(f"⚠️ No data found for {d_name}")

                if combined_dfs:
                    final_df = pd.concat(combined_dfs, ignore_index=True)
                    
                    st.session_state.groundwater_data = final_df
                    status.update(label="Comparison Ready!", state="complete")
                    
                    summary_stats = ""
                    for d in valid_districts:
                        d_stats = final_df[final_df['District'] == d]['dataValue'].describe().to_string()
                        summary_stats += f"\n--- Stats for {d} ---\n{d_stats}\n"

                    analysis_prompt = f"""
                    User Request: "{user_input}"
                    I have successfully plotted data for: {', '.join(valid_districts)}.
                    
                    Statistical Summary:
                    {summary_stats}
                    
                    Task: Compare the groundwater trends.
                    1. Which district has deeper water levels (more negative)?
                    2. Are they stable or depleting?
                    3. Highlight the key differences.
                    """
                    response = model.generate_content(analysis_prompt)
                    bot_reply = response.text
                else:
                    status.update(label="No Data Found", state="error")
                    bot_reply = "I couldn't find data for any of the requested locations. Please check the spelling or try a different date range."
        
        else:
            status.write("Searching general knowledge...")
            try:
                results = search.search(q=user_input, engine="google")
                if "ai_overview" in results:
                    snippet = results["ai_overview"]["text_blocks"][0].get('snippet', '')
                    full_summary = f"User Query: {user_input}\nEvidence: {snippet}\nAnswer user."
                    bot_reply = model.generate_content(full_summary).text
                else:
                    bot_reply = model.generate_content(user_input).text
            except Exception as e:
                bot_reply = f"Error: {e}"
            status.update(label="Replied", state="complete")

    save_message(st.session_state.session_id, "assistant", bot_reply)
    st.session_state.chat_history.append({"sender": "assistant", "content": bot_reply})
    st.rerun()