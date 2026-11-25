
import os
import json
import uuid
import re
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient

# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="AI Groundwater Agent",
    page_icon="💧",
    layout="wide"
)

# Read credentials from environment variables or Streamlit secrets

def get_secret(env_name, secret_name=None):
    value = os.getenv(env_name)

    if not value:
        try:
            value = st.secrets.get(
                secret_name or env_name
            )
        except Exception:
            value = None

    return value


GEMINI_API_KEY = get_secret(
    "GEMINI_API_KEY", "GEMINI_API_KEY"
)

MONGODB_URI = get_secret(
    "MONGODB_URI", "MONGODB_URI"
)

SERPAPI_KEY = get_secret(
    "SERPAPI_KEY", "SERPAPI_KEY"
)

# Backward compatibility with your old secrets configuration

if not GEMINI_API_KEY:
    GEMINI_API_KEY = get_secret("api_key", "api_key")

if not SERPAPI_KEY:
    SERPAPI_KEY = get_secret("SERPAPI_KEY", "api_key")


# =========================================================
# 2. INITIALIZE GEMINI
# =========================================================

if not GEMINI_API_KEY:
    st.error("Gemini API key is missing.")
    st.info(
        "Add GEMINI_API_KEY to your .env file "
        "or Streamlit secrets."
    )
    st.stop()

try:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )
except Exception as e:
    st.error(f"Gemini initialization failed: {e}")
    st.stop()


GEMINI_MODEL = "gemini-3.8-flash"


def ask_gemini(prompt):
    """Generate a response using Gemini."""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


# =========================================================
# 3. INITIALIZE MONGODB (OPTIONAL)
# =========================================================

mongo_client = None
sessions = None
messages = None

if MONGODB_URI:
    try:
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )

        mongo_client.admin.command("ping")

        db = mongo_client["chatbot"]

        sessions = db["sessions"]
        messages = db["messages"]

        st.sidebar.success("MongoDB connected")

    except Exception as e:
        st.sidebar.warning(
            f"MongoDB unavailable. Chat persistence disabled: {e}"
        )

        mongo_client = None
        sessions = None
        messages = None


# =========================================================
# 4. INITIALIZE SERPAPI (OPTIONAL)
# =========================================================

search_client = None

if SERPAPI_KEY:
    try:
        import serpapi

        search_client = serpapi.Client(
            api_key=SERPAPI_KEY
        )

    except Exception as e:
        st.sidebar.warning(
            f"SerpAPI initialization failed: {e}"
        )


# =========================================================
# 5. APPLICATION UI
# =========================================================

st.title("💧 AI Groundwater Agent")

st.caption(
    "An AI-powered groundwater monitoring and "
    "comparison system using India-WRIS data."
)

bot_greeting = (
    "Hello! I can compare groundwater trends across "
    "Indian districts. Ask me to analyze groundwater "
    "levels, compare districts, or identify trends."
)


# =========================================================
# 6. SESSION MANAGEMENT
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "sender": "assistant",
            "content": bot_greeting
        }
    ]


# =========================================================
# 7. MONGODB MESSAGE STORAGE
# =========================================================

def save_message(session_id, sender, content):
    """Save messages without crashing the application."""

    if messages is None:
        return

    try:
        messages.insert_one({
            "session_id": session_id,
            "sender": sender,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        })

        sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "updated_at": datetime.now(timezone.utc)
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc)
                }
            },
            upsert=True
        )

    except Exception as e:
        st.warning(f"Could not save chat history: {e}")


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Settings")

    st.write("**AI Model:** Gemini 2.5 Flash")

    st.write(
        "**Groundwater Source:** India-WRIS / CGWB"
    )

    st.divider()

    if st.button(
        "Clear Chat & Graphs",
        use_container_width=True
    ):

        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.chat_history = [
            {
                "sender": "assistant",
                "content": bot_greeting
            }
        ]

        if "groundwater_data" in st.session_state:
            del st.session_state.groundwater_data

        st.rerun()


# =========================================================
# 9. GEMINI LOCATION AND DATE EXTRACTION
# =========================================================

def extract_params_from_llm(user_input):
    """
    Extract multiple districts, states and date ranges
    from natural-language groundwater queries.
    """

    today = datetime.now().date()

    default_start = today - timedelta(days=30)

    system_prompt = f"""
You are an information extraction assistant
for an Indian groundwater monitoring application.

Current date: {today}

User query:
{user_input}

Determine whether the user wants groundwater
data from India-WRIS.

Return ONLY valid JSON.

If the query is not a groundwater data request:

{{
    "is_data_request": false
}}

If it is a groundwater data request:

{{
    "is_data_request": true,
    "locations": [
        {{
            "district": "District name",
            "state": "Indian state name",
            "start_date": "{default_start}",
            "end_date": "{today}"
        }}
    ]
}}

RULES:

1. Identify ALL districts mentioned.
2. Infer the state when possible.
3. If the state is ambiguous, use an empty string.
4. Default start_date to {default_start}.
5. Default end_date to {today}.
6. Convert dates into YYYY-MM-DD format.
7. If the user specifies a date range, respect it.
8. Do not invent district names.
9. For comparisons, return a separate object
   for every district.
10. Return raw JSON only. No Markdown.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=system_prompt
        )

        clean_text = response.text.strip()

        # Remove accidental Markdown code fences

        clean_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            clean_text
        )

        clean_text = re.sub(
            r"\s*```$",
            "",
            clean_text
        )

        params = json.loads(clean_text)

        if not params.get("is_data_request"):
            return {"is_data_request": False}

        locations = params.get("locations", [])

        if not isinstance(locations, list):
            return {"is_data_request": False}

        valid_locations = []

        for loc in locations:

            if not isinstance(loc, dict):
                continue

            district = str(
                loc.get("district", "")
            ).strip()

            state = str(
                loc.get("state", "")
            ).strip()

            if not district:
                continue

            try:
                start_date = datetime.strptime(
                    loc.get("start_date", str(default_start)),
                    "%Y-%m-%d"
                ).date()

                end_date = datetime.strptime(
                    loc.get("end_date", str(today)),
                    "%Y-%m-%d"
                ).date()

            except (ValueError, TypeError):
                start_date = default_start
                end_date = today

            if start_date > end_date:
                raise ValueError(
                    f"Invalid date range for {district}."
                )

            if end_date > today:
                raise ValueError(
                    "The requested end date cannot be in the future."
                )

            valid_locations.append({
                "district": district,
                "state": state,
                "start_date": str(start_date),
                "end_date": str(end_date)
            })

        return {
            "is_data_request": True,
            "locations": valid_locations
        }

    except Exception as e:

        print(f"Location extraction error: {e}")

        return {
            "is_data_request": False,
            "error": str(e)
        }


# =========================================================
# 10. FETCH GROUNDWATER DATA FROM INDIA-WRIS
# =========================================================

def fetch_groundwater_api(
    state,
    district,
    start_date,
    end_date
):
    """
    Fetch groundwater level data from India-WRIS.
    """

    url = (
        "https://indiawris.gov.in/"
        "Dataset/Ground%20Water%20Level"
    )

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
        "Accept": "application/json"
    }

    try:

        response = requests.post(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.warning(
            f"Request timed out for {district}."
        )

        return None

    except requests.exceptions.RequestException as e:

        st.warning(
            f"India-WRIS API error for {district}: {e}"
        )

        return None

    except ValueError:

        st.warning(
            f"Invalid JSON response for {district}."
        )

        return None


# =========================================================
# 11. PROCESS GROUNDWATER DATA
# =========================================================

def process_groundwater_data(
    json_input,
    district_name,
    state_name
):
    """
    Convert API JSON into a clean Pandas DataFrame.
    """

    try:

        if not json_input:
            return None, False

        data = json_input

        # Handle nested API responses

        if isinstance(data, dict):

            for key in [
                "content",
                "data",
                "result",
                "results"
            ]:

                if key in data:

                    data = data[key]
                    break

        if isinstance(data, dict):

            if "content" in data:
                data = data["content"]

        if not isinstance(data, list):
            return None, False

        if not data:
            return None, False

        df = pd.DataFrame(data)

        required_columns = [
            "dataTime",
            "dataValue"
        ]

        for col in required_columns:

            if col not in df.columns:
                return None, False

        # Convert timestamps

        df["timestamp"] = pd.to_datetime(
            df["dataTime"],
            errors="coerce"
        )

        # Convert groundwater measurements

        df["dataValue"] = pd.to_numeric(
            df["dataValue"],
            errors="coerce"
        )

        # Remove invalid records

        df = df.dropna(
            subset=["timestamp", "dataValue"]
        )

        if df.empty:
            return None, False

        df = df.sort_values(
            "timestamp"
        )

        # Add comparison identifiers

        df["District"] = district_name
        df["State"] = state_name

        if "stationName" not in df.columns:
            df["stationName"] = "Unknown"

        return df, True

    except Exception as e:

        print(
            f"Data processing error for {district_name}: {e}"
        )

        return None, False


# =========================================================
# 12. GENERAL WEB SEARCH
# =========================================================

def general_query(user_input):
    """
    Answer general questions using SerpAPI evidence
    and Gemini.
    """

    try:

        if search_client is not None:

            results = search_client.search({
                "engine": "google",
                "q": user_input
            })

            evidence = ""

            if isinstance(results, dict):

                # Google AI Overview

                ai_overview = results.get(
                    "ai_overview", {}
                )

                if isinstance(ai_overview, dict):

                    blocks = ai_overview.get(
                        "text_blocks", []
                    )

                    for block in blocks:

                        if isinstance(block, dict):

                            evidence += (
                                block.get("snippet", "") + "\n"
                            )

                # Organic search results as fallback

                if not evidence.strip():

                    organic_results = results.get(
                        "organic_results", []
                    )

                    for item in organic_results[:5]:

                        evidence += (
                            item.get("title", "") + "\n"
                            + item.get("snippet", "") + "\n"
                        )

            prompt = f"""
User question:
{user_input}

Web search evidence:
{evidence if evidence else "No web evidence available."}

Answer the user clearly and accurately.

If the evidence is insufficient, explain that.
Do not invent groundwater measurements or statistics.
"""

            return ask_gemini(prompt)

        else:

            return ask_gemini(user_input)

    except Exception as e:

        return (
            "I encountered an error while answering "
            f"your question: {str(e)}"
        )


# =========================================================
# 13. DISPLAY GROUNDWATER VISUALIZATION
# =========================================================

if "groundwater_data" in st.session_state:

    groundwater_df = st.session_state.groundwater_data

    districts = groundwater_df["District"].unique()

    title_text = " vs ".join(districts)

    st.subheader(
        f"📊 Groundwater Analysis: {title_text}"
    )

    tab1, tab2, tab3 = st.tabs([
        "Trend Graph",
        "Data Table",
        "Statistical Summary"
    ])

    # ---------------- TREND GRAPH ----------------

    with tab1:

        st.markdown(
            "**Groundwater Level Variation Over Time**"
        )

        chart_df = groundwater_df[
            ["timestamp", "dataValue", "District"]
        ].copy()

        chart_df = chart_df.sort_values(
            "timestamp"
        )

        st.line_chart(
            chart_df,
            x="timestamp",
            y="dataValue",
            color="District",
            use_container_width=True
        )

        st.caption(
            "Each line represents a district. "
            "Check the source's measurement convention "
            "before interpreting deeper or shallower levels."
        )

    # ---------------- DATA TABLE ----------------

    with tab2:

        display_columns = [
            "timestamp",
            "dataValue",
            "District",
            "State",
            "stationName"
        ]

        st.dataframe(
            groundwater_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = groundwater_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="groundwater_comparison.csv",
            mime="text/csv"
        )

    # ---------------- STATISTICS ----------------

    with tab3:

        stats_df = (
            groundwater_df
            .groupby("District")["dataValue"]
            .agg([
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std"
            ])
            .round(2)
        )

        st.dataframe(
            stats_df,
            use_container_width=True
        )

    st.divider()


# =========================================================
# 14. DISPLAY CHAT HISTORY
# =========================================================

for msg in st.session_state.chat_history:

    with st.chat_message(msg["sender"]):

        st.markdown(msg["content"])


# =========================================================
# 15. USER INPUT AND REQUEST PROCESSING
# =========================================================

user_input = st.chat_input(
    "Ask to compare groundwater levels across districts..."
)

if user_input:

    save_message(
        st.session_state.session_id,
        "user",
        user_input
    )

    st.session_state.chat_history.append({
        "sender": "user",
        "content": user_input
    })

    with st.status(
        "Processing your request...",
        expanded=True
    ) as status:

        status.write(
            "Analyzing your query using Gemini..."
        )

        params = extract_params_from_llm(
            user_input
        )

        # =================================================
        # GROUNDWATER DATA REQUEST
        # =================================================

        if params.get("is_data_request"):

            locations = params.get(
                "locations", []
            )

            if not locations:

                bot_reply = (
                    "I couldn't identify the district names. "
                    "Please mention the district and state clearly."
                )

                status.update(
                    label="Location extraction failed",
                    state="error"
                )

            else:

                combined_dfs = []
                valid_districts = []

                for loc in locations:

                    district = loc["district"]
                    state = loc["state"]

                    status.write(
                        f"Fetching groundwater data for {district}, {state}..."
                    )

                    api_response = fetch_groundwater_api(
                        state,
                        district,
                        loc["start_date"],
                        loc["end_date"]
                    )

                    df, is_valid = process_groundwater_data(
                        api_response,
                        district,
                        state
                    )

                    if is_valid:

                        combined_dfs.append(df)

                        valid_districts.append(
                            district
                        )

                    else:

                        status.write(
                            f"No valid groundwater data found for {district}."
                        )

                # =========================================
                # COMBINE DATA
                # =========================================

                if combined_dfs:

                    final_df = pd.concat(
                        combined_dfs,
                        ignore_index=True
                    )

                    st.session_state.groundwater_data = final_df

                    status.update(
                        label="Groundwater data retrieved successfully!",
                        state="complete"
                    )

                    # =====================================
                    # STATISTICAL ANALYSIS
                    # =====================================

                    summary_stats = ""

                    for district in valid_districts:

                        district_data = final_df[
                            final_df["District"] == district
                        ]["dataValue"]

                        summary_stats += (
                            f"\nDistrict: {district}\n"
                            f"Count: {district_data.count()}\n"
                            f"Mean: {district_data.mean():.2f}\n"
                            f"Median: {district_data.median():.2f}\n"
                            f"Minimum: {district_data.min():.2f}\n"
                            f"Maximum: {district_data.max():.2f}\n"
                            f"Standard Deviation: {district_data.std():.2f}\n"
                        )

                    # =====================================
                    # GEMINI COMPARISON
                    # =====================================

                    analysis_prompt = f"""
You are an expert groundwater data analyst.

User request:
{user_input}

Districts successfully retrieved:
{', '.join(valid_districts)}

Statistical summary:
{summary_stats}

Analyze the results.

Cover the following:

1. Compare the average groundwater measurements.
2. Identify districts with relatively higher
   and lower recorded values.
3. Explain variation and stability.
4. Identify possible trends if the dates support them.
5. Highlight important differences.
6. Explain limitations of the available data.

IMPORTANT:

Do not assume that negative values always mean
deeper groundwater.

Check the measurement convention before interpreting
the direction of groundwater levels.

Do not claim depletion or recovery from mean values alone.
Use chronological observations when discussing trends.

Do not invent units or measurements.

Present your answer using clear headings and bullets.
"""

                    try:

                        bot_reply = ask_gemini(
                            analysis_prompt
                        )

                    except Exception as e:

                        bot_reply = (
                            "The groundwater data was retrieved "
                            "and plotted successfully, but AI analysis "
                            f"failed: {str(e)}"
                        )

                else:

                    status.update(
                        label="No groundwater data found",
                        state="error"
                    )

                    bot_reply = (
                        "I couldn't retrieve valid groundwater data "
                        "for the requested districts. Please check "
                        "the spelling, state, or date range."
                    )

        # =================================================
        # GENERAL KNOWLEDGE REQUEST
        # =================================================

        else:

            status.write(
                "Processing your general knowledge query..."
            )

            bot_reply = general_query(
                user_input
            )

            status.update(
                label="Response generated",
                state="complete"
            )

    # =====================================================
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    save_message(
        st.session_state.session_id,
        "assistant",
        bot_reply
    )

    st.session_state.chat_history.append({
        "sender": "assistant",
        "content": bot_reply
    })

    st.rerun()
