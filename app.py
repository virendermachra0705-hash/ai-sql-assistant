import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.agents import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import fitz  # For PDF
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI SQL Analyst: Langchain", layout="wide")
st.title("AI SQL Analyst")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Sidebar
api_key = st.sidebar.text_input("Groq API Key", type="password")
radio_opt = ["USE SQLLite3 Database",
             "Connect to your SQL Database", "Upload File"]
selected_opt = st.sidebar.radio(label="Choose Database", options=radio_opt)

db_uri = None
uploaded_file = None
mysql_host = mysql_user = mysql_password = mysql_db = None
if selected_opt == "Connect to your SQL Database":
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
elif selected_opt == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, SQL, PDF", type=[
                                             "csv", "xlsx", "sql", "pdf"])
else:
    db_uri = LOCALDB

col1, col2 = st.sidebar.columns(2)
with col1:
    clear_btn = st.button("Clear History")
with col2:
    copy_btn = st.button("Copy Output")

show_sql = st.sidebar.checkbox("Show SQL query used", value=False)

if not (db_uri or uploaded_file) or not api_key:
    st.info("Please configure database or upload a file and enter API key.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key,
               model_name="Llama3-8b-8192", streaming=False)


@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = Path("student.db").absolute()
        return SQLDatabase.from_uri(f"sqlite:///{dbfilepath}")
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase.from_uri(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")


def parse_file(uploaded_file):
    if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xlsx"):
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
            ".csv") else pd.read_excel(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False, if_exists="replace")
        return SQLDatabase(engine=create_engine("sqlite://", creator=lambda: conn))
    elif uploaded_file.name.endswith(".sql"):
        sql_content = uploaded_file.read().decode("utf-8")
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.executescript(sql_content)
        conn.commit()
        return SQLDatabase(engine=create_engine("sqlite://", creator=lambda: conn))
    elif uploaded_file.name.endswith(".pdf"):
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        st.write("**Extracted PDF Content:**")
        st.text(text[:1000])
        return None
    else:
        return None


def is_general_message(msg):
    greetings = ["hi", "hello", "how are you", "who are you",
                 "good morning", "good evening", "what's up", "thank you", "thanks"]
    return any(greet in msg.lower() for greet in greetings)


def generate_followup_questions(user_query):
    prompt = f"""You are an AI assistant. A user asked: \"{user_query}\". Suggest 3 intelligent and relevant follow-up questions they might ask related to this database or file topic."""
    return llm.invoke(prompt).content.strip()


# Session state
if "messages" not in st.session_state or clear_btn:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! How can I help you with your database or uploaded data?"}]
if "last_response" not in st.session_state:
    st.session_state["last_response"] = ""

user_query = st.chat_input(
    placeholder="Ask a question from the database or uploaded file...")

if copy_btn:
    st.code(st.session_state.get("last_response",
            "Nothing to copy yet!"), language="text")

# Load DB
if uploaded_file:
    db = parse_file(uploaded_file)
    if db is None:
        st.stop()
else:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)

# SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True,
                         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Chat Logic
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if is_general_message(user_query):
                    response = llm.invoke(user_query).content.strip()
                    sql_used = None
                else:
                    result = agent.invoke({"input": user_query})
                    response = result.get("output", "No answer.")
                    sql_used = None
                    for step in result.get("intermediate_steps", []):
                        if isinstance(step[0], dict) and "tool_input" in step[0]:
                            sql_used = step[0]["tool_input"]
                            break
            except Exception as e:
                response = f"❌ Error: {e}"
                sql_used = None

            st.session_state.messages.append(
                {"role": "assistant", "content": response})
            st.session_state["last_response"] = response
            st.write(response)

            if sql_used and not is_general_message(user_query) and show_sql:
                st.subheader("SQL Query Used")
                st.code(sql_used, language="sql")

            if not is_general_message(user_query):
                suggestions = generate_followup_questions(user_query)
                st.subheader("Suggested Follow-up Questions")
                for s in suggestions.split("\n"):
                    if s.strip():
                        st.markdown(f"- {s.strip().lstrip('-')}")
