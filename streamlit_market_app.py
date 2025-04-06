import streamlit as st
from phi.agent import Agent
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import re

# === Create Agents ===
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True
    )],
    instructions=["Use tables to display data"],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use table to display data"],
    show_tool_calls=True,
    markdown=True
)

# === Streamlit UI ===
st.set_page_config(page_title="Market Insights Agent", layout="wide")

st.title("📊 Market Insights with Phi Agent Team")

query = st.text_input("Enter your query (e.g., 'Summarizse and compare analyst recommendation and fundamental for TSLA and Google')")

if st.button("Ask Agents") and query:
    with st.spinner("Agents are working..."):
        response = agent_team.run(query)
        response = str(response)
        # Clean up response text
        clean_response = re.sub(r'\\n', '\n', response)  # Convert literal \n to real newline
        clean_response = re.sub(r'\n{2,}', '\n\n', clean_response)  # Normalize extra newlines

        # Optional code/markdown block rendering
        if "```" in clean_response:
            parts = clean_response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    st.markdown(part)
                else:
                    st.code(part.strip())
        else:
            st.markdown(clean_response)