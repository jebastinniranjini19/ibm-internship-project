import os
import re
import json
import requests
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI IT Helpdesk Agent", page_icon="💻", layout="wide")

KB_PATH = Path("data/knowledge_base.txt")

def load_kb():
    text = KB_PATH.read_text(encoding="utf-8")
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    return sections

KB = load_kb()

def retrieve(query, top_k=3):
    # Lightweight keyword-based RAG for a dependency-light demo.
    q = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    scored = []
    for section in KB:
        words = set(re.findall(r"[a-zA-Z0-9]+", section.lower()))
        score = len(q & words)
        scored.append((score, section))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored[:top_k] if score > 0]

def create_ticket(issue):
    # Demo tool: stores a local support ticket.
    ticket_id = "TKT-" + str(abs(hash(issue)) % 100000).zfill(5)
    with open("tickets.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ticket_id": ticket_id, "issue": issue}) + "\n")
    return ticket_id

def choose_tool(query):
    q = query.lower()
    if any(x in q for x in ["wifi", "wi-fi", "internet", "network"]):
        return "network_diagnostic"
    if any(x in q for x in ["printer", "printing", "print"]):
        return "printer_diagnostic"
    if any(x in q for x in ["bluetooth", "mouse", "keyboard"]):
        return "device_diagnostic"
    if any(x in q for x in ["slow", "hang", "freeze", "performance"]):
        return "performance_diagnostic"
    return "knowledge_search"

def local_agent(query):
    results = retrieve(query)
    tool = choose_tool(query)

    if results:
        context = "\n\n".join(results)
        answer = (
            "### Diagnosis\n"
            f"I identified this as a **{tool.replace('_', ' ')}** issue.\n\n"
            "### Recommended troubleshooting\n"
            f"{context}\n\n"
            "If the issue continues after these steps, create a support ticket."
        )
    else:
        answer = (
            "I could not find a close match in the IT knowledge base. "
            "Please describe the device, operating system, and exact error message. "
            "You can also create a support ticket."
        )
    return answer, tool, results

def watsonx_answer(query, context):
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    region_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

    if not api_key or not project_id:
        return None

    token_url = "https://iam.cloud.ibm.com/oidc/token"
    token_data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    token_resp = requests.post(token_url, data=token_data, timeout=30)
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]

    prompt = f"""You are an IT Helpdesk Agent.
Use ONLY the provided knowledge-base context for technical recommendations.
Be concise, safe, and give numbered troubleshooting steps.
If the context is insufficient, say so and recommend creating a support ticket.

Knowledge base:
{context}

User issue:
{query}
"""

    url = region_url.rstrip("/") + "/ml/v1/text/chat?version=2025-10-25"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "project_id": project_id,
        "model_id": model_id,
        "max_completion_tokens": 700,
        "temperature": 0.2
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

st.title("💻 AI IT Helpdesk Agent")
st.caption("Agent + Knowledge Retrieval (RAG) + Tools")

with st.sidebar:
    st.header("Agent Capabilities")
    st.write("• IT issue diagnosis")
    st.write("• Knowledge-base retrieval")
    st.write("• Tool selection")
    st.write("• Troubleshooting guidance")
    st.write("• Support ticket creation")
    st.divider()
    st.info("If WATSONX_APIKEY and WATSONX_PROJECT_ID are configured, IBM Granite is used to generate the final answer. Otherwise the project runs in demo mode.")

examples = [
    "My laptop is connected to Wi-Fi but internet is not working",
    "My printer is not printing",
    "My computer is very slow",
    "Bluetooth mouse is not connecting",
    "An application is not opening"
]

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Describe your IT problem:", placeholder="Example: My laptop Wi-Fi is not working")

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("🔎 Diagnose Issue", use_container_width=True)
with col2:
    ticket = st.button("🎫 Create Support Ticket", use_container_width=True)

if run and query.strip():
    answer, tool, results = local_agent(query)
    context = "\n\n".join(results) if results else "No matching knowledge-base content."
    try:
        wx = watsonx_answer(query, context)
    except Exception as e:
        wx = None
        st.warning(f"IBM watsonx call failed, so demo mode was used: {e}")
    final_answer = wx if wx else answer

    st.session_state.history.append({
        "user": query,
        "tool": tool,
        "answer": final_answer
    })

if ticket and query.strip():
    ticket_id = create_ticket(query)
    st.success(f"Support ticket created successfully: **{ticket_id}**")

st.subheader("Try an example")
for ex in examples:
    if st.button(ex, key=ex):
        st.session_state.history.append({
            "user": ex,
            "tool": choose_tool(ex),
            "answer": local_agent(ex)[0]
        })

if st.session_state.history:
    st.subheader("Conversation / Results")
    for item in reversed(st.session_state.history):
        st.markdown(f"**User:** {item['user']}")
        st.markdown(f"**Selected tool:** `{item['tool']}`")
        st.markdown(item["answer"])
        st.divider()
else:
    st.info("Enter a technical problem above to start the agent.")
