import streamlit as st
import requests
import json
import os

# Backend API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="MINI-RAG System", page_icon="🧠", layout="wide")

# Session State Initialization
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "project_id" not in st.session_state:
    st.session_state.project_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

def get_headers():
    return {"X-API-Key": st.session_state.api_key}

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password", help="Enter your backend API Key")
    
    st.divider()
    
    st.subheader("Project Setup")
    project_action = st.radio("Action", ["Select Existing", "Create New"])
    
    if project_action == "Create New":
        new_project_id = st.text_input("New Project ID")
        if st.button("Create Project"):
            if not st.session_state.api_key:
                st.error("API Key required.")
            elif new_project_id:
                try:
                    res = requests.post(
                        f"{API_URL}/project/",
                        json={"project_id": new_project_id},
                        headers=get_headers()
                    )
                    if res.status_code == 200:
                        st.success(f"Project '{new_project_id}' created!")
                        st.session_state.project_id = new_project_id
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter a Project ID.")
    else:
        st.session_state.project_id = st.text_input("Enter Project ID", value=st.session_state.project_id)
        if st.session_state.project_id:
            st.success(f"Using project: {st.session_state.project_id}")

    st.divider()

    st.subheader("Data Ingestion")
    if not st.session_state.project_id:
        st.info("Select or create a project first.")
    else:
        # File Upload
        uploaded_files = st.file_uploader("Upload Documents (PDF, TXT, DOCX)", accept_multiple_files=True)
        if st.button("Upload Files"):
            if uploaded_files:
                with st.spinner("Uploading and processing..."):
                    files = []
                    for f in uploaded_files:
                        files.append(("files", (f.name, f.getvalue(), f.type)))
                    
                    try:
                        res = requests.post(
                            f"{API_URL}/upload/batch/{st.session_state.project_id}",
                            files=files,
                            headers=get_headers()
                        )
                        if res.status_code == 200:
                            st.success(f"Successfully uploaded {len(uploaded_files)} files!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Please select files to upload.")

        # URL Ingestion
        ingest_url = st.text_input("Ingest from URL")
        if st.button("Ingest URL"):
            if ingest_url:
                with st.spinner("Fetching and processing URL..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/ingest/url",
                            json={"project_id": st.session_state.project_id, "url": ingest_url},
                            headers=get_headers()
                        )
                        if res.status_code == 200:
                            st.success("Successfully ingested URL!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

# --- Main Chat UI ---
st.title("🧠 MINI-RAG Chat")

if not st.session_state.api_key:
    st.warning("Please enter your API Key in the sidebar.")
elif not st.session_state.project_id:
    st.warning("Please specify a Project ID in the sidebar.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # We use streaming endpoint
                res = requests.post(
                    f"{API_URL}/nlp/chat/stream",
                    json={
                        "project_id": st.session_state.project_id,
                        "query": prompt,
                        "limit": 5,
                        "session_id": st.session_state.session_id,
                        "use_hybrid": True,
                        "use_rerank": True
                    },
                    headers=get_headers(),
                    stream=True
                )
                
                if res.status_code == 200:
                    full_response = ""
                    for line in res.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    token = data_json.get("token", "")
                                    full_response += token
                                    message_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    pass
                    
                    message_placeholder.markdown(full_response)
                    # Add assistant message to state
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Backend error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
