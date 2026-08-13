import streamlit as st
import requests
import json
import os
import uuid

# ── Backend API Configuration ─────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="MINI-RAG System", page_icon="🧠", layout="wide")

# ── Custom CSS for chat bubbles in the History viewer ─────────────────────────
st.markdown("""
<style>
.msg-user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.6rem 1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.35rem 0 0.35rem 20%;
    font-size: 0.93rem;
    line-height: 1.5;
}
.msg-assistant {
    background: #f0f2f6;
    color: #1a1a2e;
    padding: 0.6rem 1rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.35rem 20% 0.35rem 0;
    font-size: 0.93rem;
    line-height: 1.5;
}
.msg-meta {
    font-size: 0.72rem;
    color: #888;
    margin-bottom: 0.5rem;
    text-align: right;
}
.msg-meta-left {
    font-size: 0.72rem;
    color: #888;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Initialisation ──────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "project_id" not in st.session_state:
    st.session_state.project_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history_sessions" not in st.session_state:
    st.session_state.history_sessions = []
if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None
if "selected_session_messages" not in st.session_state:
    st.session_state.selected_session_messages = []
if "db_hydrated" not in st.session_state:
    st.session_state.db_hydrated = False


# ── Helper functions ──────────────────────────────────────────────────────────

def get_headers() -> dict:
    return {"X-API-Key": st.session_state.api_key}


def fetch_sessions(project_id: str) -> list:
    """Call GET /api/v1/nlp/sessions/{project_id} and return the session list."""
    try:
        res = requests.get(
            f"{API_URL}/nlp/sessions/{project_id}",
            headers=get_headers(),
            timeout=10,
        )
        if res.status_code == 200:
            return res.json().get("sessions", [])
        st.error(f"Failed to load sessions: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return []


def fetch_session_messages(project_id: str, session_id: str) -> list:
    """Call GET /api/v1/nlp/sessions/{project_id}/{session_id} and return messages."""
    try:
        res = requests.get(
            f"{API_URL}/nlp/sessions/{project_id}/{session_id}",
            headers=get_headers(),
            timeout=10,
        )
        if res.status_code == 200:
            return res.json().get("messages", [])
        st.error(f"Failed to load messages: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return []


def delete_session(project_id: str, session_id: str) -> bool:
    """Call DELETE /api/v1/nlp/sessions/{project_id}/{session_id}."""
    try:
        res = requests.delete(
            f"{API_URL}/nlp/sessions/{project_id}/{session_id}",
            headers=get_headers(),
            timeout=10,
        )
        return res.status_code == 200
    except Exception as e:
        st.error(f"Connection error: {e}")
    return False


def hydrate_messages_from_db():
    """
    On the first run after a page refresh, try to restore the current chat
    session's messages from MongoDB so the conversation is not lost.
    """
    if st.session_state.db_hydrated:
        return
    if not st.session_state.project_id or not st.session_state.session_id:
        return

    msgs = fetch_session_messages(
        st.session_state.project_id, st.session_state.session_id
    )
    if msgs:
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]
    st.session_state.db_hydrated = True


def render_message_thread(messages: list):
    """Render a list of {'role', 'content', 'created_at'} dicts as styled bubbles."""
    if not messages:
        st.info("No messages found for this session.")
        return

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        ts = m.get("created_at", "")
        ts_display = ts[:19].replace("T", " ") if ts else ""

        if role == "user":
            st.markdown(
                f'<div class="msg-meta" style="text-align:right">🧑 You &nbsp;·&nbsp; {ts_display}</div>'
                f'<div class="msg-user">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg-meta-left">🤖 Assistant &nbsp;·&nbsp; {ts_display}</div>'
                f'<div class="msg-assistant">{content}</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your backend API Key",
    )

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
                        headers=get_headers(),
                    )
                    if res.status_code == 200:
                        st.success(f"Project '{new_project_id}' created!")
                        st.session_state.project_id = new_project_id
                        st.session_state.db_hydrated = False
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter a Project ID.")
    else:
        prev_project = st.session_state.project_id
        st.session_state.project_id = st.text_input(
            "Enter Project ID", value=st.session_state.project_id
        )
        if st.session_state.project_id != prev_project:
            st.session_state.db_hydrated = False
        if st.session_state.project_id:
            st.success(f"Using project: {st.session_state.project_id}")

    st.divider()

    st.subheader("Data Ingestion")
    if not st.session_state.project_id:
        st.info("Select or create a project first.")
    else:
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, DOCX)", accept_multiple_files=True
        )
        if st.button("Upload Files"):
            if uploaded_files:
                with st.spinner("Uploading and processing..."):
                    files = [
                        ("files", (f.name, f.getvalue(), f.type))
                        for f in uploaded_files
                    ]
                    try:
                        res = requests.post(
                            f"{API_URL}/upload/batch/{st.session_state.project_id}",
                            files=files,
                            headers=get_headers(),
                        )
                        if res.status_code == 200:
                            st.success(f"Successfully uploaded {len(uploaded_files)} files!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Please select files to upload.")

        ingest_url = st.text_input("Ingest from URL")
        if st.button("Ingest URL"):
            if ingest_url:
                with st.spinner("Fetching and processing URL..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/ingest/url",
                            json={
                                "project_id": st.session_state.project_id,
                                "url": ingest_url,
                            },
                            headers=get_headers(),
                        )
                        if res.status_code == 200:
                            st.success("Successfully ingested URL!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    st.divider()

    st.subheader("Current Session")
    st.caption(f"ID: `{st.session_state.session_id}`")
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.db_hydrated = False
        st.rerun()


# ── Main content: two tabs ─────────────────────────────────────────────────────
st.title("🧠 MINI-RAG Chat")

tab_chat, tab_history = st.tabs(["💬 Chat", "📜 Chat History"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.api_key:
        st.warning("Please enter your API Key in the sidebar.")
    elif not st.session_state.project_id:
        st.warning("Please specify a Project ID in the sidebar.")
    else:
        # Hydrate from DB once per page load so refresh doesn't wipe the chat
        hydrate_messages_from_db()

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    res = requests.post(
                        f"{API_URL}/nlp/index/answer/stream/{st.session_state.project_id}",
                        json={
                            "text": prompt,
                            "limit": 5,
                            "session_id": st.session_state.session_id,
                            "use_hybrid": True,
                            "use_rerank": True,
                        },
                        headers=get_headers(),
                        stream=True,
                    )

                    if res.status_code == 200:
                        full_response = ""
                        for line in res.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        if data_json.get("done"):
                                            break
                                        token = data_json.get("token", "")
                                        full_response += token
                                        message_placeholder.markdown(full_response + "▌")
                                    except json.JSONDecodeError:
                                        pass

                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )
                    else:
                        st.error(f"Backend error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat History
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.api_key:
        st.warning("Please enter your API Key in the sidebar.")
    elif not st.session_state.project_id:
        st.warning("Please specify a Project ID in the sidebar.")
    else:
        st.subheader(f"💾 Saved Sessions for **{st.session_state.project_id}**")

        col_load, col_spacer = st.columns([2, 8])
        with col_load:
            if st.button("🔃 Load Sessions", use_container_width=True):
                with st.spinner("Fetching sessions from database…"):
                    st.session_state.history_sessions = fetch_sessions(
                        st.session_state.project_id
                    )
                    st.session_state.selected_session_id = None
                    st.session_state.selected_session_messages = []

        sessions = st.session_state.history_sessions

        if not sessions:
            st.info("No sessions loaded yet. Click **Load Sessions** to fetch from the database.")
        else:
            st.caption(f"Found **{len(sessions)}** session(s).")

            # ── Session selector ──────────────────────────────────────────────
            session_labels = [
                f"{s['session_id'][:8]}…  ({s.get('message_count', 0)} msgs"
                f"  ·  {(s.get('created_at') or '')[:10]})"
                for s in sessions
            ]
            selected_idx = st.selectbox(
                "Select a session to review:",
                options=range(len(sessions)),
                format_func=lambda i: session_labels[i],
                key="history_selectbox",
            )

            selected_session = sessions[selected_idx]
            full_session_id = selected_session["session_id"]

            # Show session metadata
            with st.expander("ℹ️ Session details", expanded=False):
                st.write({
                    "session_id": full_session_id,
                    "message_count": selected_session.get("message_count", 0),
                    "created_at": selected_session.get("created_at"),
                    "updated_at": selected_session.get("updated_at"),
                    "summary": selected_session.get("summary") or "—",
                })

            # ── Action buttons ────────────────────────────────────────────────
            col_view, col_del, col_exp = st.columns([2, 2, 2])

            with col_view:
                if st.button("📖 View Messages", use_container_width=True):
                    with st.spinner("Loading messages…"):
                        st.session_state.selected_session_id = full_session_id
                        st.session_state.selected_session_messages = fetch_session_messages(
                            st.session_state.project_id, full_session_id
                        )

            with col_del:
                if st.button("🗑️ Delete Session", use_container_width=True, type="secondary"):
                    if delete_session(st.session_state.project_id, full_session_id):
                        st.success(f"Session `{full_session_id[:8]}…` deleted.")
                        st.session_state.history_sessions = [
                            s for s in sessions if s["session_id"] != full_session_id
                        ]
                        if st.session_state.selected_session_id == full_session_id:
                            st.session_state.selected_session_id = None
                            st.session_state.selected_session_messages = []
                        st.rerun()
                    else:
                        st.error("Failed to delete session.")

            with col_exp:
                msgs_for_export = st.session_state.selected_session_messages
                if msgs_for_export and st.session_state.selected_session_id == full_session_id:
                    export_data = json.dumps(
                        {
                            "session_id": full_session_id,
                            "project_id": st.session_state.project_id,
                            "messages": msgs_for_export,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    st.download_button(
                        label="⬇️ Export JSON",
                        data=export_data,
                        file_name=f"session_{full_session_id[:8]}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "⬇️ Export JSON",
                        disabled=True,
                        help="Load messages first",
                        use_container_width=True,
                    )

            # ── Message thread viewer ─────────────────────────────────────────
            if (
                st.session_state.selected_session_id == full_session_id
                and st.session_state.selected_session_messages
            ):
                st.divider()
                total = len(st.session_state.selected_session_messages)
                st.markdown(
                    f"**Conversation thread** — `{full_session_id}` · {total} message(s)"
                )
                render_message_thread(st.session_state.selected_session_messages)


