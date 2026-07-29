"""Streamlit UI State Machine: Elite Finance Terminal UI/UX."""

import os
import sys
import subprocess
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "family_offices.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "rag_index", "embeddings.npy")

# --- State Initialization ---
if "pipeline_state" not in st.session_state:
    if os.path.exists(CSV_PATH) and os.path.exists(EMBEDDINGS_PATH):
        st.session_state.pipeline_state = "READY"
    else:
        st.session_state.pipeline_state = "INIT"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="PolarityIQ", layout="wide", page_icon="📊")

# --- Elite CSS Injection ---
def load_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* --- Core Variables --- */
        :root {
            --bg-main: #08090A;
            --bg-card: #111317;
            --bg-sidebar: #0C0E10;
            --border-color: #1F2226;
            --text-main: #E6EDF3;
            --text-muted: #6E7681;
            --accent-gold: #D4AF37;
            --accent-green: #3FB950;
            --accent-red: #F85149;
            --font-head: 'Sora', sans-serif;
            --font-body: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        /* --- Global Overrides --- */
        html, body, [class*="css"] {
            font-family: var(--font-body);
            color: var(--text-main);
        }
        .stApp {
            background-color: var(--bg-main);
            background-image: radial-gradient(circle at 50% 0%, rgba(212, 175, 55, 0.05) 0%, transparent 50%);
        }
        
        /* --- Typography --- */
        h1, h2, h3, h4 {
            font-family: var(--font-head) !important;
            letter-spacing: -1px;
            color: #ffffff !important;
        }
        h1 { font-weight: 700; font-size: 2.5rem !important; }
        h3 { font-size: 1.3rem !important; font-weight: 600; }
        p, li { font-size: 0.95rem; line-height: 1.6; color: var(--text-main); }

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
        }
        section[data-testid="stSidebar"] .stMarkdown h1, 
        section[data-testid="stSidebar"] .stMarkdown h2 {
            font-family: var(--font-head);
            color: var(--accent-gold) !important;
        }

        /* --- Metric Badges --- */
        .stMetric {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px !important;
        }
        .stMetric label { font-size: 0.8rem !important; color: var(--text-muted) !important; }
        .stMetric value { font-family: var(--font-mono) !important; color: var(--text-main) !important; }

        /* --- Chat Interface --- */
        .stChatInput {
            border-color: var(--border-color) !important;
        }
        .stChatInput > div > div {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
        .stChatInput input::placeholder { color: var(--text-muted) !important; }
        
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        .stChatMessage [data-testid="stChatMessageAvatarUser"] {
            display: none; /* Hide avatars for cleaner look */
        }
        .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
            display: none;
        }

        /* --- Custom Components --- */
        .query-bubble {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 12px 20px;
            border-radius: 12px 12px 0 12px;
            margin-left: auto;
            max-width: 70%;
            text-align: right;
            color: var(--text-main);
            font-weight: 500;
        }
        
        .response-container {
            margin-top: 20px;
            border-left: 2px solid var(--accent-gold);
            padding-left: 25px;
        }
        
        .status-banner {
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            font-weight: 500;
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
        }

        .record-card {
            background: linear-gradient(180deg, var(--bg-card) 0%, #0E1013 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .record-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            border-color: var(--accent-gold);
        }
        
        .entity-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: var(--font-mono);
            background: rgba(212, 175, 55, 0.15);
            color: var(--accent-gold);
            border: 1px solid rgba(212, 175, 55, 0.3);
            margin-bottom: 8px;
        }
        
        .verified-block {
            background: rgba(63, 185, 80, 0.05);
            border-left: 3px solid var(--accent-green);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 0 8px 8px 0;
        }
        
        .verified-pill {
            color: var(--accent-green);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .unverified-block {
            background: rgba(248, 81, 73, 0.05);
            border-left: 3px solid var(--accent-red);
            padding: 15px;
            border-radius: 0 8px 8px 0;
        }
        
        .unverified-pill {
            color: var(--accent-red);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .data-row {
            margin-bottom: 12px;
        }
        .data-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 2px;
        }
        .data-value {
            font-size: 0.95rem;
            color: var(--text-main);
        }
        .muted-value {
            color: var(--text-muted);
            font-style: italic;
            font-size: 0.9rem;
        }
        
        /* --- Buttons --- */
        .stButton > button {
            background-color: var(--accent-gold);
            color: #000000;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-family: var(--font-head);
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            background-color: #E5C158;
            box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
            color: #000000;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

def render_response(res: dict):
    """Render the RAG response with structured claim verification UI."""

    if "Fallback" in res.get("model_used", ""):
        st.markdown(
            '<div class="status-banner">'
            '⚠️ System operated in degraded mode (8B model) '
            'due to rate limits.</div>',
            unsafe_allow_html=True,
        )
    if res.get("partial_answer"):
        st.markdown(
            '<div class="status-banner">'
            '⚠️ Partial answer: Some claims could not be verified '
            'against the dataset.</div>',
            unsafe_allow_html=True,
        )
        
    st.markdown(
        "<div class='response-container'>"
        f"<h3>{res['summary']}</h3></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    if res.get("verified_claims"):
        st.markdown("<h4>Verified Intelligence</h4>", unsafe_allow_html=True)
        for claim in res["verified_claims"]:
            st.markdown(f"""
            <div class="verified-block">
                <div class="verified-pill">✓ Verified Claim</div>
                <p style="font-size: 1.1rem; margin: 10px 0;">{claim['statement']}</p>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin: 0;">
                    <span style="font-family: var(--font-mono);">SOURCE:</span> {claim['field_used']} from {claim['actual_entity']}<br>
                    <a href="{claim.get('source_url', '#')}" style="color: var(--accent-gold); text-decoration: none;">View Source Documentation →</a>
                </p>
            </div>
            """, unsafe_allow_html=True)
                    
    if res.get("unverified_claims"):
        with st.expander("View Unverified Claims (Hallucinations Detected)"):
            for claim in res["unverified_claims"]:
                st.markdown(f"""
                <div class="unverified-block">
                    <div class="unverified-pill">✗ Unverified Claim</div>
                    <p style="font-size: 0.95rem; margin: 10px 0;">"{claim['statement']}"</p>
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin: 0;">Evidence could not be verified in the source data.</p>
                </div>
                """, unsafe_allow_html=True)
                    
    if res.get("raw_records"):
        st.markdown("<br><h4>Retrieved Dossiers</h4>", unsafe_allow_html=True)
        for record in res["raw_records"]:
            entity_name = record.get('entity_name', 'N/A')
            entity_type = record.get('entity_type', 'N/A')
            
            def get_field(label, val, is_link=False):
                muted = val == "Could not verify" or not val
                if muted:
                    return (
                        f'<div class="data-row">'
                        f'<div class="data-label">{label}</div>'
                        f'<div class="muted-value">Could not verify</div>'
                        f'</div>'
                    )
                if is_link:
                    return (
                        f'<div class="data-row">'
                        f'<div class="data-label">{label}</div>'
                        f'<div class="data-value">'
                        f'<a href="{val}" style="color: var(--accent-gold); '
                        f'text-decoration: none;">{val}</a></div>'
                        f'</div>'
                    )
                return (
                    f'<div class="data-row">'
                    f'<div class="data-label">{label}</div>'
                    f'<div class="data-value">{val}</div>'
                    f'</div>'
                )

            st.markdown(f"""
            <div class="record-card">
                <span class="entity-badge">{entity_type}</span>
                <h3 style="margin-top: 5px; margin-bottom: 20px;">{entity_name}</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                    <div>
                        {get_field("Principal", f"{record.get('principal_name', '')} ({record.get('principal_title', '')})")}
                        {get_field("AUM Range", record.get('aum_range', ''))}
                        {get_field("Location", record.get('address', ''))}
                    </div>
                    <div>
                        {get_field("Website", record.get('website', ''), is_link=True)}
                        {get_field("Entity LinkedIn", record.get('entity_linkedin', ''), is_link=True)}
                    </div>
                    <div>
                        {get_field("Recent Signal", record.get('recent_signal', ''))}
                        {get_field("Signal Date", record.get('signal_date', ''))}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Phase 1: INIT ---
if st.session_state.pipeline_state == "INIT":
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh; text-align: center;">
        <h1 style="font-size: 3.5rem; background: linear-gradient(135deg, #FFFFFF 0%, #D4AF37 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PolarityIQ Terminal</h1>
        <p style="font-size: 1.2rem; color: var(--text-muted); max-width: 600px; margin-bottom: 40px;">
            Actionable intelligence on 50 validated Single Family Offices. Execute the pipeline to initialize the semantic index.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Initialize Pipeline", use_container_width=True):
            st.session_state.pipeline_state = "RUNNING_PIPELINE"
            st.rerun()

# --- Phase 2: RUNNING_PIPELINE ---
elif st.session_state.pipeline_state == "RUNNING_PIPELINE":
    st.markdown("<h1 style='color: var(--accent-gold);'>Pipeline Executing...</h1>", unsafe_allow_html=True)
    
    env_vars = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": BASE_DIR}
    try:
        env_vars["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        env_vars["SEC_USER_AGENT"] = st.secrets.get("SEC_USER_AGENT", "PolarityIQ test@test.com")
    except Exception:
        pass

    script_enrich = os.path.join(BASE_DIR, "enrichment", "run_enrichment.py")
    script_index = os.path.join(BASE_DIR, "rag", "build_index.py")
    
    success = True
    
    with st.status("Executing Discovery & Enrichment Pipeline...", expanded=True) as status:
        st.write("Running enrichment script...")
        process = subprocess.Popen(
            [sys.executable, script_enrich],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_vars,
            cwd=BASE_DIR
        )
        
        for line in process.stdout:
            st.write(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            status.update(label="Pipeline Failed!", state="error")
            success = False
        else:
            st.write("Building RAG Index (generating embeddings)...")
            process2 = subprocess.Popen(
                [sys.executable, script_index],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_vars,
                cwd=BASE_DIR
            )
            
            for line in process2.stdout:
                st.write(line.strip())
                
            process2.wait()
            
            if process2.returncode != 0:
                status.update(label="Indexing Failed!", state="error")
                success = False
                
    if success:
        status.update(label="Pipeline Complete!", state="complete")
        st.session_state.pipeline_state = "READY"
        st.rerun()
    else:
        st.session_state.pipeline_state = "INIT"
        st.error("Pipeline failed. Check logs above.")

# --- Phase 3: READY (Elite Chat UI) ---
elif st.session_state.pipeline_state == "READY":
    with st.sidebar:
        st.markdown("## PolarityIQ Terminal")
        st.markdown("---")
        try:
            df_count = len(pd.read_csv(CSV_PATH, keep_default_na=False))
            st.metric(label="Validated Records", value=f"{df_count}")
            st.metric(label="Embedding Model", value="BGE-small")
            st.metric(label="LLM Engine", value="Groq 70B/8B")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        st.markdown("---")
        st.markdown("### System States")
        st.markdown("""
        <div style="font-family: var(--font-mono); font-size: 0.85rem;">
            <p style="color: var(--accent-green);">● Dataset Generated</p>
            <p style="color: var(--accent-green);">● Semantic Index Built</p>
            <p style="color: var(--accent-green);">● Chat Interface Active</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h1>Intelligence Query Interface</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='margin-top: -10px; color: var(--text-muted);'>"
        "Query the dataset of validated Single Family Offices. "
        "Answers are strictly grounded in the retrieved evidence.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                <div class="query-bubble">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_response(message["content"])
                
    # Chat input
    if prompt := st.chat_input("Query the intelligence database..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
            <div class="query-bubble">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
            
        with st.spinner("Analyzing intelligence..."):
            try:
                from rag import retrieval, generation
                results = retrieval.search(prompt)
                response_data = generation.generate_response(prompt, results)
                
                render_response(response_data)
                st.session_state.messages.append({"role": "assistant", "content": response_data})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")