import streamlit as st
import requests
import time
import pandas as pd

API_URL = "http://backend:8000/api"

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "primary": "#0e4c6e",
    "secondary": "#1a7fb8",
    "accent": "#2ecc71",
    "background": "#f8f9fa",
    "card_bg": "#ffffff",
    "text": "#2c3e50",
    "muted": "#6c757d",
    "border": "#dee2e6"
}

st.markdown(f"""
    <style>
    /* Main theme */
    .stApp {{
        background-color: {COLORS['background']};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, #0a3a54 100%);
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: rgba(255,255,255,0.85) !important;
    }}
    
    /* Custom title */
    .main-title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }}
    .subtitle {{
        font-size: 1rem;
        color: {COLORS['muted']};
        margin-bottom: 2rem;
    }}
    
    /* Card styling */
    .card {{
        background: {COLORS['card_bg']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
    }}
    .card-header {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {COLORS['primary']};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 0.75rem;
    }}
    
    /* Form styling */
    .stTextArea textarea, .stTextInput input {{
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 0 0 2px rgba(26, 127, 184, 0.15);
    }}
    
    /* Button styling */
    div.stButton > button {{
        background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26, 127, 184, 0.35);
    }}
    
    /* Loading button state */
    button[kind="secondary"]:disabled {{
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
        cursor: not-allowed;
        opacity: 0.7;
    }}
    
    /* Success/Info/Error boxes */
    .stSuccess, .stInfo, .stWarning, .stError {{
        border-radius: 8px;
    }}
    
    /* Toast styling */
    div[data-testid="stToast"] {{
        border-radius: 8px;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: {COLORS['muted']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['card_bg']};
        color: {COLORS['primary']};
        border-bottom: 3px solid {COLORS['secondary']};
    }}
    
    /* Code blocks */
    .stCodeBlock {{
        border-radius: 8px;
    }}
    
    /* Divider */
    hr {{
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid {COLORS['border']};
    }}
    
    /* DataFrame table styling */
    .dataframe {{
        font-size: 14px;
    }}
    
    /* Loading spinner */
    [data-testid="stSpinner"] {{
        text-align: center;
    }}
    
    /* Icon emoji styling */
    .icon {{
        font-size: 1.2em;
    }}
    
/* Response box animation */
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .response-box {{
        animation: slideIn 0.3s ease-out;
    }}
    </style>
""", unsafe_allow_html=True)

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

if "report_file_bytes" not in st.session_state:
    st.session_state.report_file_bytes = None

if "report_filename" not in st.session_state:
    st.session_state.report_filename = None

if "loading" not in st.session_state:
    st.session_state.loading = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sql_result_data" not in st.session_state:
    st.session_state.sql_result_data = None

if "sql_result_df" not in st.session_state:
    st.session_state.sql_result_df = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "sql_page" not in st.session_state:
    st.session_state.sql_page = 1

if "sql_total_pages" not in st.session_state:
    st.session_state.sql_total_pages = 1

SQL_PAGE_SIZE = 10


def post_request_safe(endpoint, payload=None, files=None, use_json=False, timeout=60):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload if use_json else None,
            data=payload if (payload and not use_json) else None,
            files=files,
            timeout=timeout,
        )
        if response.ok:
            try:
                return response.json()
            except Exception:
                return response.content
        else:
            st.toast(f"Error: {response.text[:100]}", icon="❌")
            return None
    except requests.exceptions.RequestException as e:
        st.toast(f"Request failed: {str(e)[:100]}", icon="❌")
        return None


def get_request_safe(endpoint, timeout=60):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=timeout,
        )
        if response.ok:
            return response
        else:
            st.toast(f"Error: {response.text[:100]}", icon="❌")
            return None
    except requests.exceptions.RequestException as e:
        st.toast(f"Request failed: {str(e)[:100]}", icon="❌")
        return None


def render_styled_table(df, page=1, page_size=10):
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    header_css = f"font-weight:600;color:#fff;background:linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);padding:14px 18px;text-align:left;border:none;"
    row_even = f"background-color:#f8fafc;padding:14px 18px;border-bottom:1px solid #e2e8f0;color:{COLORS['text']};"
    row_odd = f"background-color:#ffffff;padding:14px 18px;border-bottom:1px solid #e2e8f0;color:{COLORS['text']};"
    html = f'<div style="overflow-x:auto;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.1);border:1px solid #e2e8f0;"><table style="border-collapse:collapse;width:100%;font-size:14px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
    html += "<thead><tr>"
    for col in df.columns:
        html += f'<th style="{header_css}">{col}</th>'
    html += "</tr></thead><tbody>"
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        css = row_even if idx % 2 == 0 else row_odd
        html += "<tr>"
        for val in row:
            html += f'<td style="{css}">{val}</td>'
        html += "</tr>"
    html += "</tbody></table></div>"
    
    html += f'<div style="padding:12px;text-align:center;color:{COLORS["muted"]};font-size:13px;">'
    html += f'Showing <strong>{start_idx+1}-{end_idx}</strong> of <strong>{total_rows}</strong> rows'
    html += f' • Page <strong>{page}</strong> of <strong>{total_pages}</strong>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    return total_pages





with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:3rem;">🏥</div>
            <h2 style="margin:0.5rem 0;color:#fff !important;">MedAI Assistant</h2>
            <p style="color:rgba(255,255,255,0.7);margin:0;font-size:0.85rem;">AI-Powered Medical RAG System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    with st.container():
        col_dark1, col_dark2 = st.columns([1, 2])
        with col_dark1:
            st.markdown("<span style='color:#fff;font-size:14px;'>🌙 Dark Mode</span>", unsafe_allow_html=True)
        with col_dark2:
            st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
    
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #1a1a2e; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #16213e 0%, #0f0f23 100%); }
        .card { background: #1f1f3d; border-color: #2d2d5a; }
        .main-title, .card-header, .section-header { color: #e0e0e0; }
        .stTextArea textarea, .stTextInput input { background: #2d2d5a; border-color: #3d3d6a; color: #e0e0e0; }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    <p style="font-size:0.9rem;line-height:1.6;color:rgba(255,255,255,0.85);">
        • <strong>Patient Detection</strong> - Identify patients from text<br>
        • <strong>RAG Chat</strong> - Ask questions about medical documents<br>
        • <strong>SQL Query</strong> - Query database in natural language<br>
        • <strong>Upload PDF</strong> - Add documents to vectorstore<br>
        • <strong>Generate Report</strong> - Create comprehensive PDF reports
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    <p style="font-size:0.8rem;line-height:1.5;color:rgba(255,255,255,0.7);">
        • Press <code style="background:rgba(255,255,255,0.15);padding:2px 5px;border-radius:3px;">Ctrl+Enter</code> to submit<br>
        • Use quick action buttons in RAG Chat<br>
        • Results are paginated for large data
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <p style="font-size:0.75rem;color:rgba(255,255,255,0.5);text-align:center;">
        Powered by Ollama + FAISS
    </p>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Medical RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent medical document analysis and querying system</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Patient Detection", 
    "💬 RAG Chat", 
    "🗄️ SQL Query",
    "📄 Upload PDF",
    "📊 Generate Report"
])

with tab1:
    st.markdown("""
    <div class="card">
        <div class="card-header">👤 Patient Detection</div>
        <p style="color:#6c757d;margin-bottom:1rem;">Enter a patient ID or at least two names to identify a patient. <em>Press Ctrl+Enter to submit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("patient_detection_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id_input = st.text_input(
                "Patient ID (optional)",
                placeholder="e.g., NCH-28155 (press Ctrl+Enter)"
            )
        with col2:
            patient_names = st.text_input(
                "Patient Names (at least two)",
                placeholder="e.g., John Michael Smith (press Ctrl+Enter)"
            )
        
        submitted = st.form_submit_button("🔍 Detect Patient", use_container_width=True, disabled=st.session_state.loading.get("patient_detection", False))
        
        if submitted:
            patient_id = patient_id_input.strip() if patient_id_input else None
            names = patient_names.strip() if patient_names else None
            
            if patient_id:
                st.session_state.loading["patient_detection"] = True
                with st.spinner("Looking up patient..."):
                    result = post_request_safe("/detect_patient/", {"text": patient_id}, use_json=True)
                    if result:
                        patient_data = result
                        found_id = patient_data.get('patient_id')
                        first_name = patient_data.get('first_name')
                        last_name = patient_data.get('last_name')
                        full_name = patient_data.get('full_name')
                        if found_id:
                            if first_name and last_name:
                                st.toast(f"Patient found: {found_id} - {first_name} {last_name}", icon="✅")
                                st.success(f"✅ Patient found: **{found_id}** - {first_name} {last_name}")
                            elif full_name:
                                st.toast(f"Patient found: {found_id} - {full_name}", icon="✅")
                                st.success(f"✅ Patient found: **{found_id}** - {full_name}")
                            else:
                                st.toast(f"Patient found: {found_id}", icon="✅")
                                st.success(f"✅ Patient found: **{found_id}**")
                        else:
                            st.toast("Patient not found", icon="❌")
                            st.error("Patient not found.")
                st.session_state.loading["patient_detection"] = False
            elif names:
                name_count = len(names.split())
                if name_count >= 2:
                    st.session_state.loading["patient_detection"] = True
                    with st.spinner("Searching for patient..."):
                        result = post_request_safe("/detect_patient/", {"text": names}, use_json=True)
                        if result:
                            patient_data = result
                            found_id = patient_data.get('patient_id')
                            first_name = patient_data.get('first_name')
                            last_name = patient_data.get('last_name')
                            full_name = patient_data.get('full_name')
                            if found_id:
                                if first_name and last_name:
                                    st.toast(f"Patient found: {found_id} - {first_name} {last_name}", icon="✅")
                                    st.success(f"✅ Patient found: **{found_id}** - {first_name} {last_name}")
                                elif full_name:
                                    st.toast(f"Patient found: {found_id} - {full_name}", icon="✅")
                                    st.success(f"✅ Patient found: **{found_id}** - {full_name}")
                                else:
                                    st.toast(f"Patient found: {found_id}", icon="✅")
                                    st.success(f"✅ Patient found: **{found_id}**")
                            elif patient_data.get('suggestions'):
                                suggestions = patient_data.get('suggestions', [])
                                if suggestions:
                                    st.info("Did you mean:")
                                    for s in suggestions[:5]:
                                        s_first = s.get('first_name', '')
                                        s_last = s.get('last_name', '')
                                        s_id = s.get('patient_id', '')
                                        st.markdown(f"• **{s_id}** - {s_first} {s_last}")
                            else:
                                st.toast("No patient found with these names", icon="❌")
                                st.info("No patient found with these names.")
                    st.session_state.loading["patient_detection"] = False
                else:
                    st.toast("Please enter at least two names", icon="⚠️")
                    st.warning("Please enter at least two names.")
            else:
                st.toast("Please provide a Patient ID or at least two names", icon="⚠️")
                st.warning("Please provide a Patient ID or at least two names.")

with tab2:
    st.markdown("""
    <div class="card">
        <div class="card-header">💬 RAG Chat</div>
        <p style="color:#6c757d;margin-bottom:1rem;">Ask questions about medical documents using AI-powered retrieval. <em>Press Ctrl+Enter to submit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    quick_queries = [
        "What medications is the patient currently taking?",
        "What are the patient's recent lab results?",
        "Summarize the patient's diagnosis history",
        "What allergies does the patient have?",
        "Show the patient's vital signs summary"
    ]
    
    st.markdown("**Quick Actions:**")
    q_cols = st.columns(len(quick_queries))
    for idx, q in enumerate(quick_queries):
        if q_cols[idx % len(q_cols)].button(f"💊 {q[:25]}...", key=f"quick_{idx}", use_container_width=True):
            st.session_state.quick_query = q
    
    if "quick_query" in st.session_state:
        default_msg = st.session_state.quick_query
        del st.session_state.quick_query
    else:
        default_msg = ""
    
    with st.form("rag_chat_form"):
        col1, col2 = st.columns([1, 2])
        with col1:
            patient_id_chat = st.text_input(
                "Patient ID (optional)",
                placeholder="NCH-...",
                label_visibility="collapsed"
            )
        
        message = st.text_area(
            "Your question",
            value=default_msg,
            placeholder="What medications is the patient currently taking?\n\nExamples:\n• List all current medications\n• What were the latest lab results?\n• Summarize the diagnosis history",
            height=100,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("💭 Ask AI", use_container_width=True, disabled=st.session_state.loading.get("rag_chat", False))
        
        if submitted:
            if message.strip():
                st.session_state.loading["rag_chat"] = True
                with st.spinner("Searching knowledge base..."):
                    payload = {"query": message, "patient_id": patient_id_chat or None}
                    result = post_request_safe("/chat/", payload, timeout=300)
                    if result:
                        response_text = result.get("response", "")
                        st.session_state.chat_history.append({
                            "question": message,
                            "response": response_text
                        })
                        st.session_state.last_response = response_text
                        st.markdown("---")
                        st.markdown("### 📌 Response")
                        col_resp1, col_resp2 = st.columns([4, 1])
                        with col_resp1:
                            st.markdown(
                                f"<div class='response-box' style='background:#e9f4fb;padding:1rem;border-radius:8px;border-left:4px solid {COLORS['secondary']};'>{response_text}</div>", 
                                unsafe_allow_html=True
                            )
                        with col_resp2:
                            escaped_text = response_text.replace('`', '`').replace('\n', '\\n')
                            st.markdown(f"""
                            <button onclick="navigator.clipboard.writeText(`{escaped_text}`)" 
                            style="background:#e9ecef;border:none;padding:8px 12px;border-radius:6px;cursor:pointer;margin-top:8px;">📋 Copy</button>
                            """, unsafe_allow_html=True)
                        st.toast("Response received", icon="✅")
                st.session_state.loading["rag_chat"] = False
            else:
                st.toast("Please enter a question", icon="⚠️")
                st.warning("Please enter a question.")

    if st.session_state.chat_history:
        hist_expander = st.expander("📜 Chat History", expanded=True)
        with hist_expander:
            col_hist1, col_hist2 = st.columns([4, 1])
            with col_hist2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            for i, chat in enumerate(st.session_state.chat_history[-5:]):
                st.markdown(f"**Q{len(st.session_state.chat_history)-4+i}:** {chat['question']}")
                st.markdown(f"**A:** {chat['response'][:200]}...")
                st.markdown("---")

with tab3:
    st.markdown("""
    <div class="card">
        <div class="card-header">🗄️ SQL Query Interpreter</div>
        <p style="color:#6c757d;margin-bottom:1rem;">Describe your data needs in natural language and get instant SQL results. <em>Press Ctrl+Enter to submit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("sql_interpreter_form"):
        nl_query = st.text_area(
            "Describe your query",
            placeholder="e.g., Show all patients prescribed medication X in the last month\n\nExample queries:\n• List patients with diabetes\n• Show recent lab results for patient NCH-12345\n• Find all emergency room visits this year",
            height=100,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("⚡ Run SQL Query", use_container_width=True, disabled=st.session_state.loading.get("sql_query", False))
        
        if submitted:
            if nl_query.strip():
                st.session_state.loading["sql_query"] = True
                st.session_state.sql_result_data = None
                st.session_state.sql_result_df = None
                st.session_state.sql_page = 1
                with st.spinner("Generating and executing SQL..."):
                    result = post_request_safe("/sql_query/", {"nl_query": nl_query}, use_json=True, timeout=300)
                    if result:
                        sql = result.get("sql")
                        if sql:
                            escaped_sql = sql.replace('`', '`')
                            col_sql1, col_sql2 = st.columns([4, 1])
                            with col_sql1:
                                st.markdown("#### Generated SQL")
                            with col_sql2:
                                st.markdown(f"""
                                <button onclick="navigator.clipboard.writeText(`{escaped_sql}`)" 
                                style="background:#e9ecef;border:none;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px;">📋 Copy SQL</button>
                                """, unsafe_allow_html=True)
                            st.code(sql, language="sql")
                        
                        st.markdown("#### Query Results")
                        data = result.get("result")
                        if data and isinstance(data, list) and len(data) > 0:
                            df = pd.DataFrame(data)
                            st.session_state.sql_result_data = data
                            st.session_state.sql_result_df = df
                            st.session_state.sql_total_pages = (len(df) + SQL_PAGE_SIZE - 1) // SQL_PAGE_SIZE
                        elif result.get("error"):
                            error_msg = str(result.get('error'))
                            st.toast(f"Query error: {error_msg[:50]}", icon="❌")
                            st.error(f"Query error: {error_msg}")
                        else:
                            st.toast("No results found", icon="ℹ️")
                            st.info("No results found.")
                st.session_state.loading["sql_query"] = False
            else:
                st.toast("Please enter a query description", icon="⚠️")
                st.warning("Please enter a query description.")
    
    if st.session_state.sql_result_df is not None:
        df = st.session_state.sql_result_df
        total_pages = st.session_state.sql_total_pages
        render_styled_table(df, st.session_state.sql_page, SQL_PAGE_SIZE)
        
        st.markdown(f"**📊 {len(df)} row(s) returned**")
        col_pag1, col_pag2, col_pag3 = st.columns([1, 2, 1])
        with col_pag1:
            if st.button("⬅️ Previous", disabled=st.session_state.sql_page <= 1, use_container_width=True):
                st.session_state.sql_page = max(1, st.session_state.sql_page - 1)
                st.rerun()
        with col_pag2:
            st.markdown(f"<div style='text-align:center;padding:8px;'>Page {st.session_state.sql_page} of {total_pages}</div>", unsafe_allow_html=True)
        with col_pag3:
            if st.button("Next ➡️", disabled=st.session_state.sql_page >= total_pages, use_container_width=True):
                st.session_state.sql_page = min(total_pages, st.session_state.sql_page + 1)
                st.rerun()
        st.toast(f"{len(df)} row(s) returned", icon="📊")

    if st.session_state.sql_result_df is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            csv = st.session_state.sql_result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sql_query_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            json_str = st.session_state.sql_result_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str.encode('utf-8'),
                file_name="sql_query_results.json",
                mime="application/json",
                use_container_width=True
            )

with tab4:
    st.markdown("""
    <div class="card">
        <div class="card-header">📄 Upload PDF</div>
        <p style="color:#6c757d;margin-bottom:1rem;">Upload medical documents to add them to the vector database for AI querying. <em>Press Ctrl+Enter to submit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("upload_pdf_form"):
        patient_id_pdf = st.text_input(
            "Patient ID",
            placeholder="Enter Patient ID (e.g., NCH-28155)",
            help="The patient ID to associate with this document"
        )
        
        uploaded_file = st.file_uploader(
            "Choose PDF document",
            type="pdf",
            help="Upload medical records, lab results, or clinical notes"
        )
        
        submitted = st.form_submit_button("📤 Upload & Process", use_container_width=True, disabled=st.session_state.loading.get("upload_pdf", False))
        
        if submitted:
            if uploaded_file and patient_id_pdf:
                st.session_state.loading["upload_pdf"] = True
                with st.spinner("Processing PDF and creating embeddings..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "application/pdf",
                        )
                    }
                    payload = {"patient_id": patient_id_pdf}
                    response = post_request_safe(
                        "/upload_pdf/",
                        files=files,
                        payload=payload,
                        timeout=300,
                    )
                    if response:
                        st.toast("PDF uploaded and indexed successfully!", icon="✅")
                        st.success("✅ PDF uploaded and indexed successfully!")
                st.session_state.loading["upload_pdf"] = False
            else:
                st.toast("Please provide both a PDF file and Patient ID", icon="⚠️")
                st.warning("Please provide both a PDF file and Patient ID.")

with tab5:
    st.markdown("""
    <div class="card">
        <div class="card-header">📊 Generate PDF Report</div>
        <p style="color:#6c757d;margin-bottom:1rem;">Create comprehensive analysis reports from patient data and medical documents. <em>Press Ctrl+Enter to submit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("generate_report_form"):
        patient_id_report = st.text_input(
            "Patient ID for Report",
            placeholder="Enter Patient ID (e.g., NCH-28155)",
            help="Generate a comprehensive report for this patient"
        )
        
        submitted = st.form_submit_button("📑 Generate Report", use_container_width=True, disabled=st.session_state.loading.get("generate_report", False))
    
    if submitted:
        if not patient_id_report:
            st.toast("Please provide a valid Patient ID", icon="⚠️")
            st.warning("Please provide a valid Patient ID")
        else:
            st.session_state.loading["generate_report"] = True
            st.session_state.report_ready = False
            st.session_state.report_file_bytes = None
            
            init_response = post_request_safe(f"/report/{patient_id_report}")
            
            if not init_response:
                st.session_state.loading["generate_report"] = False
                st.error("Failed to start report generation. Please check if the Patient ID exists in the database.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.toast("Report generation started...", icon="🔄")
                
                for i in range(150):
                    time.sleep(2)
                    status = get_request_safe(f"/report/status/{patient_id_report}")
                    if status and status.json().get("status") == "completed":
                        break
                    if i == 0:
                        check_status = get_request_safe(f"/report/status/{patient_id_report}")
                        if check_status and check_status.status_code == 404:
                            st.session_state.loading["generate_report"] = False
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"Patient '{patient_id_report}' not found in database. Please verify the Patient ID.")
                            break
                    pct = min((i+1), 95)
                    progress_bar.progress(pct)
                    status_text.text(f"⚙️ Processing report... {pct}%")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.session_state.loading["generate_report"] = False
                    st.error("Report generation timed out. Please try again or check if the Patient ID is correct.")

                progress_bar.empty()
                status_text.empty()
                st.session_state.loading["generate_report"] = False
                
                response = get_request_safe(
                    f"/report/download/{patient_id_report}",
                    timeout=120,
                )
                
                if response:
                    st.session_state.report_ready = True
                    st.session_state.report_file_bytes = response.content
                    st.session_state.report_filename = f"{patient_id_report}_report.pdf"
                else:
                    st.error("Failed to download report. Please try again.")

    if st.session_state.report_ready:
        st.success("✅ Report generated successfully!")
        st.download_button(
            label="📥 Download PDF Report",
            data=st.session_state.report_file_bytes,
            file_name=st.session_state.report_filename,
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:1rem;color:{COLORS['muted']};font-size:0.8rem;">
    Medical RAG Assistant v1.0 • <a href="#" style="color:{COLORS['secondary']};">Documentation</a>
</div>
""", unsafe_allow_html=True)
