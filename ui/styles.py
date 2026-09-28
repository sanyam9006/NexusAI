"""
CSS design system for the Streamlit UI.

Defines a consistent visual language: gradients, transitions, hover
effects, and component styles.  Injected once via st.markdown().
"""

MAIN_CSS = """
<style>
    /* ── Global ── */
    .main {
        background-color: #f8f9fa;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Source Citation Boxes ── */
    .source-box {
        padding: 12px 16px;
        background-color: #eef2ff;
        border-left: 5px solid #4F46E5;
        margin-top: 8px;
        font-size: 0.88em;
        border-radius: 0 8px 8px 0;
        transition: background-color 0.2s ease;
        line-height: 1.5;
    }
    .source-box:hover {
        background-color: #e0e7ff;
    }

    /* ── Status Badges ── */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-active {
        background-color: #dcfce7;
        color: #166534;
    }
    .status-inactive {
        background-color: #fef2f2;
        color: #991b1b;
    }

    /* ── Metric Cards ── */
    .metric-card {
        padding: 16px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
        text-align: center;
    }

    /* ── Sidebar Refinements ── */
    [data-testid="stSidebar"] {
        background-color: #fafbfc;
    }

    /* ── Chat Input ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px;
    }
</style>
"""
