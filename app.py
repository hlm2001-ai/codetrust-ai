"""CodeTrust AI • 3D learning studio • GPT-4.1 mini + frozen V3.

The decorative 3D object is self-contained CSS: no CDN, tracking, or JS.
Student code is sent to OpenAI for textual analysis, never executed locally.
The API key lives only in a server-side environment variable or Streamlit secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="CodeTrust AI • Python Learning Studio",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "CodeTrust AI • Master's research prototype • Frozen Enhanced Prompt V3."},
)

PROMPT_FILE = Path(__file__).with_name("frozen_v3_prompt.json")
EXPECTED_V3_SHA256 = "b0d6afcee8e671f87c297793b0e4dc1b7dea4f2aa79e6335786b5f9968b4d9d9"
MODEL = "gpt-4.1-mini"


@st.cache_data(show_spinner=False)
def load_prompt() -> dict:
    data = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(data["prompt"].encode("utf-8")).hexdigest()
    if checksum != EXPECTED_V3_SHA256:
        raise ValueError("Frozen Enhanced V3 fingerprint mismatch. Analysis is disabled.")
    if not data.get("response_contract", "").strip():
        raise ValueError("Shared response contract is missing.")
    return data


def get_api_key() -> str:
    from_env = os.environ.get("OPENAI_API_KEY", "").strip()
    if from_env:
        return from_env
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


SAMPLES = {
    "runtime": (
        "def calculate_average(grades):\n"
        "    total = 0\n"
        "    for grade in grades:\n"
        "        total += grade\n"
        "    return total / len(grades)\n\n"
        "grades = [85, 90, 78, '95', 88]\n"
        "print(calculate_average(grades))",
        "Calculate and print the arithmetic average of five numeric grades. The grade '95' is a number stored as text.",
    ),
    "valid": (
        "items = [1, 2, 3]\n"
        "doubled = [x * 2 for x in items]\n"
        "print(doubled)",
        "Print the doubled values of the list in the original order.",
    ),
    "logic": (
        "def is_even(number):\n"
        "    return number % 2 == 1\n\n"
        "print(is_even(8))",
        "Return True for even integers and False for odd integers.",
    ),
}


def choose_example(example: str) -> None:
    code, requirements = SAMPLES[example]
    st.session_state["editor_code"] = code
    st.session_state["editor_requirements"] = requirements
    st.session_state["editor_context"] = ""
    st.session_state.pop("last_answer", None)
    st.session_state.pop("last_code", None)
    st.session_state.pop("analysis_error", None)


def parse_sections(answer: str) -> dict[str, str]:
    """Extract the four research-contract headings without using model HTML."""
    headings = ("Diagnosis", "Location", "Explanation", "Suggested Fix")
    pattern = re.compile(
        r"(?im)^\s*(?:#{1,5}\s*)?(?:\*\*)?\s*"
        r"(Diagnosis|Location|Explanation|Suggested Fix)\s*:?(?:\*\*)?\s*:?[ \t]*"
    )
    matches = list(pattern.finditer(answer))
    if len(matches) != 4 or [m.group(1) for m in matches] != list(headings):
        return {}
    results = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        results[match.group(1)] = answer[match.end():end].strip()
    if all(results.values()):
        return results
    return {}


# Static CSS/HTML only. The model's output never enters custom HTML.
st.markdown(
    """<style>
    :root { --ct-primary: #877cff; --ct-teal: #4de3d0; --ct-orange: #ffd08d; }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background: #080c18 !important;
      color: #eef2ff;
    }
    [data-testid="stAppViewContainer"] {
      background-image:
        radial-gradient(ellipse 62% 38% at 80% 2%, rgba(107, 92, 244,.13), transparent 90%),
        radial-gradient(ellipse 45% 40% at 5% 48%, rgba(18, 179, 171,.08), transparent 82%) !important;
    }
    #MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stDecoration"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] {display:none !important;}
    header[data-testid="stHeader"] {background:transparent !important; height: 10px;}
    .block-container {max-width:1330px !important; padding-top: 1.1rem !important; padding-bottom:3rem !important;}
    .stVerticalBlock {gap:.65rem;}
    .ct-navbar {display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0 0 19px;}
    .ct-brand {display:flex; align-items:center; gap:11px; font-size:19px; font-weight:800; letter-spacing:-.6px; color:#f8f8ff;}
    .ct-brand-mark {width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
       background:linear-gradient(145deg,#8d77ff,#24b9c8); font-size:20px; box-shadow:0 6px 24px #6c69ff38;}
    .ct-nav-meta {border:1px solid #26334c; border-radius:90px; padding:8px 14px; font-size:12px; color:#bbc8df; white-space:nowrap;}
    .ct-nav-dot {color:#57e9ca; margin-right:5px;}
    .ct-hero {position:relative; overflow:hidden; display:grid; grid-template-columns: 1.4fr 1fr; align-items:center;
      min-height:320px; padding:35px 48px; border-radius:26px; border:1px solid rgba(164,169,255,.17);
      background:linear-gradient(115deg,#141c35 0%,#171b35 46%,#101c33 100%); box-shadow:0 30px 90px #00000038;}
    .ct-hero::before {content:""; position:absolute; inset:0; background-image:radial-gradient(#9ea9dc 0.8px,transparent 0.8px);
      background-size:21px 21px; opacity:.09; pointer-events:none;}
    .ct-hero::after {content:""; position:absolute; width:400px; height:400px; border-radius:50%; right:12%; top:-70%;
      background:#6661f5; filter:blur(90px); opacity:.15; pointer-events:none;}
    .ct-copy {position:relative; z-index:2;}
    .ct-eyebrow {display:inline-flex; align-items:center; gap:10px; border:1px solid #5f5ba74d; color:#c7c0ff;
      background:#8c79ff13; padding:8px 12px; border-radius:20px; font-size:11px; font-weight:800;
      letter-spacing:1.3px; text-transform:uppercase;}
    .ct-dot {width:7px; height:7px; border-radius:50%; background:#53e5c6; box-shadow:0 0 10px #53e5c6; display:inline-block;}
    .ct-hero h1 {font-size:clamp(36px,4vw,59px); line-height:1.08; margin:18px 0 14px !important;
      font-weight:850; color:#f7f8ff; letter-spacing:-2.2px;}
    .ct-gradient {background:linear-gradient(100deg,#a39aff,#57d8ed 85%); background-clip:text;-webkit-background-clip:text;color:transparent;}
    .ct-hero-desc {max-width:550px;color:#b9c4d9;line-height:1.7;font-size:15px; margin:0 0 20px !important;}
    .ct-hero-tag {display:inline-flex; align-items:center; gap:8px; color:#a5b5cf; font-size:12px; padding-right:13px;}
    .ct-hero-tag b {color:#f5d39c;font-weight:700;}
    .ct-3d-stage {height:265px; position:relative; display:flex; align-items:center; justify-content:center; perspective:830px; z-index:1;}
    .ct-orbit {position:absolute; width:250px; height:250px; border-radius:50%; border:1px solid #787ee829; transform:rotateX(66deg) rotateZ(-16deg); animation:ct-orbit 13s linear infinite;}
    .ct-orbit.orbit-b {width:306px; height:306px; border-color:#38d3cc20; transform:rotateX(55deg) rotateZ(47deg); animation-duration:16s;animation-direction:reverse;}
    .ct-orbit::before {content:""; position:absolute; width:9px;height:9px;border-radius:50%; background:#72e4d9; top:13%;left:13%;box-shadow:0 0 17px #72e4d9;}
    .ct-cube {position:relative;width:140px;height:140px;transform-style:preserve-3d;
       animation:ct-spin 16s ease-in-out infinite; transition:filter .3s;
       filter:drop-shadow(0 20px 25px #0007);}
    .ct-3d-stage:hover .ct-cube {animation-play-state:paused; filter:drop-shadow(0 15px 28px #8884ff66);}
    .ct-face {position:absolute; inset:0; border:1px solid rgba(192,202,255,.50); border-radius:17px;
       display:flex;align-items:center;justify-content:center; flex-direction:column;
       background:linear-gradient(145deg,rgba(99,105,240,.78),rgba(47,166,178,.24));
       backdrop-filter:blur(8px);box-shadow:inset 0 0 40px #c8d5ff12; color:#fff;}
    .ct-face span {font-size:45px; font-weight:850; letter-spacing:-2px; text-shadow:0 5px 28px #0005;}
    .ct-face small {font-size:9px;letter-spacing:2px;color:#d9e5ff;text-transform:uppercase;font-weight:800;}
    .ct-front {transform:translateZ(70px)} .ct-back {transform:rotateY(180deg) translateZ(70px)}
    .ct-right {transform:rotateY(90deg) translateZ(70px)} .ct-left {transform:rotateY(-90deg) translateZ(70px)}
    .ct-top {transform:rotateX(90deg) translateZ(70px)} .ct-bottom {transform:rotateX(-90deg) translateZ(70px)}
    .ct-float-label {position:absolute; font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#6ee6d7;
      padding:9px 12px;border:1px solid #67d9cf35;border-radius:12px;background:#13283ba6;box-shadow:0 7px 20px #0509175c;
      animation:ct-float 4s ease-in-out infinite;}
    .ct-float-label.one {top:10%;left:2%;} .ct-float-label.two {right:0;bottom:12%;animation-delay:-1.3s;color:#e6bbff;}
    .ct-float-label.three {left:3%;bottom:10%;animation-delay:-2.1s;color:#ffd59f;}
    .ct-stages {display:flex;gap:12px;justify-content:space-between;margin:18px 0 25px;}
    .ct-stage {flex:1;display:flex;align-items:center;gap:11px; padding:13px 18px;border:1px solid #232c43;
      background:#101729b3; border-radius:15px;color:#bac7dd;font-size:13px;}
    .ct-stage-num {color:#9b94ff;font-weight:850;font-size:13px;font-family:ui-monospace,Consolas,monospace;}
    .ct-stage b {color:#edf3ff;font-weight:650;}
    .ct-area-head {display:flex;align-items:center;justify-content:space-between;margin:9px 0 4px; gap:15px;}
    .ct-section-title {font-size:24px;font-weight:800;letter-spacing:-.8px;color:#f2f4ff;}
    .ct-section-sub {color:#8e9cb5;font-size:12px;margin:2px 0 12px;}
    .ct-ai-label {display:inline-flex;align-items:center;gap:8px;background:#2f2448;border:1px solid #654c8b;
      padding:6px 11px;border-radius:25px;color:#cdbaff;font-size:11px;font-weight:800;}
    .ct-card-head {font-size:14px;font-weight:800;color:#edf2ff;margin-bottom:6px;}
    .ct-note {font-size:12px; color:#9facbf;line-height:1.6;}
    .ct-footer {display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-top:22px; border-top:1px solid #263046;
       padding:16px 0 8px;font-size:11px;color:#8190a9;}
    [data-testid="stTextArea"] label p {color:#c8d2e4 !important;font-weight:650;font-size:13px;}
    [data-testid="stTextArea"] textarea {background:#111a2b !important; color:#f1f6ff !important;
       border:1px solid #303c59 !important; border-radius:13px !important; font-family:ui-monospace,"Cascadia Code",Consolas,monospace;
       font-size:13px; line-height:1.6;}
    [data-testid="stTextArea"] textarea:focus {border-color:#8f86ff !important;box-shadow:0 0 0 1px #7775f544 !important;}
    div.stButton > button[kind="primary"] {border:0 !important;border-radius:13px !important;
      background:linear-gradient(105deg,#7875ff,#5e64ea 47%,#3da9c3) !important;
      color:white !important;font-weight:800 !important;min-height:49px !important;
      box-shadow:0 8px 20px #615df52e;transition: transform .18s,box-shadow .18s;}
    div.stButton > button[kind="primary"]:hover {transform:translateY(-2px);box-shadow:0 12px 22px #615df54d;}
    div.stButton > button[kind="secondary"] {background:#151e31;border:1px solid #2c3954;border-radius:11px;color:#d8e4fa;font-weight:650;}
    [data-testid="stExpander"] {background:#11182a;border:1px solid #273450;border-radius:12px;}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:16px !important;}
    [data-testid="stMarkdownContainer"] code {color:#7de3d5;}
    [data-testid="stCode"] {border:1px solid #2a3653;border-radius:13px;}
    .ct-empty {border:1px dashed #384966;background:linear-gradient(150deg,#111b2e,#111627);min-height:256px;
      border-radius:20px;display:flex;justify-content:center;align-items:center;text-align:center;flex-direction:column;gap:12px;padding:25px;}
    .ct-empty-icon {font-size:33px; background:#6e6dfb23;border-radius:16px;padding:12px;}
    .ct-empty-title {font-size:17px;color:#eaf0ff;font-weight:750;}
    .ct-empty-desc {max-width:310px;font-size:12px;color:#93a3bb;line-height:1.7;}
    @keyframes ct-spin {0%,100% {transform:rotateX(-19deg) rotateY(-32deg)} 50% {transform:rotateX(15deg) rotateY(148deg)}}
    @keyframes ct-orbit {from {rotate:0deg} to {rotate:360deg}}
    @keyframes ct-float {0%,100%{translate:0 0}50%{translate:0 -8px}}
    @media(max-width:850px) {
      .ct-hero {grid-template-columns:1fr; padding:26px 24px; min-height:auto;}
      .ct-hero h1 {font-size:37px;}
      .ct-3d-stage {height:205px; transform:scale(.78);}
      .ct-stages {gap:7px;flex-direction:column;}
      .ct-stage {padding:9px 13px;}
      .ct-navbar .ct-nav-meta {display:none;}
      .block-container{padding-left:1.0rem!important;padding-right:1.0rem!important;}
    }
    @media(prefers-reduced-motion:reduce) {
      .ct-cube,.ct-orbit,.ct-float-label {animation:none !important;}
      div.stButton > button {transition:none !important;}
    }
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="ct-navbar">
      <div class="ct-brand"><div class="ct-brand-mark">⌘</div> CodeTrust AI</div>
      <div class="ct-nav-meta"><span class="ct-nav-dot">●</span> PYTHON LEARNING STUDIO &nbsp; · &nbsp; ENHANCED V3</div>
    </div>
    <section class="ct-hero" aria-label="CodeTrust AI Python debugger">
      <div class="ct-copy">
        <div class="ct-eyebrow"><span class="ct-dot"></span> YOUR AI CODING COMPANION</div>
        <h1>Every bug is a clue.<br><span class="ct-gradient">Every fix is progress.</span></h1>
        <p class="ct-hero-desc">Turn Python errors into learning moments. Paste your code, explore the explanation, and build confidence one line at a time.</p>
        <div><span class="ct-hero-tag">✦ <b>Diagnose</b> smarter</span><span class="ct-hero-tag">✦ <b>Learn</b> faster</span><span class="ct-hero-tag">✦ <b>Code</b> confidently</span></div>
      </div>
      <div class="ct-3d-stage" role="img" aria-label="Animated three-dimensional Python learning cube">
        <div class="ct-orbit"></div><div class="ct-orbit orbit-b"></div>
        <div class="ct-cube">
          <div class="ct-face ct-front"><span>Py</span><small>PYTHON</small></div>
          <div class="ct-face ct-back"><span>{ }</span><small>LOGIC</small></div>
          <div class="ct-face ct-right"><span>AI</span><small>ANALYZE</small></div>
          <div class="ct-face ct-left"><span>✓</span><small>LEARN</small></div>
          <div class="ct-face ct-top"><span>&lt;/&gt;</span><small>CODE</small></div>
          <div class="ct-face ct-bottom"><span>01</span><small>DEBUG</small></div>
        </div>
        <div class="ct-float-label one">def grow():</div>
        <div class="ct-float-label two">✔ fixed</div>
        <div class="ct-float-label three">{ learn }</div>
      </div>
    </section>
    <div class="ct-stages">
      <div class="ct-stage"><span class="ct-stage-num">01</span><span><b>Paste</b> your Python code</span></div>
      <div class="ct-stage"><span class="ct-stage-num">02</span><span><b>Analyze</b> the behavior</span></div>
      <div class="ct-stage"><span class="ct-stage-num">03</span><span><b>Understand</b> the fix</span></div>
    </div>""",
    unsafe_allow_html=True,
)

try:
    frozen = load_prompt()
except Exception as exc:
    st.error(f"Cannot load frozen V3 prompt: {exc}")
    st.stop()

if "editor_code" not in st.session_state:
    st.session_state["editor_code"] = ""
if "editor_requirements" not in st.session_state:
    st.session_state["editor_requirements"] = ""
if "editor_context" not in st.session_state:
    st.session_state["editor_context"] = ""

left, right = st.columns([1.07, 1], gap="large")

with left:
    st.markdown('<div class="ct-area-head"><div class="ct-section-title">✍️ Your coding space</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ct-section-sub">Start with your code or choose a quick example.</div>', unsafe_allow_html=True)
    button_cols = st.columns(3, gap="small")
    with button_cols[0]:
        st.button("🐞 Runtime bug", key="sample_runtime", on_click=choose_example, args=("runtime",), use_container_width=True)
    with button_cols[1]:
        st.button("✅ Valid code", key="sample_valid", on_click=choose_example, args=("valid",), use_container_width=True)
    with button_cols[2]:
        st.button("🧠 Logic bug", key="sample_logic", on_click=choose_example, args=("logic",), use_container_width=True)

    code = st.text_area(
        "Python source code",
        key="editor_code",
        height=272,
        placeholder="def greet(name):\n    print('Hello ' + name)\n\ngreet(None)",
        help="Paste Python code; this app analyzes the text and does not execute it.",
    )
    task = st.text_area(
        "What should your code do? (optional)",
        key="editor_requirements",
        height=98,
        placeholder="Example: Calculate the average score and print it.",
    )
    with st.expander("⚙️ Advanced: Python version & assumptions", expanded=False):
        assumptions = st.text_area(
            "Execution assumptions or missing context (optional)",
            key="editor_context",
            height=84,
            placeholder="E.g. Python 3.11; external input is guaranteed to be a list.",
        )
    analyze = st.button("✦ Analyze my code", type="primary", use_container_width=True)
    st.markdown('<div class="ct-note">🔒 Analysis only: your code is sent to OpenAI for diagnosis, never executed by this website.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="ct-area-head"><div class="ct-section-title">✨ Your AI insight</div><span class="ct-ai-label">✦ ENHANCED V3</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ct-section-sub">Learn what happened, where, why, and how to improve.</div>', unsafe_allow_html=True)

    if analyze:
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_code", None)
        if not code.strip():
            st.session_state["analysis_error"] = "Paste a Python code snippet first, or choose an example."
        elif len(code) > 30000:
            st.session_state["analysis_error"] = "For this demonstration, code must be 30,000 characters or fewer."
        elif not get_api_key():
            st.session_state["analysis_error"] = "The application owner needs to configure the API key. Run python setup_key.py on this computer."
        else:
            model_input = json.dumps(
                {
                    "code": code,
                    "task_description": task.strip() or "No explicit task requirements provided. Do not invent intended behavior.",
                    "execution_assumptions": assumptions.strip() or (
                        "Python 3.11; no unspecified external state is guaranteed. Treat missing facts as unknown."
                    ),
                },
                ensure_ascii=False,
            )
            try:
                with st.spinner("Exploring your code with CodeTrust AI…"):
                    response = OpenAI(api_key=get_api_key(), timeout=60.0, max_retries=1).responses.create(
                        model=MODEL,
                        instructions=frozen["prompt"] + "\n\n" + frozen["response_contract"],
                        input=model_input,
                        temperature=0,
                        max_output_tokens=1400,
                        store=False,
                    )
                answer = response.output_text or ""
                if response.status != "completed" or not answer.strip():
                    st.session_state["analysis_error"] = "The analysis was incomplete. Please try again."
                else:
                    st.session_state["last_answer"] = answer
                    st.session_state["last_code"] = code
            except Exception as exc:
                # Do not display arbitrary exception text; it could contain secrets.
                st.session_state["analysis_error"] = (
                    f"API request failed ({type(exc).__name__}). Check your internet connection, API key, or API billing."
                )

    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])

    answer = st.session_state.get("last_answer", "")
    if answer:
        sections = parse_sections(answer)
        if sections:
            descriptors = {
                "Diagnosis": ("🔎", "What's happening"),
                "Location": ("📍", "Where it happens"),
                "Explanation": ("💡", "Understand the cause"),
                "Suggested Fix": ("🛠️", "Suggested next step"),
            }
            for title in ("Diagnosis", "Location", "Explanation", "Suggested Fix"):
                symbol, label = descriptors[title]
                with st.container(border=True):
                    st.markdown(f"**{symbol} {title}** · {label}")
                    st.markdown(sections[title])
        else:
            with st.container(border=True):
                st.markdown(answer)
        st.download_button(
            "⬇️ Save my analysis",
            data=answer,
            file_name="codetrust_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption("AI-generated guidance can be inaccurate. Verify any diagnosis and suggested fix before relying on it.")
    elif not st.session_state.get("analysis_error"):
        st.markdown(
            '<div class="ct-empty"><div class="ct-empty-icon">⌁</div>'
            '<div class="ct-empty-title">Your next breakthrough starts here.</div>'
            '<div class="ct-empty-desc">Choose an example or paste your Python code, then click <b>Analyze my code</b> to reveal the explanation.</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="ct-footer"><span>© 2026 CodeTrust AI · Research prototype</span>'
    '<span>Powered by GPT-4.1 mini · Frozen Enhanced Prompt V3 · Explanations require verification</span></div>',
    unsafe_allow_html=True,
)
