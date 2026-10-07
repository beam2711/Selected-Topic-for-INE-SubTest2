"""
รู้ทันโจร — แชตบอต RAG ให้ความรู้เรื่องภัยมิจฉาชีพออนไลน์และสินเชื่อดิจิทัล
Streamlit + PyThaiNLP + sentence-transformers (multilingual-e5-small) + FAISS + Groq
"""

import html
import re

import streamlit as st
from groq import Groq

import rag_core as rc

st.set_page_config(page_title="รู้ทันโจร", page_icon="🛡️", layout="centered",
                   initial_sidebar_state="collapsed")

LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# คำถามตัวอย่าง จัดกลุ่มตามสถานการณ์ที่ผู้ใช้เจอจริง
STARTERS = {
    "เพิ่งโดนหลอก": [
        "ถูกหลอกโอนเงินไปแล้ว ต้องทำอะไรก่อน",
        "สงสัยว่าติดตั้งแอปดูดเงินไปแล้ว ต้องทำอย่างไร",
    ],
    "ไม่แน่ใจว่าโจรไหม": [
        "มีคนโทรมาอ้างเป็นตำรวจให้โอนเงินไปบัญชีปลายทางเพื่อตรวจสอบ จริงไหม",
        "งานกดไลก์ได้เงินจริงตอนแรก แล้วให้โอนเงินทำภารกิจพิเศษ เป็นมิจฉาชีพไหม",
    ],
    "เรื่องเงินกู้และธนาคาร": [
        "แอปกู้เงินส่งข้อความทวงหนี้ไปหาเพื่อนในโทรศัพท์เรา ทำได้ไหม",
        "โอนเงินเท่าไหร่ถึงต้องสแกนหน้า",
    ],
}

# ------------------------------------------------------------------ styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@400;500;600;700&display=swap');

:root{
  --ink:#14213D; --ink-soft:#4A5672; --paper:#F6F7F9; --card:#FFFFFF;
  --line:#DCE1E8; --amber:#FFB703; --amber-soft:#FFF4D6; --alert:#B3261E;
  --trust:#1F7A6D; --trust-soft:#E3F2EF;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family:'Anuphan', system-ui, sans-serif !important;
}
.stApp{ background:var(--paper); color:var(--ink); }
.block-container{ padding-top:2.2rem; padding-bottom:7rem; max-width:780px; }
[data-testid="stHeader"]{ background:transparent; }

/* hazard tape — the one loud element */
.tape{
  height:14px; border-radius:3px; margin-bottom:1.4rem;
  background:repeating-linear-gradient(-45deg,var(--amber) 0 18px,var(--ink) 18px 36px);
}
.hero h1{
  font-size:2.6rem; line-height:1.15; font-weight:700; color:var(--ink);
  margin:0 0 .5rem 0; padding:0; letter-spacing:-.01em;
}
.hero p{ font-size:1.05rem; color:var(--ink-soft); margin:0 0 1.2rem 0; max-width:36em; }

/* emergency hotline strip */
.hotline{
  display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
  background:var(--amber-soft); border:2px solid var(--amber); border-radius:10px;
  padding:.85rem 1.1rem; margin-bottom:1.6rem;
}
.hotline .num{
  font-size:2rem; font-weight:700; color:var(--alert); line-height:1;
  text-decoration:none; font-variant-numeric:tabular-nums;
}
.hotline .txt{ color:var(--ink); font-size:.98rem; line-height:1.45; flex:1; min-width:14em; }
.hotline .txt b{ font-weight:600; }

/* starter questions */
.group-title{ font-weight:600; color:var(--ink); font-size:.95rem; margin:.4rem 0 .35rem; }
div[data-testid="stMainBlockContainer"] .stButton > button,
section.main .stButton > button{
  width:100%; text-align:left; justify-content:flex-start;
  background:var(--card); color:var(--ink); border:1px solid var(--line);
  border-radius:8px; padding:.65rem .85rem; font-size:.95rem; min-height:3.2rem;
}
.stButton > button:hover{ border-color:var(--ink); color:var(--ink); background:var(--card); }
.stButton > button:focus-visible{ outline:3px solid var(--amber); outline-offset:2px; }
.stButton > button p, .stButton > button div{
  text-align:left !important; white-space:normal !important;
  overflow:visible !important; text-overflow:clip !important; width:100%;
}
.stButton > button > div{ justify-content:flex-start !important; }

/* chat */
[data-testid="stChatMessage"]{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.9rem 1rem; margin-bottom:.8rem;
}
[data-testid="stChatMessage"]:has(.is-user){
  background:transparent; border:none; padding:.2rem 1rem;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{ font-size:1rem; line-height:1.7; }
[data-testid="stChatInput"]{ border-radius:12px; }
[data-testid="stChatInput"] textarea{ font-size:1rem; }
.query-note{ font-size:.85rem; color:var(--ink-soft); margin:-.2rem 0 .5rem; }
.refuse{
  border-left:4px solid var(--ink-soft); background:var(--paper);
  padding:.7rem .9rem; border-radius:6px; color:var(--ink);
}

.is-user{ display:none; }
[data-testid="stChatMessage"]:has(.is-user) [data-testid="stElementContainer"]:has(.is-user){ display:none; }

/* sources */
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:10px; background:var(--paper); }
[data-testid="stExpander"] summary p{ font-weight:600; font-size:.92rem; }
.src{ padding:.7rem 0; border-bottom:1px solid var(--line); }
.src:last-child{ border-bottom:none; }
.src-head{ line-height:1.5; }
.src-head span{ margin-right:.35rem; }
.src-no{ font-weight:700; color:var(--ink); }
.src-title{ font-weight:600; color:var(--ink); }
.src-sec{ color:var(--ink-soft); }
.src-meta{ display:flex; gap:.45rem; flex-wrap:wrap; margin:.35rem 0; }
.tag{ font-size:.78rem; padding:.1rem .5rem; border-radius:999px; border:1px solid var(--line); color:var(--ink-soft); background:var(--card); }
.tag.used{ background:var(--trust-soft); color:var(--trust); border-color:var(--trust); font-weight:600; }
.src-text{ font-size:.9rem; color:var(--ink-soft); line-height:1.6; white-space:pre-line; }
.src-from{ font-size:.8rem; color:var(--ink-soft); margin-top:.3rem; }

/* sidebar */
[data-testid="stSidebar"]{ background:var(--card); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .stButton > button{ width:100%; }
.foot{ font-size:.82rem; color:var(--ink-soft); text-align:center; margin-top:2rem; }

@media (max-width:640px){
  .hero h1{ font-size:2rem; }
  .hotline .num{ font-size:1.7rem; }
}
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; animation:none !important; } }
</style>
"""


# ----------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="กำลังเตรียมคลังความรู้ (ครั้งแรกใช้เวลาประมาณ 1 นาที)...")
def load_knowledge_base():
    docs = rc.load_documents()
    chunks = rc.build_chunks(docs)
    embedder = rc.load_embedder()
    vectors = rc.embed_passages(embedder, [c.embed_text for c in chunks])
    index = rc.build_index(vectors)
    return docs, chunks, embedder, index


@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None
    return Groq(api_key=api_key)


# ----------------------------------------------------------------- LLM calls
def chat_completion(client, model, messages, stream=False, max_tokens=1200):
    kwargs = dict(model=model, messages=messages, temperature=0.1, max_tokens=max_tokens, stream=stream)
    try:
        # โมเดล gpt-oss รองรับ reasoning_effort; ตั้ง low เพื่อตอบเร็ว
        return client.chat.completions.create(reasoning_effort="low", **kwargs)
    except TypeError:
        return client.chat.completions.create(**kwargs)


def condense_question(client, model, history, question):
    """เขียนคำถามต่อเนื่องให้เป็นคำถามเดี่ยว เพื่อใช้ค้นหาเอกสาร (รองรับการคุยต่อเนื่อง)"""
    turns = [m for m in history if m["role"] in ("user", "assistant")][-4:]
    if not turns:
        return question
    hist = "\n".join(
        f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:400]}" for m in turns
    )
    try:
        resp = chat_completion(
            client, model,
            [{"role": "user", "content": rc.CONDENSE_PROMPT.format(history=hist, question=question)}],
            max_tokens=300,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception:
        return question


def stream_answer(client, model, history, question, results):
    messages = [{"role": "system", "content": rc.SYSTEM_PROMPT}]
    # ใส่ประวัติสั้น ๆ (ไม่รวม context เดิม) เพื่อให้คุยต่อเนื่องได้
    for m in history[-4:]:
        messages.append({"role": m["role"], "content": m["content"][:800]})
    messages.append({"role": "user", "content": rc.build_user_prompt(question, results)})
    stream = chat_completion(client, model, messages, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


# ------------------------------------------------------------------ helpers
def render_sources(sources, answer_text):
    if not sources:
        st.caption("ไม่มีเอกสารที่ตรงกับคำถามพอ ระบบจึงไม่ได้ส่งคำถามให้ AI ตอบ")
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    n_used = len([i for i in range(1, len(sources) + 1) if i in cited])
    label = f"เอกสารที่ใช้ตอบ {n_used} จาก {len(sources)} ชิ้นที่ค้นเจอ"
    with st.expander(label, expanded=False):
        parts = []
        for i, s in enumerate(sources, start=1):
            used = i in cited
            text = html.escape(s["text"][:420]) + ("…" if len(s["text"]) > 420 else "")
            parts.append(
                f'<div class="src"><div class="src-head">'
                f'<span class="src-no">[{i}]</span>'
                f'<span class="src-title">{html.escape(s["title"])}</span>'
                f'<span class="src-sec">· {html.escape(s["section"])}</span></div>'
                f'<div class="src-meta">'
                f'<span class="tag{" used" if used else ""}">{"ใช้ในคำตอบ" if used else "ค้นเจอแต่ไม่ได้ใช้"}</span>'
                f'<span class="tag">ความใกล้เคียง {s["score"]:.2f}</span>'
                f'<span class="tag">{html.escape(s["file"])}</span></div>'
                f'<div class="src-text">{text}</div>'
                f'<div class="src-from">ที่มา: {html.escape(s["source"])}</div></div>'
            )
        st.markdown("".join(parts), unsafe_allow_html=True)


def refusal_html():
    return (f'<div class="refuse"><b>{rc.NOT_FOUND_TEXT}</b><br>'
            "ผู้ช่วยนี้ตอบได้เฉพาะเรื่องกลโกงออนไลน์ บัญชีม้า มาตรการธนาคาร Virtual Bank "
            "และสินเชื่อดิจิทัล ลองถามใหม่ในหัวข้อเหล่านี้</div>")


# ------------------------------------------------------------------- layout
st.markdown(CSS, unsafe_allow_html=True)

docs, chunks, embedder, index = load_knowledge_base()
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ตั้งค่าการค้นหา")
    model = st.selectbox("โมเดล AI (Groq)", LLM_MODELS, index=0)
    top_k = st.slider("จำนวนเอกสารที่ค้นคืน (top-k)", 2, 8, 6)
    min_score = st.slider("คะแนนความใกล้เคียงขั้นต่ำ", 0.60, 0.95, 0.78, 0.01,
                          help="ถ้าไม่มีเอกสารชิ้นใดผ่านเกณฑ์ ระบบจะตอบว่าไม่พบข้อมูลทันที "
                               "ลดค่าลงถ้าระบบปฏิเสธบ่อยเกินไป")
    use_condense = st.checkbox("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", value=True)
    st.divider()
    st.markdown("### คลังความรู้")
    c1, c2 = st.columns(2)
    c1.metric("เอกสาร", f"{len(docs)} ไฟล์")
    c2.metric("Chunks", f"{len(chunks)}")
    with st.expander("รายชื่อเอกสาร"):
        for d in docs:
            st.markdown(f"- {d.title}")
    st.divider()
    if st.button("เริ่มบทสนทนาใหม่", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# hero
st.markdown(
    """
<div class="tape" aria-hidden="true"></div>
<div class="hero">
  <h1>มีคนให้โอนเงิน? หยุดก่อน แล้วถามที่นี่</h1>
  <p>ถามเรื่องกลโกงออนไลน์ บัญชีม้า มาตรการธนาคาร และสินเชื่อดิจิทัลได้เลย
  ทุกคำตอบมาจากเอกสารของหน่วยงานทางการ และบอกแหล่งที่มาให้ตรวจสอบได้</p>
</div>
<div class="hotline" role="note">
  <a class="num" href="tel:1441">1441</a>
  <div class="txt"><b>เพิ่งโอนเงินให้มิจฉาชีพ?</b> โทรสายด่วน AOC ทันที ตลอด 24 ชั่วโมง
  เพื่อขอระงับบัญชีปลายทางก่อนเงินถูกโอนต่อ</div>
</div>
""",
    unsafe_allow_html=True,
)

if client is None:
    st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY — เปิด Manage app › Settings › Secrets "
             "แล้วเพิ่ม `GROQ_API_KEY = \"gsk_...\"` จากนั้นรีเฟรชหน้านี้")
    st.stop()

question = st.chat_input("พิมพ์สิ่งที่เจอ เช่น ได้ SMS จากธนาคารให้กดลิงก์ ควรทำอย่างไร")
if not question:
    question = st.session_state.pop("pending_question", None)

# empty state: starter questions grouped by situation
if not st.session_state.messages and not question:
    cols = st.columns(len(STARTERS))
    for col, (group, qs) in zip(cols, STARTERS.items()):
        with col:
            st.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)
            for q in qs:
                if st.button(q, key=f"starter-{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🛡️" if m["role"] == "assistant" else "🙋"):
        if m["role"] == "user":
            st.markdown('<span class="is-user"></span>', unsafe_allow_html=True)
        if m.get("refused"):
            st.markdown(refusal_html(), unsafe_allow_html=True)
        else:
            st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("search_query"):
                st.markdown(f'<div class="query-note">ค้นหาด้วย: {html.escape(m["search_query"])}</div>',
                            unsafe_allow_html=True)
            render_sources(m.get("sources", []), m["content"])

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown('<span class="is-user"></span>', unsafe_allow_html=True)
        st.markdown(question)

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("กำลังค้นหาในเอกสาร..."):
            search_query = question
            if use_condense and history:
                search_query = condense_question(client, model, history, question)
            results = rc.search(index, rc.embed_query(embedder, search_query), chunks,
                                top_k=top_k, min_score=min_score)

        refused = not results
        if refused:
            answer = rc.NOT_FOUND_TEXT
            st.markdown(refusal_html(), unsafe_allow_html=True)
        else:
            try:
                answer = st.write_stream(stream_answer(client, model, history, question, results))
            except Exception as e:
                answer = f"เรียก AI ไม่สำเร็จ: {e} — ลองส่งคำถามอีกครั้ง หรือเปลี่ยนโมเดลในแถบตั้งค่า"
                st.error(answer)

        if search_query != question:
            st.markdown(f'<div class="query-note">ค้นหาด้วย: {html.escape(search_query)}</div>',
                        unsafe_allow_html=True)
        sources = [dict(file=c.file, title=c.title, section=c.section, source=c.source,
                        text=c.text, score=s) for c, s in results]
        render_sources(sources, answer)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources, "refused": refused,
        "search_query": search_query if search_query != question else "",
    })

st.markdown('<div class="foot">ข้อมูลเพื่อการศึกษา · ตั้งค่าการค้นหาได้ที่แถบด้านซ้าย (กด › มุมซ้ายบน)</div>',
            unsafe_allow_html=True)
