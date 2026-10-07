"""
รู้ทันโจร — แชตบอต RAG ให้ความรู้เรื่องภัยมิจฉาชีพออนไลน์และสินเชื่อดิจิทัล
Streamlit + PyThaiNLP + sentence-transformers (multilingual-e5-small) + FAISS + Groq
"""

import re

import streamlit as st
from groq import Groq

import rag_core as rc

st.set_page_config(page_title="รู้ทันโจร | Scam & Digital Loan RAG", page_icon="🛡️", layout="centered")

LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
EXAMPLE_QUESTIONS = [
    "ถูกหลอกโอนเงินไปแล้ว ต้องทำอะไรก่อน",
    "มีคนโทรมาอ้างเป็นตำรวจให้โอนเงินไปบัญชีปลายทางเพื่อตรวจสอบ จริงไหม",
    "รับจ้างเปิดบัญชีธนาคารให้คนอื่นใช้ ผิดกฎหมายไหม โทษเท่าไหร่",
    "โอนเงินเท่าไหร่ถึงต้องสแกนหน้า",
    "Virtual Bank รายแรกของไทยคือใคร",
    "แอปกู้เงินทวงหนี้กับเพื่อนในโทรศัพท์เรา ทำได้ไหม",
]


# ----------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="กำลังโหลดเอกสาร สร้าง Embedding และ FAISS Index (ครั้งแรกใช้เวลาประมาณ 1 นาที)...")
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
        st.caption("📄 ไม่มีเอกสารที่เกี่ยวข้องเพียงพอ จึงไม่ได้ส่งคำถามให้ LLM")
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    with st.expander(f"📚 เอกสารอ้างอิงที่ค้นพบ ({len(sources)} ชิ้น)", expanded=False):
        for i, s in enumerate(sources, start=1):
            mark = "✅ ใช้ในคำตอบ" if i in cited else "ค้นพบแต่ไม่ได้อ้างถึง"
            st.markdown(f"**[{i}] {s['title']}** — {s['section']}  \n"
                        f"`{s['file']}` · ความใกล้เคียง {s['score']:.3f} · {mark}")
            st.markdown(f"> {s['text'][:500].replace(chr(10), chr(10) + '> ')}"
                        + ("…" if len(s['text']) > 500 else ""))
            st.caption(f"แหล่งที่มา: {s['source']}")


# ------------------------------------------------------------------- layout
docs, chunks, embedder, index = load_knowledge_base()
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🛡️ รู้ทันโจร")
    st.write("ผู้ช่วยตอบคำถามเรื่องภัยมิจฉาชีพออนไลน์ บัญชีม้า มาตรการธนาคาร "
             "Virtual Bank และสินเชื่อดิจิทัล โดยตอบจากคลังเอกสารเท่านั้น")
    st.metric("เอกสารความรู้", f"{len(docs)} ไฟล์")
    st.metric("Chunks ใน FAISS", f"{len(chunks)} ชิ้น")

    st.subheader("ลองถามคำถามตัวอย่าง")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    with st.expander("⚙️ ตั้งค่าการค้นหา"):
        model = st.selectbox("LLM (Groq)", LLM_MODELS, index=0)
        top_k = st.slider("จำนวนเอกสารที่ค้นคืน (top-k)", 2, 8, 4)
        min_score = st.slider("คะแนนความใกล้เคียงขั้นต่ำ", 0.60, 0.95, 0.78, 0.01,
                              help="ถ้าไม่มี chunk ใดผ่านเกณฑ์ ระบบจะตอบว่าไม่พบข้อมูลทันที "
                                   "(ลดค่าลงถ้าระบบปฏิเสธบ่อยเกินไป)")
        use_condense = st.checkbox("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", value=True)

    if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("⚠️ ข้อมูลเพื่อการศึกษา หากถูกหลอกโอนเงิน โทร **1441** ทันที (24 ชม.)")

st.title("🛡️ รู้ทันโจร")
st.caption("แชตบอต RAG รู้ทันภัยมิจฉาชีพออนไลน์และสินเชื่อดิจิทัล · ตอบจากเอกสาร พร้อมแหล่งอ้างอิงทุกครั้ง")

if client is None:
    st.error("ไม่พบ GROQ_API_KEY ใน Secrets — ไปที่ Settings → Secrets ของแอปบน Streamlit Cloud "
             "แล้วเพิ่ม `GROQ_API_KEY = \"...\"`")
    st.stop()

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown("สวัสดีครับ ถามเรื่องกลโกงออนไลน์ บัญชีม้า วิธีรับมือเมื่อถูกหลอก "
                    "มาตรการธนาคาร หรือสินเชื่อดิจิทัลได้เลย หรือกดคำถามตัวอย่างทางซ้ายมือ")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🛡️" if m["role"] == "assistant" else "🙋"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("search_query"):
                st.caption(f"🔎 คำค้นที่ใช้: {m['search_query']}")
            render_sources(m.get("sources", []), m["content"])

question = st.chat_input("พิมพ์คำถาม เช่น ได้ SMS จากธนาคารให้กดลิงก์ ควรทำอย่างไร")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("กำลังค้นหาเอกสาร..."):
            search_query = question
            if use_condense and history:
                search_query = condense_question(client, model, history, question)
            results = rc.search(index, rc.embed_query(embedder, search_query), chunks,
                                top_k=top_k, min_score=min_score)

        if search_query != question:
            st.caption(f"🔎 คำค้นที่ใช้: {search_query}")

        if not results:
            answer = (f"**{rc.NOT_FOUND_TEXT}** — ระบบนี้ตอบได้เฉพาะเรื่องภัยมิจฉาชีพออนไลน์ "
                      "บัญชีม้า มาตรการธนาคาร Virtual Bank และสินเชื่อดิจิทัลตามเอกสารในคลังความรู้")
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(stream_answer(client, model, history, question, results))
            except Exception as e:
                answer = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
                st.error(answer)

        sources = [dict(file=c.file, title=c.title, section=c.section, source=c.source,
                        text=c.text, score=s) for c, s in results]
        render_sources(sources, answer)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
        "search_query": search_query if search_query != question else "",
    })
