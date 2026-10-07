"""
ประเมินส่วน Retrieval ด้วย test_questions.csv (ไม่ต้องใช้ API Key)
  - Hit@k   : คำถามที่มีคำตอบ ค้นเจอไฟล์ที่ถูกต้องใน top-k หรือไม่
  - คะแนนสูงสุดของคำถามที่ไม่มีคำตอบ ใช้ช่วยเลือกค่า min_score ในแอป

รัน:  python evaluate.py
"""
import pandas as pd

import rag_core as rc

TOP_K = 4

docs = rc.load_documents()
chunks = rc.build_chunks(docs)
embedder = rc.load_embedder()
index = rc.build_index(rc.embed_passages(embedder, [c.embed_text for c in chunks]))
print(f"documents={len(docs)}  chunks={len(chunks)}  "
      f"total_chars={sum(len(d.text) for d in docs):,}")

df = pd.read_csv("test_questions.csv")
hits, rows = 0, []
for _, r in df.iterrows():
    res = rc.search(index, rc.embed_query(embedder, r.question), chunks, top_k=TOP_K)
    files = [c.file for c, _ in res]
    top = res[0][1] if res else 0.0
    hit = r.source_file in files if r.answerable == "yes" else None
    hits += bool(hit)
    rows.append({"id": r.id, "answerable": r.answerable, "top1_score": round(top, 3),
                 "hit@k": hit, "top1_file": files[0] if files else "-"})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
n_yes = (df.answerable == "yes").sum()
print(f"\nHit@{TOP_K} = {hits}/{n_yes} = {hits / n_yes:.0%}")
print("min top1 (answerable)   =", out[out.answerable == "yes"].top1_score.min())
print("max top1 (unanswerable) =", out[out.answerable == "no"].top1_score.max())
