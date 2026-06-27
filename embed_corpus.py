"""
corpus_data[i]
     │
     ├── doc["doc_id"] → saved as doc_ids[i] in doc_ids.json
     │
     └── " ".join(doc["abstract"]) → embedded → vector at position i in FAISS index

Query time:
claim text → embed → query_vec → FAISS search → positions → doc_ids[position] → doc_ids
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Script started")

# ── 1. Load corpus ──────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

corpus_data = load_jsonl("data/corpus.jsonl")
print(f"Loaded {len(corpus_data)} corpus docs")

# ── 2. Prepare texts ────────────────────────────────────────────────────────
# abstract is a list of sentence strings — join into one passage per doc

doc_ids   = [str(doc["doc_id"]) for doc in corpus_data]   # keep as strings (matches evidence keys)
abstracts = [" ".join(doc["abstract"]) for doc in corpus_data]

print(f"Example abstract (first 120 chars):\n  {abstracts[0][:120]}...")

# ── 3. Embed ────────────────────────────────────────────────────────────────

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"\nEncoding {len(abstracts)} abstracts...")

embeddings = model.encode(
    abstracts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # unit vectors → IndexFlatIP = cosine sim
)

print(f"Embedding matrix shape: {embeddings.shape}")   # (5183, 384)
print(f"First vector norm: {np.linalg.norm(embeddings[0]):.4f}")   # should be ~1.0

# ── 4. Build FAISS index ────────────────────────────────────────────────────

dim   = embeddings.shape[1]    # 384 for MiniLM-L6-v2
index = faiss.IndexFlatIP(dim) # inner product on normalized vecs = cosine sim
index.add(embeddings)

print(f"\nFAISS index built — {index.ntotal} vectors, dim={dim}")

# ── 5. Persist ──────────────────────────────────────────────────────────────

faiss.write_index(index, "data/abstract_index.faiss")

with open("data/doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

print("Saved: data/abstract_index.faiss")
print("Saved: data/doc_ids.json")

# ── 6. Sanity check ─────────────────────────────────────────────────────────
# Query with a real train claim and confirm the right abstract comes back

claims_train = load_jsonl("data/claims_train.jsonl")

# grab first claim that has evidence (non-NEI)
test_claim = next(ex for ex in claims_train if ex["evidence"])
gold_doc_id = list(test_claim["evidence"].keys())[0]

query_vec = model.encode(
    [test_claim["claim"]],
    normalize_embeddings=True,
    convert_to_numpy=True,
)

scores, indices = index.search(query_vec, k=5)

retrieved_ids = [doc_ids[i] for i in indices[0]]
scores_list   = scores[0].tolist()

print(f"\n── Sanity check ──")
print(f"Claim : {test_claim['claim']}")
print(f"Gold  : {gold_doc_id}")
print(f"Top-5 retrieved doc_ids + scores:")
for rank, (doc_id, score) in enumerate(zip(retrieved_ids, scores_list), 1):
    hit = " ✓" if doc_id == gold_doc_id else ""
    print(f"  {rank}. {doc_id}  (score={score:.4f}){hit}")

