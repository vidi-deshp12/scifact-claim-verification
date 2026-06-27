import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── 1. Load index and doc_ids ────────────────────────────────────────────────
index = faiss.read_index("data/abstract_index.faiss")
with open("data/doc_ids.json") as f:
    doc_ids = json.load(f)

print(f"Index loaded: {index.ntotal} vectors")
print(f"Doc IDs loaded: {len(doc_ids)}")

# ── 2. Load model and claims ─────────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

claims_dev = load_jsonl("data/claims_dev.jsonl")
print(f"Dev claims loaded: {len(claims_dev)}")

# ── 3. Separate NEI from non-NEI ─────────────────────────────────────────────
# Recall@k is only meaningful for claims with gold evidence docs
non_nei = [ex for ex in claims_dev if ex["evidence"]]
nei     = [ex for ex in claims_dev if not ex["evidence"]]
print(f"Non-NEI claims (eval targets): {len(non_nei)}")
print(f"NEI claims (excluded from retrieval eval): {len(nei)}")

# ── 4. Get gold doc ids ──────────────────────────────────────────────────────
def get_gold_doc_ids(example):
    """All gold evidence doc_ids for a claim — any one counts as a hit."""
    return set(example["evidence"].keys())  # already strings

# ── 5. Embed all dev claims in one batch ─────────────────────────────────────
claim_texts = [ex["claim"] for ex in non_nei]
print(f"\nEmbedding {len(claim_texts)} non-NEI claims...")

claim_embeddings = model.encode(
    claim_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"Claim embeddings shape: {claim_embeddings.shape}")

# ── 6. Search FAISS ───────────────────────────────────────────────────────────
K = 10  # retrieve top-10, evaluate at k=1,3,5,10
print(f"\nSearching index with k={K}...")
scores, indices = index.search(claim_embeddings, K)
print(f"Search complete. Results shape: {scores.shape}")

# ── 7. Compute Recall@k ──────────────────────────────────────────────────────
# Recall@k = fraction of claims where gold doc appears in top-k results
results = []

for i, ex in enumerate(non_nei):
    gold_ids = get_gold_doc_ids(ex)
    retrieved = [doc_ids[idx] for idx in indices[i]]  # top-K doc_ids
    top_scores = scores[i].tolist()

    # Check at each k
    hit_at = {}
    for k in [1, 3, 5, 10]:
        hit_at[k] = any(doc_id in gold_ids for doc_id in retrieved[:k])

    # Find rank of first gold doc
    rank = None
    for r, doc_id in enumerate(retrieved, 1):
        if doc_id in gold_ids:
            rank = r
            break

    results.append({
        "id": ex["id"],
        "claim": ex["claim"],
        "gold_doc_ids": list(gold_ids),
        "retrieved_doc_ids": retrieved,
        "top_scores": top_scores,
        "hit_at_1": hit_at[1],
        "hit_at_3": hit_at[3],
        "hit_at_5": hit_at[5],
        "hit_at_10": hit_at[10],
        "gold_rank": rank  # None if not in top-10
    })

# ── 8. Aggregate metrics ─────────────────────────────────────────────────────
n = len(results)
recall_at = {k: sum(r[f"hit_at_{k}"] for r in results) / n 
             for k in [1, 3, 5, 10]}
found_in_top10 = sum(1 for r in results if r["gold_rank"] is not None)

print(f"\n── Retrieval Evaluation (Dev set, non-NEI only, n={n}) ──")
for k in [1, 3, 5, 10]:
    print(f"  Recall@{k:2d}: {recall_at[k]:.4f} ({sum(r[f'hit_at_{k}'] for r in results)}/{n})")
print(f"  Gold in top-10: {found_in_top10}/{n} ({found_in_top10/n:.4f})")

# ── 9. Save results for later analysis ───────────────────────────────────────
with open("results/retrieval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved detailed results to results/retrieval_results.json")

# ── 10. Sanity check — print a few examples ──────────────────────────────────
print(f"\n── Sample results (first 3 non-NEI claims) ──")
for r in results[:3]:
    print(f"\nClaim: {r['claim']}")
    print(f"Gold:  {r['gold_doc_ids']}")
    print(f"Top-3: {r['retrieved_doc_ids'][:3]}")
    print(f"Scores:{[f'{s:.4f}' for s in r['top_scores'][:3]]}")
    print(f"Hit@1={r['hit_at_1']} Hit@3={r['hit_at_3']} Hit@5={r['hit_at_5']}")
    print(f"Gold rank: {r['gold_rank']}")