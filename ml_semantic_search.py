import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# --- 1. Завантаження моделі та даних wrong_usage
MODEL_ID = os.getenv("HF_MODEL_ID", "KyzenLamar/term-analysis-embedder")
# MODEL_PATH = "fine_tuned_semantic_model_mnrl"

# Load model (один раз при старті програми!)
# model = SentenceTransformer(MODEL_PATH)
print(f"[INFO] Loading SentenceTransformer model: {MODEL_ID}")
model = SentenceTransformer(MODEL_ID)
# model = SentenceTransformer("KyzenLamar/term-analysis-embedder")

def load_wrong_usages(csv_path):
    df = pd.read_csv(csv_path)
    sentences = list(df["wrong_usage"])
    terms = list(df["approved_term"])
    comments = list(df["comment"]) if "comment" in df.columns else [""]*len(df)
    # embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    return sentences, terms, comments, embeddings

# --- 2. Semantic search function
def semantic_search(sentence, wrong_sentences, terms, comments, wrong_embeds, topn=5, threshold=0.7):
    # query_emb = model.encode(sentence, convert_to_tensor=True)
    query_emb = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.cos_sim(query_emb, wrong_embeds)[0]
    top_ids = hits.argsort(descending=True)[:topn]
    results = []
    for idx in top_ids:
        score = hits[idx].item()
        if score >= threshold:
            results.append({
                "score": round(score, 3),
                "wrong_usage": wrong_sentences[idx],
                "approved_term": terms[idx],
                "comment": comments[idx] if comments else ""
            })
    return results