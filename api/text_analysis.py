"""
text_analysis.py
FastAPI router for Text Analysis.

Endpoints:
    POST /api/analysis/text             — Full pipeline: tokens + TF-IDF + sentiment + topics + NER
    POST /api/analysis/text/sentiment   — Sentiment analysis only
    POST /api/analysis/text/topics      — Topic modeling (LDA-style)
    POST /api/analysis/text/ner         — Named entity recognition
    POST /api/analysis/text/similarity  — Document similarity matrix
    POST /api/analysis/text/keywords    — TF-IDF keyword extraction
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import traceback
import warnings
import math
import re
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """Convert numpy/pandas types to plain Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ── Stopwords ──────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "it", "my", "your", "his",
    "her", "our", "its", "not", "no", "so", "yet", "both", "either",
    "neither", "as", "if", "when", "while", "because", "since", "although",
    "very", "just", "also", "more", "most", "some", "any", "all", "each",
    "every", "only", "too", "even", "than", "then", "there", "here",
    "about", "after", "before", "between", "into", "through", "until",
    "without", "within", "across", "up", "out", "down", "off", "over",
    "such", "much", "many", "few", "other", "same", "own", "like",
    "well", "back", "still", "way", "take", "get", "make", "go", "know",
    "re", "s", "t", "ve", "ll", "d", "m",
}


def _tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer, removes stopwords."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _detect_language(text: str) -> str:
    """Heuristic language detection based on character composition."""
    ascii_chars = len(re.findall(r"[a-zA-Z]", text))
    total_chars = max(len(text.replace(" ", "")), 1)
    return "en" if ascii_chars / total_chars > 0.4 else "other"


# ── TF-IDF (pure Python, no sklearn required) ─────────────────

def _compute_tfidf_corpus(
    docs: List[str], max_features: int = 50
) -> Dict[str, float]:
    """
    Compute corpus-level TF-IDF scores.
    Returns a dict of {word: score} sorted by relevance.
    """
    tokenized = [_tokenize(d) for d in docs]
    n = max(len(tokenized), 1)

    # Term frequency per document
    tf_list: List[Dict[str, float]] = []
    for tokens in tokenized:
        freq = Counter(tokens)
        total = max(len(tokens), 1)
        tf_list.append({w: c / total for w, c in freq.items()})

    # Document frequency
    df: Counter = Counter()
    for tf in tf_list:
        df.update(tf.keys())

    # Inverse document frequency (smoothed)
    idf = {
        w: math.log((n + 1) / (cnt + 1)) + 1
        for w, cnt in df.items()
    }

    # Aggregate TF-IDF score across all documents
    scores: Dict[str, float] = defaultdict(float)
    for tf in tf_list:
        for w, v in tf.items():
            scores[w] += v * idf.get(w, 1.0)

    top = sorted(scores.items(), key=lambda x: -x[1])[:max_features]
    return {w: round(s, 4) for w, s in top}


def _doc_tfidf_vectors(
    tokenized: List[List[str]],
) -> List[Dict[str, float]]:
    """Build per-document TF-IDF vectors for similarity computation."""
    n = max(len(tokenized), 1)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    idf = {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}
    vectors = []
    for tokens in tokenized:
        freq = Counter(tokens)
        total = max(len(tokens), 1)
        vectors.append({w: (c / total) * idf.get(w, 1.0) for w, c in freq.items()})
    return vectors


# ── Lexicon-based Sentiment ────────────────────────────────────

POSITIVE_LEXICON = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "superb", "outstanding", "perfect", "love", "best", "awesome",
    "brilliant", "impressive", "happy", "pleased", "satisfied",
    "recommend", "fast", "quick", "easy", "helpful", "friendly",
    "professional", "smooth", "reliable", "nice", "beautiful",
    "exceptional", "superior", "delightful", "effective", "efficient",
    "exceeded", "quality", "genuine", "secure", "accurate", "strong",
    "clear", "innovative", "affordable", "valuable", "trustworthy",
}

NEGATIVE_LEXICON = {
    "bad", "terrible", "awful", "horrible", "worst", "poor",
    "disappointing", "broken", "useless", "waste", "fail", "slow",
    "difficult", "hard", "problem", "issue", "wrong", "mistake",
    "error", "fake", "cheap", "ugly", "confusing", "unhelpful",
    "rude", "overpriced", "delayed", "missing", "damaged",
    "frustrating", "annoying", "regret", "refund", "complaint",
    "dissatisfied", "defective", "inferior", "unreliable", "unclear",
    "expensive", "complicated", "broke", "failed", "returned",
}


def _score_sentiment(text: str) -> Dict[str, Any]:
    """
    Lexicon-based sentiment scoring.
    Returns compound score in [-1, 1] and a label.
    Replace with a transformer model (e.g. cardiffnlp/twitter-roberta)
    for production use.
    """
    tokens = set(_tokenize(text))
    pos_hits = tokens & POSITIVE_LEXICON
    neg_hits = tokens & NEGATIVE_LEXICON
    pos = len(pos_hits)
    neg = len(neg_hits)
    denom = max(pos + neg, 1)
    compound = round((pos - neg) / denom, 4)

    if compound >= 0.1:
        label = "positive"
    elif compound <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": compound,
        "positiveScore": round(pos / denom, 4),
        "negativeScore": round(neg / denom, 4),
        "label": label,
        "positiveWords": list(pos_hits)[:8],
        "negativeWords": list(neg_hits)[:8],
    }


# ── Topic Modeling (word co-occurrence based) ─────────────────

TOPIC_SEED_WORDS: List[Dict] = [
    {
        "id": "T1",
        "label": "Product Quality",
        "seeds": {"quality", "material", "build", "durable", "design",
                  "solid", "sturdy", "well", "made", "construction"},
    },
    {
        "id": "T2",
        "label": "Customer Service",
        "seeds": {"service", "support", "customer", "response", "helpful",
                  "staff", "team", "representative", "contact", "resolved"},
    },
    {
        "id": "T3",
        "label": "Shipping & Delivery",
        "seeds": {"shipping", "delivery", "arrived", "packaging", "shipped",
                  "fast", "quick", "days", "received", "tracking"},
    },
    {
        "id": "T4",
        "label": "Value & Price",
        "seeds": {"price", "value", "money", "worth", "cost",
                  "affordable", "expensive", "cheap", "paid", "refund"},
    },
]


def _run_topic_model(docs: List[str]) -> List[Dict]:
    """
    Seed-word based topic assignment.
    Each document is assigned to the topic with the highest seed overlap.
    For production use, replace with sklearn LatentDirichletAllocation
    or BERTopic.
    """
    tokenized = [set(_tokenize(d)) for d in docs]
    topic_doc_counts: Dict[str, int] = {t["id"]: 0 for t in TOPIC_SEED_WORDS}
    topic_all_words: Dict[str, List[str]] = {t["id"]: [] for t in TOPIC_SEED_WORDS}

    for tokens in tokenized:
        best_id = None
        best_score = -1
        for t in TOPIC_SEED_WORDS:
            score = len(tokens & t["seeds"])
            if score > best_score:
                best_score = score
                best_id = t["id"]
        if best_id and best_score > 0:
            topic_doc_counts[best_id] += 1
            topic_all_words[best_id].extend(list(tokens))

    results = []
    for t in TOPIC_SEED_WORDS:
        top_words = [w for w, _ in Counter(topic_all_words[t["id"]]).most_common(8)]
        coverage = round(topic_doc_counts[t["id"]] / max(len(docs), 1) * 100, 1)
        results.append({
            "id": t["id"],
            "label": t["label"],
            "words": top_words,
            "coverage": coverage,
            "docCount": topic_doc_counts[t["id"]],
        })

    return sorted(results, key=lambda x: -x["coverage"])


# ── Pattern-based Named Entity Recognition ────────────────────

_DATE_RE    = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_MONEY_RE   = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY)\b", re.I)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_EMAIL_RE   = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE     = re.compile(r"https?://[^\s]+")


def _extract_entities(text: str) -> List[Dict]:
    """Rule-based NER for dates, money, percentages, emails, and URLs."""
    entities = []
    for pattern, etype in [
        (_DATE_RE,    "DATE"),
        (_MONEY_RE,   "MONEY"),
        (_PERCENT_RE, "PERCENT"),
        (_EMAIL_RE,   "EMAIL"),
        (_URL_RE,     "URL"),
    ]:
        for m in pattern.finditer(text):
            entities.append({
                "text":  m.group().strip(),
                "type":  etype,
                "start": m.start(),
                "end":   m.end(),
            })
    return entities


def _aggregate_entities(docs: List[str]) -> List[Dict]:
    """Count entity occurrences across all documents."""
    counts: Dict[str, Dict] = {}
    for doc in docs:
        for ent in _extract_entities(doc):
            key = f"{ent['type']}:{ent['text'].lower()}"
            if key not in counts:
                counts[key] = {"text": ent["text"], "type": ent["type"], "count": 0}
            counts[key]["count"] += 1
    return sorted(counts.values(), key=lambda x: -x["count"])


# ── Readability (Flesch Reading Ease) ─────────────────────────

def _readability(text: str) -> Dict[str, Any]:
    """
    Flesch Reading Ease score (English only).
    Score 0–100: higher = easier to read.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[a-zA-Z]+", text)
    syllables = sum(
        max(1, len(re.findall(r"[aeiouAEIOU]", w))) for w in words
    )
    n_sent = max(len(sentences), 1)
    n_word = max(len(words), 1)

    fre = 206.835 - 1.015 * (n_word / n_sent) - 84.6 * (syllables / n_word)
    fre = round(max(0.0, min(100.0, fre)), 1)

    if fre >= 80:
        level = "Very Easy"
    elif fre >= 60:
        level = "Standard"
    elif fre >= 40:
        level = "Difficult"
    else:
        level = "Very Difficult"

    return {
        "fleschScore": fre,
        "level": level,
        "sentenceCount": n_sent,
        "wordCount": n_word,
        "avgSentenceLength": round(n_word / n_sent, 1),
        "avgSyllablesPerWord": round(syllables / n_word, 2),
    }


# ── Cosine Similarity ─────────────────────────────────────────

def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    keys = set(v1) | set(v2)
    dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1   = math.sqrt(sum(x ** 2 for x in v1.values()))
    n2   = math.sqrt(sum(x ** 2 for x in v2.values()))
    return round(dot / (n1 * n2 + 1e-9), 4)


# ══════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════

class TextAnalysisRequest(BaseModel):
    data:           List[Dict[str, Any]]   # list of document records
    textCol:        str                    # column name for text content
    docIdCol:       Optional[str] = None   # column name for document ID
    timestampCol:   Optional[str] = None   # column name for timestamp
    labelCol:       Optional[str] = None   # column name for existing labels
    maxKeywords:    int = 30
    nTopics:        int = 4


class SentimentRequest(BaseModel):
    data:    List[Dict[str, Any]]
    textCol: str
    docIdCol: Optional[str] = None


class TopicRequest(BaseModel):
    data:    List[Dict[str, Any]]
    textCol: str
    nTopics: int = 4


class NERRequest(BaseModel):
    data:    List[Dict[str, Any]]
    textCol: str
    docIdCol: Optional[str] = None


class SimilarityRequest(BaseModel):
    data:      List[Dict[str, Any]]
    textCol:   str
    docIdCol:  Optional[str] = None
    topN:      int = 5          # return top-N similar pairs


class KeywordsRequest(BaseModel):
    data:        List[Dict[str, Any]]
    textCol:     str
    maxFeatures: int = 50


# ══════════════════════════════════════════════════════════════
# Full Pipeline Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text")
async def run_text_analysis(request: TextAnalysisRequest):
    """
    Full text analysis pipeline.
    Returns: tokens, TF-IDF keywords, per-doc sentiment,
             topic model, aggregated NER, readability, and corpus stats.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        if request.textCol not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.textCol}' not found. Available: {list(df.columns)}",
            )

        texts = df[request.textCol].fillna("").astype(str).tolist()
        doc_ids = (
            df[request.docIdCol].astype(str).tolist()
            if request.docIdCol and request.docIdCol in df.columns
            else [f"D{i+1:04d}" for i in range(len(texts))]
        )

        # ── 1. Tokenization stats ─────────────────────────────
        tokenized = [_tokenize(t) for t in texts]
        all_tokens = [tok for tokens in tokenized for tok in tokens]
        vocab = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_tokens = len(vocab)
        avg_doc_length = round(total_tokens / max(len(texts), 1), 1)

        # ── 2. TF-IDF Keywords ────────────────────────────────
        keywords = _compute_tfidf_corpus(texts, request.maxKeywords)

        # ── 3. Per-document Sentiment ─────────────────────────
        sentiment_results = []
        label_dist: Counter = Counter()
        for doc_id, text in zip(doc_ids, texts):
            s = _score_sentiment(text)
            sentiment_results.append({"docId": doc_id, "text": text[:120], **s})
            label_dist[s["label"]] += 1

        compound_scores = [r["compound"] for r in sentiment_results]
        avg_sentiment = round(float(np.mean(compound_scores)), 4) if compound_scores else 0.0

        # ── 4. Topic Modeling ─────────────────────────────────
        topics = _run_topic_model(texts)

        # ── 5. Named Entity Recognition ───────────────────────
        entities = _aggregate_entities(texts)

        # ── 6. Readability (per-doc, summarized) ─────────────
        readability_scores = [_readability(t) for t in texts]
        avg_flesch = round(
            float(np.mean([r["fleschScore"] for r in readability_scores])), 1
        )

        # ── 7. Language detection ─────────────────────────────
        lang_counts: Counter = Counter(_detect_language(t) for t in texts)

        # ── 8. Token frequency distribution ──────────────────
        token_freq = [{"word": w, "count": c, "rank": i + 1}
                      for i, (w, c) in enumerate(vocab.most_common(50))]

        # ── 9. Corpus stats ───────────────────────────────────
        stats = {
            "totalDocuments":  len(texts),
            "totalTokens":     total_tokens,
            "uniqueTokens":    unique_tokens,
            "avgDocLength":    avg_doc_length,
            "avgSentiment":    avg_sentiment,
            "sentimentDist":   dict(label_dist),
            "avgFleschScore":  avg_flesch,
            "dominantTopic":   topics[0]["label"] if topics else "—",
            "topEntity":       entities[0]["text"] if entities else "—",
            "languages":       dict(lang_counts),
        }

        result = {
            "keywords":          keywords,
            "sentimentResults":  sentiment_results,
            "topics":            topics,
            "entities":          entities[:20],
            "tokenFreq":         token_freq,
            "readability":       readability_scores,
            "stats":             stats,
        }

        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Sentiment Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text/sentiment")
async def run_sentiment(request: SentimentRequest):
    """
    Sentiment analysis only.
    Returns per-document scores and corpus-level distribution.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        texts = df[request.textCol].fillna("").astype(str).tolist()
        doc_ids = (
            df[request.docIdCol].astype(str).tolist()
            if request.docIdCol and request.docIdCol in df.columns
            else [f"D{i+1:04d}" for i in range(len(texts))]
        )

        results = []
        label_dist: Counter = Counter()
        for doc_id, text in zip(doc_ids, texts):
            s = _score_sentiment(text)
            results.append({"docId": doc_id, "text": text[:120], **s})
            label_dist[s["label"]] += 1

        compounds = [r["compound"] for r in results]
        avg = round(float(np.mean(compounds)), 4) if compounds else 0.0

        return _to_native({
            "results": {
                "documents":    results,
                "distribution": dict(label_dist),
                "avgCompound":  avg,
                "totalDocs":    len(results),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Topic Modeling Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text/topics")
async def run_topics(request: TopicRequest):
    """
    Topic modeling — seed-word based assignment.
    Replace _run_topic_model() with sklearn LDA or BERTopic for production.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        texts = df[request.textCol].fillna("").astype(str).tolist()

        topics = _run_topic_model(texts)

        return _to_native({
            "results": {
                "topics":     topics,
                "totalDocs":  len(texts),
                "nTopics":    len(topics),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# NER Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text/ner")
async def run_ner(request: NERRequest):
    """
    Named entity recognition — rule-based patterns for dates, money,
    percentages, emails, and URLs.
    For richer NER (PERSON, ORG, GPE) add spaCy: pip install spacy.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        texts = df[request.textCol].fillna("").astype(str).tolist()
        doc_ids = (
            df[request.docIdCol].astype(str).tolist()
            if request.docIdCol and request.docIdCol in df.columns
            else [f"D{i+1:04d}" for i in range(len(texts))]
        )

        # Per-document entities
        per_doc = []
        for doc_id, text in zip(doc_ids, texts):
            ents = _extract_entities(text)
            per_doc.append({"docId": doc_id, "entities": ents, "count": len(ents)})

        # Aggregated across corpus
        aggregated = _aggregate_entities(texts)

        # Type distribution
        type_dist: Counter = Counter(e["type"] for e in aggregated)

        return _to_native({
            "results": {
                "perDocument": per_doc,
                "aggregated":  aggregated[:30],
                "typeDist":    dict(type_dist),
                "totalEntities": sum(e["count"] for e in aggregated),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Document Similarity Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text/similarity")
async def run_similarity(request: SimilarityRequest):
    """
    Cosine similarity matrix across all documents using TF-IDF vectors.
    Returns the full N×N matrix (capped at 50 docs) and the top-N similar pairs.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        texts = df[request.textCol].fillna("").astype(str).tolist()
        doc_ids = (
            df[request.docIdCol].astype(str).tolist()
            if request.docIdCol and request.docIdCol in df.columns
            else [f"D{i+1:04d}" for i in range(len(texts))]
        )

        # Cap at 50 documents to keep matrix size manageable
        max_docs = 50
        if len(texts) > max_docs:
            texts   = texts[:max_docs]
            doc_ids = doc_ids[:max_docs]

        tokenized = [_tokenize(t) for t in texts]
        vectors   = _doc_tfidf_vectors(tokenized)
        n         = len(vectors)

        # Build similarity matrix
        matrix: List[List[float]] = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(1.0 if i == j else _cosine(vectors[i], vectors[j]))
            matrix.append(row)

        # Extract top-N similar pairs (excluding self-similarity)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "doc1":  doc_ids[i],
                    "doc2":  doc_ids[j],
                    "score": matrix[i][j],
                })
        pairs.sort(key=lambda x: -x["score"])
        top_pairs = pairs[:request.topN]

        return _to_native({
            "results": {
                "matrix":    matrix,
                "docIds":    doc_ids,
                "topPairs":  top_pairs,
                "totalDocs": n,
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Keywords Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/text/keywords")
async def run_keywords(request: KeywordsRequest):
    """
    TF-IDF keyword extraction.
    Returns per-document top keywords and corpus-level ranking.
    """
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        df = pd.DataFrame(rows)
        texts = df[request.textCol].fillna("").astype(str).tolist()

        # Corpus-level keywords
        corpus_keywords = _compute_tfidf_corpus(texts, request.maxFeatures)
        keyword_list = [
            {"word": w, "score": s, "rank": i + 1}
            for i, (w, s) in enumerate(corpus_keywords.items())
        ]

        # Per-document top-5 keywords
        tokenized = [_tokenize(t) for t in texts]
        vectors   = _doc_tfidf_vectors(tokenized)
        per_doc   = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            top5 = sorted(vec.items(), key=lambda x: -x[1])[:5]
            per_doc.append({
                "docIndex": i,
                "preview":  text[:80],
                "keywords": [{"word": w, "score": round(s, 4)} for w, s in top5],
            })

        return _to_native({
            "results": {
                "corpusKeywords": keyword_list,
                "perDocument":    per_doc,
                "totalDocs":      len(texts),
                "vocabSize":      len(corpus_keywords),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
