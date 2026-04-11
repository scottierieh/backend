from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class SentimentRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nTexts: int = 300
    seed: Optional[int] = None
    # Column mapping
    colText: Optional[str] = None
    colDate: Optional[str] = None
    colCampaign: Optional[str] = None
    colChannel: Optional[str] = None
    # Config
    rollingWindow: int = 7
    topKeywords: int = 15


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception: return default


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

CAMPAIGNS = ['Summer Sale 2025', 'Brand Awareness Q1', 'Product Launch X', 'Holiday Promo', 'Retargeting Wave']
CHANNELS = ['Instagram', 'Facebook', 'Google Ads', 'TikTok', 'Email', 'YouTube']
AD_GROUPS = ['Video_A', 'Carousel_B', 'Static_C', 'Story_D', 'Search_E']

POSITIVE_TEXTS = [
    "Love this product! Best purchase I've made this year.",
    "Amazing ad, really caught my attention. Already ordered!",
    "Great quality and fast shipping. Highly recommend.",
    "This brand never disappoints. Customer for life!",
    "Perfect timing with this sale. Got exactly what I needed.",
    "The ad was so creative, I had to share it with friends.",
    "Exceeded my expectations. Will definitely buy again.",
    "Best customer service experience I've ever had.",
    "This product changed my daily routine for the better.",
    "Incredible value for the price. Five stars!",
    "So glad I clicked on that ad. Worth every penny.",
    "The quality is outstanding. Premium feel at affordable price.",
    "Already recommended to all my friends and family.",
    "Finally a brand that delivers on its promises!",
    "Smooth checkout, fast delivery, beautiful packaging.",
]

NEUTRAL_TEXTS = [
    "Saw the ad. Product looks okay, might check it out later.",
    "Decent product, nothing special but does the job.",
    "Average experience. Not bad, not great either.",
    "Got the product. It's fine for the price point.",
    "The ad was interesting but I'm not sure I need this.",
    "Standard quality. Meets basic expectations.",
    "Packaging was normal. Product as described.",
    "It works as advertised. No complaints.",
    "Received my order on time. Product is adequate.",
    "Not bad for what I paid. Could be better though.",
]

NEGATIVE_TEXTS = [
    "Terrible quality. Broke after one week of use.",
    "Misleading ad! Product looks nothing like the pictures.",
    "Worst customer service ever. No response for days.",
    "Complete waste of money. Do not buy this.",
    "Shipping took forever and the product arrived damaged.",
    "False advertising. Very disappointed with this brand.",
    "Regret this purchase. Returning immediately.",
    "The ad was annoying and repetitive. Stop showing it!",
    "Poor quality materials. Feels cheap and fragile.",
    "Overpriced for what you get. Not worth it at all.",
    "Still waiting for my refund after 3 weeks. Unacceptable.",
    "Product stopped working after a few uses. Junk.",
]


def generate_texts(n: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp('2025-01-01')

    for i in range(n):
        # Sentiment distribution: 40% pos, 30% neu, 30% neg
        roll = rng.random()
        if roll < 0.40:
            text = rng.choice(POSITIVE_TEXTS)
            # Add slight variation
            suffix = rng.choice(['', ' 😊', ' ❤️', ' 👍', ' Absolutely love it!', ''])
        elif roll < 0.70:
            text = rng.choice(NEUTRAL_TEXTS)
            suffix = rng.choice(['', ' 🤷', ' It is what it is.', ''])
        else:
            text = rng.choice(NEGATIVE_TEXTS)
            suffix = rng.choice(['', ' 😡', ' 👎', ' Never again.', ' So frustrated.', ''])

        text = text + suffix
        campaign = rng.choice(CAMPAIGNS)
        channel = rng.choice(CHANNELS)
        ad_group = rng.choice(AD_GROUPS)
        date = base_date + pd.Timedelta(days=int(rng.integers(0, 120)))

        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'text': text,
            'campaign': campaign,
            'channel': channel,
            'ad_group': ad_group,
            'platform_engagement': int(rng.integers(0, 500)),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Sentiment Engine
# ══════════════════════════════════════════════════════════════

def analyze_sentiment_vader(texts: List[str]) -> List[Dict[str, Any]]:
    """VADER sentiment analysis."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        results.append({
            'compound': safe_float(compound),
            'pos': safe_float(scores['pos']),
            'neu': safe_float(scores['neu']),
            'neg': safe_float(scores['neg']),
            'label': label,
        })
    return results


def analyze_sentiment_textblob(texts: List[str]) -> List[Dict[str, Any]]:
    """TextBlob sentiment analysis."""
    from textblob import TextBlob
    results = []
    for text in texts:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        results.append({
            'polarity': safe_float(polarity),
            'subjectivity': safe_float(subjectivity),
            'label': label,
        })
    return results


def extract_keywords(texts: List[str], top_n: int = 15) -> List[Dict[str, Any]]:
    """TF-IDF-like keyword extraction using term frequency."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
        )
        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.mean(axis=0).A1

        top_indices = scores.argsort()[-top_n:][::-1]
        return [{'keyword': str(feature_names[i]), 'score': safe_float(scores[i])}
                for i in top_indices if scores[i] > 0]
    except Exception:
        # Fallback: simple word frequency
        words = []
        for t in texts:
            words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()))
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'was', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'one', 'our', 'out', 'has', 'have', 'been', 'from', 'its', 'they', 'were', 'will', 'would', 'there', 'their', 'what', 'about', 'which', 'when', 'make', 'like', 'just', 'over', 'such', 'very', 'after'}
        words = [w for w in words if w not in stopwords]
        freq = Counter(words).most_common(top_n)
        total = len(words) or 1
        return [{'keyword': w, 'score': safe_float(c / total)} for w, c in freq]


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/sentiment-analysis")
async def sentiment_analysis(request: SentimentRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_texts(request.nTexts, request.seed)
            col_text = 'text'
            col_date = 'date'
            col_campaign = 'campaign'
            col_channel = 'channel'
        else:
            df = pd.DataFrame(request.data)
            col_text = request.colText or next((c for c in df.columns if 'text' in c.lower() or 'comment' in c.lower() or 'review' in c.lower() or 'message' in c.lower() or 'content' in c.lower()), None)
            col_date = request.colDate or next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
            col_campaign = request.colCampaign or next((c for c in df.columns if 'campaign' in c.lower() or 'camp' in c.lower()), None)
            col_channel = request.colChannel or next((c for c in df.columns if 'channel' in c.lower() or 'platform' in c.lower() or 'source' in c.lower()), None)

            if not col_text:
                raise HTTPException(status_code=400, detail="Cannot find text column. Please map it manually.")

        df = df.dropna(subset=[col_text])
        texts = df[col_text].astype(str).tolist()
        n = len(texts)
        if n < 5:
            raise HTTPException(status_code=400, detail=f"Need >=5 texts. Got {n}.")

        # ── 2. Sentiment Analysis ──
        vader_results = analyze_sentiment_vader(texts)
        textblob_results = analyze_sentiment_textblob(texts)

        # Ensemble: average compound scores
        sentiments = []
        for i in range(n):
            v = vader_results[i]
            t = textblob_results[i]
            # Ensemble score: weighted average (VADER 60%, TextBlob 40%)
            ensemble_score = v['compound'] * 0.6 + t['polarity'] * 0.4
            if ensemble_score >= 0.05:
                ensemble_label = 'Positive'
            elif ensemble_score <= -0.05:
                ensemble_label = 'Negative'
            else:
                ensemble_label = 'Neutral'

            sentiments.append({
                'vader_compound': v['compound'],
                'vader_label': v['label'],
                'textblob_polarity': t['polarity'],
                'textblob_subjectivity': t['subjectivity'],
                'textblob_label': t['label'],
                'ensemble_score': safe_float(ensemble_score),
                'ensemble_label': ensemble_label,
            })

        df['sentiment_score'] = [s['ensemble_score'] for s in sentiments]
        df['sentiment_label'] = [s['ensemble_label'] for s in sentiments]

        # ── 3. Overall Distribution ──
        label_counts = df['sentiment_label'].value_counts().to_dict()
        total = n
        distribution = {
            'Positive': {'count': int(label_counts.get('Positive', 0)), 'pct': safe_float(label_counts.get('Positive', 0) / total * 100)},
            'Neutral': {'count': int(label_counts.get('Neutral', 0)), 'pct': safe_float(label_counts.get('Neutral', 0) / total * 100)},
            'Negative': {'count': int(label_counts.get('Negative', 0)), 'pct': safe_float(label_counts.get('Negative', 0) / total * 100)},
        }

        # ── 4. Keywords by Sentiment ──
        pos_texts = [texts[i] for i in range(n) if sentiments[i]['ensemble_label'] == 'Positive']
        neg_texts = [texts[i] for i in range(n) if sentiments[i]['ensemble_label'] == 'Negative']

        pos_keywords = extract_keywords(pos_texts, request.topKeywords)
        neg_keywords = extract_keywords(neg_texts, request.topKeywords)

        # ── 5. Campaign Breakdown ──
        campaign_chart = []
        if col_campaign and col_campaign in df.columns:
            for camp in df[col_campaign].unique():
                mask = df[col_campaign] == camp
                camp_df = df[mask]
                camp_counts = camp_df['sentiment_label'].value_counts()
                camp_total = len(camp_df)
                campaign_chart.append({
                    'campaign': str(camp),
                    'positive': safe_float(camp_counts.get('Positive', 0) / camp_total * 100),
                    'neutral': safe_float(camp_counts.get('Neutral', 0) / camp_total * 100),
                    'negative': safe_float(camp_counts.get('Negative', 0) / camp_total * 100),
                    'avg_score': safe_float(camp_df['sentiment_score'].mean()),
                    'count': int(camp_total),
                })

        # ── 6. Channel Breakdown ──
        channel_chart = []
        if col_channel and col_channel in df.columns:
            for ch in df[col_channel].unique():
                mask = df[col_channel] == ch
                ch_df = df[mask]
                ch_counts = ch_df['sentiment_label'].value_counts()
                ch_total = len(ch_df)
                channel_chart.append({
                    'channel': str(ch),
                    'positive': safe_float(ch_counts.get('Positive', 0) / ch_total * 100),
                    'neutral': safe_float(ch_counts.get('Neutral', 0) / ch_total * 100),
                    'negative': safe_float(ch_counts.get('Negative', 0) / ch_total * 100),
                    'avg_score': safe_float(ch_df['sentiment_score'].mean()),
                    'count': int(ch_total),
                })

        # ── 7. Time Series ──
        time_chart = []
        if col_date and col_date in df.columns:
            try:
                df['_date'] = pd.to_datetime(df[col_date])
                daily = df.groupby('_date').agg(
                    avg_score=('sentiment_score', 'mean'),
                    count=('sentiment_score', 'count'),
                    pos_pct=('sentiment_label', lambda x: (x == 'Positive').mean() * 100),
                    neg_pct=('sentiment_label', lambda x: (x == 'Negative').mean() * 100),
                ).sort_index()

                if len(daily) > 1:
                    w = min(request.rollingWindow, len(daily))
                    daily['rolling_score'] = daily['avg_score'].rolling(w, min_periods=1).mean()

                    for date, row in daily.iterrows():
                        time_chart.append({
                            'date': str(date.strftime('%Y-%m-%d')),
                            'avg_score': safe_float(row['avg_score']),
                            'rolling_score': safe_float(row.get('rolling_score', row['avg_score'])),
                            'count': int(row['count']),
                            'pos_pct': safe_float(row['pos_pct']),
                            'neg_pct': safe_float(row['neg_pct']),
                        })
            except Exception:
                pass

        # ── 8. Score Distribution Histogram ──
        scores = df['sentiment_score'].values
        bins = np.linspace(-1, 1, 30)
        score_hist = []
        for j in range(len(bins) - 1):
            count = int(((scores >= bins[j]) & (scores < bins[j + 1])).sum())
            mid = (bins[j] + bins[j + 1]) / 2
            score_hist.append({
                'range': f'{mid:.2f}',
                'count': count,
                'sentiment': 'Positive' if mid > 0.05 else ('Negative' if mid < -0.05 else 'Neutral'),
            })

        # ── 9. Top Texts ──
        df_sorted = df.sort_values('sentiment_score')
        top_positive = []
        for _, row in df_sorted.tail(5).iterrows():
            entry = {'text': str(row[col_text])[:200], 'score': safe_float(row['sentiment_score'])}
            if col_campaign and col_campaign in df.columns:
                entry['campaign'] = str(row.get(col_campaign, ''))
            top_positive.append(entry)
        top_positive.reverse()

        top_negative = []
        for _, row in df_sorted.head(5).iterrows():
            entry = {'text': str(row[col_text])[:200], 'score': safe_float(row['sentiment_score'])}
            if col_campaign and col_campaign in df.columns:
                entry['campaign'] = str(row.get(col_campaign, ''))
            top_negative.append(entry)

        # ── 10. Model Agreement ──
        agree = sum(1 for s in sentiments if s['vader_label'] == s['textblob_label'])
        agreement_pct = safe_float(agree / n * 100)

        # ── Response ──
        results = {
            'n_texts': n,
            'columns_used': {
                'text': col_text,
                'date': col_date,
                'campaign': col_campaign,
                'channel': col_channel,
            },
            'summary': {
                'avg_score': safe_float(scores.mean()),
                'median_score': safe_float(np.median(scores)),
                'std_score': safe_float(scores.std()),
                'model_agreement_pct': agreement_pct,
            },
            'distribution': distribution,
            'charts': {
                'score_histogram': score_hist,
                'campaign': campaign_chart,
                'channel': channel_chart,
                'time_series': time_chart,
                'pos_keywords': pos_keywords,
                'neg_keywords': neg_keywords,
            },
            'top_positive': top_positive,
            'top_negative': top_negative,
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
