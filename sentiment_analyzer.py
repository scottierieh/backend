
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import string
import io
import base64
import warnings
import platform
import matplotlib.font_manager as fm
from pathlib import Path

# NLTK for English sentiment
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Konlpy for Korean sentiment
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.bool_): return bool(obj)
    return obj

def get_font_path():
    """Find a suitable font on the system, prioritizing Korean fonts."""
    system = platform.system()
    
    font_paths_to_check = {
        'Windows': ['C:/Windows/Fonts/malgun.ttf'],
        'Darwin': ['/System/Library/Fonts/Apple SD Gothic Neo.ttc'],
        'Linux': ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    }
    
    if system in font_paths_to_check:
        for path in font_paths_to_check[system]:
            if Path(path).exists():
                return path

    # Fallback to matplotlib font manager
    try:
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        korean_fonts = [f for f in font_list if any(keyword in f.lower() for keyword in ['nanum', 'malgun', 'gothic'])]
        if korean_fonts:
            return korean_fonts[0]
    except Exception:
        pass
        
    return None


class SentimentAnalyzer:
    def __init__(self):
        self.font_path = get_font_path()
        if self.font_path:
            plt.rcParams['font.family'] = fm.FontProperties(fname=self.font_path).get_name()
        else:
            warnings.warn("No suitable Korean font found. Non-English characters may not render correctly.")
        
        plt.rcParams['axes.unicode_minus'] = False
        
        if NLTK_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        
        if KONLPY_AVAILABLE:
            self.okt = Okt()
        
        # Simple Korean sentiment dictionary
        self.korean_sentiment_dict = {
            '좋다': 1, '최고': 2, '만족': 1.5, '추천': 1.5, '훌륭하다': 2, '친절하다': 1,
            '나쁘지 않다': 0.5, '괜찮다': 0.5,
            '별로': -1, '실망': -1.5, '최악': -2, '불편하다': -1, '아쉽다': -0.5, '늦다': -1,
        }

    def analyze_english(self, text):
        scores = self.vader.polarity_scores(text)
        return scores['compound']

    def analyze_korean(self, text):
        tokens = self.okt.morphs(text, stem=True)
        score = 0
        for token in tokens:
            if token in self.korean_sentiment_dict:
                score += self.korean_sentiment_dict[token]
        return score / (len(tokens)**0.5) if tokens else 0 # Normalize by root of token count

    def analyze_comprehensive(self, text):
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in text)
        
        if is_korean and KONLPY_AVAILABLE:
            score = self.analyze_korean(text)
            if score > 0.3: sentiment = 'positive'
            elif score < -0.3: sentiment = 'negative'
            else: sentiment = 'neutral'
            confidence = min(1.0, abs(score))
        elif not is_korean and NLTK_AVAILABLE:
            score = self.analyze_english(text)
            if score >= 0.05: sentiment = 'positive'
            elif score <= -0.05: sentiment = 'negative'
            else: sentiment = 'neutral'
            confidence = abs(score)
        else:
            return {"error": "Required library (NLTK or Konlpy) not available."}
            
        return {
            'text': text,
            'consensus': {
                'sentiment': sentiment,
                'confidence': confidence,
                'score': score
            },
        }

    def analyze_batch(self, texts, show_progress=False):
        return [self.analyze_comprehensive(text) for text in texts]

    def create_sentiment_report(self, results):
        if not results:
            return None

        df = pd.DataFrame([r['consensus'] for r in results if 'consensus' in r])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Overall Sentiment Distribution
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        ax1 = axes[0, 0]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[s] for s in sentiment_counts.index])
        ax1.set_title('Overall Sentiment Distribution')
        ax1.set_ylabel('Count')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom')

        # 2. Confidence Score Distribution
        ax2 = axes[0, 1]
        ax2.hist(df['confidence'], bins=10, color='skyblue', edgecolor='black')
        ax2.axvline(df['confidence'].mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {df['confidence'].mean():.3f}")
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Sentiment Strength Distribution
        ax3 = axes[1, 0]
        ax3.hist(df['score'], bins=15, color='purple', alpha=0.7)
        ax3.set_title('Sentiment Strength Distribution')
        ax3.set_xlabel('Sentiment Score (-1 to 1)')
        ax3.set_ylabel('Frequency')

        # 4. Sentiment vs Confidence
        ax4 = axes[1, 1]
        for sentiment_type, color in colors.items():
            subset = df[df['sentiment'] == sentiment_type]
            ax4.scatter(subset['sentiment'], subset['confidence'], alpha=0.6, label=sentiment_type, color=color)
        ax4.set_title('Sentiment vs Confidence')
        ax4.set_xlabel('Sentiment')
        ax4.set_ylabel('Confidence Score')
        ax4.legend()


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def export_results(self, results, format='dict'):
        df = pd.DataFrame([
            {
                'text': r.get('text', ''),
                'sentiment': r.get('consensus', {}).get('sentiment', 'N/A'),
                'confidence': r.get('consensus', {}).get('confidence', 0.0),
                'score': r.get('consensus', {}).get('score', 0.0)
            } for r in results
        ])
        if format == 'dict':
            return df.to_dict('records')
        elif format == 'csv':
            return df.to_csv(index=False)
        return None


def main():
    try:
        payload = json.load(sys.stdin)
        texts = payload.get('texts', [])

        if not texts:
            raise ValueError("Input 'texts' array is empty.")
            
        analyzer = SentimentAnalyzer()
        
        results = analyzer.analyze_batch(texts)
        report_plot = analyzer.create_sentiment_report(results)
        
        response = {
            "results": analyzer.export_results(results, format='dict'),
            "plot": f"data:image/png;base64,{report_plot}"
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
