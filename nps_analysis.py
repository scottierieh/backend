
import sys
import json
import numpy as np

def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def get_nps_interpretation(score, promoters, passives, detractors, total):
    if total == 0:
        return "Not enough data to generate an interpretation."

    promoter_pct = (promoters / total) * 100
    detractor_pct = (detractors / total) * 100

    level = ""
    if score >= 70:
        level = "Excellent"
    elif score >= 50:
        level = "Good"
    elif score >= 30:
        level = "Fair"
    elif score >= 0:
        level = "Fair"
    else:
        level = "Poor"


    interpretation = (
        f"The Net Promoter Score (NPS) is calculated by subtracting the percentage of Detractors from the percentage of Promoters. "
        f"Out of {total} respondents, there were **{promoters} Promoters** ({promoter_pct:.1f}%), **{passives} Passives**, and **{detractors} Detractors** ({detractor_pct:.1f}%).\n\n"
        f"This results in an NPS of **{score:.1f}**, which is considered **{level}**. "
    )

    if level == "Excellent":
        interpretation += "Your customers are highly loyal and are actively promoting your brand, which is a strong indicator of future growth."
    elif level == "Good":
        interpretation += "You have a strong base of satisfied customers. Focusing on converting Passives can elevate your score further."
    elif level == "Fair":
        interpretation += "Your customer base is lukewarm. It's crucial to understand the reasons for dissatisfaction among Detractors to prevent negative word-of-mouth."
    else: # Needs Improvement
        interpretation += "Your company has more detractors than promoters, indicating significant issues with customer satisfaction that need immediate attention."

    return interpretation


def calculate_nps(scores):
    if not scores:
        return {}
        
    scores = np.array(scores)
    promoters = np.sum(scores >= 9)
    detractors = np.sum(scores <= 6)
    passives = len(scores) - promoters - detractors
    total = len(scores)

    promotersP = (promoters / total) * 100 if total > 0 else 0
    detractorsP = (detractors / total) * 100 if total > 0 else 0
    
    nps_score = promotersP - detractorsP

    return {
        "npsScore": nps_score,
        "promoters": int(promoters),
        "passives": int(passives),
        "detractors": int(detractors),
        "total": int(total),
        "interpretation": get_nps_interpretation(nps_score, promoters, passives, detractors, total)
    }

def main():
    try:
        payload = json.load(sys.stdin)
        scores = payload.get('scores')

        if scores is None:
            raise ValueError("Missing 'scores' data")

        results = calculate_nps(scores)
        
        response = { "results": results }
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
