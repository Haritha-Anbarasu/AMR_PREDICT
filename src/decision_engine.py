"""
Sections 9 & 20: Hybrid Decision Engine

Combines similarity-screening evidence with ML prediction evidence to
produce one of four final categories, rather than trusting either
source alone.
"""


def final_classification(similarity_result: str, ml_prediction: str, ml_probability: float) -> dict:
    """
    similarity_result: "Known ARG" | "Candidate ARG" | "Unknown"  (from similarity_screening.py)
    ml_prediction:      "ARG" | "Non-ARG"                          (from prediction.py)
    ml_probability:     float 0-1

    Returns the final category + confidence per Section 21 of the plan:
        Category 1: Confirmed/known ARG
        Category 2: Potential ARG
        Category 3: Unclassified candidate
        Category 4: Non-ARG-like sequence
    """
    if similarity_result == "Known ARG":
        category = "Confirmed ARG"
        confidence = "High"

    elif similarity_result == "Candidate ARG" and ml_prediction == "ARG" and ml_probability >= 0.70:
        category = "Potential ARG"
        confidence = "High" if ml_probability >= 0.90 else "Moderate"

    elif similarity_result == "Unknown" and ml_prediction == "ARG":
        if ml_probability >= 0.90:
            category = "Potential ARG"
            confidence = "High"
        elif ml_probability >= 0.70:
            category = "Unclassified candidate"
            confidence = "Moderate"
        else:
            category = "Unclassified candidate"
            confidence = "Low"

    else:
        category = "Non-ARG-like sequence"
        confidence = "High" if ml_probability < 0.30 else "Moderate"

    return {"final_category": category, "confidence": confidence}


if __name__ == "__main__":
    print(final_classification("Candidate ARG", "ARG", 0.93))
