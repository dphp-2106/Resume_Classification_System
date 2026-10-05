"""
Clean Python import alias for 06_predict.py
Allows standard import syntax:
    from resume_classifier.predict import predict_resume, predict_batch
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_mod = importlib.import_module("06_predict")
predict_resume = _mod.predict_resume
predict_batch = _mod.predict_batch

if __name__ == '__main__':
    test_text = "Senior Python and AWS software engineer with Docker, Kubernetes, microservices, and SQL experience."
    res = predict_resume(test_text)
    print("Test Prediction:", res['predicted_category'])
    print("Confidence:", f"{res['confidence']*100:.1f}%")
    print("Method:", res['method'])
    print("Top 3:", res['top_5'][:3])
