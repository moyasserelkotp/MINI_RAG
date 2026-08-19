import json
import os
import requests

API_URL = "http://localhost:8000/api/v1/nlp/answer/HealthPlus_Company_Guide"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../assets/benchmark_dataset.json")

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_rag_response(question):
    payload = {
        "text": question,
        "limit": 3,
        "use_hybrid": True,
        "use_cache": False
    }
    response = requests.post(API_URL, json=payload, headers={"accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    return data

def calculate_context_recall(keywords, full_answer, contexts_text):
    """
    Check recall against BOTH the generated answer AND the full context strings.
    This avoids false 0.0 recalls caused by 300-char truncation in the API response.
    """
    if not keywords:
        return 1.0  # N/A for negative test cases
        
    combined = (full_answer + " " + " ".join(contexts_text)).lower()
    found_count = sum(1 for kw in keywords if kw.lower() in combined)
    return round(found_count / len(keywords), 2)


def main():
    print("Loading benchmark dataset...")
    data = load_dataset()
    
    results = []

    for idx, item in enumerate(data):
        q = item["question"]
        print(f"[{idx+1}/{len(data)}] Querying API for: {q}")
        try:
            api_data = get_rag_response(q)
            
            ans = api_data.get("answer", "")
            sources = api_data.get("sources", [])
            contexts = [src.get("text", "") for src in sources]
            
            # Extract scores for logging
            scores = [src.get("score", 0.0) for src in sources]
            
            # Calculate automated Context Recall (checks full answer + full contexts)
            recall_score = calculate_context_recall(item.get("keywords", []), ans, contexts)
            
            is_negative = item.get("is_negative", False)
            anti_hallucination_pass = None
            if is_negative:
                anti_hallucination_pass = "cannot find the answer" in ans.lower() or "not explicitly stated" in ans.lower()
                
            results.append({
                "question": q,
                "expected_ground_truth": item["ground_truth"],
                "rag_generated_answer": ans,
                "retrieved_contexts": contexts,
                "context_scores": scores,
                "context_recall": recall_score,
                "is_negative": is_negative,
                "anti_hallucination_pass": anti_hallucination_pass
            })
            
            print(f"  -> Context Recall: {recall_score:.2f}")
            if is_negative:
                print(f"  -> Anti-Hallucination Pass: {anti_hallucination_pass}")
                
        except Exception as e:
            print(f"Error querying API for '{q}': {e}")
            continue

    # Write full report for ChatGPT evaluation of precision/relevancy
    output_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSuccess! Exported {len(results)} items to: {output_path}")
    print("You can now review the Context Recall scores above, and copy eval_report.json into ChatGPT for full evaluation.")

if __name__ == "__main__":
    main()
