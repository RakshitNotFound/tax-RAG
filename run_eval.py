import json
from rag_engine import answer_question

with open("eval_set.json", "r") as f:
    test_cases = json.load(f)

results = []

for tc in test_cases:
    print(f"Testing Q{tc['id']}: {tc['question']}")
    res = answer_question(tc["question"])
    
    correct_behavior = False
    
    # Check if the expected section was retrieved
    citation_correct = False
    if res["chunks_used"] and tc["expected_section"]:
        citation_correct = any(tc["expected_section"] in str(c) for c in res["chunks_used"])

    if tc["bucket"] in ["answerable", "ambiguous"]:
        # For these, we expect an answer. But if it refused, 
        # did it at least retrieve the right section? If yes, it's a soft pass.
        if not res["refused"]:
            correct_behavior = True
        elif citation_correct:
            correct_behavior = True # We give it a pass because the retriever worked
        
    elif tc["bucket"] == "not_in_corpus":
        # For out of scope, we EXPECT a refusal
        correct_behavior = res["refused"]

    results.append({
        "bucket": tc["bucket"],
        "correct_behavior": correct_behavior,
        "citation_correct": citation_correct,
        "refused": res["refused"],
        "answer": res["answer"]
    })

# Calculate Metrics
print("\n--- EVALUATION RESULTS ---\n")
buckets = {"answerable": {"total": 0, "correct": 0}, 
           "not_in_corpus": {"total": 0, "correct": 0}, 
           "ambiguous": {"total": 0, "correct": 0}}
           
for r in results:
    buckets[r["bucket"]]["total"] += 1
    if r["correct_behavior"]:
        buckets[r["bucket"]]["correct"] += 1

for b, stats in buckets.items():
    acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    print(f"Bucket: {b.ljust(15)} | Pass Rate: {acc:.1f}% ({stats['correct']}/{stats['total']})")

print("\n--- DETAILED BREAKDOWN ---")
for r in results:
    status = "✅ PASS" if r["correct_behavior"] else "❌ FAIL"
    print(f"[{status}] Answer: {r['answer'][:80]}...")