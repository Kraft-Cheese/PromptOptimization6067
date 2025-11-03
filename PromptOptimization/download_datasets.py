from datasets import load_dataset
import json
import os

# Create data directory
os.makedirs("data", exist_ok=True)

def download_piqa(n=100):
    """PIQA: Physical commonsense reasoning (A/B choice)"""
    dataset = load_dataset("piqa", split="validation")
    
    data = []
    for i, ex in enumerate(dataset):
        if i >= n: break
        data.append({
            "goal": ex["goal"],
            "sol1": ex["sol1"],
            "sol2": ex["sol2"],
            "label": str(ex["label"])  # "0" or "1"
        })
    
    with open("data/piqa.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} PIQA examples")
    return len(data)

def download_hellaswag(n=100):
    """HellaSwag: Commonsense NLI (4-way choice)"""
    dataset = load_dataset("hellaswag", split="validation")
    
    data = []
    for i, ex in enumerate(dataset):
        if i >= n: break
        data.append({
            "context": ex["ctx"],
            "endings": ex["endings"],  # list of 4 endings
            "label": str(ex["label"])  # "0", "1", "2", or "3"
        })
    
    with open("data/hellaswag.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} HellaSwag examples")
    return len(data)

def download_boolq(n=100):
    """BoolQ: Yes/No question answering"""
    print("\nDownloading BoolQ...")
    dataset = load_dataset("google/boolq", split="validation")
    
    data = []
    for i, ex in enumerate(dataset):
        if i >= n: break
        data.append({
            "question": ex["question"],
            "passage": ex["passage"],
            "answer": ex["answer"]  # True or False
        })
    
    with open("data/boolq.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} BoolQ examples")
    return len(data)

def download_gsm8k(n=100):
    """GSM8K: Grade school math word problems"""
    dataset = load_dataset("gsm8k", "main", split="test")
    
    data = []
    for i, ex in enumerate(dataset):
        if i >= n: break
        
        # Extract numeric answer
        answer_text = ex["answer"]
        # GSM8K answers end with "#### <number>"
        if "####" in answer_text:
            answer = float(answer_text.split("####")[-1].strip().replace(",", ""))
        else:
            continue
        
        data.append({
            "question": ex["question"],
            "answer": answer,
            "solution": answer_text  # answer for evaluation + reference
        })
    
    with open("data/gsm8k.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} GSM8K examples")
    return len(data)

if __name__ == "__main__":
    print("Downloading Benchmark Datasets")
    total = 0
    total += download_piqa(100)
    total += download_hellaswag(100)
    total += download_boolq(100)
    total += download_gsm8k(100)