# Redrob Intelligent Candidate Discovery
## India Runs Hackathon — Track 1: Data & AI Challenge

**Team:** Poovarasu S | KIOT Tamil Nadu | Founder, SaforaX  
**Approach:** Multi-signal semantic ranker with behavioral availability modifier

---

## How to run

```bash
pip install -r requirements.txt
python rank.py --candidates candidates.jsonl --out submission.csv --top 100
```

Runs in ~60 seconds on CPU. No GPU required. No API calls.

---

## Approach

### The core insight from the JD
The job description explicitly warned: *"The right answer is NOT to find candidates whose skills section contains the most AI keywords. That's a trap."*

We designed our ranker around this. Three key design decisions:

**1. Title-career coherence check**  
A candidate with "Marketing Manager" as title but 9 AI skills listed is a keyword stuffer. Our ranker applies a coherence penalty when high skill counts contradict career history.

**2. Behavioral availability multiplier**  
A perfect-on-paper candidate with 0% recruiter response rate and 6 months inactive is not actually hireable. We apply a multiplicative availability modifier so behavioral signals gate, not just add to, the final score.

**3. Endorsement trust scoring**  
Skills with zero endorsements despite claimed "advanced" proficiency are discounted. Real skills get real endorsements.

---

## Scoring Architecture

| Dimension | Weight | Key signals |
|-----------|--------|-------------|
| Skills (semantic) | 35% | Must-have AI/ML skills, proficiency, endorsements, duration |
| Career history | 35% | Title match, YOE 5-9 ideal, product vs services company, ML production evidence in descriptions |
| Behavioral signals | 20% | Last active date, response rate, notice period, open to work, GitHub activity |
| Education | 10% | Institution tier, CS/ML field relevance |

**Availability multiplier:** `final_score = raw_score × (0.5 + 0.5 × response_rate)`

---

## Must-have skills (from JD parsing)

- Embeddings-based retrieval: sentence-transformers, BGE, E5, OpenAI embeddings
- Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch
- Ranking evaluation: NDCG, MRR, MAP, A/B testing
- LLM stack: fine-tuning, LoRA, QLoRA, PEFT, RAG
- Strong Python, PyTorch, production ML

---

## Files

```
rank.py                    # Main ranking script
submission.csv             # Top 100 ranked candidates
submission_metadata.yaml   # Submission metadata
validate_submission.py     # Official validation script
requirements.txt           # Dependencies
README.md                  # This file
```

---

## Reproduce command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~60 seconds | CPU only | No network calls | Python 3.10+
