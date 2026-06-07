"""
REDROB INTELLIGENT CANDIDATE DISCOVERY — India Runs Hackathon
Team: Poovarasu S | KIOT Tamil Nadu | SaforaX
Approach: Multi-signal semantic ranker with behavioral modifier
"""

import json
import csv
import math
from datetime import datetime, date
from pathlib import Path
import argparse

# ─── JD INTELLIGENCE ─────────────────────────────────────────────────────────
# Parsed from the actual job description — Senior AI Engineer, Founding Team

MUST_HAVE_SKILLS = {
    # embeddings & retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "vector search",
    "semantic search", "dense retrieval", "hybrid retrieval", "hybrid search",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "annoy", "vector database", "vector db",
    # ranking & IR
    "ranking", "information retrieval", "learning to rank", "ltr",
    "bm25", "ndcg", "mrr", "map", "recall@k",
    # LLM & ML
    "llm", "large language model", "fine-tuning", "fine tuning", "lora", "qlora",
    "peft", "rag", "retrieval augmented generation", "reranking", "re-ranking",
    "transformers", "bert", "nlp", "natural language processing",
    # production ML
    "mlops", "ml pipeline", "model serving", "model deployment",
    "a/b testing", "ab testing", "evaluation framework", "offline evaluation",
    "python", "pytorch", "tensorflow",
}

NICE_TO_HAVE_SKILLS = {
    "xgboost", "lightgbm", "learning to rank", "neural ranking",
    "distributed systems", "spark", "kafka", "airflow",
    "recommendation system", "recommender", "search engine",
    "open source", "github", "research", "paper",
}

# Disqualifying title patterns (from JD: "Marketing Manager" with AI keywords = trap)
DISQUALIFY_TITLES = {
    "marketing manager", "graphic designer", "content writer",
    "accountant", "civil engineer", "mechanical engineer",
    "customer support", "customer service", "sales executive",
    "operations manager", "hr manager", "business analyst",
    "project manager",
}

# Strong positive title signals
STRONG_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ml engineer", "senior ai engineer", "staff ml engineer",
    "data scientist", "applied scientist", "research engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "backend engineer",  # acceptable if ML skills strong
    "software engineer",  # acceptable if ML skills strong
    "junior ml engineer", "senior machine learning engineer",
}

# Good company types (product companies preferred over pure services)
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis",
}

# Preferred locations (from JD)
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "ncr", "gurgaon", "gurugram", "chennai", "india",
}

# ─── SCORING ENGINE ───────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> tuple[float, int, list]:
    """Score candidate skills against JD requirements. Returns (score, count, matched)"""
    skills_list = candidate.get("skills", [])
    if not skills_list:
        return 0.0, 0, []

    skill_score = 0.0
    matched = []
    ai_core_count = 0

    for skill in skills_list:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "beginner").lower()
        endorsements = min(skill.get("endorsements", 0), 50)  # cap at 50
        duration = skill.get("duration_months", 0)

        # Proficiency multiplier
        prof_mult = {"expert": 1.5, "advanced": 1.2, "intermediate": 1.0, "beginner": 0.6}.get(proficiency, 0.8)

        # Endorsement trust multiplier (catches keyword stuffers)
        # Low endorsements on "advanced" skill = suspicious
        endorse_trust = min(1.0, (endorsements + 1) / 15)

        # Duration signal
        dur_mult = min(1.3, 1.0 + duration / 60)

        # Check against must-have
        is_must = any(kw in name for kw in MUST_HAVE_SKILLS)
        is_nice = any(kw in name for kw in NICE_TO_HAVE_SKILLS)

        if is_must:
            ai_core_count += 1
            skill_weight = 3.0
            matched.append(name)
        elif is_nice:
            skill_weight = 1.5
        else:
            skill_weight = 0.2

        skill_score += skill_weight * prof_mult * endorse_trust * dur_mult

    # Normalize — max theoretical score per candidate
    normalized = min(1.0, skill_score / 35.0)
    return normalized, ai_core_count, matched[:10]


def score_career(candidate: dict) -> tuple[float, list]:
    """Score career history for relevance — semantic, not keyword matching"""
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    yoe = profile.get("years_of_experience", 0)

    if not history:
        return 0.0, []

    title_score = 0.0
    reasons = []

    # Current title scoring
    is_strong = any(t in current_title for t in STRONG_TITLES)
    is_disqualified = any(t in current_title for t in DISQUALIFY_TITLES)

    if is_strong:
        title_score += 0.4
        reasons.append(f"strong title: {profile.get('current_title', '')}")
    elif is_disqualified:
        title_score -= 0.3
        reasons.append(f"title mismatch: {profile.get('current_title', '')}")

    # YOE scoring (JD wants 5-9 years, sweet spot is 6-8)
    if 5 <= yoe <= 9:
        yoe_score = 0.3
        reasons.append(f"{yoe:.1f} yrs (ideal range)")
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.2
        reasons.append(f"{yoe:.1f} yrs (near range)")
    elif yoe < 4:
        yoe_score = 0.05
        reasons.append(f"{yoe:.1f} yrs (too junior)")
    else:
        yoe_score = 0.1
        reasons.append(f"{yoe:.1f} yrs (overqualified risk)")

    # Career history quality
    career_score = 0.0
    product_company_bonus = 0.0
    ml_production_signal = 0.0

    for job in history:
        comp = job.get("company", "").lower()
        role = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        dur = job.get("duration_months", 0)
        industry = job.get("industry", "").lower()

        # Services company penalty
        is_services = any(s in comp for s in SERVICES_COMPANIES)
        if is_services:
            career_score -= 0.05
        else:
            # Product company bonus
            if dur > 18:  # stayed 18+ months at product company
                product_company_bonus = min(0.15, product_company_bonus + 0.05)

        # ML production signals in description
        ml_terms = ["embedding", "retrieval", "ranking", "recommendation", "search",
                    "model", "pipeline", "inference", "deploy", "production", "scale"]
        ml_hits = sum(1 for t in ml_terms if t in desc)
        ml_production_signal += min(0.15, ml_hits * 0.02)

        # Job stability signal (JD says title-chasers are disqualified)
        if dur < 12 and not job.get("is_current"):
            career_score -= 0.02  # penalize short stints

    career_total = title_score + yoe_score + career_score + product_company_bonus + ml_production_signal
    return min(1.0, max(0.0, career_total)), reasons


def score_behavioral(candidate: dict) -> tuple[float, list]:
    """Score behavioral/activity signals — availability and engagement"""
    signals = candidate.get("redrob_signals", {})
    reasons = []

    # ── Availability signals ──────────────────────────────────────────────
    # Last active date
    last_active_str = signals.get("last_active_date", "")
    days_inactive = 999
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (date(2026, 6, 7) - last_active).days
        except:
            pass

    if days_inactive <= 14:
        active_score = 1.0
        reasons.append("active last 2 weeks")
    elif days_inactive <= 30:
        active_score = 0.85
        reasons.append("active last month")
    elif days_inactive <= 90:
        active_score = 0.6
        reasons.append("active last 3 months")
    elif days_inactive <= 180:
        active_score = 0.3
        reasons.append("inactive 3-6 months")
    else:
        active_score = 0.05
        reasons.append("inactive 6+ months — availability risk")

    # Open to work
    open_to_work = signals.get("open_to_work_flag", False)
    otw_score = 0.15 if open_to_work else 0.0
    if open_to_work:
        reasons.append("open to work")

    # Notice period (JD wants sub-30 days)
    notice = signals.get("notice_period_days", 90)
    if notice <= 0:
        notice_score = 0.15
        reasons.append("immediate joiner")
    elif notice <= 30:
        notice_score = 0.12
        reasons.append(f"{notice}d notice (ideal)")
    elif notice <= 60:
        notice_score = 0.06
        reasons.append(f"{notice}d notice")
    else:
        notice_score = 0.0
        reasons.append(f"{notice}d notice (long)")

    # ── Engagement signals ────────────────────────────────────────────────
    # Recruiter response rate (key signal from JD)
    response_rate = signals.get("recruiter_response_rate", 0)
    response_score = response_rate * 0.2  # max 0.2

    # Interview completion rate
    interview_completion = signals.get("interview_completion_rate", 0)
    interview_score = interview_completion * 0.1

    # Saved by recruiters (social proof)
    saved = min(signals.get("saved_by_recruiters_30d", 0), 20)
    saved_score = (saved / 20) * 0.08

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0) / 100
    completeness_score = completeness * 0.05

    # GitHub activity (JD explicitly values open source)
    github = signals.get("github_activity_score", 0) / 100
    github_score = github * 0.1

    # Location match
    location = candidate.get("profile", {}).get("location", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    location_score = 0.0
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        location_score = 0.08
        reasons.append("preferred location")
    elif willing_to_relocate:
        location_score = 0.04
        reasons.append("willing to relocate")

    total = (active_score * 0.35 + otw_score + notice_score + response_score +
             interview_score + saved_score + completeness_score + github_score + location_score)

    return min(1.0, total), reasons


def score_education(candidate: dict) -> float:
    """Score education — tier and relevance"""
    education = candidate.get("education", [])
    if not education:
        return 0.1

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "tier_3").lower()
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_score = {"tier_1": 0.4, "tier_2": 0.3, "tier_3": 0.2}.get(tier, 0.15)

        field_bonus = 0.0
        if any(f in field for f in ["computer science", "machine learning", "ai",
                                      "data science", "statistics", "mathematics"]):
            field_bonus = 0.15
        elif any(f in field for f in ["engineering", "information technology", "electronics"]):
            field_bonus = 0.08

        edu_score = tier_score + field_bonus
        best_score = max(best_score, edu_score)

    return min(1.0, best_score)


def rank_candidate(candidate: dict) -> tuple[float, str]:
    """
    Master scoring function.
    Weights tuned to match JD intent:
    - Skills (semantic, not keyword): 35%
    - Career history + title: 35%
    - Behavioral signals: 20%
    - Education: 10%
    """
    # Score each dimension
    skill_score, ai_core_count, matched_skills = score_skills(candidate)
    career_score, career_reasons = score_career(candidate)
    behavioral_score, behavioral_reasons = score_behavioral(candidate)
    edu_score = score_education(candidate)

    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    response_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0)

    # Hard disqualification — title is completely wrong AND few AI skills
    title_lower = current_title.lower()
    is_disqualified_title = any(t in title_lower for t in DISQUALIFY_TITLES)

    # Keyword stuffer trap — high skills count but wrong title, no career evidence
    if is_disqualified_title and ai_core_count >= 8 and career_score < 0.15:
        # This is the honeypot the JD warned about
        skill_score *= 0.3
        career_score *= 0.5

    # Availability multiplier — a perfect candidate who won't respond is useless
    availability_mult = 0.5 + (0.5 * min(1.0, response_rate + 0.3))

    # Weighted composite
    raw_score = (
        skill_score   * 0.35 +
        career_score  * 0.35 +
        behavioral_score * 0.20 +
        edu_score     * 0.10
    )

    # Apply availability multiplier
    final_score = raw_score * availability_mult

    # Build reasoning string
    skill_str = f"{ai_core_count} AI core skills"
    if matched_skills:
        skill_str += f" ({', '.join(matched_skills[:3])})"

    career_str = "; ".join(career_reasons[:2]) if career_reasons else current_title
    behavioral_str = f"response rate {response_rate:.2f}"
    if behavioral_reasons:
        behavioral_str += f"; {behavioral_reasons[0]}"

    reasoning = f"{current_title} | {yoe:.1f}yrs | {skill_str} | {behavioral_str}"

    return round(final_score, 4), reasoning


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(candidates_path: str, output_path: str, top_n: int = 100):
    print(f"Loading candidates from {candidates_path}...")

    results = []
    total = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                score, reasoning = rank_candidate(candidate)
                results.append({
                    "candidate_id": candidate["candidate_id"],
                    "score": score,
                    "reasoning": reasoning,
                })
                total += 1
                if total % 10000 == 0:
                    print(f"  Processed {total:,} candidates...")
            except Exception as e:
                continue

    print(f"Scored {total:,} candidates. Sorting...")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Write top N
    top_results = results[:top_n]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top_results, 1):
            writer.writerow([r["candidate_id"], i, r["score"], r["reasoning"]])

    print(f"\n✅ Top {top_n} candidates written to {output_path}")
    print("\nTop 10 preview:")
    for r in top_results[:10]:
        print(f"  #{top_results.index(r)+1} {r['candidate_id']} | {r['score']:.4f} | {r['reasoning'][:80]}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Intelligent Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of top candidates to output")
    args = parser.parse_args()

    main(args.candidates, args.out, args.top)
