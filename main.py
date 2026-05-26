"""
SkillSync AI Matching Engine — Weighted Reciprocal Jaccard Algorithm
=====================================================================

Step 1 — Two directional Jaccard scores:
    ScoreA→B = |Teach_A ∩ Learn_B| / |Teach_A ∪ Learn_B|
    ScoreB→A = |Teach_B ∩ Learn_A| / |Teach_B ∪ Learn_A|

Step 2 — Weighted reciprocal formula:
    Total = (ScoreA→B × W1) + (ScoreB→A × W2)
    W1 = 0.5, W2 = 0.5  (equal weight, balanced exchange)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="SkillSync AI Matching Engine",
    description="Weighted Reciprocal Jaccard matching for peer learning",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

W1 = 0.5   # weight for A→B (current user teaches candidate)
W2 = 0.5   # weight for B→A (candidate teaches current user)


class User(BaseModel):
    user_id: int
    skills_teach: List[str] = []
    skills_learn: List[str] = []


class Candidate(BaseModel):
    user_id: int
    name: str
    bio: str = ""
    avatar_color: str = "blue"
    skills_teach: List[str] = []
    skills_learn: List[str] = []


class MatchRequest(BaseModel):
    user: User
    candidates: List[Candidate]
    limit: int = 10


class MatchedUser(BaseModel):
    user_id: int
    name: str
    bio: str
    avatar_color: str
    skills_teach: List[str]
    skills_learn: List[str]
    match_percent: int
    can_teach_me: List[str]        # candidate teaches → user learns
    can_learn_from_me: List[str]   # user teaches → candidate learns
    is_top_match: bool
    score_a_to_b: float            # ScoreA→B (user teaches candidate)
    score_b_to_a: float            # ScoreB→A (candidate teaches user)


class MatchResponse(BaseModel):
    matches: List[MatchedUser]
    algorithm: str = "weighted_reciprocal_jaccard_v2"
    weights: dict = {"W1": W1, "W2": W2}


def _norm(skills: List[str]) -> set:
    return {s.strip().lower() for s in skills if s.strip()}


def _restore(matched_set: set, source_list: List[str]) -> List[str]:
    return [s for s in source_list if s.strip().lower() in matched_set]


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_score(user: User, cand: Candidate) -> dict:
    """
    ScoreA→B = jaccard(Teach_A, Learn_B)   — user can teach what candidate wants
    ScoreB→A = jaccard(Teach_B, Learn_A)   — candidate can teach what user wants
    Total    = ScoreA→B * W1 + ScoreB→A * W2
    """
    teach_a = _norm(user.skills_teach)
    learn_a = _norm(user.skills_learn)
    teach_b = _norm(cand.skills_teach)
    learn_b = _norm(cand.skills_learn)

    score_a_to_b = jaccard(teach_a, learn_b)   # A teaches → B learns
    score_b_to_a = jaccard(teach_b, learn_a)   # B teaches → A learns

    total = score_a_to_b * W1 + score_b_to_a * W2
    percent = int(round(total * 100))

    fwd_set = teach_b & learn_a  # candidate teaches, user learns
    bwd_set = teach_a & learn_b  # user teaches, candidate learns

    return {
        "score_a_to_b": round(score_a_to_b, 4),
        "score_b_to_a": round(score_b_to_a, 4),
        "percent": percent,
        "can_teach_me": _restore(fwd_set, cand.skills_teach),
        "can_learn_from_me": _restore(bwd_set, user.skills_teach),
    }


@app.get("/")
def root():
    return {
        "service": "SkillSync AI Matching Engine",
        "algorithm": "weighted_reciprocal_jaccard_v2",
        "formula": "Total = ScoreA→B * 0.5 + ScoreB→A * 0.5",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    scored = []
    for cand in req.candidates:
        s = compute_score(req.user, cand)
        if not s["can_teach_me"] and not s["can_learn_from_me"]:
            continue
        scored.append(MatchedUser(
            user_id=cand.user_id,
            name=cand.name,
            bio=cand.bio,
            avatar_color=cand.avatar_color,
            skills_teach=cand.skills_teach,
            skills_learn=cand.skills_learn,
            match_percent=s["percent"],
            can_teach_me=s["can_teach_me"],
            can_learn_from_me=s["can_learn_from_me"],
            is_top_match=(s["percent"] >= 70),
            score_a_to_b=s["score_a_to_b"],
            score_b_to_a=s["score_b_to_a"],
        ))

    scored.sort(key=lambda m: m.match_percent, reverse=True)
    return MatchResponse(matches=scored[: req.limit])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
