"""Simulated systems for the categorical arms (dry runs and tests; no network). The sender fabricates the raw response
each route returns (an OpenRouter chat completion carrying the JSON answer; Jev's `answers` object with probabilities
and confidence), so requests, the raw store and the parsers run exactly as in a live run. Each system has a fixed
accuracy; a few replies are malformed on purpose (not JSON, probabilities off the sum rule, an unknown label).
Deterministic: every draw is seeded by the request hash."""
from __future__ import annotations

import json

import numpy as np

from .categorical import ArmSpec, Item
from .clients import MODELS, _sha

SKILL = {"jev": 0.72, "gpt": 0.86, "claude": 0.84, "gemini": 0.80, "muse": 0.74, "glm": 0.82, "gpt_free": 0.76,
         "claude_free": 0.78, "gemini_free": 0.68, "medgemma": 0.66, "gemma": 0.60}
LATENCY = {"jev": 3.5, "gpt": 9.0, "claude": 6.0, "gemini": 8.0, "muse": 12.0, "glm": 40.0, "gpt_free": 5.0,
           "claude_free": 5.0, "gemini_free": 2.0, "medgemma": 3.4, "gemma": 5.8}
TOKENS_OUT = {"gpt": 180, "claude": 60, "gemini": 520, "muse": 900, "glm": 3000, "gpt_free": 150, "claude_free": 65,
              "gemini_free": 15, "medgemma": 30, "gemma": 30}
REASONING = {"gpt": 0.0, "claude": 0.6, "gemini": 0.75, "muse": 0.95, "glm": 0.95, "gpt_free": 0.4, "claude_free": 0.0,
             "gemini_free": 0.0}         # share of output tokens that are reasoning; the Hugging Face router reports none


class SimulatedSender:
    def __init__(self, arm: ArmSpec, items: list[Item], skill: dict | None = None, malformed: float = 0.03):
        self.arm, self.skill, self.malformed = arm, {**SKILL, **(skill or {})}, malformed
        self.items = {(it.item_id, it.variant): it for it in items}

    def __call__(self, system: str, req: dict) -> dict:
        it = self.items[(req["_item"], req.get("_variant", "main"))]
        rng = np.random.default_rng(int(_sha(req)[:12], 16))
        fam = system.split("@")[0]
        k, truth = len(it.presented), it.presented[it.canonical.index(it.truth)]
        right = rng.random() < self.skill.get(fam, 0.7)
        pick = truth if right else self._wrong(it, truth, rng)
        conc = 6.0 if right else 2.5
        p = rng.dirichlet(np.full(k, 0.4))
        p = 0.25 * p
        p[it.presented.index(pick)] += 0.75 * (1 - np.exp(-conc * rng.random()))
        p = p / p.sum()
        lat = float(LATENCY.get(fam, 5.0) * rng.lognormal(0, 0.3))
        if system == "jev":
            return {**self._jev(it, pick, p, req), "_sim_latency_s": lat}
        return {**self._chat(fam, it, pick, p, rng, req), "_sim_latency_s": lat}

    def _wrong(self, it: Item, truth: str, rng) -> str:
        if self.arm.kind == "score":     # a near miss is the common error on an ordered scale
            t = it.presented.index(truth)
            steps = [d for d in (-2, -1, -1, 1, 1, 2) if 0 <= t + d < len(it.presented)]
            return it.presented[t + int(rng.choice(steps))]
        other = [l for l in it.presented if l != truth]
        opp = {"A": "B", "B": "A"}.get(it.to_canonical(truth))   # arm 2: the opposite action is the common error
        w = np.array([4.0 if it.to_canonical(l) == opp else 1.0 for l in other])
        return other[int(rng.choice(len(other), p=w / w.sum()))]

    def _jev(self, it: Item, pick: str, p: np.ndarray, req: dict) -> dict:
        p = np.round(p, 2)
        p[int(np.argmax(p))] += round(1 - p.sum(), 2)
        conf = float(np.round(max(0.0, (p.max() - 1 / len(p)) / (1 - 1 / len(p))), 2))
        if self.arm.kind == "choice":
            ans = {"type": "choice", "choice": it.presented[int(np.argmax(p))],
                   "probabilities": {l: float(x) for l, x in zip(it.presented, p)}, "confidence": conf}
        else:
            ans = {"type": "score", "score": float(np.round((np.arange(len(p)) * p).sum(), 2)), "confidence": conf,
                   "legend": {str(i): c for i, c in enumerate(req["questions"][self.arm.jev_question_id]["criteria"])},
                   "probabilities": {str(i): float(x) for i, x in enumerate(p)}}
        tin = max(1, len(json.dumps(req["state"]) + json.dumps(req["questions"])) // 4)
        return {"model": "typesafe/jev-1.13-20260917", "answers": {self.arm.jev_question_id: ans},
                "usage": {"input_tokens": tin, "output_tokens": 60, "cost": tin * MODELS["jev"]["price_per_mtok_input"] / 1e6},
                "id": f"gen-dec-sim-{_sha(req)[:12]}", "provider": "TypeSafe"}

    def _chat(self, fam: str, it: Item, pick: str, p: np.ndarray, rng, req: dict) -> dict:
        probs = {l: round(float(x), 3) for l, x in zip(it.presented, p)}
        u = rng.random()
        if u < self.malformed:
            content = f"The best answer is {pick}."                                  # not JSON: unusable
        elif u < 2 * self.malformed:
            content = json.dumps({self.arm.answer_key: pick, "probabilities": {l: x * 0.8 for l, x in probs.items()}})
        elif u < 3 * self.malformed:
            content = "```json\n" + json.dumps({self.arm.answer_key: pick.lower(), "probabilities": probs}) + "\n```"
        else:
            content = json.dumps({self.arm.answer_key: pick, "probabilities": probs})
        tin = max(1, sum(len(m["content"]) for m in req["messages"]) // 4)
        spec = MODELS["families"][fam]
        usage = {"prompt_tokens": tin, "completion_tokens": TOKENS_OUT.get(fam, 100)}
        if fam in REASONING:
            usage["completion_tokens_details"] = {"reasoning_tokens": int(REASONING[fam] * usage["completion_tokens"])}
        return {"id": f"gen-sim-{_sha(req)[:12]}", "model": spec["slug"],
                "provider": (spec.get("provider_only") or ["featherless-ai"])[0],
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": usage}
