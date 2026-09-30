"""API clients. Every call writes the raw response verbatim to runs/<run>/<sha>.json before parsing."""
from __future__ import annotations

import hashlib
import json
import math
import re
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODELS = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
PROMPTS = yaml.safe_load((ROOT / "config" / "prompts.yaml").read_text(encoding="utf-8"))
OPENROUTER = "https://openrouter.ai"
HF_ROUTER = "https://router.huggingface.co/v1"
TYPESAFE = "https://api.typesafe.ai"
BATCH_MISSING = "batch result not collected"


def active_families(tiers: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """LLM families that are run, in config order, without those marked disabled; optionally only those whose `tier`
    is in `tiers` (primary, free, pair)."""
    return [f for f, s in MODELS["families"].items()
            if not s.get("disabled") and (tiers is None or s.get("tier", "primary") in tiers)]


EFFORT_SEP = "@"


def split_system(name: str) -> tuple[str, str | None]:
    """'gpt@low' -> ('gpt', 'low'): a reasoning-arm system is a family plus a unified reasoning.effort level."""
    fam, _, eff = name.partition(EFFORT_SEP)
    return fam, (eff or None)


def arm_systems(arm: str) -> list[str]:
    """System names (calls.model values) planned for an arm (config `arms`). Reasoning: every active primary family at
    each effort, except a family/effort whose answers are reused from the default run (`reuse_default`)."""
    a = MODELS["arms"][arm]
    fams = active_families((a["tier"],))
    if arm != "reasoning":
        return fams
    reuse = a.get("reuse_default") or {}
    return [f"{f}{EFFORT_SEP}{e}" for f in fams for e in a["efforts"] if reuse.get(f) != e]


def system_tier(name: str) -> str:
    """'jev', 'primary', 'free', 'pair' or 'reasoning' for a calls.model value (a disabled family keeps its config
    tier)."""
    if name.startswith("jev"):
        return "jev"
    fam, eff = split_system(name)
    if eff:
        return "reasoning"
    return (MODELS["families"].get(fam) or {}).get("tier", "primary")


def _key(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} missing from .env")
    return v


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


@dataclass
class RawStore:
    run_dir: Path

    def path(self, req: dict) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / f"{_sha(req)}.json"

    def get(self, req: dict) -> dict | None:
        p = self.path(req)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, req: dict, resp: dict, meta: dict) -> bool:
        """First write wins: a stored response is never overwritten, so the designated primary answer is immutable.
        Written to a temporary file and moved into place atomically; returns False if an answer already existed."""
        dst = self.path(req)
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"request": req, "response": resp, "meta": meta}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        try:
            if os.name == "nt":
                os.rename(tmp, dst)            # fails if dst exists
            else:
                os.link(tmp, dst)              # fails if dst exists
                tmp.unlink()
            return True
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            return False

    def lock(self, req: dict) -> threading.Lock:
        """One in-flight request per payload within a process (single flight): a second worker with an identical
        payload waits and then reads the stored answer."""
        key = str(self.path(req))
        with _LOCK_GUARD:
            return _LOCKS.setdefault(key, threading.Lock())


ON_RETRY_STATUS = None    # optional callable(status, url, body): lets a runner slow down on HTTP 429; requests unchanged


def _post(url: str, headers: dict, body: dict, retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=120) as c:
                r = c.post(url, headers=headers, json=body)
            if r.status_code in (429, 500, 502, 503, 529):
                if ON_RETRY_STATUS is not None:
                    ON_RETRY_STATUS(r.status_code, url, body)
                time.sleep(delay); delay *= 2; continue
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError,) as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay); delay *= 2
    raise RuntimeError("retries exhausted")


_NUM = r"(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)"                 # a decimal in [0, 1]; no sign, exponent or percent
_TEXT_P = re.compile(r'\{?\s*"?p"?\s*[:=]\s*' + _NUM + r'\s*\}?\.?', re.I)
_TEXT_BARE = re.compile(_NUM + r"\.?")
_TEXT_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\.?")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)


def _real01(v) -> float | None:
    """A finite real in [0, 1]; booleans and strings are not numbers."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def parse_probability_detail(content: str | None) -> tuple[float | None, str]:
    """Strict grammar, whole-response matches only: a JSON object with key "p" or a bare JSON number (finite, in
    [0, 1], not boolean); otherwise the whole text must be `p: x` / `{"p": x}` without quotes, a bare decimal in
    [0, 1], or one percentage. Anything else (refusals, explanations, ranges, several numbers) is unusable. Returns
    (p, category); the category is stored with every row."""
    if content is None or not content.strip():
        return None, "empty"
    txt = content.strip()
    m = _FENCE.match(txt)
    if m:
        txt = m.group(1).strip()
    try:
        v = json.loads(txt)
    except (ValueError, RecursionError):
        v = _MISSING
    if v is not _MISSING:
        if isinstance(v, dict):
            if "p" not in v:
                return None, "json_no_p"
            p = _real01(v["p"])
            return (p, "json") if p is not None else (None, "json_invalid_p")
        p = _real01(v)
        return (p, "json_number") if p is not None else (None, "json_invalid")
    for rx, cat in ((_TEXT_P, "text_p"), (_TEXT_BARE, "text_number")):
        m = rx.fullmatch(txt)
        if m:
            return float(m.group(1)), cat
    m = _TEXT_PCT.fullmatch(txt)
    if m and float(m.group(1)) <= 100:
        return float(m.group(1)) / 100, "text_percent"
    return None, "unparsed"


def parse_probability(content: str | None) -> float | None:
    return parse_probability_detail(content)[0]


_MISSING = object()
VARIANTS = ("raw", "M1", "F1")


def drop_sentence(text: str, sentence: str) -> str:
    """`text` (whitespace normalised) without `sentence`; an absent sentence is an error, never a silent no-op."""
    t, s = " ".join(text.split()), " ".join(sentence.split())
    if s not in t:
        raise ValueError(f"sentence not found in the system message: {s!r}")
    return " ".join(t.replace(s, "", 1).split())


class LLMClient:
    """OpenRouter chat completions: one pinned provider, no fallbacks, provider defaults for reasoning and temperature
    (a parameter is sent only when config sets it), seed where supported, JSON schema where the pinned endpoint
    supports it and a text parser otherwise. Population "eicu" (credentialed) is refused outright."""

    def __init__(self, family: str, store: RawStore, population: str = "nhanes", repeat: int = 0, variant: str = "raw",
                 standard: bool = False, effort: str | None = None):
        family, eff = split_system(family)
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.family, self.store, self.repeat, self.population, self.variant = family, store, repeat, population, variant
        self.effort = effort or eff     # reasoning arm only: unified reasoning.effort; the family's config is untouched
        self.spec = MODELS["families"][family]
        if self.spec.get("disabled"):
            raise RuntimeError(f"family {family} is disabled in config/models.yaml")
        self.transport = self.spec.get("transport", "openrouter")     # openrouter | hf_router
        if self.effort is not None and (self.effort not in ("low", "high") or self.transport != "openrouter"):
            raise ValueError(f"reasoning effort {self.effort!r} is defined for OpenRouter families at low or high only")
        self.route = MODELS["routing"]
        self.gen = MODELS["generation"]
        self.batch = None if (standard or self.transport != "openrouter") else self._batch_spec()   # standard=True: synchronous

    use_batch = True    # FillClient (synchronous diagnostic) sets False

    def _batch_spec(self) -> dict | None:
        """The family's batch route if enabled for this population, else None (standard route)."""
        b = MODELS.get("batch") or {}
        if not (self.use_batch and b.get("enabled") and self.family in (b.get("families") or {})):
            return None
        if self.population not in b.get("populations", []):
            return None
        assert self.population != "eicu", "eICU never goes through the batch route"
        return b["families"][self.family]

    def _system(self) -> str:
        parts = [PROMPTS["llm"]["system"]]
        src = PROMPTS["llm"].get("record_source") or {}
        if self.population != "nhanes" and self.population in src:    # eICU: the record's source
            t, old = " ".join(parts[0].split()), src["nhanes"]
            if old not in t:
                raise ValueError(f"phrase not found in the system message: {old!r}")
            parts = [t.replace(old, src[self.population], 1)]
        if self.variant == "F1":     # framing sensitivity analysis: the same system message without the study sentence
            parts = [drop_sentence(parts[0], PROMPTS["framing_sensitivity"]["drop"])]
        if self.variant == "M1":
            parts.append(PROMPTS["mitigation"]["M1_instruction"])
        parts.append(PROMPTS["llm"]["reply"])
        return " ".join(" ".join(x.split()) for x in parts)

    def request(self, record_text: str, statement: str) -> dict:
        if self.population == "eicu":
            raise RuntimeError("credentialed eICU rows never go to any model; use population eicu_demo")
        if self.transport == "hf_router":   # Hugging Face router: the provider is the model-id suffix; no provider block
            provider = None
        elif self.batch is not None:     # batch: provider.only is a batch-level field; the API rejects fallbacks/zdr keys
            provider = {"only": list(self.batch["provider_only"])}
        else:
            provider = {"only": list(self.spec["provider_only"]), "allow_fallbacks": self.route["allow_fallbacks"]}
            if self.route["zdr"]:
                provider["zdr"] = True
        req = {
            # base slug on both OpenRouter routes (the batch API rejects the :batch suffix); HF: model id with provider suffix
            "model": self.spec["hf_model"] if self.transport == "hf_router" else self.spec["slug"],
            "messages": [
                {"role": "system", "content": self._system()},
                {"role": "user", "content": PROMPTS["llm"]["user_template"].format(record=record_text, statement=statement)},
            ],
            "max_tokens": self.spec["max_tokens"],
            "provider": provider,
            "_repeat": self.repeat,   # cache key only; stripped before sending
        }
        if provider is None:
            req.pop("provider")
        if self.gen.get("temperature") is not None:        # provider default unless config sets one
            req["temperature"] = self.gen["temperature"]
        if self.spec.get("reasoning"):                      # provider default unless config sets one
            req["reasoning"] = dict(self.spec["reasoning"])
        if self.effort is not None:                         # reasoning arm: the same named level for every model
            req["reasoning"] = {"effort": self.effort}
        if self.spec.get("seed") and self.gen.get("seed") is not None:
            req["seed"] = int(self.gen["seed"])
        if self.spec.get("structured_outputs"):
            req["response_format"] = {"type": "json_schema", "json_schema": PROMPTS["llm"]["json_schema"]}
        if self.batch is not None:
            req["_route"] = "batch"
        return req

    def probability(self, record_text: str, statement: str) -> dict:
        req = self.request(record_text, statement)
        cached = self.store.get(req)
        if cached is None and self.batch is not None:   # never sent synchronously: submit and collect via jevity.batch
            if self.family in (MODELS.get("batch") or {}).get("fallback_to_standard", []):
                return LLMClient(self.family, self.store, self.population, self.repeat, self.variant,
                                 standard=True, effort=self.effort).probability(record_text, statement)
            return {"p": None, "valid": False, "provider": None, "model_reported": None, "usage": None,
                    "raw_path": str(self.store.path(req)), "error": BATCH_MISSING}
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        resp = cached["response"]
        try:
            content = resp["choices"][0]["message"].get("content")
        except Exception:
            content = None
        return self._result(req, cached, resp, content)

    def _send(self, req: dict) -> dict:
        body = {k: v for k, v in req.items() if not k.startswith("_")}
        if self.population == "eicu":
            raise RuntimeError("credentialed eICU rows never go to any model")
        t0 = time.time()
        if self.transport == "hf_router":
            resp = _post(f"{HF_ROUTER}/chat/completions",
                         {"Authorization": f"Bearer {_key('HF_TOKEN')}", "Content-Type": "application/json"}, body)
        else:
            resp = _post(f"{OPENROUTER}/api/v1/chat/completions",
                         {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}", "Content-Type": "application/json"}, body)
        meta = {"ts": time.time(), "latency_s": time.time() - t0, "family": self.family, "route": "standard",
                "transport": self.transport,
                "provider_reported": resp.get("provider"),
                "model_reported": resp.get("model"), "usage": resp.get("usage")}
        if not self.store.put(req, resp, meta):
            return self.store.get(req)                 # another process answered first; its answer stands
        return {"request": req, "response": resp, "meta": meta}

    def _result(self, req: dict, cached: dict, resp: dict, content: str | None) -> dict:
        p, parse = parse_probability_detail(content)
        meta = cached.get("meta") or {}
        prov = resp.get("provider") or meta.get("provider_pinned")   # batch bodies may omit it
        try:
            finish = resp["choices"][0].get("finish_reason")
        except Exception:
            finish = None
        return {"p": p, "valid": p is not None, "provider": prov, "model_reported": resp.get("model"),
                "usage": resp.get("usage"), "raw_path": str(self.store.path(req)), "parse": parse,
                "finish_reason": finish, "route": meta.get("route", "standard"), "response_id": resp.get("id"),
                "latency_s": meta.get("latency_s")}


def parse_value(content: str | None) -> float | None:
    """A masked-field answer's value: the whole reply as JSON ({"value": x} or a bare number), else the last
    {"value": x} object in the reply; a reply with neither (a refusal, or prose with no answer object) is missing."""
    import re
    if not content:
        return None
    txt = content.strip().strip("`")
    if txt.lower().startswith("json"):
        txt = txt[4:]
    try:
        v = json.loads(txt)
        if isinstance(v, dict) and "value" in v:
            return float(v["value"])
        if isinstance(v, (int, float)):
            return float(v)
        return None
    except Exception:
        pass
    m = re.findall(r'\{\s*"value"\s*:\s*"?(-?\d+(?:\.\d+)?)"?\s*\}', txt)
    return float(m[-1]) if m else None


class FillClient(LLMClient):
    """Masked-field recovery for the exposure diagnostic: standard route (synchronous), same settings as LLMClient,
    different prompt."""
    use_batch = False

    def request(self, record_text: str, field: str, unit: str | None = None) -> dict:
        req = super().request(record_text, "")
        ex = PROMPTS["exposure"]
        unit = ex["fields"][field]["unit"] if unit is None else unit      # eICU fields pass their unit (config/eicu.yaml)
        req["messages"] = [{"role": "system", "content": " ".join(ex["system"].split())},
                           {"role": "user", "content": ex["user_template"].format(record=record_text, field=field, unit=unit)}]
        req.pop("response_format", None)
        req["_task"] = "fill"
        return req

    def fill(self, record_text: str, field: str, unit: str | None = None) -> dict:
        req = self.request(record_text, field, unit)
        cached = self.store.get(req)
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        try:
            content = cached["response"]["choices"][0]["message"].get("content")
        except Exception:
            content = None
        v = parse_value(content)
        return {"value": v, "valid": v is not None, "raw_path": str(self.store.path(req))}


class JevClient:
    """Jev through OpenRouter's TypeSafe-compatible endpoint (POST /api/v1/systemone; alternative
    /api/alpha/decisions) or TypeSafe direct (POST https://api.typesafe.ai/v1/systemone, model jev-1.13.0).
    `question` = "primary" (Noul, the death statement), "risk_bands" (Choice over bands; expected value),
    "complement" (Noul on the complementary statement; coherence diagnostic) or "bundle" (the three in one request:
    does the death answer change when other questions share the pass?)."""

    def __init__(self, store: RawStore, population: str = "nhanes", repeat: int = 0, transport: str | None = None,
                 question: str = "primary", variant: str = "raw"):
        if variant not in ("raw", "M1"):        # F1 changes an LLM system message; Jev has none
            raise ValueError(f"variant {variant!r} is not defined for Jev")
        self.store, self.repeat, self.population, self.question, self.variant = store, repeat, population, question, variant
        self.spec = MODELS["jev"]
        self.route = MODELS["routing"]
        self.transport = transport or self.spec["transport"]

    def _questions(self, statement: str) -> dict:
        if self.question == "primary":
            q = {"type": "noul", "instructions": statement}     # identical in every arm; M1 goes into the state
            return {PROMPTS["jev"]["primary"]["question_id"]: q}
        if self.question == "complement":
            return {"alive": {"type": "noul", "instructions": PROMPTS["complements"][self.population]}}
        rb = PROMPTS["jev"]["risk_bands"]
        assert self.population in rb["instructions"], "the ten-year outcome question is defined for NHANES only"
        crit = {k: _band_text(v) for k, v in rb["options"].items()}
        bands = {rb["question_id"]: {"type": "choice", "instructions": rb["instructions"][self.population], "criteria": crit}}
        if self.question == "bundle":
            return {PROMPTS["jev"]["primary"]["question_id"]: {"type": "noul", "instructions": statement},
                    "alive": {"type": "noul", "instructions": PROMPTS["complements"][self.population]}, **bands}
        return bands

    def request(self, record_text: str, statement: str) -> dict:
        if self.population == "eicu":
            raise RuntimeError("credentialed eICU rows never go to any model; use population eicu_demo")
        model = self.spec["slug"] if self.transport == "openrouter" else "jev-1.13.0"
        if self.variant == "M1":   # Jev has no system prompt: the instruction opens the state, the question is unchanged
            record_text = f"Instruction: {' '.join(PROMPTS['mitigation']['M1_instruction'].split())}\n\n{record_text}"
        req = {"model": model, "state": record_text, "questions": self._questions(statement), "_repeat": self.repeat}
        if self.transport == "openrouter":
            req["provider"] = {"allow_fallbacks": False, **({"zdr": True} if self.route["zdr"] else {})}
        return req

    def probability(self, record_text: str, statement: str) -> dict:
        req = self.request(record_text, statement)
        cached = self.store.get(req)
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        return self._result(req, cached)

    def _send(self, req: dict) -> dict:
        body = {k: v for k, v in req.items() if not k.startswith("_")}
        if self.transport == "openrouter":
            url = f"{OPENROUTER}{self.spec['openrouter_path']}"
            headers = {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}", "Content-Type": "application/json"}
        elif self.transport == "typesafe_direct":
            url = f"{TYPESAFE}/v1/systemone"
            headers = {"Authorization": f"Bearer {_key('TYPESAFE_API_KEY')}", "Content-Type": "application/json"}
            body.pop("provider", None)
        else:
            raise ValueError(self.transport)
        t0 = time.time()
        resp = _post(url, headers, body)
        meta = {"ts": time.time(), "latency_s": time.time() - t0, "transport": self.transport, "route": "standard",
                "provider_reported": resp.get("provider"), "model_reported": resp.get("model"),
                "usage": resp.get("usage")}
        if not self.store.put(req, resp, meta):
            return self.store.get(req)
        return {"request": req, "response": resp, "meta": meta}

    def _result(self, req: dict, cached: dict) -> dict:
        resp = cached["response"]
        p, extra, parse = None, {}, "unparsed"
        try:
            ans = resp["answers"]
            if self.question == "bundle":
                p = _real01(ans[PROMPTS["jev"]["primary"]["question_id"]]["noul"])
                pb, _, extra = _risk_band_mean(ans[PROMPTS["jev"]["risk_bands"]["question_id"]])
                extra = {**extra, "p_alive": _real01((ans.get("alive") or {}).get("noul")), "p_bands": pb}
                parse = "bundle" if p is not None else "noul_invalid"
            elif self.question in ("primary", "complement"):
                qid = PROMPTS["jev"]["primary"]["question_id"] if self.question == "primary" else "alive"
                p = _real01(ans[qid]["noul"])
                parse = "noul" if p is not None else "noul_invalid"
                if ans[qid].get("confidence") is not None:
                    extra = {"confidence": ans[qid].get("confidence")}
            else:
                p, parse, extra = _risk_band_mean(ans[PROMPTS["jev"]["risk_bands"]["question_id"]])
        except Exception:
            p, parse = None, "missing_answer"
        return {"p": p, "valid": p is not None, "provider": resp.get("provider"), "model_reported": resp.get("model"),
                "usage": resp.get("usage"), "raw_path": str(self.store.path(req)), "parse": parse,
                "route": (cached.get("meta") or {}).get("route", "standard"), "response_id": resp.get("id"),
                "latency_s": (cached.get("meta") or {}).get("latency_s"), **extra}


def _band_text(years) -> str:
    """Criteria for one outcome option: death in [from, to) years after the examination, or alive at ten years."""
    if years is None:
        return "Alive ten years after this examination."
    lo, hi = years
    if lo == 0:
        return f"Died from any cause within {hi} years of this examination."
    return f"Died from any cause at least {lo} and under {hi} years after this examination."


def _risk_band_mean(a: dict) -> tuple[float | None, str, dict]:
    """Probability of death within ten years from Jev's Choice over outcomes: the mass on the three death options
    (= 1 - P(alive at ten years)). The full distribution and the confidence are kept. Unknown labels, values outside
    [0, 1] or a total outside [0.99, 1.01] make the answer unusable; nothing is renormalised."""
    rb = PROMPTS["jev"]["risk_bands"]["options"]
    probs = a.get("probabilities") or {}
    extra = {"band_probabilities": probs, "confidence": a.get("confidence")}
    if not set(probs) <= set(rb):
        return None, "bands_labels", extra
    vals = {k: _real01(probs.get(k, 0.0)) for k in rb}           # an omitted option carries zero mass
    if any(v is None for v in vals.values()) or not 0.99 <= sum(vals.values()) <= 1.01:
        return None, "bands_mass", extra
    return min(1.0, sum(vals[k] for k, yrs in rb.items() if yrs is not None)), "bands", extra


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
