"""KDIGO 2012 AKI stage by the serum creatinine criteria, applied to exactly the values a record shows (eICU AKI set;
rule in config/eicu_aki.yaml `kdigo`). Pure functions on (minutes from unit admission, value as
shown); decimal arithmetic on the shown values, so a ratio of exactly 1.5 or a rise of exactly 0.3 counts.

A value taken 0-72 h after unit admission is assessed against the values taken strictly before it:
  - rise: the value minus the lowest value in the preceding 48 h is at least 0.3 mg/dL (stage 1);
  - ratio: the value over the lowest value in the preceding 7 days; 1.5-1.9 stage 1, 2.0-2.9 stage 2, 3.0 or more
    stage 3;
  - a value of 4.0 mg/dL or more with an acute rise (either criterion met at that value): stage 3.
Dialysis started at or before 72 h: stage 3. The record's stage is the highest over its assessed values and dialysis;
values before unit admission serve as reference values only."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

LEVELS = ("no AKI", "stage 1", "stage 2", "stage 3")


@dataclass(frozen=True)
class Rule:
    assess_from: int
    assess_to: int
    rise: Decimal
    rise_window: int
    ratio_window: int
    ratio_stages: tuple[tuple[Decimal, int], ...]      # ascending thresholds
    stage3_value: Decimal
    dialysis_stage: int

    @classmethod
    def from_config(cls, k: Mapping) -> "Rule":
        return cls(int(k["assess_from_min"]), int(k["assess_to_min"]), Decimal(str(k["rise_mg_dl"])),
                   int(k["rise_window_min"]), int(k["ratio_window_min"]),
                   tuple(sorted((Decimal(str(t)), int(s)) for t, s in k["ratio_stages"])),
                   Decimal(str(k["stage3_mg_dl"])), int(k["dialysis_stage"]))


@dataclass(frozen=True)
class Finding:
    stage: int                      # 0-3, index into LEVELS
    criterion: str                  # "none", "rise", "ratio", "value_4_with_rise", "dialysis"
    at_min: int | None              # the assessed value's time (or dialysis start)
    value: Decimal | None
    reference: Decimal | None       # the lowest preceding value the criterion used


def assess(t: int, v: Decimal, earlier: Sequence[tuple[int, Decimal]], rule: Rule) -> Finding:
    """Stage carried by the value v at minute t, given every value of the record (any order); only values strictly
    before t and inside each window count."""
    w48 = [u for s, u in earlier if s < t and t - s <= rule.rise_window]
    w7 = [u for s, u in earlier if s < t and t - s <= rule.ratio_window]
    stage, crit, ref = 0, "none", None
    rise = bool(w48) and v - min(w48) >= rule.rise
    ratio_ok = False
    if w7:
        m = min(w7)
        for thr, st in rule.ratio_stages:
            if v >= thr * m:
                ratio_ok = True
                if st > stage:
                    stage, crit, ref = st, "ratio", m
    if rise and stage < 1:
        stage, crit, ref = 1, "rise", min(w48)
    if (rise or ratio_ok) and v >= rule.stage3_value and stage < 3:
        stage, crit = 3, "value_4_with_rise"
        ref = min(w48) if rise else min(w7)
    return Finding(stage, crit, t if stage else None, v if stage else None, ref)


def highest_stage(values: Sequence[tuple[int, Decimal]], dialysis_min: int | None, rule: Rule) -> Finding:
    """The record's stage: the highest over its assessed values (0-72 h) and dialysis started at or before 72 h. Ties go
    to the earliest finding; dialysis wins a tie with a value."""
    best = Finding(0, "none", None, None, None)
    for t, v in sorted(values, key=lambda x: x[0]):
        if rule.assess_from <= t <= rule.assess_to:
            f = assess(t, v, values, rule)
            if f.stage > best.stage:
                best = f
    if dialysis_min is not None and dialysis_min <= rule.assess_to and rule.dialysis_stage >= best.stage:
        best = Finding(rule.dialysis_stage, "dialysis", dialysis_min, None, None)
    return best


def highest_stage_either_side(values: Sequence[tuple[int, Decimal]], dialysis_min: int | None, rule: Rule) -> int:
    """For the sensitivity analysis only (the key is highest_stage): the stage when each assessed value's
    ratio is taken against the lowest other value within 7 days before or after it (a baseline read from the whole
    record, as a reader might for a falling creatinine). The rise and 4.0 criteria and dialysis are unchanged."""
    best = highest_stage(values, dialysis_min, rule).stage
    for t, v in values:
        if not rule.assess_from <= t <= rule.assess_to:
            continue
        around = [u for s, u in values if s != t and abs(s - t) <= rule.ratio_window]
        if around:
            m = min(around)
            best = max([best] + [st for thr, st in rule.ratio_stages if v >= thr * m])
    return best


def fmt_time(minutes: int) -> str:
    """-3130 -> '-52 h 10 min'; 45 -> '+0 h 45 min'."""
    sign = "-" if minutes < 0 else "+"
    h, m = divmod(abs(int(minutes)), 60)
    return f"{sign}{h} h {m:02d} min"


def fmt_value(v: Decimal) -> str:
    return f"{v:.2f}"
