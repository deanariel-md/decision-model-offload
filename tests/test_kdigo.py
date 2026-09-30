"""The KDIGO key of the eICU AKI set (src/jevity/kdigo.py, config/eicu_aki.yaml kdigo) on hand-worked cases. Times in
minutes from unit admission; values in mg/dL as shown. No network, no data."""
from decimal import Decimal as D
from pathlib import Path

import pytest
import yaml

from jevity import kdigo as KD

ROOT = Path(__file__).resolve().parents[1]
RULE = KD.Rule.from_config(yaml.safe_load((ROOT / "config" / "eicu_aki.yaml").read_text(encoding="utf-8"))["kdigo"])
H = 60


def stage(values, dialysis=None):
    f = KD.highest_stage([(t, D(v)) for t, v in values], dialysis, RULE)
    return KD.LEVELS[f.stage], f.criterion


def test_the_rule_in_the_config_is_kdigo_2012():
    assert (RULE.assess_from, RULE.assess_to) == (0, 72 * H)
    assert RULE.rise == D("0.3") and RULE.rise_window == 48 * H and RULE.ratio_window == 7 * 24 * H
    assert RULE.ratio_stages == ((D("1.5"), 1), (D("2.0"), 2), (D("3.0"), 3))
    assert RULE.stage3_value == D("4.0") and RULE.dialysis_stage == 3


@pytest.mark.parametrize("values, want", [
    # rise of 0.3 within 48 h: exactly 0.3 counts, 0.29 does not
    ([(-1 * H, "1.00"), (10 * H, "1.30")], ("stage 1", "rise")),
    ([(-1 * H, "1.00"), (10 * H, "1.29")], ("no AKI", "none")),
    # the 48 h window: exactly 48 h counts, 48 h 1 min does not (and the ratio 1.3 is below 1.5)
    ([(-2 * H, "1.00"), (46 * H, "1.30")], ("stage 1", "rise")),
    ([(-2 * H, "1.00"), (46 * H + 1, "1.30")], ("no AKI", "none")),
    # ratio to the lowest value in the preceding 7 days, outside 48 h: the stage boundaries
    ([(-80 * H, "1.00"), (10 * H, "1.49")], ("no AKI", "none")),
    ([(-80 * H, "1.00"), (10 * H, "1.50")], ("stage 1", "ratio")),
    ([(-80 * H, "1.00"), (10 * H, "1.99")], ("stage 1", "ratio")),
    ([(-80 * H, "1.00"), (10 * H, "2.00")], ("stage 2", "ratio")),
    ([(-80 * H, "1.00"), (10 * H, "2.99")], ("stage 2", "ratio")),
    ([(-80 * H, "1.00"), (10 * H, "3.00")], ("stage 3", "ratio")),
    # the lowest value, not the first: 0.80 then 1.00 then 1.60 is a ratio of 2.0
    ([(-100 * H, "0.80"), (-60 * H, "1.00"), (10 * H, "1.60")], ("stage 2", "ratio")),
    # the 7-day window: a value 7 days and 1 minute earlier is not a reference
    ([(-7 * 24 * H, "0.50"), (-80 * H, "1.00"), (1, "1.60")], ("stage 1", "ratio")),
    ([(-7 * 24 * H, "0.50"), (-80 * H, "1.00"), (0, "1.60")], ("stage 3", "ratio")),
    # 4.0 mg/dL or more: stage 3 with an acute rise, nothing without one
    ([(-10 * H, "3.70"), (2 * H, "4.00")], ("stage 3", "value_4_with_rise")),
    ([(-10 * H, "4.40"), (2 * H, "4.50")], ("no AKI", "none")),
    ([(-80 * H, "2.70"), (2 * H, "4.05")], ("stage 3", "value_4_with_rise")),   # ratio 1.5 is the acute rise
    # values before unit admission are references only: a rise before admission is not assessed ...
    ([(-50 * H, "1.00"), (-2 * H, "2.50")], ("no AKI", "none")),
    # ... but a high value in the unit is assessed against them
    ([(-50 * H, "1.00"), (-2 * H, "2.50"), (1 * H, "2.40")], ("stage 2", "ratio")),
    # only preceding values count: a falling creatinine is no AKI by this rule
    ([(-1 * H, "3.00"), (1 * H, "1.00"), (50 * H, "1.10")], ("no AKI", "none")),
    # values after 72 h are not shown; a value at exactly 72 h is assessed
    ([(-10 * H, "1.00"), (72 * H, "1.30")], ("no AKI", "none")),   # 82 h apart, ratio 1.3
    ([(30 * H, "1.00"), (72 * H, "1.30")], ("stage 1", "rise")),
    # the highest over the assessed values
    ([(-80 * H, "1.00"), (10 * H, "1.60"), (40 * H, "2.10"), (60 * H, "1.20")], ("stage 2", "ratio")),
])
def test_hand_worked_cases(values, want):
    assert stage(values) == want


def test_dialysis_started_by_72_hours_is_stage_3():
    assert stage([(-10 * H, "1.00"), (2 * H, "1.05")], dialysis=30 * H) == ("stage 3", "dialysis")
    assert stage([(-10 * H, "1.00"), (2 * H, "1.05")], dialysis=72 * H) == ("stage 3", "dialysis")
    assert stage([(-10 * H, "1.00"), (2 * H, "1.05")], dialysis=72 * H + 1) == ("no AKI", "none")
    f = KD.highest_stage([(-80 * H, D("1.00")), (5 * H, D("3.20"))], 20 * H, RULE)
    assert (f.stage, f.criterion, f.at_min) == (3, "dialysis", 20 * H)          # dialysis wins a tie


def test_finding_records_the_value_and_reference():
    f = KD.highest_stage([(-80 * H, D("0.80")), (-60 * H, D("1.00")), (10 * H, D("1.60"))], None, RULE)
    assert (f.stage, f.at_min, f.value, f.reference) == (2, 10 * H, D("1.60"), D("0.80"))


def test_a_baseline_read_either_side_only_raises_falling_records():
    """The sensitivity key: a falling creatinine (2.30 at +3 h, 0.70 at +34 h) is no AKI by the key and stage 3 with a
    baseline taken after it; a rising record is the same either way."""
    falling = [(-6 * H, D("2.20")), (3 * H, D("2.30")), (34 * H, D("0.70")), (57 * H, D("0.90"))]
    assert KD.highest_stage(falling, None, RULE).stage == 0 and KD.highest_stage_either_side(falling, None, RULE) == 3
    rising = [(-51 * H, D("1.30")), (8 * H, D("2.90")), (55 * H, D("2.00"))]
    assert KD.highest_stage(rising, None, RULE).stage == KD.highest_stage_either_side(rising, None, RULE) == 2
    # only values 0-72 h are assessed: a fall before admission alone changes nothing
    assert KD.highest_stage_either_side([(-50 * H, D("3.00")), (-10 * H, D("1.00")), (5 * H, D("1.05"))], None, RULE) == 0


def test_time_and_value_format():
    assert KD.fmt_time(-(52 * H + 10)) == "-52 h 10 min" and KD.fmt_time(45) == "+0 h 45 min"
    assert KD.fmt_time(0) == "+0 h 00 min" and KD.fmt_value(D("1.3")) == "1.30"
