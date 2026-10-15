"""Independent decimal calculations cover the recovered worksheet rule families."""
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import math

import pytest

from fdm.live_experiment import ExperimentSession, MaterialSnapshot, summarize
from fdm.live_experiment.statistics import METHODS, round_cu


# An explicit mapping independent of the production registry catches dispatch
# mistakes, including methods which share their numerical formula.
BRANCHES = {
    "GB/T 16988-2013": "variance", "ISO 17751-1:2016(E)": "variance",
    "CNS 2339-1 L3050-1": "variance", "CAN CGSB-4.2 No.14-2005": "variance",
    "JIS L 1030-2：2012": "variance", "SN/T 1062-2002": "mean_square",
    "FZ/T 01101-2008": "square_mean", "AATCC 20A-2018": "square_mean",
    "SN/T 0756-1999": "bast_upper", "FZ/T 32004-2009": "bast_mid",
    "FZ/T 30003-2009": "empirical",
}


def D(value):
    return Decimal(str(value))


def _oracle(method, rows, width, grouped=True):
    """Recompute from the complete raw sample using exact decimal moments.

    No production aggregates, grouping, rounding, or branch definitions are used.
    """
    kind = BRANCHES[method]
    weights = []
    for family, density, values, extra in rows:
        samples = [D(x) for x in values]
        bins = [(v / D(width)).to_integral_value(rounding=ROUND_FLOOR) for v in samples]
        if grouped:
            samples = [(i + D("0.5")) * D(width) for i in bins]
        count = len(samples)
        mean = sum(samples) / count
        square_mean = sum(x*x for x in samples) / count
        variance = sum((x-mean)**2 for x in samples) / (count-1) if count > 1 else D(0)
        if kind == "variance":
            quantity = mean*mean + variance
        elif kind in ("mean_square", "empirical"):
            quantity = mean*mean
        elif kind == "square_mean":
            quantity = square_mean
        elif family == "ramie":
            terms = []
            for i, x in zip(bins, samples):
                ref = (i+1)*D(width) if kind == "bast_upper" else x
                coefficient = D("0.3202") if ref < 15 else D("0.2652") if ref < 40 else D("0.221")
                terms.append(x*x*coefficient)
            quantity = sum(terms) / count
        else:
            coefficient = {"cotton": D("0.2939"), "flax": D("0.42"), "hemp": D("0.35"), "apocynum": D("0.39")}[family]
            quantity = square_mean*coefficient
        weights.append(quantity*(count+extra)*D(density))
    values = [x/sum(weights)*100 for x in weights]
    if kind == "empirical":
        for i, (family, _, _, _) in enumerate(rows[1:], 1):
            x = values[i]
            if family == "hemp":
                x = D("1.3402")*x-D("0.0034")*x*x
            elif family == "flax":
                x = D("1.3731")*x-D("0.0037")*x*x
            values[i] = max(D(0), min(D(100), x))
        values[0] = 100-sum(values[1:])
    return values


def _fixture(method, width, profile):
    session = ExperimentSession(method=method, group_width=width, profile=profile)
    if BRANCHES[method] in ("bast_mid", "bast_upper", "empirical"):
        materials = [MaterialSnapshot("cotton", "棉", 1.52, "cotton"),
                     MaterialSnapshot("ramie", "苎麻", 1.50, "ramie"),
                     MaterialSnapshot("hemp", "大麻", 1.48, "hemp"),
                     MaterialSnapshot("flax", "亚麻", 1.50, "flax")]
        if BRANCHES[method] != "empirical":
            materials.append(MaterialSnapshot("apocynum", "罗布麻", 1.45, "apocynum"))
    else:
        materials = [MaterialSnapshot("wool", "绵羊毛", 1.31, "wool"), MaterialSnapshot("cashmere", "山羊绒", 1.30, "cashmere")]
    expected = []
    for index in range(3):
        op = session.add_operator(f"操作者{index}")
        session.set_weights(70-index*10, 30+index*10, operator_id=op)
        direction_results = []
        for direction in ("warp", "weft"):
            session.set_materials(materials, operator_id=op, direction=direction)
            raw = []
            for i, material in enumerate(materials):
                shift = index + (3 if direction == "weft" else 0)
                values = [14.0 + i + shift, 21.0 + 2*i + shift, 38.0 + i + shift]
                extra = 20 if material.family == "cotton" else i + 1
                for value in values:
                    session.add_measurement(value, material.id, operator_id=op, direction=direction)
                session.adjust_count(material.id, extra, operator_id=op, direction=direction)
                raw.append((material.family, material.density, values, extra))
            direction_results.append(_oracle(method, raw, width))
        expected.append((op, direction_results))
    return session, expected


@pytest.mark.parametrize("method", list(BRANCHES))
@pytest.mark.parametrize("width", [1, 1.5, 2, 2.5])
@pytest.mark.parametrize("profile", ["normal", "decolorization_free"])
@pytest.mark.parametrize("report_type", ["fiber", "fabric"])
def test_176_complete_configurations_against_decimal_reference(method, width, profile, report_type):
    with localcontext() as ctx:
        ctx.prec = 45
        session, expected = _fixture(method, width, profile)
        result = summarize(session)
        per_operator = []
        for index, (op, directions) in enumerate(expected):
            for direction, numeric in zip(("warp", "weft"), directions):
                actual = result["operators"][op][direction]
                assert actual["status"] == "ready", actual["reasons"]
                assert [row["content"] for row in actual["categories"]] == pytest.approx([float(x) for x in numeric], abs=1e-10, rel=1e-12)
            if report_type == "fiber":
                per_operator.append(directions[0])
            else:
                w = D(70-index*10)/100
                per_operator.append([a*w+b*(1-w) for a, b in zip(*directions)])
        raw = [sum(values)/3 for values in zip(*per_operator)]
        rounded = [x.quantize(D("0.01"), rounding=ROUND_HALF_EVEN) for x in raw]
        rounded[0] = 100-sum(rounded[1:])
        if profile != "normal":
            rounded = [D(0)] + [x.quantize(D("0.1"), rounding=ROUND_HALF_EVEN) for x in rounded[1:]]
            rounded[0] = 100-sum(rounded[1:])
        actual = result[report_type]
        assert actual["status"] == "ready"
        assert [row["raw_content"] for row in actual["categories"]] == pytest.approx([float(x) for x in raw], abs=1e-10)
        assert [row["content"] for row in actual["categories"]] == pytest.approx([float(x) for x in rounded], abs=1e-10)


def test_recovered_real_gui_example_and_counts_do_not_change_measured_mean():
    s = ExperimentSession()
    op = s.add_operator("A")
    s.set_materials([MaterialSnapshot("w", "绵羊毛", 1.31), MaterialSnapshot("c", "山羊绒", 1.30)])
    s.add_measurement(15, "w"); s.add_measurement(30, "w"); s.add_measurement(18, "c")
    s.adjust_count("w", 1)
    rows = summarize(s)["operators"][op]["warp"]["categories"]
    assert rows[0]["mean"] == 22.5
    assert rows[0]["calculation_mean"] == 23.75
    assert rows[0]["total_count"] == 3
    assert rows[0]["content"] == pytest.approx(85.33239059467755, abs=1e-12)
    assert len(s.measurements) == 3


@pytest.mark.parametrize("method", [m for m, family in BRANCHES.items() if family in ("variance", "mean_square", "square_mean")])
def test_raw_statistics_and_single_sample_are_defined(method):
    s = ExperimentSession(method=method, grouped=False)
    op = s.add_operator("A")
    s.set_materials([MaterialSnapshot("a", "A", 1.2), MaterialSnapshot("b", "B", 1.4)])
    s.add_measurement(18, "a")
    s.add_measurement(14, "b"); s.add_measurement(30, "b")
    result = summarize(s)["operators"][op]["warp"]
    expected = _oracle(method, [("other", 1.2, [18], 0), ("other", 1.4, [14, 30], 0)], 2.5, False)
    if BRANCHES[method] == "variance":
        assert result["status"] == "not_ready"
        assert result["categories"][0]["content"] is None
        assert "sample_variance_requires_two" in [r["code"] for r in result["reasons"]]
        return
    assert result["status"] == "ready"
    assert result["categories"][0]["stddev"] is None
    assert [r["content"] for r in result["categories"]] == pytest.approx([float(x) for x in expected])


@pytest.mark.parametrize("method", ["SN/T 0756-1999", "FZ/T 32004-2009"])
@pytest.mark.parametrize("value", [12.5, 14.999999, 15, 37.5, 39.999999, 40, 42.5])
def test_ramie_group_boundary_and_coefficient_reference(method, value):
    s = ExperimentSession(method=method)
    op = s.add_operator("A")
    s.set_materials([MaterialSnapshot("r", "苎麻", 1.5), MaterialSnapshot("c", "棉", 1.5)])
    s.add_measurement(value, "r"); s.add_measurement(20, "c")
    expected = _oracle(method, [("ramie", 1.5, [value], 0), ("cotton", 1.5, [20], 0)], 2.5)
    actual = summarize(s)["operators"][op]["warp"]
    assert actual["categories"][0]["content"] == pytest.approx(float(expected[0]), abs=1e-10)


@pytest.mark.parametrize("count", [1487, 1488, 1489, 1499, 1500, 1501, 3000])
def test_no_worksheet_row_or_bucket_truncation(count):
    s = ExperimentSession(method="FZ/T 01101-2008", grouped=False)
    op = s.add_operator("A")
    s.set_materials([MaterialSnapshot("a", "A", 1), MaterialSnapshot("b", "B", 1)])
    for _ in range(count-1):
        s.add_measurement(20, "a")
    s.add_measurement(600, "a")
    s.add_measurement(20, "b")
    rows = summarize(s)["operators"][op]["warp"]["categories"]
    assert rows[0]["measured_count"] == count
    # q*N with c=0 is exactly the sum of all squared diameters.
    mass = (count-1)*400+360000
    assert rows[0]["content"] == pytest.approx(100*mass/(mass+400), abs=1e-10)
    s.configure(grouped=True)
    grouped = summarize(s)["operators"][op]["warp"]["categories"]
    mass = (count-1)*21.25**2+601.25**2
    assert grouped[0]["content"] == pytest.approx(100*mass/(mass+21.25**2), abs=1e-10)


def test_named_operator_denominator_and_material_identity():
    s = ExperimentSession(grouped=False, method="SN/T 1062-2002")
    a = s.add_operator("A")
    mats = [MaterialSnapshot("a", "A", 1), MaterialSnapshot("b", "B", 1)]
    s.set_materials(mats)
    s.add_measurement(20, "a"); s.add_measurement(20, "b")
    b = s.add_operator("B")
    s.set_materials(list(reversed(mats)))
    s.add_measurement(20, "b")
    result = summarize(s)["fiber"]
    rows = {row["id"]: row for row in result["categories"]}
    assert rows["a"]["raw_content"] == 25
    assert rows["b"]["raw_content"] == 75
    assert result["operator_count"] == 2


def test_rounding_half_tolerance_includes_both_boundaries():
    assert round_cu(1.225) == 1.22
    assert round_cu(1.235) == 1.24
    assert round_cu(-1.235) == -1.24
    for value in (1.225-1e-10, 1.225+1e-10, 1.235-1e-10, 1.235+1e-10):
        assert 1.2 < round_cu(value) < 1.3  # original VBA had an unassigned/default-zero boundary


def test_registry_has_all_eleven_recovered_method_labels():
    assert {m.label for m in METHODS} == set(BRANCHES)


def test_mt_first_component_uses_already_rounded_neighbouring_cells():
    s = ExperimentSession(grouped=False, profile="cu6mt", method="SN/T 1062-2002")
    s.add_operator("A")
    s.set_materials([MaterialSnapshot(key, key.upper(), 1) for key in "abc"])
    for key in "abc":
        s.add_measurement(20, key)
    values = [row["content"] for row in summarize(s)["fiber"]["categories"]]
    assert values == [33.4, 33.3, 33.3]
    assert math.fsum(values) == 100
