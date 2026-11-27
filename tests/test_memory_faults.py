import os
import tempfile

import pytest

from proofpilot.faults import corrupt_value, make_fault_hook
from proofpilot.memory import MemoryStore, memory_key
from proofpilot.tools.base import ToolOutput


def test_key_ignores_argument_order_but_not_values():
    assert memory_key("t", {"a": 1, "b": 2}) == memory_key("t", {"b": 2, "a": 1})
    assert memory_key("t", {"a": 1}) != memory_key("t", {"a": 2})
    assert memory_key("t", {"a": 1}) != memory_key("u", {"a": 1})


def test_put_get_roundtrip_and_hit_counter():
    m = MemoryStore()
    assert m.get("t", {"a": 1}) is None
    m.put("t", {"a": 1}, {"x": [1, 2]}, "disp", "meth", "det")
    got = m.get("t", {"a": 1})
    assert got == {"value": {"x": [1, 2]}, "display": "disp", "method": "meth", "detail": "det"}
    m.get("t", {"a": 1})
    assert m.count() == 1 and m.entries()[0]["hits"] == 2


def test_overwrite_keeps_hits_and_clear():
    m = MemoryStore()
    m.put("t", {}, 1, "a", "m", "d")
    m.get("t", {})
    m.put("t", {}, 2, "b", "m", "d")
    assert m.count() == 1 and m.entries()[0]["hits"] == 1 and m.get("t", {})["value"] == 2
    m.clear()
    assert m.count() == 0


def test_persistence_across_instances():
    path = os.path.join(tempfile.mkdtemp(), "mem.db")
    MemoryStore(path).put("t", {"k": 1}, "v", "d", "m", "x")
    assert MemoryStore(path).get("t", {"k": 1})["value"] == "v"


@pytest.mark.parametrize("value", [3, 2.5, "x**2", [1, 2], {"a": 1.0, "b": 2}, True])
def test_corrupt_value_always_changes_the_value(value):
    assert corrupt_value(value) != value


def test_fault_hook_is_deterministic_and_rate_bounds():
    out = ToolOutput(5, "5")
    a = [make_fault_hook(0.5, seed=1)("t", out).value for _ in range(1)]
    b = [make_fault_hook(0.5, seed=1)("t", out).value for _ in range(1)]
    assert a == b
    never = make_fault_hook(0.0)
    always = make_fault_hook(1.0)
    assert all(never("t", out).value == 5 for _ in range(50))
    assert all(always("t", out).value != 5 for _ in range(50))
