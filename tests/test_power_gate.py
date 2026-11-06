from datetime import datetime

from clipfarm.core import power


class Cfg:
    def __init__(self, **kv):
        self.kv = kv

    def get(self, k, d=None):
        return self.kv.get(k, d)


def test_parse_adapter_watts():
    assert power.parse_adapter_watts(" Wattage = 140W\n Current = 4990mA") == 140
    assert power.parse_adapter_watts("") is None and power.parse_adapter_watts("nothing") is None


def test_window_overnight_and_across_midnight():
    c = Cfg(**{"power.heavy_window": ["00:30", "06:00"]})
    assert power.in_window(c, datetime(2026, 10, 22, 0, 30)) and power.in_window(c, datetime(2026, 10, 22, 5, 59))
    assert not power.in_window(c, datetime(2026, 10, 22, 0, 29)) and not power.in_window(c, datetime(2026, 10, 22, 6, 0)) and not power.in_window(c, datetime(2026, 10, 22, 15, 0))
    x = Cfg(**{"power.heavy_window": ["22:00", "02:00"]})
    assert power.in_window(x, datetime(2026, 10, 22, 23, 0)) and power.in_window(x, datetime(2026, 10, 22, 1, 0)) and not power.in_window(x, datetime(2026, 10, 22, 12, 0))
    assert power.in_window(Cfg(), datetime(2026, 10, 22, 12, 0))                             # no window = always


def test_heavy_ok_needs_ac_charger_watts_and_window(monkeypatch):
    c = Cfg(**{"power.heavy_window": ["00:30", "06:00"], "power.min_adapter_watts": 90})
    night, day = datetime(2026, 10, 22, 2, 0), datetime(2026, 10, 22, 14, 0)
    monkeypatch.setattr(power, "on_ac_power", lambda: True)
    monkeypatch.setattr(power, "adapter_watts", lambda: 140)
    assert power.heavy_ok(c, night) == (True, "ok")
    assert power.heavy_ok(c, day)[0] is False
    monkeypatch.setattr(power, "adapter_watts", lambda: 65)
    ok, why = power.heavy_ok(c, night)
    assert not ok and "65 W" in why                                                           # a weak dock cannot carry a full render
    monkeypatch.setattr(power, "adapter_watts", lambda: None)
    assert power.heavy_ok(c, night)[0] is False
    monkeypatch.setattr(power, "on_ac_power", lambda: False)
    assert power.heavy_ok(c, night) == (False, "on battery power")
