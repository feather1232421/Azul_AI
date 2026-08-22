from types import SimpleNamespace

import get_dataset


class FakeKernel32:
    def __init__(self):
        self.calls = []

    def SetThreadExecutionState(self, flags):
        self.calls.append(flags)
        return 1


def test_windows_sleep_block_keeps_display_timeout_available(monkeypatch):
    kernel32 = FakeKernel32()
    monkeypatch.setattr(get_dataset.sys, "platform", "win32")
    monkeypatch.setattr(
        get_dataset.ctypes,
        "windll",
        SimpleNamespace(kernel32=kernel32),
    )

    assert get_dataset.set_system_sleep_blocked(True)
    assert get_dataset.set_system_sleep_blocked(False)
    assert kernel32.calls == [
        get_dataset.ES_CONTINUOUS | get_dataset.ES_SYSTEM_REQUIRED,
        get_dataset.ES_CONTINUOUS,
    ]


def test_sleep_block_is_a_noop_outside_windows(monkeypatch):
    monkeypatch.setattr(get_dataset.sys, "platform", "linux")
    assert not get_dataset.set_system_sleep_blocked(True)
