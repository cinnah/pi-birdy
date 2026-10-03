import dataclasses
import threading

from birdy.live import LiveConfig


def test_get_returns_initial_config(config):
    live = LiveConfig(config)
    assert live.get() is config
    assert live.version == 0


def test_replace_bumps_version_and_swaps(config):
    live = LiveConfig(config)
    new = dataclasses.replace(config, motion=dataclasses.replace(config.motion, min_area=999))
    live.replace(new)
    assert live.version == 1
    assert live.get().motion.min_area == 999
    assert live.get() is new


def test_concurrent_readers_and_writers_never_see_torn_state(config):
    live = LiveConfig(config)
    stop = threading.Event()
    errors: list[Exception] = []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            live.replace(dataclasses.replace(config, motion=dataclasses.replace(config.motion, min_area=i)))

    def reader():
        try:
            while not stop.is_set():
                cfg = live.get()
                assert cfg.motion.min_area >= 0
                _ = live.version
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    threading.Event().wait(0.2)
    stop.set()
    for t in threads:
        t.join(2)
    assert errors == []
    assert live.version > 0
