"""H173 OOD scoring may be consumed only once, even by racing processes."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from h173.native_reward import start_consumption_lock


def test_owner_ood_lock_is_exclusive_and_never_clobbered(tmp_path):
    lock = tmp_path / "private-h173" / "H173_UNSEEN_ORIGINAL_OOD_PERMANENTLY_CONSUMED.lock"
    receipt = start_consumption_lock(lock, "a" * 64, "b" * 64)
    first = lock.read_bytes()
    assert receipt["total_browser_rewards_so_far"] == 0
    assert json.loads(first)["status"] == "H173_ONE_SHOT_STARTED_AND_IRREVOCABLY_CONSUMED"
    with pytest.raises(ValueError, match="already started or permanently consumed"):
        start_consumption_lock(lock, "c" * 64, "d" * 64)
    assert lock.read_bytes() == first


def test_two_simultaneous_owner_ood_scorers_cannot_both_start(tmp_path):
    lock = tmp_path / "private-h173" / "H173_UNSEEN_ORIGINAL_OOD_PERMANENTLY_CONSUMED.lock"
    start = Barrier(3)

    def attempt():
        start.wait()
        try:
            return start_consumption_lock(lock, "a" * 64, "b" * 64)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(attempt)
        second = pool.submit(attempt)
        start.wait()
        results = [first.result(timeout=10), second.result(timeout=10)]

    assert sum(result is not None for result in results) == 1
    assert json.loads(lock.read_text("utf8"))["status"] == (
        "H173_ONE_SHOT_STARTED_AND_IRREVOCABLY_CONSUMED"
    )
