"""Short, owned thread-only CPU contention trial for the H40 fake-child matrix.

All stress threads are joined. No external process termination, native UI, GPU
or real model. Designed for diagnosis, not a stable unit-test gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

from evaluation.h40_cpu_lifecycle_stress import run


def loaded_trial(output:Path, *, rounds:int=8, workers:int=2)->dict:
    if type(workers) is not int or workers not in (1,2,3):
        raise ValueError("Only bounded 1..3 CPU stress workers")
    stop=threading.Event()
    threads=[]
    def busy():
        while not stop.is_set():
            hashlib.pbkdf2_hmac("sha256",b"hala-h40-local-fixture",b"fixed-bounded-salt",50000)
    try:
        for _ in range(workers):
            thread=threading.Thread(target=busy,daemon=True)
            thread.start()
            threads.append(thread)
        result=run(output,rounds=rounds)
        result["stress_worker_threads"]=workers
        result["cpu_stress_fixture"]="thread-local pbkdf2_sha256; no I/O or GPU"
        return result
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("H40 owned stress thread not joined")

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--rounds",type=int,default=8)
    a.add_argument("--workers",type=int,default=2)
    n=a.parse_args()
    print(json.dumps(loaded_trial(n.output,rounds=n.rounds,workers=n.workers),sort_keys=True))
