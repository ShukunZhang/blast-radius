"""Replay process-pool jobs in order with the same pickle boundaries."""
import hashlib
import pickle
from concurrent.futures import Future


class InlinePool:
    baseline_jobs = {}
    recording = None
    impact = None

    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, function, /, *args, **kwargs):
        future = Future()
        try:
            packed = pickle.dumps((function, args, kwargs))
            cacheable = function.__module__ == "tests.test_optimizer" and function.__name__ == "parse_and_optimize"
            key = hashlib.sha256(packed).hexdigest() if cacheable else ""
            cached = self.baseline_jobs.get(key)
            if cached is not None and self.impact is not None and not type(self).impact(cached[1]):
                payload = bytes.fromhex(cached[0])
            else:
                fn, args, kwargs = pickle.loads(packed)
                if cacheable and (record := type(self).recording) is not None:
                    payload = record(fn, args, kwargs, key)
                else:
                    payload = pickle.dumps(fn(*args, **kwargs))
            result = pickle.loads(payload)
        except Exception as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        return future

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
