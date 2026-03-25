# timing.py
import time
from functools import wraps

def timeit(label="Execution"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            out = func(*args, **kwargs)
            t2 = time.time()
            print(f"[TIME] {label}: {t2 - t1:.4f}s")
            return out
        return wrapper
    return decorator
