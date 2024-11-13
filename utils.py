from time import time
def time_benchmark(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        name = func.__name__
        print(f"{name} took {time() - t0} seconds")
        return res
    return wrapper