import time

class Timer:
    def __init__(self, label="Task"):
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc, tb):
        print(f"[⏱] {self.label}: {time.time() - self.start:.2f}s")
