import time
from collections import defaultdict

import torch


class TimeProfiler:
    def __init__(self) -> None:
        self.stack = []
        self.count = defaultdict(lambda: 0)
        self.total_time = defaultdict(lambda: 0)
        self.children = defaultdict(set)
        self.enabled = False
        self.synchronize_cuda = True

    def start(self, key: str):
        if not self.enabled:
            return

        parent = None if len(self.stack) == 0 else self.stack[-1][0]
        key = "$$".join([s[0] for s in self.stack] + [key])
        self.children[parent].add(key)
        self.stack.append([key, time.time()])

    def end(self, end_key: str):
        if not self.enabled:
            return

        if len(self.stack) == 0:
            raise ValueError("Stack is empty, missing a call to start()")

        if self.synchronize_cuda:
            torch.cuda.synchronize()

        key, start_time = self.stack.pop()
        if key.split("$$")[-1] != end_key:
            raise ValueError(f"start key is {key}, but end key is {end_key}; check your start() calls")
        self.total_time[key] += time.time() - start_time
        self.count[key] += 1

    def get_results_string(self):
        if len(self.stack) > 0:
            raise ValueError("Not all profiling has ended!")

        columns = [[], [], [], []]  # key, total, count, avg

        keys = [[k, 0] for k in reversed(sorted(self.children[None]))]
        while len(keys) > 0:
            key, depth = keys.pop()
            keys += list([[k, depth + 1] for k in reversed(sorted(self.children[key]))])

            total_time = self.total_time[key]
            count = self.count[key]

            columns[0].append("  " * depth + key.split("$$")[-1])
            columns[1].append(f"{total_time:.2f}s")
            columns[2].append(f"{count}")
            columns[3].append(f"{total_time*1000/count:.2f}ms")

        lengths = [max([len(s) for s in col]) for col in columns]
        return "\n".join(
            [
                " ".join(
                    [f"{columns[0][i]:<{lengths[0]}}"]
                    + [f"{columns[j][i]:>{lengths[j]}}" for j in range(1, len(columns))]
                )
                for i in range(len(columns[0]))
            ]
        )


timer = TimeProfiler()
