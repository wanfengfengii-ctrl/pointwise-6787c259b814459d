"""反复增删资源实测：时间、内存、压缩率与局部性。

仅使用标准库（time / tracemalloc / resource）。
输出可复现，随机种子固定。
"""

from __future__ import annotations

import os
import random
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lexicon


def measure(label: str) -> None:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  {label:<34} 峰值RSS={rss_kb / 1024:8.1f} MiB")


def main() -> int:
    rng = random.Random(7)

    # ---------- 1. 构建吞吐：5 万 URL 风格词 ----------
    n = 50_000
    words = [f"https://todo.example/{rng.randrange(8):x}/"
             f"item/{rng.getrandbits(40):010x}" for _ in range(n)]
    words = list(dict.fromkeys(words))  # 去重但保持顺序

    idx = lexicon.Index()
    t0 = time.perf_counter()
    added = 0
    for w in words:
        added += int(idx.add(w))
    t_build = time.perf_counter() - t0
    assert added == len(words)
    states = len(idx.export()["nodes"])
    raw_bytes = sum(len(w.encode("utf-8")) for w in words)

    print("== 构建吞吐 ==")
    print(f"  词数={len(words)}  原始字节={raw_bytes/1024/1024:.2f} MiB")
    print(f"  耗时={t_build:.3f}s  "
          f"平均={t_build/len(words)*1e6:.2f} μs/次 add")
    print(f"  自动机状态数={states}（含根）")

    # 朴素未压缩前缀树状态数（独立计算，仅用于对比压缩率）
    trie_states = 1
    seen: set[tuple] = set()
    for w in words:
        cur: tuple = ()
        for b in w.encode("utf-8"):
            cur = cur + (b,)
            if cur not in seen:
                seen.add(cur)
                trie_states += 1
    print(f"  未压缩前缀树状态数={trie_states}  "
          f"压缩率={trie_states/states:.1f}x  "
          f"每词摊薄={states/len(words):.3f} 状态")

    # ---------- 2. 稳态增删延迟（局部操作，不随词集规模线性增长）----------
    short_words = [f"url/{i:05d}" for i in range(2000)]
    small = lexicon.Index()
    for w in short_words:
        small.add(w)
    vocab = short_words + [f"url/{rng.randrange(2000):05d}" for _ in range(500)]

    def median(xs: list[float]) -> float:
        xs = sorted(xs)
        return xs[len(xs) // 2]

    samples_add: list[float] = []
    samples_del: list[float] = []
    rounds = 20_000
    t0 = time.perf_counter()
    for _ in range(rounds):
        w = rng.choice(vocab)
        if small.contains(w):
            t1 = time.perf_counter()
            small.discard(w)
            samples_del.append(time.perf_counter() - t1)
        else:
            t1 = time.perf_counter()
            small.add(w)
            samples_add.append(time.perf_counter() - t1)
    elapsed = time.perf_counter() - t0
    print("\n== 稳态增删延迟（2000 词规模下 2 万次混合操作）==")
    print(f"  总耗时={elapsed:.3f}s  吞吐={rounds/elapsed:,.0f} 次/秒")
    print(f"  add     中位延迟={median(samples_add)*1e6:7.2f} μs "
          f"最大={max(samples_add)*1e6:8.2f} μs（n={len(samples_add)}）")
    print(f"  discard 中位延迟={median(samples_del)*1e6:7.2f} μs "
          f"最大={max(samples_del)*1e6:8.2f} μs（n={len(samples_del)}）")

    # ---------- 3. 局部性证据：延迟不随总词数膨胀 ----------
    print("\n== 局部性：固定长词增删延迟 vs 词集规模 ==")
    probe = "url/99999"
    for scale in (1_000, 10_000, 50_000):
        big = lexicon.Index()
        pool = [f"pad/{i:06d}" for i in range(scale)]
        for w in pool:
            big.add(w)
        # 预热
        big.add(probe)
        big.discard(probe)
        t1 = time.perf_counter()
        for _ in range(200):
            big.add(probe)
            big.discard(probe)
        dt = (time.perf_counter() - t1) / 400
        print(f"  词集规模={scale:6d}  探针增删平均={dt*1e6:7.2f} μs/次")

    # ---------- 4. 反复增删有界性：废状态不积累 ----------
    print("\n== 有界性：10 万次反复增删 ==")
    churn_words = [f"churn/{i:04d}" for i in range(500)]
    ch = lexicon.Index()
    for w in churn_words:
        ch.add(w)
    base_states = len(ch.export()["nodes"])
    base_nodes = len(ch._nodes)          # type: ignore[attr-defined]
    base_registry = len(ch._registry)    # type: ignore[attr-defined]
    t0 = time.perf_counter()
    for _ in range(100_000):
        w = rng.choice(churn_words)
        if ch.contains(w):
            ch.discard(w)
        else:
            ch.add(w)
    t_churn = time.perf_counter() - t0
    for w in churn_words:
        ch.add(w)
    end_states = len(ch.export()["nodes"])
    end_nodes = len(ch._nodes)           # type: ignore[attr-defined]
    end_registry = len(ch._registry)     # type: ignore[attr-defined]
    print(f"  10 万次操作耗时={t_churn:.3f}s "
          f"({100_000/t_churn:,.0f} 次/秒)")
    print(f"  状态数 {base_states} -> {end_states}（必须相等）")
    print(f"  存活节点 {base_nodes} -> {end_nodes}；"
          f"等价索引项 {base_registry} -> {end_registry}（必须相等）")
    assert (base_states, base_nodes, base_registry) == (
        end_states, end_nodes, end_registry)

    # ---------- 5. 内存实测 ----------
    print("\n== 内存（tracemalloc 跟踪 Python 对象）==")
    tracemalloc.start()
    mem_idx = lexicon.Index()
    for w in words:
        mem_idx.add(w)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  5 万词索引：当前={current/1024/1024:.2f} MiB "
          f"峰值={peak/1024/1024:.2f} MiB")
    measure("进程级")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
