"""真实演示：规定操作序列与每一步的导出图。"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lexicon


def show(index: lexicon.Index, title: str) -> None:
    graph = index.export()
    print(f"\n=== {title}：{len(graph['nodes'])} 个状态 ===")
    print(json.dumps(graph, ensure_ascii=False, separators=(",", ":")))


def main() -> int:
    idx = lexicon.Index()
    show(idx, "空索引")

    steps = [
        ("add('ab')", lambda: idx.add("ab")),
        ("add('cb')", lambda: idx.add("cb")),
        ("discard('ab')", lambda: idx.discard("ab")),
        ("add('ad')", lambda: idx.add("ad")),
    ]
    for desc, op in steps:
        changed = op()
        show(idx, f"{desc} -> 返回 {changed}")

    print("\n=== contains 精确核验 ===")
    for word in ("cb", "ad", "cd", "ab", "a", ""):
        print(f"  contains({word!r:5}) = {idx.contains(word)}")

    print("\n=== 布尔返回值语义（重复操作图不变）===")
    print(f"  add('ad') 再次  -> {idx.add('ad')}")
    print(f"  discard('zzz') -> {idx.discard('zzz')}")
    show(idx, "重复操作后图保持不变")

    print("\n=== Unicode / 空串 / NUL 字节（按 UTF-8 字节处理，不做规范化）===")
    for w in ("", "https://例子.test/路径", "a\x00b", "🙂"):
        idx.add(w)
        print(f"  add({w!r}) -> contains = {idx.contains(w)}; "
              f"UTF-8 = {w.encode('utf-8')!r}")
    show(idx, "加入 Unicode/空串/NUL 后")

    idx.debug_check_invariants()
    print("\n内部不变量核验通过；演示结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
