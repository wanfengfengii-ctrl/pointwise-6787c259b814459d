"""独立正确性核验：不采信 lexicon 自身的内部结论。

参考模型是 Python set；另独立构造参考集合的最小无环确定自动机
（朴素前缀树 + 自底向上 Myhill-Nerode 等价合并），逐项比对：
  1. 规定剧本（ab/cb/ad/cd）
  2. 随机增删模糊测试（含空串、NUL、多字节 Unicode）
  3. 长词（验证迭代实现无递归深度问题）
  4. 返回布尔值语义
  5. 导出图的结构、语言、无环/确定性、JSON 往返、最小状态数
  6. lexicon 内部不变量
"""

from __future__ import annotations

import json
import os
import random
import sys

# -I 隔离模式（本环境隐含 -P）下显式加入脚本自身目录（可信代码），
# 同时仍阻止 cwd 中植入的同名模块被导入。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lexicon


# --------------------------------------------------------------------- #
# 独立参考模型 1：集合 + 朴素最小无环 DFA 构造
# --------------------------------------------------------------------- #
def reference_minimal_state_count(words: set[bytes]) -> int:
    """从字节词集合独立构造最小无环 DFA，返回其状态数。"""
    trie: dict[int, dict] = {0: {"final": False, "edges": {}}}
    next_id = 1
    for w in words:
        cur = 0
        for b in w:
            edges = trie[cur]["edges"]
            if b not in edges:
                edges[b] = next_id
                trie[next_id] = {"final": False, "edges": {}}
                next_id += 1
            cur = edges[b]
        trie[cur]["final"] = True

    # 迭代后序（无环图），自底向上按 Myhill-Nerode 签名合并
    remap: dict[int, int] = {}
    sig_to_canon: dict[tuple, int] = {}
    canon_count = 0
    stack = [(0, False)]
    postorder: list[int] = []
    while stack:
        nid, processed = stack.pop()
        if processed:
            postorder.append(nid)
            continue
        stack.append((nid, True))
        for child in trie[nid]["edges"].values():
            stack.append((child, False))

    for nid in postorder:
        node = trie[nid]
        sig = (
            node["final"],
            tuple(sorted((b, remap[c]) for b, c in node["edges"].items())),
        )
        if sig not in sig_to_canon:
            sig_to_canon[sig] = canon_count
            canon_count += 1
        remap[nid] = sig_to_canon[sig]

    # 注：根的规范编号不要求为 0——例如集合仅含空串时，
    # 根签名 (True, ()) 可能先由叶节点登记。canon_count 才是最小状态数。
    return canon_count


def exported_language(graph: dict) -> set[bytes]:
    """对导出图做独立 DFS，枚举出它接纳的全部字节串。"""
    nodes = graph["nodes"]
    result: set[bytes] = set()
    stack: list[tuple[int, bytes]] = [(graph["root"], b"")]
    while stack:
        nid, prefix = stack.pop()
        node = nodes[nid]
        if node["final"]:
            result.add(prefix)
        for byte, target in node["edges"]:
            stack.append((target, prefix + bytes([byte])))
    return result


def check_export_structure(graph: dict) -> None:
    assert set(graph.keys()) == {"root", "nodes"}
    assert graph["root"] == 0
    nodes = graph["nodes"]
    assert isinstance(nodes, list) and nodes
    for i, node in enumerate(nodes):
        assert set(node.keys()) == {"final", "edges"}
        assert isinstance(node["final"], bool)
        labels = [e[0] for e in node["edges"]]
        assert labels == sorted(labels), "边标签必须升序"
        assert len(labels) == len(set(labels)), "边标签不得重复"
        for byte, target in node["edges"]:
            assert isinstance(byte, int) and 0 <= byte <= 255
            assert isinstance(target, int) and 0 <= target < len(nodes)
    # 无环 + 全可达（标准三色 DFS：灰=在当前 DFS 栈，黑=已完成）
    seen = {0}
    color = {0: 1}
    dfs_stack = [(0, iter(nodes[0]["edges"]))]
    while dfs_stack:
        nid, it = dfs_stack[-1]
        nxt = next(it, None)
        if nxt is None:
            color[nid] = 2
            dfs_stack.pop()
            continue
        _, target = nxt
        assert color.get(target, 0) != 1, "图中存在环"
        if target not in seen:
            seen.add(target)
            color[target] = 1
            dfs_stack.append((target, iter(nodes[target]["edges"])))
    assert seen == set(range(len(nodes))), "存在不可达节点"
    # JSON 可往返
    again = json.loads(json.dumps(graph, ensure_ascii=False))
    assert again == graph


# --------------------------------------------------------------------- #
# 针对一个参考集合做全量比对
# --------------------------------------------------------------------- #
def cross_check(index: lexicon.Index, reference: set[str]) -> None:
    # 1. contains 精确匹配
    encoded = {w.encode("utf-8") for w in reference}
    for w in reference:
        assert index.contains(w), f"漏纳: {w!r}"
    # 采样负例（含前缀陷阱）
    negatives = {b"", b"\x00", b"a", b"ab", b"cd", b"ad", b"cb",
                 b"abc", b"ABD", "中文".encode(), "🙂".encode()}
    for w in encoded:
        # 翻转一个字节、追加一个字节，制造近似负例
        if w:
            flipped = bytes([w[0] ^ 0x01]) + w[1:]
            negatives.add(flipped)
        negatives.add(w + b"\x00")
    for raw in negatives:
        try:
            w = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert index.contains(w) == (w in reference), f"误判: {w!r}"

    graph = index.export()
    check_export_structure(graph)
    # 2. 接纳语言恰好等于参考集合的 UTF-8 编码
    assert exported_language(graph) == encoded
    # 3. 状态数与独立构造的最小 DFA 一致
    minimal = reference_minimal_state_count(encoded)
    assert len(graph["nodes"]) == minimal, (
        f"非最小: 实际 {len(graph['nodes'])} 状态, 最小应为 {minimal}"
    )
    # 4. 内部不变量
    index.debug_check_invariants()


# --------------------------------------------------------------------- #
# 用例 1：规定剧本
# --------------------------------------------------------------------- #
def scenario_scripted() -> None:
    idx = lexicon.Index()
    g0 = idx.export()
    assert g0 == {"root": 0, "nodes": [{"final": False, "edges": []}]}
    assert idx.contains("") is False

    assert idx.add("ab") is True
    assert idx.add("ab") is False          # 重复添加不变
    assert idx.add("cb") is True
    assert idx.discard("ab") is True
    assert idx.discard("ab") is False      # 删除不存在的词不变
    assert idx.discard("nonexistent") is False
    idx.add("ad")

    assert idx.contains("cb") is True
    assert idx.contains("ad") is True
    assert idx.contains("cd") is False
    assert idx.contains("ab") is False
    assert idx.contains("a") is False      # 前缀不算命中
    cross_check(idx, {"cb", "ad"})

    # 空串作为词
    assert idx.add("") is True
    assert idx.contains("") is True
    assert idx.add("") is False
    cross_check(idx, {"cb", "ad", ""})
    assert idx.discard("") is True
    assert idx.contains("") is False
    cross_check(idx, {"cb", "ad"})

    # 清空后回到“仅非接纳根”
    assert idx.discard("cb") is True
    assert idx.discard("ad") is True
    assert idx.export() == {"root": 0,
                            "nodes": [{"final": False, "edges": []}]}
    print("  [ok] 规定剧本（ab/cb -> 删ab -> ad，cd 不命中；空串；清空）")


# --------------------------------------------------------------------- #
# 用例 2：随机模糊测试
# --------------------------------------------------------------------- #
def random_word(rng: random.Random) -> str:
    kind = rng.randrange(7)
    if kind == 0:
        return ""
    alphabets = [
        [97, 98, 99, 100],                       # ASCII a-d
        [0, 1, 97, 255],                         # 含 NUL 与边界字节
        [0x4E2D, 0x6587, 0x1F642, 0xE9],         # 多字节 Unicode
        [0x80, 0x7FF, 0x800, 0xFFFF],            # 2/3 字节边界
    ]
    alpha = alphabets[min(kind - 1, 3)] if kind <= 4 else alphabets[rng.randrange(4)]
    length = rng.randrange(0, 12)
    while True:
        s = "".join(chr(rng.choice(alpha)) for _ in range(length))
        try:
            s.encode("utf-8")  # 排除代理对
        except UnicodeEncodeError:
            continue
        return s


def scenario_fuzz(rounds: int, ops_per_round: int, seed: int) -> None:
    rng = random.Random(seed)
    for r in range(rounds):
        idx = lexicon.Index()
        reference: set[str] = set()
        vocab = [random_word(rng) for _ in range(rng.randrange(1, 24))]
        for _ in range(ops_per_round):
            w = rng.choice(vocab)
            if rng.random() < 0.55:
                changed = idx.add(w)
                assert changed == (w not in reference)
                reference.add(w)
            else:
                changed = idx.discard(w)
                assert changed == (w in reference)
                reference.discard(w)
            if rng.random() < 0.25:
                cross_check(idx, reference)
        cross_check(idx, reference)
    print(f"  [ok] 随机模糊：{rounds} 轮 × {ops_per_round} 操作"
          f"（空串/NUL/多字节 Unicode，seed={seed}）")


# --------------------------------------------------------------------- #
# 用例 3：长词（迭代实现，不得触发递归限制）
# --------------------------------------------------------------------- #
def scenario_long_words() -> None:
    idx = lexicon.Index()
    long_a = "a" * 8000
    long_b = "a" * 7999 + "b"
    assert idx.add(long_a) is True
    assert idx.add(long_b) is True
    assert idx.contains(long_a) and idx.contains(long_b)
    cross_check(idx, {long_a, long_b})
    assert idx.discard(long_a) is True
    cross_check(idx, {long_b})
    assert idx.discard(long_b) is True
    assert idx.export() == {"root": 0,
                            "nodes": [{"final": False, "edges": []}]}

    # 多字节长词：字节长度远超字符长度
    uni = "中" * 4000
    assert idx.add(uni) is True
    assert idx.contains(uni)
    assert idx.discard(uni) is True
    print("  [ok] 长词：8000 字符 ASCII 与 12000 字节 Unicode 增删查")


# --------------------------------------------------------------------- #
# 用例 4：反复增删的有界性（不积累废状态）
# --------------------------------------------------------------------- #
def scenario_churn() -> None:
    idx = lexicon.Index()
    words = [f"url/{i:04d}/item" for i in range(200)]
    for w in words:
        idx.add(w)
    baseline = len(idx.export()["nodes"])
    rng = random.Random(42)
    for _ in range(4000):
        w = rng.choice(words)
        if idx.contains(w):
            idx.discard(w)
        else:
            idx.add(w)
    for w in words:
        idx.add(w)
    assert len(idx.export()["nodes"]) == baseline, "反复增删后状态数膨胀"
    cross_check(idx, set(words))
    print(f"  [ok] 4000 次反复增删后状态数仍为 {baseline}（无废状态积累）")


def scenario_large_stress() -> None:
    """较大词表上的随机顺序压力测试：每步全量比对参考集合。"""
    rng = random.Random(99)
    vocab = {
        f"https://todo.example/{rng.randrange(16):x}/"
        f"{rng.getrandbits(32):08x}"
        for _ in range(1500)
    }
    vocab = list(vocab)
    idx = lexicon.Index()
    reference: set[str] = set()
    for step in range(8000):
        w = rng.choice(vocab)
        if rng.random() < 0.6:
            assert idx.add(w) == (w not in reference)
            reference.add(w)
        else:
            assert idx.discard(w) == (w in reference)
            reference.discard(w)
        if step % 1000 == 999:
            cross_check(idx, reference)
    # 最终再做一次完整比对（语言、结构、最小状态数、不变量）
    cross_check(idx, reference)
    print(f"  [ok] 大规模压力：8000 次随机增删、{len(reference)} 个存活词，"
          f"状态数与独立最小 DFA 一致")


def main() -> int:
    smoke = "--smoke" in sys.argv
    print("lexicon.Index 独立核验")
    scenario_scripted()
    if smoke:
        scenario_fuzz(20, 200, seed=20261114)
    else:
        scenario_fuzz(60, 500, seed=20261114)
        scenario_large_stress()
    scenario_long_words()
    scenario_churn()
    print("全部核验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
