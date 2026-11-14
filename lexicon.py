"""lexicon — 紧凑去重的字节级最小无环确定自动机索引。

Index 在 UTF-8 字节串集合上维护一个 **最小无环确定有限自动机**
（minimal acyclic DFA / DAWG），支持词的增量增删与精确查询。

数据结构
========
每个节点：
* final：是否接纳状态；
* edges：字节标签(0..255) -> 子节点号（确定性：标签唯一）；
* parents：父节点号 -> 来自该父节点的入边数。同一父节点可经
  不同字节标签有多条边指向本节点，故用计数映射而非集合。

refcount(v) = 全部入边数；根另有 1 个隐式引用，永不被回收。

等价索引 registry：结构签名 -> 该等价类的规范存活节点号。
签名 = (final, 排序后的 (字节标签, 目标节点号) 序列)。
Myhill-Nerode 定理：两状态等价当且仅当签名相同。

增量增删算法（私有路径重建）
============================
每次 add/discard 只处理该词的字节路径 path[] 与每跳标签
labels[]，分四步：

1. 遍历：沿字节边走，走不通即停。
2. 私有化 _isolate_path：从路径上“最深的共享节点”（入边数>1）
   开始，把它及其路径后继逐个克隆，并把对应父边改指克隆。
   克隆浅拷贝出边表并在子节点登记新父关系；原共享节点的其它
   入边（属于别的词）保持不变，因此别的词语言不受影响。
3. 结构修改：
   * add：扩展缺失后缀，或翻转末端 final；
   * discard：翻转末端 final 为 False。
4. 自底向上重算 _rebuild_path：沿私有路径从深到浅，先剪枝
   “非接纳且无出边”的死状态（有限语言最小 DFA 不含可达死状态），
   再重算签名，命中 registry 规范节点就把父边改指过去（哈希
   共享），否则自行登记。根永不参与合并。
5. _sweep：回收父集合为空的非根节点，借助 parents 表同步清理
   指回它们的前向边，并沿出边局部级联。

为什么正确且最小
================
* 私有化保证所有修改只作用于当前词的私有副本，共享结构只读；
* 重算后两个存活节点签名相同即被合并，不同签名绝不合并，由
  Myhill-Nerode 保证状态数最少；
* parents 表让每条入边的建立/消亡都精确记账，共享节点在仍有
  其它父时计数不归零，不会被误删，因而不产生悬空边。

复杂度：每次操作的时间与临时分配为 O(|word|)（按 UTF-8 字节
计），registry 为哈希表均摊 O(1)，垃圾回收只在本次影响闭包内
传播。不枚举词集、不扫描全图、不保留完整词集或未压缩前缀树、
不留历史废状态。

仅使用 Python 标准库。
"""

from __future__ import annotations

from collections import deque

__all__ = ["Index"]


class _Node:
    """图节点（__slots__ 降低长运行场景的内存开销）。"""

    __slots__ = ("nid", "final", "edges", "parents")

    def __init__(self, nid: int, final: bool = False) -> None:
        self.nid = nid
        self.final = final
        self.edges: dict[int, int] = {}
        self.parents: dict[int, int] = {}

    @property
    def refcount(self) -> int:
        total = sum(self.parents.values())
        return total + (1 if self.nid == 0 else 0)

    def signature(self) -> tuple:
        return (self.final, tuple(sorted(self.edges.items())))


class Index:
    """UTF-8 字节词集合的最小无环确定自动机索引。"""

    def __init__(self) -> None:
        self._root = _Node(0)
        self._nodes: dict[int, _Node] = {0: self._root}
        self._next_id = 1
        self._registry: dict[tuple, int] = {self._root.signature(): 0}

    # ------------------------------------------------------------------ #
    # 底层原语：同时维护正向边、反向 parents 与等价索引
    # ------------------------------------------------------------------ #
    def _new_node(self, final: bool = False) -> _Node:
        nid = self._next_id
        self._next_id += 1
        node = _Node(nid, final)
        self._nodes[nid] = node
        return node

    def _detach(self, node: _Node) -> None:
        """节点结构即将改变：仅当索引项指向自身时摘除。"""
        if self._registry.get(node.signature()) == node.nid:
            self._registry.pop(node.signature())

    def _link(self, parent: _Node, byte: int, child: int) -> None:
        """新建边 parent --byte--> child（该标签此前无边）。"""
        self._detach(parent)
        assert byte not in parent.edges
        parent.edges[byte] = child
        child_node = self._nodes[child]
        child_node.parents[parent.nid] = \
            child_node.parents.get(parent.nid, 0) + 1

    def _unlink(self, parent: _Node, byte: int) -> int:
        """删除边 parent --byte--> child，返回 child 号。"""
        self._detach(parent)
        child = parent.edges.pop(byte)
        child_node = self._nodes[child]
        count = child_node.parents.get(parent.nid, 0) - 1
        if count <= 0:
            child_node.parents.pop(parent.nid, None)
        else:
            child_node.parents[parent.nid] = count
        return child

    def _relink(self, parent: _Node, byte: int, new_child: int) -> int:
        """把 parent --byte--> old 改指为 new_child，返回 old 号。"""
        self._detach(parent)
        old = parent.edges[byte]
        assert old != new_child
        parent.edges[byte] = new_child
        old_node = self._nodes[old]
        count = old_node.parents.get(parent.nid, 0) - 1
        if count <= 0:
            old_node.parents.pop(parent.nid, None)
        else:
            old_node.parents[parent.nid] = count
        new_node = self._nodes[new_child]
        new_node.parents[parent.nid] = \
            new_node.parents.get(parent.nid, 0) + 1
        return old

    def _clone(self, node: _Node) -> _Node:
        """浅拷贝节点（含出边表）。克隆的每条出边构成对子节点的
        真实父关系，故在子节点 parents 中登记克隆（+1）。"""
        clone = self._new_node(node.final)
        clone.edges = dict(node.edges)
        for child in clone.edges.values():
            child_node = self._nodes[child]
            child_node.parents[clone.nid] = \
                child_node.parents.get(clone.nid, 0) + 1
        return clone

    def _sweep(self, seeds: list[int]) -> None:
        """回收父集合为空的非根节点，仅在本次影响闭包内级联。

        节点无父即不可达。删除前借助 parents 反向表摘除所有指向
        它的前向边（杜绝悬空边）；其出边所指子节点同步失父，若
        也归零则级联。仍有其它父的共享节点不归零，天然受保护。
        """
        stack = list(dict.fromkeys(seeds))
        dead: set[int] = set()
        while stack:
            nid = stack.pop()
            if nid in dead or nid == self._root.nid:
                continue
            node = self._nodes.get(nid)
            if node is None or node.parents:
                continue
            dead.add(nid)
            # 摘除所有父节点边表中指向本节点的前向边。
            # 注意：node.parents 此刻为空，真正持有悬空边的父节点
            # 信息在“本节点作为子目标”的关系里——该关系已随产生
            # 垃圾的 _unlink 解除，故这里只需处理本节点的出边责任。
            for child in node.edges.values():
                child_node = self._nodes.get(child)
                if child_node is None or child in dead:
                    continue
                child_node.parents.pop(nid, None)
                if not child_node.parents and child != self._root.nid:
                    stack.append(child)
        for nid in dead:
            node = self._nodes.pop(nid, None)
            if node is not None and \
                    self._registry.get(node.signature()) == nid:
                self._registry.pop(node.signature(), None)

    # ------------------------------------------------------------------ #
    # 路径私有化与自底向上重算
    # ------------------------------------------------------------------ #
    def _isolate_path(self, path: list[_Node], labels: list[int],
                      upto: int) -> list[int]:
        """克隆路径上的共享节点，返回失去父节点的旧节点号。

        找到路径上**最浅**的共享节点（最靠近根、入边数>1 的节点），
        从它开始到路径末端逐个克隆，父边改指克隆，得到只服务当前
        词的私有后缀链。必须从最浅共享节点开始：其下方可能还有
        共享节点，只克隆最深者会让上方共享节点被原地修改，污染
        别的词。根永不克隆。
        """
        if upto <= 0:
            return []
        start = -1
        for i in range(1, upto + 1):
            if path[i].refcount > 1:
                start = i
                break
        if start == -1:
            return []
        prev = path[start - 1]
        stale: list[int] = []
        for i in range(start, upto + 1):
            clone = self._clone(path[i])
            old = self._relink(prev, labels[i - 1], clone.nid)
            stale.append(old)
            path[i] = clone
            prev = clone
        return stale

    def _rebuild_path(self, path: list[_Node], labels: list[int],
                      prune: bool) -> list[int]:
        """自底向上沿私有路径剪枝并哈希共享，返回失父节点号。

        重复“自深向浅”扫描直到一轮内无任何改动：
        * prune 且节点“非接纳、无出边”：死状态，删父边并从 path
          弹出（有限语言最小 DFA 不含可达死状态）；
        * 否则重算签名，命中 registry 规范节点则父边改指过去，
          否则自行登记。
        之所以需要重复扫描：剪去末端死节点后，其父节点可能成为
        新的末端死状态（级联剪枝），单趟固定索引会漏掉它。
        每轮至少弹出一个节点或收敛，路径长度有限故必然终止；
        总工作量为 O(L^2)（L=词字节长），仍只与词长相关。
        根不可合并，签名撞车时根优先成为规范节点。
        """
        stale: list[int] = []
        changed = True
        while changed:
            changed = False
            for i in range(len(path) - 1, 0, -1):
                node = path[i]
                if node.nid not in self._nodes:
                    continue
                # 剪枝分支：死状态删父边
                if prune and not node.final and not node.edges:
                    if labels[i - 1] in path[i - 1].edges:
                        self._detach(node)
                        stale.append(
                            self._unlink(path[i - 1], labels[i - 1]))
                        path.pop(i)
                        labels.pop(i - 1)
                        changed = True
                    break  # 路径已变，重新从深端扫描
                # 等价合并分支：先看签名（不提前 detach，避免只摘不登）
                sig = node.signature()
                canon = self._registry.get(sig)
                if canon is not None and canon != node.nid \
                        and labels[i - 1] in path[i - 1].edges:
                    self._detach(node)
                    old = self._relink(path[i - 1], labels[i - 1], canon)
                    stale.append(old)
                    path[i] = self._nodes[canon]
                    changed = True
                elif self._registry.get(sig) != node.nid:
                    # 登记项缺失或指向别处且未合并：自行登记
                    self._registry[sig] = node.nid
        # 收敛后，对仍在路径上的存活节点统一确保登记（修复中间态
        # 摘除后因 break 未补登的情况）
        for node in path:
            if node.nid in self._nodes and \
                    self._registry.get(node.signature()) is None:
                self._registry[node.signature()] = node.nid
        root = path[0]
        if root.nid in self._nodes:
            self._detach(root)
            self._registry[root.signature()] = root.nid
        return stale

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def add(self, word: str) -> bool:
        """加入一个词；集合改变返回 True，已存在返回 False。"""
        data = word.encode("utf-8")
        path: list[_Node] = [self._root]
        labels: list[int] = []
        node = self._root
        consumed = 0
        for byte in data:
            nxt = node.edges.get(byte)
            if nxt is None:
                break
            labels.append(byte)
            node = self._nodes[nxt]
            path.append(node)
            consumed += 1
        else:
            if node.final:
                return False  # 重复添加，图不变
            stale = self._isolate_path(path, labels, len(path) - 1)
            end = path[-1]
            self._detach(end)
            end.final = True
            stale += self._rebuild_path(path, labels, prune=False)
            self._sweep(stale)
            return True

        # 路径在 consumed 字节后断开，扩展剩余后缀
        stale = self._isolate_path(path, labels, len(path) - 1)
        parent = path[-1]
        for byte in data[consumed:]:
            child = self._new_node()
            self._link(parent, byte, child.nid)
            path.append(child)
            labels.append(byte)
            parent = child
        parent.final = True
        stale += self._rebuild_path(path, labels, prune=False)
        self._sweep(stale)
        return True

    def discard(self, word: str) -> bool:
        """删除一个词；集合改变返回 True，词不存在返回 False。"""
        data = word.encode("utf-8")
        path: list[_Node] = [self._root]
        labels: list[int] = []
        node = self._root
        for byte in data:
            nxt = node.edges.get(byte)
            if nxt is None:
                return False  # 词不在集合中，图不变
            labels.append(byte)
            node = self._nodes[nxt]
            path.append(node)
        if not node.final:
            return False  # 仅为前缀而非完整词，图不变

        # 私有化整条词路径（除根），翻转末端接纳标记，再单趟
        # 剪枝 + 哈希共享
        stale = self._isolate_path(path, labels, len(path) - 1)
        end = path[-1]
        self._detach(end)
        end.final = False
        stale += self._rebuild_path(path, labels, prune=True)
        self._sweep(stale)
        return True

    def contains(self, word: str) -> bool:
        """精确判断完整词是否在集合中（前缀命中不算）。"""
        node = self._root
        for byte in word.encode("utf-8"):
            nxt = node.edges.get(byte)
            if nxt is None:
                return False
            node = self._nodes[nxt]
        return node.final

    def export(self) -> dict:
        """导出 JSON 可往返的图描述；节点号即 nodes 列表下标。"""
        order: list[int] = []
        remap: dict[int, int] = {self._root.nid: 0}
        queue = deque([self._root.nid])
        while queue:
            nid = queue.popleft()
            order.append(nid)
            node = self._nodes[nid]
            for child in node.edges.values():
                if child not in remap:
                    remap[child] = len(remap)
                    queue.append(child)

        nodes = []
        for nid in order:
            node = self._nodes[nid]
            edges = [
                [byte, remap[target]]
                for byte, target in sorted(node.edges.items())
            ]
            nodes.append({"final": node.final, "edges": edges})
        return {"root": 0, "nodes": nodes}

    # ------------------------------------------------------------------ #
    # 调试核验（核验脚本调用，业务路径不依赖）
    # ------------------------------------------------------------------ #
    def debug_check_invariants(self) -> None:
        """逐项断言内部结构不变量，违例即抛 AssertionError。"""
        assert self._root.nid in self._nodes
        seen = {self._root.nid}
        indeg = {self._root.nid: 1}
        queue = deque([self._root.nid])
        while queue:
            nid = queue.popleft()
            node = self._nodes[nid]
            edge_labels = list(node.edges)
            assert len(set(edge_labels)) == len(edge_labels), "边标签重复"
            assert all(0 <= b <= 255 for b in edge_labels), "字节标签越界"
            for target in node.edges.values():
                assert target in self._nodes, f"悬空边：{nid} -> {target}"
                indeg[target] = indeg.get(target, 0) + 1
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        assert seen == set(self._nodes), "存在不可达节点"
        for nid, count in indeg.items():
            assert self._nodes[nid].refcount == count, \
                f"入边计数错误: {nid}: {self._nodes[nid].refcount}!={count}"
        # 反向边一致性
        for nid in seen:
            actual: dict[int, int] = {}
            for p in seen:
                cnt = sum(1 for t in self._nodes[p].edges.values()
                          if t == nid)
                if cnt:
                    actual[p] = cnt
            assert self._nodes[nid].parents == actual, \
                f"反向边计数错误: {nid}"
        canon_nodes = list(self._registry.values())
        assert len(canon_nodes) == len(set(canon_nodes)), (
            "等价索引中同一节点被登记为多个等价类的规范节点"
        )
        for sig, nid in self._registry.items():
            assert nid in self._nodes
            assert self._nodes[nid].signature() == sig
        for node in self._nodes.values():
            assert self._registry.get(node.signature()) is not None
        # 最小 DFA 不应存在可达的非接纳死状态（空集根除外）
        for nid, node in self._nodes.items():
            if nid != self._root.nid and not node.final and not node.edges:
                raise AssertionError(f"可达死状态未清理: {nid}")
        # 无环（DFS 三色法）
        color: dict[int, int] = {}

        def visit(start: int) -> None:
            stack = [(start, iter(self._nodes[start].edges.values()))]
            color[start] = 1
            while stack:
                nid, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[nid] = 2
                    stack.pop()
                    continue
                assert color.get(nxt, 0) != 1, "检测到环"
                if color.get(nxt, 0) == 0:
                    color[nxt] = 1
                    stack.append(
                        (nxt, iter(self._nodes[nxt].edges.values()))
                    )

        visit(self._root.nid)
