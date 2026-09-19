"""Discovery tree / grid record.

Dream-RSI (arXiv 2609.14858) Sec. 3: a discovery tree is rooted at r (initial
workspace).  Each non-root node v has exactly one primary parent; v records the
outcome of one generation-evaluation attempt: filesystem snapshot, artifact,
evaluation diagnostics and score s_v.

We additionally give every node a grid coordinate (branch, attempt) because the
replay simulator addresses cells as branch x attempt slots (Appendix B.2).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Node:
    """One generation-evaluation attempt."""

    id: str                     # "b{b}#a{a}"
    branch: int
    attempt: int
    parent_id: Optional[str]    # None for the root
    score: float
    evaluated: bool = True
    valid: bool = True
    fail_class: str = "ok"      # ok | compile | runtime | timeout | invalid | env
    error: Optional[str] = None
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    artifact: Optional[str] = None       # relative path of the produced program
    workspace: Optional[str] = None      # attempt dir (filesystem snapshot)
    tags: List[str] = field(default_factory=list)
    seq: int = 0                # global creation order (used for root opening)
    created_at: float = field(default_factory=time.time)
    proposal: Optional[str] = None       # relative path of proposal.md

    @property
    def cell(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DiscoveryTree:
    """Recorded discovery history: root + nodes with realized outcomes."""

    root_id: str = "root"
    nodes: Dict[str, Node] = field(default_factory=dict)
    baseline_score: float = 0.0
    branch_count: int = 0
    refine_count: int = 0
    round_index: int = 0
    task: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- mutation
    def add_node(self, branch: int, attempt: int, parent_id: Optional[str],
                 score: float, **kw: Any) -> Node:
        node_id = kw.pop("id", None) or f"b{branch}#a{attempt}"
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        node = Node(id=node_id, branch=branch, attempt=attempt,
                    parent_id=parent_id, score=score,
                    seq=len(self.nodes) + 1, **kw)
        self.nodes[node_id] = node
        self.branch_count = max(self.branch_count, branch + 1)
        self.refine_count = max(self.refine_count, attempt + 1)
        return node

    def set_baseline(self, score: float) -> None:
        self.baseline_score = float(score)

    # ---------------------------------------------------------------- queries
    def get(self, cell: str) -> Optional[Node]:
        return self.nodes.get(cell)

    def children_of(self, cell: str) -> List[Node]:
        return sorted((n for n in self.nodes.values() if n.parent_id == cell),
                      key=lambda n: n.seq)

    def leaves(self) -> List[Node]:
        has_child = {n.parent_id for n in self.nodes.values() if n.parent_id}
        return sorted((n for n in self.nodes.values() if n.id not in has_child),
                      key=lambda n: n.seq)

    def branch_nodes(self, branch: int) -> List[Node]:
        return sorted((n for n in self.nodes.values() if n.branch == branch),
                      key=lambda n: n.attempt)

    def max_score(self) -> float:
        return max((n.score for n in self.nodes.values()), default=-math.inf)

    def best(self) -> Optional[Node]:
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: (n.score, -n.seq))

    def size(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------- (de)serial
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "baseline_score": self.baseline_score,
            "branch_count": self.branch_count,
            "refine_count": self.refine_count,
            "round_index": self.round_index,
            "task": self.task,
            "meta": self.meta,
            "nodes": [n.to_dict() for n in
                      sorted(self.nodes.values(), key=lambda n: n.seq)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryTree":
        t = cls(root_id=d.get("root_id", "root"),
                baseline_score=d.get("baseline_score", 0.0),
                branch_count=d.get("branch_count", 0),
                refine_count=d.get("refine_count", 0),
                round_index=d.get("round_index", 0),
                task=d.get("task", ""),
                meta=d.get("meta", {}))
        for nd in d.get("nodes", []):
            node = Node.from_dict(nd)
            t.nodes[node.id] = node
        return t

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DiscoveryTree":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ digest
    def summary(self) -> Dict[str, Any]:
        leaves = self.leaves()
        return {
            "task": self.task,
            "round": self.round_index,
            "nodes": self.size(),
            "branches": self.branch_count,
            "max_attempts": self.refine_count,
            "best": round(self.max_score(), 6) if self.nodes else None,
            "baseline": round(self.baseline_score, 6),
            "best_cell": (self.best().id if self.best() else None),
            "leaves": [n.id for n in leaves],
            "failures": sum(1 for n in self.nodes.values()
                            if n.fail_class not in ("ok",)),
        }

    def compact_rows(self) -> List[Dict[str, Any]]:
        """Token-efficient table for LLM prompts."""
        rows = []
        for n in sorted(self.nodes.values(), key=lambda n: n.seq):
            rows.append({
                "cell": n.id,
                "b": n.branch,
                "a": n.attempt,
                "parent": n.parent_id,
                "s": round(n.score, 6),
                "ok": bool(n.evaluated),
                "valid": bool(n.valid),
                "fc": n.fail_class,
                "err": (n.error or "")[:120] or None,
            })
        return rows


def load_trees(paths: Iterable[str | Path]) -> List[DiscoveryTree]:
    return [DiscoveryTree.load(p) for p in paths]
