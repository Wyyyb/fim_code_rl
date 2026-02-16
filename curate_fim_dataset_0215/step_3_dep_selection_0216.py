#!/usr/bin/env python3
"""
depfim.py — Dependency-Aware Fill-in-the-Middle Function Selection

Given a list of Python code samples (dicts with 'code_content'),
selects optimal functions for FIM masking based on program dependency
graph analysis and information-theoretic scoring.

Usage:
    python depfim.py                              # run built-in demo
    python depfim.py input.json output.json       # process JSON file

Programmatic:
    from depfim import process_samples, save_results
    results = process_samples(samples, verbose=True)
    save_results(results, 'output.json')
"""

import ast
import math
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from tqdm import tqdm


# ================================================================
# Data Structures
# ================================================================

@dataclass
class FunctionInfo:
    """Metadata and computed metrics for a single function/method."""
    name: str                    # Qualified: "ClassName.method" or "func"
    lineno: int                  # Start line (1-indexed)
    end_lineno: int              # End line (1-indexed, inclusive)
    source_text: str             # Raw source code
    loc: int                     # Lines of code
    params: List[str]            # Parameter names (including self/cls)
    has_return_type: bool
    has_param_types: bool
    has_docstring: bool
    class_name: Optional[str]    # Owning class, None for module-level
    decorators: List[str] = field(default_factory=list)
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    cyclomatic_complexity: int = 1
    max_ast_depth: int = 0
    num_variables: int = 0


@dataclass
class FIMCandidate:
    """A function selected as a FIM mask target, with all scores."""
    func_name: str
    start_line: int
    end_line: int
    source_text: str
    loc: int
    complexity: float
    inferability: float
    fim_score: float
    difficulty: float


# ================================================================
# Step 1 — Dependency Graph Builder
# ================================================================

class DependencyGraphBuilder:
    """
    Parse a Python source file and construct a dependency graph.

    Nodes  = functions / methods
    Edges  = call edges  | sibling edges (same-class methods)
    """

    def __init__(self, source: str):
        self.source = source
        self.source_lines = source.splitlines()
        self.tree: Optional[ast.Module] = None
        self.functions: Dict[str, FunctionInfo] = {}
        self.ast_nodes: Dict[str, ast.AST] = {}       # name -> AST node
        self.edges: List[Tuple[str, str, str]] = []    # (src, dst, type)
        self._short_to_qname: Dict[str, List[str]] = defaultdict(list)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def build(self) -> bool:
        """Parse source and build graph.  Returns True on success."""
        try:
            self.tree = ast.parse(self.source)
        except SyntaxError:
            return False

        # Pass 1 — collect every function / method definition
        self._collect_functions(self.tree, current_class=None)

        # Short-name index for fuzzy call resolution
        for fname in self.functions:
            short = fname.split(".")[-1]
            self._short_to_qname[short].append(fname)

        # Pass 2 — extract calls inside each function body
        for fname, finfo in self.functions.items():
            self._analyze_calls(fname, finfo)

        # Pass 3 — create call edges (exact + fuzzy matching)
        self._build_call_edges()

        # Pass 4 — sibling edges (same class)
        self._add_sibling_edges()

        # Pass 5 — compute complexity metrics
        for fname in self.functions:
            self._compute_metrics(fname)

        return True

    # ----------------------------------------------------------------
    # Function collection
    # ----------------------------------------------------------------

    def _collect_functions(self, node: ast.AST, current_class: Optional[str]):
        """Walk top-level and class-level statements to find func defs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._collect_functions(child, current_class=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register_function(child, current_class)

    def _register_function(self, node, current_class: Optional[str]):
        qname = f"{current_class}.{node.name}" if current_class else node.name

        # Source text extraction
        start_idx = node.lineno - 1                       # 0-indexed
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            end_line = self._fallback_end(node)            # Python < 3.8
        source_text = "\n".join(self.source_lines[start_idx:end_line])
        loc = end_line - start_idx

        # Collect all argument objects
        all_args = (
            node.args.args
            + getattr(node.args, "posonlyargs", [])
            + node.args.kwonlyargs
        )

        # Docstring check
        body0 = node.body[0] if node.body else None
        has_docstring = (
            isinstance(body0, ast.Expr)
            and isinstance(getattr(body0, "value", None), ast.Constant)
            and isinstance(body0.value.value, str)
        )
        # Fallback for Python 3.7 ast.Str
        if not has_docstring and hasattr(ast, "Str"):
            has_docstring = (
                isinstance(body0, ast.Expr)
                and isinstance(body0.value, ast.Str)
            )

        # Decorators
        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(d.attr)
            elif isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                decorators.append(d.func.id)

        self.functions[qname] = FunctionInfo(
            name=qname,
            lineno=node.lineno,
            end_lineno=end_line,
            source_text=source_text,
            loc=loc,
            params=[a.arg for a in node.args.args],
            has_return_type=(node.returns is not None),
            has_param_types=any(a.annotation is not None for a in all_args),
            has_docstring=has_docstring,
            class_name=current_class,
            decorators=decorators,
        )
        self.ast_nodes[qname] = node

    def _fallback_end(self, node) -> int:
        """Estimate end line for Python < 3.8 (no end_lineno)."""
        m = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                m = max(m, child.lineno)
        return m

    # ----------------------------------------------------------------
    # Call analysis
    # ----------------------------------------------------------------

    def _analyze_calls(self, fname: str, finfo: FunctionInfo):
        """Walk function body AST and record every call target."""
        node = self.ast_nodes.get(fname)
        if node is None:
            return
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                resolved = self._resolve_call(child, finfo.class_name)
                if resolved:
                    finfo.calls.add(resolved)

    def _resolve_call(
        self, call: ast.Call, current_class: Optional[str]
    ) -> Optional[str]:
        func = call.func

        if isinstance(func, ast.Name):
            name = func.id
            # Class instantiation  → __init__
            init_qname = f"{name}.__init__"
            if init_qname in self.functions:
                return init_qname
            # Direct module-level function
            if name in self.functions:
                return name
            return name  # keep for fuzzy matching later

        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                obj, attr = func.value.id, func.attr
                # self.method() / cls.method()
                if obj in ("self", "cls") and current_class:
                    return f"{current_class}.{attr}"
                # ClassName.static_method()
                candidate = f"{obj}.{attr}"
                if candidate in self.functions:
                    return candidate
            # Fallback: just the attribute name
            return func.attr

        return None

    def _build_call_edges(self):
        """Create directed call edges, using fuzzy short-name matching."""
        for fname, finfo in self.functions.items():
            for callee in finfo.calls:
                targets: Set[str] = set()
                if callee in self.functions:
                    targets.add(callee)
                else:
                    short = callee.split(".")[-1]
                    for qn in self._short_to_qname.get(short, []):
                        if qn != fname:
                            targets.add(qn)
                for t in targets:
                    self.edges.append((fname, t, "call"))
                    self.functions[t].called_by.add(fname)

    def _add_sibling_edges(self):
        """Undirected edges between methods of the same class."""
        cls_map: Dict[str, List[str]] = defaultdict(list)
        for fn, fi in self.functions.items():
            if fi.class_name:
                cls_map[fi.class_name].append(fn)
        for methods in cls_map.values():
            for i, a in enumerate(methods):
                for b in methods[i + 1 :]:
                    self.edges.append((a, b, "sibling"))
                    self.edges.append((b, a, "sibling"))

    # ----------------------------------------------------------------
    # Complexity metrics
    # ----------------------------------------------------------------

    def _compute_metrics(self, fname: str):
        fi = self.functions[fname]
        node = self.ast_nodes.get(fname)
        if node is None:
            return
        fi.cyclomatic_complexity = self._cyclomatic_complexity(node)
        fi.max_ast_depth = self._max_ast_depth(node)
        fi.num_variables = self._count_variables(node)

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        cc = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                cc += 1
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                cc += 1
            elif isinstance(child, ast.ExceptHandler):
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
            elif isinstance(
                child,
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                cc += len(child.generators)
        return cc

    @staticmethod
    def _max_ast_depth(node: ast.AST, depth: int = 0) -> int:
        max_d = depth
        nesting = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.With, ast.AsyncWith, ast.Try,
        )
        if hasattr(ast, "TryStar"):
            nesting = nesting + (ast.TryStar,)
        for child in ast.iter_child_nodes(node):
            nd = depth + 1 if isinstance(child, nesting) else depth
            max_d = max(max_d, DependencyGraphBuilder._max_ast_depth(child, nd))
        return max_d

    @staticmethod
    def _count_variables(node: ast.AST) -> int:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        names.update(
                            e.id for e in t.elts if isinstance(e, ast.Name)
                        )
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    names.add(child.target.id)
            elif isinstance(child, ast.AnnAssign):
                if isinstance(getattr(child, "target", None), ast.Name):
                    names.add(child.target.id)
        return len(names)


# ================================================================
# Steps 2–4 — FIM Scoring & Threshold Selection
# ================================================================

class FIMSelector:
    """
    Compute information-theoretic FIM scores and select mask targets.

    Uses **threshold-based** filtering — a file with no suitable
    functions produces an empty target list.
    """

    DEFAULT_CONFIG = {
        # --- LOC bounds ---
        "min_loc": 5,
        "max_loc": 150,
        # --- Complexity weights  (Ĥ) ---
        "w_loc": 0.4,
        "w_cc": 0.4,
        "w_depth": 0.2,
        # --- Inferability weights (Î) ---
        "alpha_caller": 0.30,
        "beta_callee": 0.25,
        "gamma_sig": 0.20,
        "delta_doc": 0.10,
        "epsilon_class": 0.15,
        # --- Difficulty penalty (one-sided) ---
        "difficulty_ceiling": 0.5,
        "difficulty_sigma": 0.20,
        # --- Hard thresholds ---
        "score_threshold": 0.015,
        "min_complexity": 0.12,
    }

    # Dunder methods: usually required as context, not mask targets
    SKIP_NAMES = frozenset({
        "__init__", "__new__", "__del__", "__repr__", "__str__",
        "__len__", "__getitem__", "__setitem__", "__delitem__",
        "__iter__", "__next__", "__contains__",
        "__enter__", "__exit__", "__hash__",
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
        "__bool__", "__call__", "__format__", "__sizeof__",
    })

    def __init__(
        self,
        functions: Dict[str, FunctionInfo],
        edges: List[Tuple[str, str, str]],
        ast_nodes: Dict[str, ast.AST],
        config: Optional[Dict] = None,
    ):
        self.functions = functions
        self.ast_nodes = ast_nodes
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        # Build adjacency indexes
        self.callers: Dict[str, Set[str]] = defaultdict(set)
        self.callees: Dict[str, Set[str]] = defaultdict(set)
        self.siblings: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, etype in edges:
            if etype == "call":
                self.callers[dst].add(src)
                self.callees[src].add(dst)
            elif etype == "sibling":
                self.siblings[src].add(dst)

    # ----------------------------------------------------------------
    # Ĥ(v) — intrinsic complexity proxy
    # ----------------------------------------------------------------

    def compute_complexity(self, fi: FunctionInfo) -> float:
        c = self.cfg
        norm_loc = min(fi.loc / 50.0, 2.0)
        norm_cc = min(fi.cyclomatic_complexity / 10.0, 2.0)
        norm_depth = min(fi.max_ast_depth / 5.0, 2.0)
        return (
            c["w_loc"] * norm_loc
            + c["w_cc"] * norm_cc
            + c["w_depth"] * norm_depth
        )

    # ----------------------------------------------------------------
    # Î(v) — contextual inferability proxy
    # ----------------------------------------------------------------

    def compute_inferability(self, fname: str) -> float:
        c = self.cfg
        fi = self.functions[fname]

        # 1) Caller information  ──────────────────────────────────
        caller_sp = 0.0
        for caller in self.callers.get(fname, set()):
            if caller in self.functions:
                caller_sp += self._call_site_specificity(caller, fname)
        caller_score = min(caller_sp / 3.0, 1.0)

        # 2) Callee information (file-internal only) ──────────────
        internal_callees = [
            x for x in self.callees.get(fname, set()) if x in self.functions
        ]
        callee_score = min(len(internal_callees) / 4.0, 1.0)

        # 3) Signature information ────────────────────────────────
        sig_score = 0.0
        if fi.has_return_type:
            sig_score += 0.30
        if fi.has_param_types:
            sig_score += 0.25
        raw_name = fname.split(".")[-1]
        if not raw_name.startswith("_"):
            parts = [p for p in raw_name.split("_") if p]
            sig_score += min(len(parts) / 5.0, 0.25)
        non_self_params = [p for p in fi.params if p not in ("self", "cls")]
        sig_score += min(len(non_self_params) / 6.0, 0.20)
        sig_score = min(sig_score, 1.0)

        # 4) Documentation ────────────────────────────────────────
        doc_score = 0.5 if fi.has_docstring else 0.0

        # 5) Class context ────────────────────────────────────────
        class_score = 0.0
        if fi.class_name:
            n_sibs = len(self.siblings.get(fname, set()))
            class_score = min(n_sibs / 5.0, 1.0)
            init_name = f"{fi.class_name}.__init__"
            if init_name in self.functions and init_name != fname:
                class_score = min(class_score + 0.3, 1.0)

        return (
            c["alpha_caller"] * caller_score
            + c["beta_callee"] * callee_score
            + c["gamma_sig"] * sig_score
            + c["delta_doc"] * doc_score
            + c["epsilon_class"] * class_score
        )

    def _call_site_specificity(self, caller: str, callee: str) -> float:
        """
        Estimate information provided by the call site(s) in `caller`
        that invoke `callee`.  Considers argument types and counts.
        """
        node = self.ast_nodes.get(caller)
        if node is None:
            return 0.5

        short = callee.split(".")[-1]
        total_sp = 0.0
        found = False

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            # Match call target
            n = None
            if isinstance(child.func, ast.Name):
                n = child.func.id
            elif isinstance(child.func, ast.Attribute):
                n = child.func.attr
            if n != short:
                continue

            found = True
            sp = 0.50  # base: being called at all
            for arg in child.args:
                if isinstance(arg, ast.Constant):
                    sp += 0.15        # literal → strong constraint
                elif isinstance(arg, ast.Name):
                    sp += 0.05        # variable → weak constraint
                else:
                    sp += 0.08        # expression → medium
            sp += 0.12 * len(child.keywords)  # keyword args are informative
            sp += 0.10                        # return value usage (approx)
            total_sp += sp

        return min(total_sp, 1.5) if found else 0.5

    # ----------------------------------------------------------------
    # FIM score & selection
    # ----------------------------------------------------------------

    def select_targets(self) -> List[FIMCandidate]:
        """
        Score every function and return those above the threshold.

        FIM_Score = Ĥ(v) × Î(v) / (Ĥ(v) + Î(v))
        penalised by a one-sided Gaussian when difficulty > ceiling.
        """
        c = self.cfg
        results: List[FIMCandidate] = []

        for fname, fi in self.functions.items():
            # ---- hard filters ----
            if fi.loc < c["min_loc"] or fi.loc > c["max_loc"]:
                continue
            if fname.split(".")[-1] in self.SKIP_NAMES:
                continue

            h = self.compute_complexity(fi)
            if h < c["min_complexity"]:
                continue

            i_val = self.compute_inferability(fname)

            # ---- information-theoretic score ----
            # Mutual-information proxy: H * I / (H + I)
            raw_score = h * (i_val / (h + i_val + 1e-8))

            # Difficulty: fraction of complexity unexplained by context
            difficulty = max(0.0, h - i_val) / (h + 1e-8)

            # One-sided Gaussian penalty for too-hard functions
            if difficulty > c["difficulty_ceiling"]:
                excess = difficulty - c["difficulty_ceiling"]
                factor = math.exp(
                    -(excess ** 2) / (2.0 * c["difficulty_sigma"] ** 2)
                )
            else:
                factor = 1.0

            final_score = raw_score * factor

            # ---- threshold filter ----
            if final_score < c["score_threshold"]:
                continue

            results.append(
                FIMCandidate(
                    func_name=fname,
                    start_line=fi.lineno,
                    end_line=fi.end_lineno,
                    source_text=fi.source_text,
                    loc=fi.loc,
                    complexity=round(h, 4),
                    inferability=round(i_val, 4),
                    fim_score=round(final_score, 4),
                    difficulty=round(difficulty, 4),
                )
            )

        results.sort(key=lambda x: x.fim_score, reverse=True)
        return results


# ================================================================
# Pipeline
# ================================================================

def process_samples(
    samples: List[Dict[str, Any]],
    config: Optional[Dict] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main entry point.  For each sample dict:
      1. parse code_content
      2. build dependency graph
      3. score & select FIM targets
      4. attach 'mask_targets' list to the dict

    Returns a new list (input is not mutated).
    """
    results: List[Dict[str, Any]] = []

    for idx, sample in tqdm(enumerate(samples)):
        out = dict(sample)  # shallow copy
        code = out.get("code_content", "")
        sid = out.get("sample_id", idx)

        # ---- empty / missing code ----
        if not code or not code.strip():
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            results.append(out)
            if verbose:
                print(f"  [{sid}] Empty code_content → skipped")
            continue

        # ---- build dependency graph ----
        builder = DependencyGraphBuilder(code)
        if not builder.build():
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            out["parse_error"] = True
            results.append(out)
            if verbose:
                print(f"  [{sid}] SyntaxError → skipped")
            continue

        if not builder.functions:
            out["mask_targets"] = []
            out["mask_target_count"] = 0
            results.append(out)
            if verbose:
                print(f"  [{sid}] No functions found → skipped")
            continue

        # ---- select targets ----
        selector = FIMSelector(
            builder.functions, builder.edges, builder.ast_nodes, config
        )
        candidates = selector.select_targets()

        # ---- serialize into output dict ----
        targets = [
            {
                "func_name": c.func_name,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "source_text": c.source_text,
                "loc": c.loc,
                "complexity": c.complexity,
                "inferability": c.inferability,
                "fim_score": c.fim_score,
                "difficulty": c.difficulty,
            }
            for c in candidates
        ]

        out["mask_targets"] = targets
        out["mask_target_count"] = len(targets)
        out["graph_stats"] = {
            "total_functions": len(builder.functions),
            "call_edges": sum(1 for _, _, t in builder.edges if t == "call"),
            "sibling_pairs": sum(1 for _, _, t in builder.edges if t == "sibling") // 2,
        }
        results.append(out)

        if verbose:
            n_funcs = len(builder.functions)
            n_sel = len(targets)
            print(
                f"  [{sid}] {n_funcs} functions found, "
                f"{n_sel} selected as mask targets"
            )
            for t in targets:
                print(
                    f"      {t['func_name']:35s}  LOC={t['loc']:<3d}  "
                    f"score={t['fim_score']:.4f}  "
                    f"Ĥ={t['complexity']:.3f}  "
                    f"Î={t['inferability']:.3f}  "
                    f"diff={t['difficulty']:.3f}"
                )

    return results


def save_results(results: List[Dict[str, Any]], path: str):
    """Write results list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples → {path}")


# ================================================================
# Demo / Tests
# ================================================================

# ---- Test sample 1: realistic ML class with rich dependencies ----
KMEANS_CODE = '''\
import numpy as np


class KMeansClustering:
    """K-Means clustering algorithm implemented from scratch."""

    def __init__(self, k=3, max_iterations=100, tol=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None
        self.clusters = None

    def _init_centroids(self, X):
        """Initialize centroids using the k-means++ strategy."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))

        # First centroid chosen uniformly at random
        idx = np.random.randint(0, n_samples)
        centroids[0] = X[idx]

        # Remaining centroids chosen with distance-weighted probability
        for i in range(1, self.k):
            distances = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids[:i])
                for x in X
            ])
            probabilities = distances / distances.sum()
            cumulative = np.cumsum(probabilities)
            r = np.random.random()
            for j, p in enumerate(cumulative):
                if r < p:
                    centroids[i] = X[j]
                    break

        return centroids

    def _compute_distances(self, X, centroids):
        """Compute Euclidean distance from each sample to each centroid."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

    def _assign_clusters(self, distances):
        """Assign each sample to the nearest centroid."""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Recompute each centroid as the mean of its assigned samples."""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # Reinitialize empty clusters randomly
                centroids[i] = X[np.random.randint(0, X.shape[0])]
        return centroids

    def fit(self, X):
        """Run the K-Means algorithm on dataset X."""
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iterations):
            distances = self._compute_distances(X, self.centroids)
            labels = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)

            # Convergence check
            shift = np.linalg.norm(new_centroids - self.centroids)
            if shift < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            self.centroids = new_centroids

        self.clusters = labels
        return self

    def predict(self, X):
        """Predict cluster labels for new samples."""
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)

    def _compute_inertia(self, X, labels):
        """Compute within-cluster sum of squared distances (inertia)."""
        inertia = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

    def score(self, X):
        """Return negative inertia (higher is better)."""
        labels = self.predict(X)
        return -self._compute_inertia(X, labels)


def find_optimal_k(X, k_range=range(2, 11)):
    """Find the optimal k using the elbow method."""
    inertias = []
    for k in k_range:
        model = KMeansClustering(k=k)
        model.fit(X)
        labels = model.predict(X)
        inertia = model._compute_inertia(X, labels)
        inertias.append(inertia)

    # Detect elbow via maximum second-order difference
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(np.abs(diffs2)) + 2
    optimal_k = list(k_range)[elbow_idx]

    return optimal_k, inertias


def normalize_data(X):
    """Normalize features to zero mean and unit variance."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std
'''

# ---- Test sample 2: trivial utility file (expect NO targets) ----
TRIVIAL_CODE = '''\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b
'''


def run_demo():
    """Run the built-in demo with two test samples."""
    samples = [
        {
            "sample_id": 0,
            "repo_id": "demo",
            "file_path": "kmeans.py",
            "func_num": 11,
            "quality_rating": "extra-high",
            "code_content": KMEANS_CODE,
        },
        {
            "sample_id": 1,
            "repo_id": "demo",
            "file_path": "trivial_utils.py",
            "func_num": 4,
            "quality_rating": "low",
            "code_content": TRIVIAL_CODE,
        },
    ]

    print("=" * 72)
    print("DepFIM — Dependency-Aware FIM Function Selection (Demo)")
    print("=" * 72)
    print()

    results = process_samples(samples, verbose=True)

    print()
    print("-" * 72)
    print("Summary")
    print("-" * 72)
    for r in results:
        sid = r["sample_id"]
        fpath = r.get("file_path", "?")
        n_targets = r["mask_target_count"]
        gs = r.get("graph_stats", {})
        print(
            f"  Sample {sid} ({fpath}): "
            f"{gs.get('total_functions', '?')} funcs, "
            f"{gs.get('call_edges', '?')} call edges, "
            f"{gs.get('sibling_pairs', '?')} sibling pairs → "
            f"{n_targets} mask target(s)"
        )

    print()
    print("-" * 72)
    print("Detailed mask targets")
    print("-" * 72)
    for r in results:
        sid = r["sample_id"]
        fpath = r.get("file_path", "?")
        targets = r["mask_targets"]
        if not targets:
            print(f"\n  Sample {sid} ({fpath}): (no targets selected)")
            continue
        print(f"\n  Sample {sid} ({fpath}):")
        for i, t in enumerate(targets, 1):
            print(f"    [{i}] {t['func_name']}")
            print(f"        Lines {t['start_line']}–{t['end_line']}  "
                  f"(LOC={t['loc']})")
            print(f"        Ĥ(complexity)  = {t['complexity']:.4f}")
            print(f"        Î(inferability) = {t['inferability']:.4f}")
            print(f"        FIM score       = {t['fim_score']:.4f}")
            print(f"        Difficulty      = {t['difficulty']:.4f}")
            # Show first 3 lines of source
            src_lines = t["source_text"].splitlines()
            preview = src_lines[:3]
            print(f"        Source preview:")
            for line in preview:
                print(f"          {line}")
            if len(src_lines) > 3:
                print(f"          ... ({len(src_lines) - 3} more lines)")

    # Save demo output
    save_results(results, "depfim_demo_output.json")

    return results


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments → run demo
        run_demo()
    elif len(sys.argv) == 3:
        # input.json output.json
        input_path, output_path = sys.argv[1], sys.argv[2]
        with open(input_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from {input_path}")
        results = process_samples(samples, verbose=True)
        save_results(results, output_path)
    else:
        print("Usage:")
        print("  python depfim.py                          # run demo")
        print("  python depfim.py input.json output.json   # process file")
        # python step_3_dep_selection_0216.py /data/yubo/datasets/process_data_output_0215/extracted_python_files_0215.json /data/yubo/datasets/process_data_output_0215/step_3_selected_fim_functions_0215.json
        sys.exit(1)