
import os, random, math
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from docplex.mp.model import Model as CplexModel
    _HAS_CPLEX = True
except Exception:
    _HAS_CPLEX = False

PUB_COLS = ["age", "sex", "blood", "admission"]
SENSITIVE = "result"

def load_data(csv_path: Optional[str] = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rename = {}
        for c in df.columns:
            lc = c.strip().lower().replace(" ", "_").replace("-", "_")
            if lc in ["age"]:
                rename[c] = "age"
            elif lc in ["sex", "gender"]:
                rename[c] = "sex"
            elif lc in ["bloodtype", "blood_type", "blood", "blood type"]:
                rename[c] = "blood"
            elif lc in ["admissiontype", "admission_type", "admission type", "admission"]:
                rename[c] = "admission"
            elif lc in ["test results", "test_result", "test_results", "result", "results"]:
                rename[c] = "result"
        if rename: df = df.rename(columns=rename)
        missing = set(PUB_COLS + [SENSITIVE]) - set(df.columns)
        if missing: raise ValueError(f"Missing required columns: {missing}")
        return df[PUB_COLS + [SENSITIVE]].copy()
    n = 100
    ages = rng.integers(0, 101, size=n)
    sex = rng.integers(0, 2, size=n)
    blood = rng.integers(0, 8, size=n)
    admission = rng.integers(0, 3, size=n)
    logits = -2.0 + 0.02 * ages + 0.3 * (sex == 1) + 0.15 * np.isin(blood, [2,4]).astype(int) + 0.4 * (admission == 2)
    probs = 1 / (1 + np.exp(-logits))
    result = (rng.random(n) < probs).astype(int)
    return pd.DataFrame({"age": ages, "sex": sex, "blood": blood, "admission": admission, "result": result})

class RandomPredicate:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._clauses = []
        if self._rng.random() < 0.8:
            thr = self._rng.randint(0, 100)
            if self._rng.random() < 0.5:
                self._clauses.append(lambda df, thr=thr: df["age"] > thr)
            else:
                self._clauses.append(lambda df, thr=thr: df["age"] <= thr)
        if self._rng.random() < 0.5:
            val = self._rng.choice([0,1])
            self._clauses.append(lambda df, val=val: df["sex"] == val)
        if self._rng.random() < 0.6:
            k = self._rng.randint(1, 4)
            vals = self._rng.sample(list(range(8)), k)
            self._clauses.append(lambda df, vals=vals: df["blood"].isin(vals))
        if self._rng.random() < 0.6:
            val = self._rng.choice([0,1,2])
            self._clauses.append(lambda df, val=val: df["admission"] == val)
        if not self._clauses:
            self._clauses.append(lambda df: pd.Series([True]*len(df), index=df.index))
    def mask(self, data_pub: pd.DataFrame) -> np.ndarray:
        m = None
        for c in self._clauses:
            cur = c(data_pub)
            m = cur if m is None else (m & cur)
        return m.values.astype(bool)

def make_random_predicate(seed: Optional[int] = None) -> RandomPredicate:
    return RandomPredicate(seed)

def build_A(data_pub: pd.DataFrame, predicates: List[RandomPredicate]) -> np.ndarray:
    A = np.zeros((len(predicates), len(data_pub)), dtype=int)
    for i, q in enumerate(predicates):
        A[i, :] = q.mask(data_pub).astype(int)
    return A

def execute_subsetsums_exact(data: pd.DataFrame, predicates: List[RandomPredicate]) -> np.ndarray:
    A = build_A(data[ PUB_COLS ], predicates)
    x = data[ SENSITIVE ].to_numpy().astype(int)
    return (A @ x).astype(float)

def execute_subsetsums_round(R: int, data: pd.DataFrame, predicates: List[RandomPredicate]) -> np.ndarray:
    y = execute_subsetsums_exact(data, predicates)
    return np.round(y / R) * R

def execute_subsetsums_noise(sigma: float, data: pd.DataFrame, predicates: List[RandomPredicate], seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = execute_subsetsums_exact(data, predicates)
    return y + rng.normal(0.0, sigma, size=len(y))

def execute_subsetsums_sample(t: int, data: pd.DataFrame, predicates: List[RandomPredicate], seed: Optional[int] = None) -> np.ndarray:
    n = len(data)
    if t < 1 or t > n: raise ValueError(f"t must be in [1, {n}]")
    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=t, replace=False)
    y_sub = execute_subsetsums_exact(data.iloc[keep], predicates)
    return y_sub * (n / t)

def reconstruction_attack(data_pub: pd.DataFrame, predicates: List[RandomPredicate], answers: np.ndarray, solver: str = "auto") -> np.ndarray:
    A = build_A(data_pub, predicates).astype(float)
    b = np.asarray(answers, dtype=float)
    if (solver in ["auto","docplex"]) and _HAS_CPLEX:
        try:
            return _reconstruct_cplex(A, b)
        except Exception:
            pass
    return _reconstruct_ls(A, b)

def _reconstruct_cplex(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, n = A.shape
    mdl = CplexModel("reconstruction_l1")
    x = mdl.binary_var_list(n, name="x")
    e = mdl.continuous_var_list(m, lb=0, name="e")
    for i in range(m):
        expr = mdl.sum(A[i,j]*x[j] for j in range(n))
        mdl.add_constraint(e[i] >= expr - b[i])
        mdl.add_constraint(e[i] >= -(expr - b[i]))
    mdl.minimize(mdl.sum(e))
    sol = mdl.solve(log_output=False)
    if sol is None:
        raise RuntimeError("CPLEX failed")
    x_hat = np.array([round(sol.get_value(v)) for v in x], dtype=int)
    return x_hat

def _reconstruct_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    lam = 1e-6
    ATA = A.T @ A + lam*np.eye(A.shape[1])
    ATb = A.T @ b
    x = np.linalg.solve(ATA, ATb)
    x = np.clip(x, 0, 1)
    return (x >= 0.5).astype(int)

def rmse(y, y_true) -> float:
    y = np.asarray(y, float); y_true = np.asarray(y_true, float)
    return float(np.sqrt(np.mean((y - y_true)**2)))

def success_rate(x_hat, x_true) -> float:
    x_hat = np.asarray(x_hat, int); x_true = np.asarray(x_true, int)
    return float(np.mean(x_hat == x_true))
