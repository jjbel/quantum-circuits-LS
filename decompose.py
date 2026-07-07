"""Decompose an arbitrary unitary into the universal basis {H, T, CNOT}.

A unitary is
lowered in stages:

    1. twolevel_decomposition:  Unitary     -> list[TwoLevel]
    2. decompose_twolevel:      TwoLevel    -> SingleQubitGate + ControlledU
    3. decompose_controlledU:   ControlledU -> CU + CNOT
    4. decompose_cu:            CU          -> SingleQubitGate + CNOT
    5. decompose_to_ht:         SingleQubitGate -> H / T words (using rotation.py)

Numpy types: a `Unitary` (N x N) and a 2x2 gate block are
both np.ndarray (complex128); a `ComplexVec` is a 1-D np.ndarray. A `Circuit` is a
Python list of gate objects, each exposing `to_unitary()`; gates are stored in
order of application (the first gate is applied first, i.e. the rightmost matrix
factor).

Every function/method below is a stub for you to implement; See "03 - Completing the Decomposition.pdf" for the recommended order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    n = 0
    while (1 << n) < N:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Gate representations
#
# Each is a sparse description of an operation with a `to_unitary()` returning the
# full N x N matrix. As the decomposition progresses, gates get rewritten into
# simpler ones. The 2x2 block `unitary` is a (2, 2) complex ndarray.
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand to the full `size` x `size` matrix: identity except the 2x2 block
        placed at rows/cols (level0, level1).
        """
        out = np.eye(self.size, dtype=rotation.DTYPE)
        i, j = self.level0, self.level1
        out[i, i] = self.unitary[0, 0]
        out[i, j] = self.unitary[0, 1]
        out[j, i] = self.unitary[1, 0]
        out[j, j] = self.unitary[1, 1]
        return out


@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand the 2x2 to N dimensions: for each basis index whose `qubit` bit is
        0, fill the 2x2 block linking it to its partner (that bit = 1).
        """
        N = 1 << self.n
        bit = 1 << self.qubit
        out = np.eye(N, dtype=rotation.DTYPE)
        U = self.unitary
        for idx in range(N):
            if idx & bit:
                continue
            partner = idx | bit
            out[idx, idx] = U[0, 0]
            out[idx, partner] = U[0, 1]
            out[partner, idx] = U[1, 0]
            out[partner, partner] = U[1, 1]
        return out


@dataclass
class ControlledU:
    """A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity everywhere except the single controlled block: the pair (all
        ones except the target bit, all ones).
        """
        N = 1 << self.n
        out = np.eye(N, dtype=rotation.DTYPE)
        tbit = 1 << self.target
        all_ones = N - 1
        i = all_ones & ~tbit
        j = all_ones
        U = self.unitary
        out[i, i] = U[0, 0]
        out[i, j] = U[0, 1]
        out[j, i] = U[1, 0]
        out[j, j] = U[1, 1]
        return out


@dataclass
class CU:
    """A singly-controlled single-qubit gate C(U): apply the 2x2 `unitary` to
    `target` iff `control` is 1. The full U(2) (global phase kept) is stored, since
    under a control the global phase becomes a physical relative phase. This is the
    recursion leaf of decompose_controlled.
    """

    n: int
    control: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where `unitary` acts on `target`."""
        N = 1 << self.n
        out = np.eye(N, dtype=rotation.DTYPE)
        cbit = 1 << self.control
        tbit = 1 << self.target
        U = self.unitary
        for base in range(N):
            if not (base & cbit) or (base & tbit):
                continue
            i = base
            j = base | tbit
            out[i, i] = U[0, 0]
            out[i, j] = U[0, 1]
            out[j, i] = U[1, 0]
            out[j, j] = U[1, 1]
        return out


@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1. Its 2x2 is fixed to
    Pauli-X, so (unlike CU) it stores no unitary.
    """

    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where X swaps the target's 0/1
        amplitudes.
        """
        N = 1 << self.n
        out = np.eye(N, dtype=rotation.DTYPE)
        cbit = 1 << self.control
        tbit = 1 << self.target
        for base in range(N):
            if not (base & cbit) or (base & tbit):
                continue
            i = base
            j = base | tbit
            out[i, i] = 0.0
            out[j, j] = 0.0
            out[i, j] = 1.0
            out[j, i] = 1.0
        return out


@dataclass
class Swap:
    """A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`. `control_vals` has size n and is
    indexed by qubit; control_vals[target] is unused.
    """

    target: int
    control_vals: list[bool]


# A gate is any of the sparse representations above; a circuit is a list of gates.
Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list  # list[Gate]
TwoLevels = list  # list[TwoLevel]


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    """Full N x N unitary of a whole circuit. Gates are stored in order of
    application, so the product premultiplies (first gate is the rightmost factor):
    result = g_last @ ... @ g_1. Assumes the circuit is non-empty.
    """
    result = circuit[0].to_unitary()
    for g in circuit[1:]:
        result = g.to_unitary() @ result
    return result


def to_circuit(two_levels: TwoLevels) -> Circuit:
    """Wrap a two-level sequence as a circuit, so decompose_unitary /
    twolevel_decomposition output flows straight into a Circuit.
    """
    return list(two_levels)


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Elementwise difference between two same-size unitaries, ignoring an overall
    global phase: align b to a by the phase of their Hilbert-Schmidt overlap
    <b, a> = sum conj(b_ij) a_ij, then compare. ~0 means equal up to global phase.
    """
    overlap = np.sum(np.conj(b) * a)
    phase = np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(a - b * phase))


# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries (see cpp/src/TwoLevel.h)
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    """The 2x2 unitary [[conj(x), conj(y)], [-y, x]] / norm. Premultiplying it onto
    a column with entries (x, y) at two levels rotates the amplitude at the second
    level onto the first, leaving the real `norm` there and 0 below.
    """
    return np.array(
        [[np.conj(x), np.conj(y)], [-y, x]], dtype=rotation.DTYPE
    ) / norm


def decompose_vector(vec: np.ndarray) -> TwoLevels:
    """Given the first column of a unitary, return a sequence of two-levels which,
    when premultiplied onto the unitary, make its first column be (1, 0, 0, ...).
    Walk from the bottom up, using `align` at each pivot to zero out one entry; the
    running pivot holds the accumulated real norm after the first rotation.
    """
    n = vec.shape[0]
    v = vec.astype(rotation.DTYPE).copy()
    gates: TwoLevels = []
    pivot = 0
    for row in range(n - 1, 0, -1):
        x = v[pivot]
        y = v[row]
        r = np.hypot(abs(x), abs(y))
        if r < 1e-12:
            continue
        g = align(x, y, r)
        gates.append(TwoLevel(n, pivot, row, g))
        new_x = (g[0, 0] * x + g[0, 1] * y)
        v[pivot] = new_x
        v[row] = 0.0
    # residual phase at the pivot
    phase = v[pivot]
    if abs(abs(phase) - 1.0) > 1e-12 or abs(np.angle(phase)) > 1e-12:
        g = np.array([[np.conj(phase), 0], [0, phase]], dtype=rotation.DTYPE)
        gates.append(TwoLevel(n, pivot, 1, g))
    return gates


def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    """Expand each TwoLevel to n dimensions by shifting size, level0, level1 up by
    the offset (n - tl.size). Used to lift a sub-block decomposition back to full n.
    """
    offset = n - input[0].size if input else 0
    return [
        TwoLevel(n, tl.level0 + offset, tl.level1 + offset, tl.unitary)
        for tl in input
    ]


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    """Full matrix of a two-level sequence: premultiply each two-level's matrix in
    order (result = tl.to_unitary() @ result), reproducing the application order.
    """
    result = np.eye(two_levels[0].size, dtype=rotation.DTYPE)
    for tl in two_levels:
        result = tl.to_unitary() @ result
    return result


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    """Adjoint of a single two-level: same levels, adjoint (conjugate transpose) of
    the 2x2 block.
    """
    return TwoLevel(tl.size, tl.level0, tl.level1, tl.unitary.conj().T)


def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    """Adjoint of a sequence: reverse the order and take the adjoint of each, since
    (A_k ... A_1)^dagger = A_1^dagger ... A_k^dagger.
    """
    return [adjoint_twolevel(tl) for tl in reversed(two_levels)]


def decompose_unitary(u: np.ndarray) -> TwoLevels:
    """Repeat decompose_vector on successive sub-columns to reduce u to identity.
    At step k, columns/rows 0..k-1 are already reduced, so work on the lower-right
    (n-k) block: clear column k below the diagonal. Finally append a phase two-level
    on the last two levels to cancel the residual phase, so the product is identity.
    Returns the sequence S with prod(S) @ u == I (i.e. prod(S) = u^dagger).
    """
    n = u.shape[0]
    work = u.astype(rotation.DTYPE).copy()
    gates: TwoLevels = []
    for k in range(n - 1):
        col = work[k:, k]
        sub = decompose_vector(col)
        sub = expand_twolevels(sub, n)
        for tl in sub:
            work = tl.to_unitary() @ work
            gates.append(tl)
    # last entry residual phase
    phase = work[n - 1, n - 1]
    if abs(abs(phase) - 1.0) > 1e-12 or abs(np.angle(phase)) > 1e-12:
        g = np.array([[np.conj(phase), 0], [0, phase]], dtype=rotation.DTYPE)
        gates.append(TwoLevel(n, n - 2, n - 1, g))
    return gates


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    """The two-level decomposition of u itself: decompose_unitary returns the
    sequence S that reduces u to identity (prod(S) = u^dagger), so its adjoint is
    the sequence whose product is u.
    """
    return adjoint_twolevels(decompose_unitary(u))


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate (see cpp/src/ABC.h)
# ---------------------------------------------------------------------------


@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)


def abc_decompose(u: np.ndarray) -> ABC:
    """Build the ABC decomposition of u (Corollary 4.2). Take the ZYZ Euler angles
    (alpha, beta, gamma, delta) of u, then set
        A = Rz(beta) Ry(gamma/2)
        B = Ry(-gamma/2) Rz(-(delta+beta)/2)
        C = Rz((delta-beta)/2)
    Using X Ry(t) X = Ry(-t) and X Rz(t) X = Rz(-t), these satisfy A B C = I and
    e^{i alpha} A X B X C = u.
    """
    alpha, beta, gamma, delta = rotation.euler_angles_zyz(u)
    A = rotation.Rz(beta) @ rotation.Ry(gamma / 2.0)
    B = rotation.Ry(-gamma / 2.0) @ rotation.Rz(-(delta + beta) / 2.0)
    C = rotation.Rz((delta - beta) / 2.0)
    return ABC(alpha, A, B, C)


def abc_reconstruct(d: ABC) -> np.ndarray:
    """Reassemble e^{i alpha} A X B X C from an ABC (inverse of abc_decompose)."""
    X = rotation.X
    return np.exp(1j * d.alpha) * (d.A @ X @ d.B @ X @ d.C)


# ---------------------------------------------------------------------------
# Gray code and controlled circuits (see cpp/src/Swap.h, cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def gray_code(tl: TwoLevel) -> list[Swap]:
    """Gray code connecting level0 and level1 of a two-level, as the sequence of
    single-qubit flips walking from level0 to level1 (one Swap per differing bit).
    At each step the Swap records which qubit flips and the current code's values on
    the other qubits (the control pattern).
    """
    n = num_qubits(tl.size)
    a, b = tl.level0, tl.level1
    cur = a
    swaps: list[Swap] = []
    diff = a ^ b
    bits = [q for q in range(n) if diff & (1 << q)]
    for q in bits:
        cur ^= 1 << q
        vals = [(cur >> qq) & 1 == 1 for qq in range(n)]
        swaps.append(Swap(target=q, control_vals=vals))
    return swaps


def decompose_swap(swap: Swap) -> Circuit:
    """Decompose a Swap (multi-controlled NOT) into a Circuit: a controlled-X with
    the swap's arbitrary control values.
    """
    n = len(swap.control_vals)
    return controlled_circuit(n, swap.target, swap.control_vals, rotation.X)


def controlled_circuit(
    n: int, target: int, control_vals: list[bool], unitary: np.ndarray
) -> Circuit:
    """Circuit applying the 2x2 `unitary` to `target` iff every non-target qubit
    equals control_vals[q]. Realized as a fully-controlled-U core (ControlledU,
    controls = all 1) sandwiched by X gates on the qubits conditioned on 0, so those
    become 1-controls. The sandwich is symmetric (X is its own inverse).
    """
    circuit: Circuit = [SingleQubitGate(n, q, rotation.X) for q in range(n)
                        if q != target and not control_vals[q]]
    circuit.append(ControlledU(n, target, unitary))
    circuit += [SingleQubitGate(n, q, rotation.X) for q in range(n)
                if q != target and not control_vals[q]]
    return circuit


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline (see cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Lower a TwoLevel to a Circuit (Nielsen-Chuang 4.5.2): walk a gray code so
    level0 becomes adjacent to level1, apply the controlled-U on that last
    transition, then undo the walk. Orient the 2x2 so a00 (level0's corner) sits on
    the target value the second-to-last code has.
    """
    n = num_qubits(tl.size)
    swaps = gray_code(tl)

    walk = swaps[:-1]
    last = swaps[-1]

    forward: Circuit = []
    for s in walk:
        forward += decompose_swap(s)

    oriented = _orient_twolevel(tl, swaps)
    forward += controlled_circuit(
        n, last.target, last.control_vals, oriented
    )

    backward: Circuit = []
    for s in reversed(walk):
        backward += decompose_swap(s)

    return forward + backward


def _orient_twolevel(tl: TwoLevel, swaps: list[Swap]) -> np.ndarray:
    """Orient tl's 2x2 so its (0,0) corner lands on the basis state level0 reaches
    just before the final adjacent transition (the second-to-last code).
    """
    U = tl.unitary
    last = swaps[-1]
    tbit = 1 << last.target
    # basis state level0 sits at just before the last flip
    if len(swaps) >= 2:
        before_last = swaps[-2].control_vals
        code = 0
        for q in range(len(before_last)):
            if before_last[q]:
                code |= 1 << q
    else:
        code = tl.level0
    if code & tbit:
        return np.array([[U[1, 1], U[1, 0]], [U[0, 1], U[0, 0]]], dtype=rotation.DTYPE)
    return U.copy()


def decompose_controlled(
    n: int, controls: list[int], target: int, u: np.ndarray
) -> Circuit:
    """Decompose C^k(U) (k = len(controls)) into singly-controlled gates C(U) and
    CNOTs (Nielsen-Chuang fig 4.8). Base cases: no control -> a plain SingleQubitGate;
    one control -> a CNOT if U == X else a CU. Otherwise, with V = sqrt(U):
        a. C(V) on target
        b. C^{k-1}(X) onto the pivot control
        c. C(V dagger) on target
        d. repeat b
        e. C^{k-1}(V) on target
    Phases are kept throughout.
    """
    if len(controls) == 0:
        return [SingleQubitGate(n, target, u)]

    pivot = controls[-1]
    rest = controls[:-1]

    if len(controls) == 1:
        if np.allclose(u, rotation.X, atol=1e-9):
            return [CNOT(n, pivot, target)]
        return [CU(n, pivot, target, u)]

    V = rotation.unitary2_sqrt(u)
    Vdag = np.linalg.inv(V)

    circuit: Circuit = []
    circuit += decompose_controlled(n, [pivot], target, V)
    circuit += decompose_controlled(n, rest, pivot, rotation.X)
    circuit += decompose_controlled(n, [pivot], target, Vdag)
    circuit += decompose_controlled(n, rest, pivot, rotation.X)
    circuit += decompose_controlled(n, rest, target, V)
    return circuit


def decompose_controlledU(g: ControlledU) -> Circuit:
    """Lower a ControlledU (controlled on all other qubits) into CNOTs + C(U): build
    the list of all non-target qubits as controls and call decompose_controlled.
    """
    controls = [q for q in range(g.n) if q != g.target]
    return decompose_controlled(g.n, controls, g.target, g.unitary)


def decompose_cu(g: CU) -> Circuit:
    """Lower a singly-controlled C(U) into single-qubit gates + 2 CNOTs
    (Nielsen-Chuang Corollary 4.2 / fig 4.6). With U = e^{i alpha} A X B X C and
    A B C = I, emit: C, CNOT, B, CNOT, A on the target, plus a diag(1, e^{i alpha})
    phase on the control line. control=0: CNOTs vanish, target sees A B C = I;
    control=1: CNOTs act as X, target sees A X B X C = U with phase e^{i alpha}.
    """
    n = g.n
    ctl, tgt = g.control, g.target
    d = abc_decompose(g.unitary)

    phase = np.array([[1, 0], [0, np.exp(1j * d.alpha)]], dtype=rotation.DTYPE)
    circuit: Circuit = [
        SingleQubitGate(n, tgt, d.C),
        CNOT(n, ctl, tgt),
        SingleQubitGate(n, tgt, d.B),
        CNOT(n, ctl, tgt),
        SingleQubitGate(n, tgt, d.A),
        SingleQubitGate(n, ctl, phase),
    ]
    return circuit


def _rewrite(circuit: Circuit, rule) -> Circuit:
    out: Circuit = []
    for g in circuit:
        out.extend(rule(g))
    return out


def decompose_to_basis(u: np.ndarray) -> Circuit:
    """Fully lower a Unitary to a Circuit of only SingleQubitGate and CNOT, running
    the four stages in sequence:
        1. twolevel_decomposition: Unitary     -> TwoLevels
        2. decompose_twolevel:     TwoLevel    -> SingleQubitGate + ControlledU
        3. decompose_controlledU:  ControlledU -> CU + CNOT
        4. decompose_cu:           CU          -> SingleQubitGate + CNOT
    Each stage rewrites only its own gate type and passes the rest through unchanged.
    """
    tls = twolevel_decomposition(u)
    circuit = to_circuit(tls)
    circuit = _rewrite(circuit, lambda g: decompose_twolevel(g) if isinstance(g, TwoLevel) else [g])
    circuit = _rewrite(circuit, lambda g: decompose_controlledU(g) if isinstance(g, ControlledU) else [g])
    circuit = _rewrite(circuit, lambda g: decompose_cu(g) if isinstance(g, CU) else [g])
    return circuit


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    """Expand a flat H/T word into a Circuit of SingleQubitGate H/T gates on `qubit`.
    The word (leftmost char = leftmost matrix factor) is pushed in reverse so the
    circuit's application order (first gate first = rightmost factor) reproduces
    rotation.gates_to_unitary(word).
    """
    circuit: Circuit = []
    for c in reversed(word):
        U = rotation.T if c == "T" else rotation.H
        circuit.append(SingleQubitGate(n, qubit, U))
    return circuit


def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    """Fully lower a Unitary to a Circuit of only H, T, and CNOT gates (the discrete
    fault-tolerant basis): run decompose_to_basis, then replace each arbitrary
    SingleQubitGate with its {H, T} word from rotation.approximate_in_ht (CNOTs pass
    through). `error` is the per-gate angular tolerance (smaller -> longer, more
    accurate). Each word matches its gate up to a global phase; those per-gate phases
    factor out into one overall global phase, so the result reconstructs u up to
    global phase (compare with error_up_to_phase).
    """
    circuit = decompose_to_basis(u)
    out: Circuit = []
    for g in circuit:
        if isinstance(g, SingleQubitGate):
            word = rotation.approximate_in_ht(g.unitary, error)
            out.extend(ht_gates(g.n, g.qubit, word))
        else:
            out.append(g)
    return out
