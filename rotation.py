import numpy as np
from dataclasses import dataclass

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


@dataclass
class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

    G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """
    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def _wrap_angle(angle: float) -> float:
    return angle % TWO_PI


def _angle_distance(a: float, b: float) -> float:
    d = abs(_wrap_angle(a) - _wrap_angle(b))
    return min(d, TWO_PI - d)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """SO(3) rotation matrix for rotation by `angle` about unit `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ], dtype=float)


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    g = np.asarray(g, dtype=DTYPE)

    det_g = np.linalg.det(g)
    alpha = 0.5 * np.angle(det_g)

    su2 = g * np.exp(-1j * alpha)

    tr = np.trace(su2)
    cos_half_theta = np.clip(np.real(tr) / 2.0, -1.0, 1.0)
    theta = 2.0 * np.arccos(cos_half_theta)

    if np.isclose(theta, 0.0, atol=1e-12):
        n = np.array([1.0, 0.0, 0.0], dtype=float)
        theta = 0.0
        return Bloch(alpha=float(alpha), n=n, theta=float(theta))

    sin_half_theta = np.sin(theta / 2.0)

    if np.isclose(sin_half_theta, 0.0, atol=1e-12):
        n = np.array([1.0, 0.0, 0.0], dtype=float)
        return Bloch(alpha=float(alpha), n=n, theta=float(theta))

    a = su2
    nx = np.real(1j * (a[0, 1] + a[1, 0]) / (2.0 * sin_half_theta))
    ny = np.real((a[1, 0] - a[0, 1]) / (2.0 * sin_half_theta))
    nz = np.real(1j * (a[1, 1] - a[0, 0]) / (2.0 * sin_half_theta))

    n = np.array([nx, ny, nz], dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        n = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n = n / norm_n

    return Bloch(alpha=float(alpha), n=n, theta=float(theta))


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
n1 = np.array([1.0, 0.0, 0.0], dtype=float)
n2 = np.array([0.0, 1.0, 0.0], dtype=float)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
    u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    n = np.asarray(b.n, dtype=float)
    theta = float(b.theta)
    global_phase = float(b.alpha)

    # Build the SO(3) rotation corresponding to the Bloch rotation.
    R = _rotation_matrix(n, theta)

    # Express R in the {a1, a2, a3} frame.
    B = np.column_stack([a1, a2, a3])
    Rf = B.T @ R @ B

    beta = np.arccos(np.clip(Rf[0, 0], -1.0, 1.0))
    sin_beta = np.sin(beta)

    if abs(sin_beta) > 1e-12:
        alpha = np.arctan2(Rf[1, 0], -Rf[2, 0])
        gamma = np.arctan2(Rf[0, 1], Rf[0, 2])
    else:
        # Gimbal-lock style degeneracy: beta ≈ 0 or beta ≈ pi.
        if np.isclose(beta, 0.0, atol=1e-12):
            alpha = np.arctan2(Rf[2, 1], Rf[1, 1])
            gamma = 0.0
        else:
            alpha = np.arctan2(Rf[1, 2], Rf[1, 1])
            gamma = 0.0

    return (
        _wrap_angle(alpha),
        _wrap_angle(beta),
        _wrap_angle(gamma),
        global_phase,
    )


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
    (k * LAMBDA_PI) mod 2*pi ~= angle (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
    * wrap an angle into [0, 2 pi)
    * the angular distance between two wrapped angles a, b is
    min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    target = _wrap_angle(angle)
    k = 1
    while True:
        candidate = _wrap_angle(k * LAMBDA_PI)
        if _angle_distance(candidate, target) <= tolerance:
            return k
        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

    u ~= M1^k * M2^l * M1^m (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

    1. Get the Bloch form of u (to_bloch), then factor its rotation into the
    three frame angles with n1n2n1_angles:
    alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
    alpha and gamma are rotations about a1 (realized by powers of M1);
    beta is a rotation about a2 (realized by powers of M2).

    2. Convert each angle to an integer power with approx_angle_with_tolerance:
    k = approx_angle_with_tolerance(alpha, tolerance) # power of M1
    l = approx_angle_with_tolerance(beta, tolerance) # power of M2
    m = approx_angle_with_tolerance(gamma, tolerance) # power of M1
    (Mind the relationship between a target rotation angle and the base
    angle each application of M1/M2 adds.)

    3. Return (k, l, m).
    """
    alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    return k, l, m
