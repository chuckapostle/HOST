"""
PV Concentrator Ray-Optics Engine
==================================
Covers 400–1100 nm (silicon PV absorption range).

Structure
---------
1. Physics primitives  : ray_sphere_intersect(), vector_snell()
2. Objects             : Material, Ray, Surface, LayeredLensSystem
3. Illumination        : make_input_bundle()
4. Metrics             : rms_spot_radius(), concentration_cost()
5. System builder      : build_2layer_system(params)
6. Optimizer           : optimize_lens()
7. Unit tests          : run_tests()
8. Visualization       : plot_spot_diagrams(), plot_focus_shift(),
                         plot_optimized_results()
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSICS PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def ray_sphere_intersect(origin, direction, center, radius):
    """
    Intersect a ray with a sphere.

    Parameters
    ----------
    origin    : (3,) array  – ray origin O
    direction : (3,) array  – unit ray direction D
    center    : (3,) array  – sphere center C
    radius    : float       – sphere radius R (signed radius allowed)

    Returns
    -------
    t : float or None  – parameter of nearest positive intersection
    p : (3,) array or None  – intersection point
    n : (3,) array or None  – outward unit normal at p
    """
    O = np.asarray(origin,    dtype=float)
    D = np.asarray(direction, dtype=float)
    C = np.asarray(center,    dtype=float)
    R = float(radius)

    oc = O - C
    a  = np.dot(D, D)            # 1 if D is normalised
    b  = 2.0 * np.dot(D, oc)
    c  = np.dot(oc, oc) - R * R
    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return None, None, None

    sqrt_disc = np.sqrt(disc)
    t = (-b - sqrt_disc) / (2.0 * a)  # nearest root

    if t <= 1e-9:                      # try the farther root
        t = (-b + sqrt_disc) / (2.0 * a)
    if t <= 1e-9:
        return None, None, None

    p = O + t * D
    n = (p - C) / np.linalg.norm(p - C)
    return t, p, n


def vector_snell(k_in, n_hat, n1, n2):
    """
    Refract a ray at an interface using the vector form of Snell's law.

    Parameters
    ----------
    k_in  : (3,) unit vector – incident ray direction
    n_hat : (3,) unit vector – surface normal pointing from medium 1 → 2
    n1    : float – index of incidence medium
    n2    : float – index of transmission medium

    Returns
    -------
    k_out : (3,) unit vector, or None if total internal reflection occurs
    """
    k_in  = np.asarray(k_in,  dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    eta = n1 / n2
    ci  = -np.dot(k_in, n_hat)          # cos(θ_i), must be ≥ 0

    # Ensure normal points toward the incident side (ci > 0)
    if ci < 0.0:
        n_hat = -n_hat
        ci    = -ci

    k_sq = 1.0 - eta**2 * (1.0 - ci**2)  # cos²(θ_t)

    if k_sq < 0.0:
        return None                        # total internal reflection

    k_out = eta * k_in + (eta * ci - np.sqrt(k_sq)) * n_hat
    norm  = np.linalg.norm(k_out)
    if norm < 1e-12:
        return None
    return k_out / norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. OBJECTS
# ─────────────────────────────────────────────────────────────────────────────

# ── 2.1  Material ─────────────────────────────────────────────────────────────

class Material:
    """
    Optical glass characterised by a Sellmeier dispersion formula.

        n²(λ) = 1 + Σ  B_i λ² / (λ² − C_i)

    λ is in **metres**.

    Built-in presets
    ----------------
    Material.BK7()      – Schott N-BK7  (crown, nD ≈ 1.517)
    Material.F2()       – Schott F2     (dense flint, nD ≈ 1.620)
    Material.SF11()     – Schott SF11   (extra-dense flint, nD ≈ 1.785)
    Material.LASF9()    – Schott N-LASF9 (lanthanum flint, nD ≈ 1.850)
    Material.FK51A()    – Schott N-FK51A (low-dispersion crown, nD ≈ 1.487)
    Material.AIR()      – vacuum / air  (n = 1.000)
    """

    def __init__(self, name: str, B: list, C: list):
        self.name = name
        self.B = np.asarray(B, dtype=float)   # Sellmeier B coefficients
        self.C = np.asarray(C, dtype=float)   # Sellmeier C coefficients [m²]

    def n(self, lam: float) -> float:
        """Refractive index at wavelength lam [m]."""
        lam2 = lam * lam
        n2   = 1.0 + np.sum(self.B * lam2 / (lam2 - self.C))
        if n2 <= 0.0:
            return 1.0
        return float(np.sqrt(n2))

    def abbe_number(self, lam_d=587.6e-9, lam_f=486.1e-9, lam_c=656.3e-9):
        """Abbe V-number: V = (n_d − 1) / (n_f − n_c)."""
        nd = self.n(lam_d)
        nf = self.n(lam_f)
        nc = self.n(lam_c)
        denom = nf - nc
        if abs(denom) < 1e-12:
            return float("inf")
        return (nd - 1.0) / denom

    # ── Presets ────────────────────────────────────────────────────────────
    @staticmethod
    def BK7():
        return Material("N-BK7",
            B=[1.03961212, 0.231792344, 1.01046945],
            C=[6.00069867e-15, 2.00179144e-14, 1.03560653e-10])

    @staticmethod
    def F2():
        return Material("F2",
            B=[1.34533359, 0.209073176, 0.937357162],
            C=[9.97743871e-15, 4.70450767e-14, 1.11886764e-10])

    @staticmethod
    def SF11():
        return Material("SF11",
            B=[1.73759695, 0.313747346, 1.89878101],
            C=[1.31887070e-14, 6.23068142e-14, 1.55236290e-10])

    @staticmethod
    def LASF9():
        return Material("N-LASF9",
            B=[2.00029547, 0.298926886, 1.80691843],
            C=[1.21426017e-14, 5.34549042e-14, 1.56971974e-10])

    @staticmethod
    def FK51A():
        return Material("N-FK51A",
            B=[0.971247817, 0.216901417, 0.904651666],
            C=[4.72301995e-15, 1.53575612e-14, 1.68681330e-10])

    @staticmethod
    def AIR():
        """Air / vacuum: n = 1.000 at all wavelengths."""
        return _AirMaterial()

    def __repr__(self):
        return f"Material({self.name}, V={self.abbe_number():.1f})"


class _AirMaterial(Material):
    """Special-case material with n ≡ 1.000."""
    def __init__(self):
        super().__init__("AIR", [], [])

    def n(self, lam: float) -> float:
        return 1.0

    def abbe_number(self, **_):
        return float("inf")


AIR = Material.AIR()


# ── 2.2  Ray ──────────────────────────────────────────────────────────────────

class Ray:
    """
    Geometric ray in 3-D.

    Attributes
    ----------
    o : (3,) ndarray – current origin
    d : (3,) ndarray – unit direction
    """

    def __init__(self, origin, direction):
        self.o = np.array(origin,    dtype=float)
        self.d = np.array(direction, dtype=float)
        norm   = np.linalg.norm(self.d)
        if norm < 1e-12:
            raise ValueError("Ray direction must be non-zero.")
        self.d /= norm

    def at(self, t: float):
        """Point along the ray at parameter t."""
        return self.o + t * self.d

    def propagate_to_z(self, z: float):
        """
        Advance the origin to the plane z = const.
        Modifies self.o in-place and returns self.
        """
        if abs(self.d[2]) < 1e-12:
            raise ValueError("Ray is nearly parallel to the z=const plane.")
        t      = (z - self.o[2]) / self.d[2]
        self.o = self.at(t)
        return self

    def copy(self):
        return Ray(self.o.copy(), self.d.copy())

    def __repr__(self):
        return f"Ray(o={self.o}, d={self.d})"


# ── 2.3  Surface ──────────────────────────────────────────────────────────────

class Surface:
    """
    Spherical refracting surface separating two optical media.

    Parameters
    ----------
    center    : (3,) – sphere centre (defines vertex position and z-location)
    radius    : float – signed radius of curvature
                  > 0  →  centre of curvature is to the right (+z)
                  < 0  →  centre of curvature is to the left  (−z)
                  ±inf →  flat surface (treated as very large sphere)
    mat_left  : Material on the –z side  (None → AIR)
    mat_right : Material on the +z side  (None → AIR)
    """

    def __init__(self, center, radius, mat_left=None, mat_right=None):
        self.center    = np.array(center, dtype=float)
        self.R         = float(radius)
        self.mat_left  = mat_left  or AIR
        self.mat_right = mat_right or AIR

    # ── Intersection ──────────────────────────────────────────────────────
    def intersect(self, ray: Ray):
        """
        Returns (point, normal) of the nearest intersection, or (None, None).
        Normal is oriented to point from left medium toward right medium.
        """
        t, p, n = ray_sphere_intersect(ray.o, ray.d, self.center, self.R)
        if p is None:
            return None, None

        # Ensure n points from mat_left (−z) to mat_right (+z): i.e. n_z > 0
        if n[2] < 0.0:
            n = -n
        return p, n

    # ── Refraction ────────────────────────────────────────────────────────
    def refract(self, ray: Ray, lam: float, going_forward: bool = True):
        """
        Refract *ray* through this surface at wavelength *lam* [m].

        Parameters
        ----------
        going_forward : True  → light travels in +z direction (left→right)
                        False → light travels in −z direction (right→left)

        Returns
        -------
        Refracted Ray, or None if TIR or miss.
        """
        p, n = self.intersect(ray)
        if p is None:
            return None

        if going_forward:
            n1 = self.mat_left.n(lam)
            n2 = self.mat_right.n(lam)
            # n already points toward mat_right; flip if ray comes from right
        else:
            n1 = self.mat_right.n(lam)
            n2 = self.mat_left.n(lam)
            n  = -n          # now points toward mat_left

        k_out = vector_snell(ray.d, n, n1, n2)
        if k_out is None:
            return None     # TIR

        return Ray(p, k_out)


# ── 2.4  LayeredLensSystem ────────────────────────────────────────────────────

class LayeredLensSystem:
    """
    Sequence of spherical surfaces followed by propagation to a target plane.

    Parameters
    ----------
    surfaces : list[Surface]  – in order along the optical axis
    z_target : float          – z-coordinate of the detector / PV plane
    """

    def __init__(self, surfaces, z_target: float):
        self.surfaces = surfaces
        self.z_target = z_target

    def trace_ray(self, ray: Ray, lam: float, going_forward: bool = True):
        """
        Trace a single ray through all surfaces and propagate to z_target.

        Returns the final Ray (origin = impact point on PV plane), or None.
        """
        for surf in self.surfaces:
            new_ray = surf.refract(ray, lam, going_forward=going_forward)
            if new_ray is None:
                return None
            ray = new_ray

        try:
            ray.propagate_to_z(self.z_target)
        except ValueError:
            return None
        return ray

    def trace_bundle(self, rays, lam: float, going_forward: bool = True):
        """
        Trace a list of rays at wavelength *lam*. Returns list of output Rays.
        """
        out = []
        for r in rays:
            rr = self.trace_ray(r.copy(), lam, going_forward=going_forward)
            if rr is not None:
                out.append(rr)
        return out

    def find_paraxial_focus(self, lam: float, z_search=None):
        """
        Scan z to find the z-plane that minimises RMS spot radius for lam.
        Returns (z_focus, rms_at_focus).
        """
        if z_search is None:
            z_search = np.linspace(
                self.z_target * 0.5, self.z_target * 2.0, 300)

        rays_in = make_input_bundle()
        best_z, best_rms = z_search[0], float("inf")

        for z in z_search:
            sys_tmp = LayeredLensSystem(self.surfaces, z)
            rays_out = sys_tmp.trace_bundle(rays_in, lam)
            if not rays_out:
                continue
            rms = rms_spot_radius(rays_out)
            if rms < best_rms:
                best_rms = rms
                best_z   = z

        return best_z, best_rms


# ─────────────────────────────────────────────────────────────────────────────
# 3. ILLUMINATION & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def make_input_bundle(radius: float = 1e-3, n_rays: int = 25) -> list:
    """
    Create a square grid of paraxial rays (direction +z) over a circular
    aperture approximation.

    Parameters
    ----------
    radius : aperture half-width [m]
    n_rays : approximate total number of rays (nearest perfect square used)

    Returns
    -------
    list[Ray]
    """
    side = int(np.round(np.sqrt(n_rays)))
    xs   = np.linspace(-radius, radius, side)
    ys   = np.linspace(-radius, radius, side)
    rays = []
    for x in xs:
        for y in ys:
            rays.append(Ray(origin=[x, y, 0.0], direction=[0.0, 0.0, 1.0]))
    return rays


def rms_spot_radius(rays) -> float:
    """
    RMS radial spot size on the detector plane.

    Parameters
    ----------
    rays : list[Ray] – after propagation; ray.o is the impact point

    Returns
    -------
    float – RMS radius [m]
    """
    if not rays:
        return float("inf")
    pts = np.array([r.o for r in rays])
    x   = pts[:, 0]
    y   = pts[:, 1]
    return float(np.sqrt(np.mean(x**2 + y**2)))


def pv_weight(lam: float) -> float:
    """
    Spectral weight: 2× for the peak Si PV response band (500–900 nm),
    1× elsewhere.
    """
    return 2.0 if 500e-9 <= lam <= 900e-9 else 1.0


# Default wavelength grid: 15 points from 400 nm to 1100 nm
LAMBDA_GRID = np.linspace(400e-9, 1100e-9, 15)


def concentration_cost(params, system_builder, lam_list=None) -> float:
    """
    Weighted sum of RMS² spot sizes across the wavelength grid.

    Parameters
    ----------
    params         : 1-D array of design variables
    system_builder : callable(params) → LayeredLensSystem
    lam_list       : wavelength grid [m]; defaults to LAMBDA_GRID

    Returns
    -------
    float – cost (lower is better)
    """
    if lam_list is None:
        lam_list = LAMBDA_GRID

    system   = system_builder(params)
    rays_in  = make_input_bundle()
    E        = 0.0

    for lam in lam_list:
        rays_out = system.trace_bundle(rays_in, lam)
        if not rays_out:
            E += 1e6           # heavy penalty for lost rays
            continue
        rms = rms_spot_radius(rays_out)
        E  += pv_weight(lam) * rms**2

    return E


# ─────────────────────────────────────────────────────────────────────────────
# 4. SYSTEM BUILDER  (2-layer apochromatic doublet + singlet)
# ─────────────────────────────────────────────────────────────────────────────
#
# Layout (all distances in metres, z increases to the right):
#
#   z=0       z=t1     z=t1+t2      z=t1+t2+t3
#   |<-- air -->|<- mat1 ->|<-- mat2 -->|<--- air --->|
#  S0         S1          S2          S3
#
#   S0: entry surface  (air | mat1),  centre at (0,0, z0 + R0)
#   S1: mid   surface  (mat1 | mat2), centre at (0,0, z1 + R1)
#   S2: exit  surface  (mat2 | air),  centre at (0,0, z2 + R2)
#
# params = [R0, R1, R2, t1, t2, f]
#   R0, R1, R2 : radii of curvature [m]
#   t1, t2     : thicknesses of mat1, mat2 [m]
#   f          : target focal length ≈ z_target [m]

def build_2layer_system(params,
                        mat1: Material = None,
                        mat2: Material = None) -> LayeredLensSystem:
    """
    Build a 3-surface, 2-material layered lens system.

    Default materials: N-BK7 (crown) + F2 (flint) → achromatic base.
    """
    if mat1 is None:
        mat1 = Material.BK7()
    if mat2 is None:
        mat2 = Material.F2()

    R0, R1, R2, t1, t2, f = params

    # Guard: keep thicknesses positive and sensible
    t1 = max(t1, 0.5e-3)
    t2 = max(t2, 0.5e-3)

    # Vertex z-positions
    z0 = 0.0
    z1 = z0 + t1
    z2 = z1 + t2

    # Sphere centres: for a paraxial lens, the centre of curvature is offset
    # from the vertex along z by the radius.
    c0 = np.array([0.0, 0.0, z0 + R0])
    c1 = np.array([0.0, 0.0, z1 + R1])
    c2 = np.array([0.0, 0.0, z2 + R2])

    surfaces = [
        Surface(c0, R0, AIR,  mat1),
        Surface(c1, R1, mat1, mat2),
        Surface(c2, R2, mat2, AIR),
    ]

    return LayeredLensSystem(surfaces, z_target=f)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

def optimize_lens(mat1: Material = None,
                  mat2: Material = None,
                  x0=None,
                  method: str = "Nelder-Mead",
                  maxiter: int = 1500) -> dict:
    """
    Minimise concentration_cost for a 2-layer doublet + singlet lens.

    Parameters
    ----------
    mat1, mat2 : materials for the two layers
    x0         : initial params [R0, R1, R2, t1, t2, f] (metres)
    method     : scipy.optimize method
    maxiter    : maximum optimiser iterations

    Returns
    -------
    dict with keys 'params', 'cost', 'result', 'system'
    """
    if mat1 is None:
        mat1 = Material.BK7()
    if mat2 is None:
        mat2 = Material.F2()

    if x0 is None:
        x0 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])

    def builder(p):
        return build_2layer_system(p, mat1=mat1, mat2=mat2)

    # ── Fast coarse cost (fewer rays + coarser λ grid) used during search ──
    FAST_LAMS = np.linspace(400e-9, 1100e-9, 7)   # 7 wavelengths instead of 15

    def fast_bundle():
        return make_input_bundle(radius=1e-3, n_rays=9)  # 9 rays instead of 25

    def fast_cost(params):
        system = builder(params)
        rays_in = fast_bundle()
        E = 0.0
        for lam in FAST_LAMS:
            rays_out = system.trace_bundle(rays_in, lam)
            if not rays_out:
                E += 1e6
                continue
            E += pv_weight(lam) * rms_spot_radius(rays_out)**2
        return E

    # ── Progress callback ──────────────────────────────────────────────────
    _state = {"nit": 0, "last_print": 0}

    def callback(_):
        _state["nit"] += 1
        n = _state["nit"]
        if n - _state["last_print"] >= 50:   # print every 50 iterations
            _state["last_print"] = n
            print(f"  ... iteration {n}", flush=True)

    print(f"\nOptimising {mat1.name} / {mat2.name} lens ...")
    init_cost = concentration_cost(x0, builder)   # full-res for display only
    print(f"  Initial cost : {init_cost:.6e}")
    print(f"  Running up to {maxiter} iterations (progress every 50) ...")

    result = minimize(
        fast_cost,
        x0,
        method=method,
        callback=callback,
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-14,
                 "adaptive": True},
    )

    # Re-evaluate winner at full resolution for an honest final cost
    final_cost = concentration_cost(result.x, builder)
    print(f"  Iterations   : {_state['nit']}  (converged={result.success})")
    print(f"  Final cost   : {final_cost:.6e}")

    system = builder(result.x)
    return {
        "params": result.x,
        "cost":   final_cost,
        "result": result,
        "system": system,
        "mat1":   mat1,
        "mat2":   mat2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _assert_close(a, b, tol=1e-9, label=""):
    if abs(a - b) > tol:
        raise AssertionError(
            f"FAIL [{label}]: {a:.6e} != {b:.6e}  (tol={tol:.1e})")
    print(f"  PASS [{label}]")


def run_tests():
    """Run a suite of physics-level unit tests."""
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    # ── T1: ray-sphere intersection – head-on ─────────────────────────────
    t, p, n = ray_sphere_intersect(
        origin=[0, 0, -5], direction=[0, 0, 1],
        center=[0, 0, 0], radius=2.0)
    _assert_close(t, 3.0,  tol=1e-12, label="T1 hit t")
    _assert_close(p[2], -2.0, tol=1e-12, label="T1 hit z")
    _assert_close(n[2], -1.0, tol=1e-12, label="T1 normal z")

    # ── T2: ray-sphere miss ───────────────────────────────────────────────
    t2, p2, n2 = ray_sphere_intersect(
        origin=[0, 5, -5], direction=[0, 0, 1],
        center=[0, 0, 0], radius=2.0)
    assert t2 is None, "FAIL [T2]: expected miss"
    print("  PASS [T2 miss]")

    # ── T3: Snell straight-through (normal incidence) ─────────────────────
    k_in  = np.array([0.0, 0.0, 1.0])
    n_hat = np.array([0.0, 0.0, 1.0])
    k_out = vector_snell(k_in, n_hat, 1.0, 1.5)
    _assert_close(k_out[2], 1.0, tol=1e-12, label="T3 normal incidence k_z")

    # ── T4: Snell 45° air→glass ──────────────────────────────────────────
    # θ_i = 45°, n1=1, n2=1.5  → sin θ_t = sin(45°)/1.5
    theta_i  = np.radians(45.0)
    k_in4    = np.array([np.sin(theta_i), 0.0, np.cos(theta_i)])
    n_hat4   = np.array([0.0, 0.0, 1.0])
    k_out4   = vector_snell(k_in4, n_hat4, 1.0, 1.5)
    sin_out  = k_out4[0]
    sin_expected = np.sin(theta_i) / 1.5
    _assert_close(sin_out, sin_expected, tol=1e-10, label="T4 Snell x-component")

    # ── T5: TIR detection ────────────────────────────────────────────────
    theta_c  = np.arcsin(1.0 / 1.5)           # critical angle
    theta_tir = theta_c + 0.05
    k_in5    = np.array([np.sin(theta_tir), 0.0, np.cos(theta_tir)])
    n_hat5   = np.array([0.0, 0.0, 1.0])
    k_out5   = vector_snell(k_in5, n_hat5, 1.5, 1.0)
    assert k_out5 is None, "FAIL [T5]: expected TIR"
    print("  PASS [T5 TIR]")

    # ── T6: Single refracting surface – plano-convex air→glass ───────────
    lam = 550e-9
    glass = Material.BK7()
    surf = Surface(center=[0, 0, 50e-3], radius=50e-3, mat_left=AIR, mat_right=glass)
    ray  = Ray([0.0, 1e-3, 0.0], [0.0, 0.0, 1.0])   # 1 mm off-axis paraxial
    r_out = surf.refract(ray, lam)
    assert r_out is not None, "FAIL [T6]: expected refraction"
    # refracted ray should bend toward axis (y-component of direction becomes negative)
    assert r_out.d[1] < 0.0, "FAIL [T6]: paraxial ray not converging"
    print("  PASS [T6 plano-convex convergence]")

    # ── T7: Sellmeier dispersion – BK7 nD ≈ 1.5168 ───────────────────────
    bk7  = Material.BK7()
    nD   = bk7.n(587.6e-9)
    _assert_close(nD, 1.5168, tol=5e-4, label="T7 BK7 nD")

    # ── T8: Sellmeier dispersion – F2 nD ≈ 1.6200 ────────────────────────
    f2 = Material.F2()
    nD = f2.n(587.6e-9)
    _assert_close(nD, 1.6200, tol=5e-3, label="T8 F2 nD")

    # ── T9: Air-glass-air slab → ray returns to same angle ────────────────
    # Flat surfaces (very large R), parallel slab
    big_R = 1e6
    slab_mat = Material.BK7()
    s_in  = Surface([0, 0, big_R + 10e-3], big_R, AIR,      slab_mat)
    s_out = Surface([0, 0, big_R + 15e-3], big_R, slab_mat, AIR)
    theta  = np.radians(10.0)
    ray9   = Ray([0, 0, 0], [np.sin(theta), 0, np.cos(theta)])
    r1 = s_in.refract(ray9, 550e-9)
    assert r1 is not None
    r2 = s_out.refract(r1, 550e-9)
    assert r2 is not None
    _assert_close(r2.d[0], ray9.d[0], tol=1e-6, label="T9 slab x-dir restored")

    print("\nAll tests PASSED.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _wavelength_to_rgb(lam: float):
    """Approximate wavelength [m] → (R, G, B) for plotting."""
    nm = lam * 1e9
    if   nm < 380:              return (0.5, 0.0, 0.5)
    elif nm < 440:
        r = (440 - nm) / 60.0
        return (r, 0.0, 1.0)
    elif nm < 490:
        g = (nm - 440) / 50.0
        return (0.0, g, 1.0)
    elif nm < 510:
        b = (510 - nm) / 20.0
        return (0.0, 1.0, b)
    elif nm < 580:
        r = (nm - 510) / 70.0
        return (r, 1.0, 0.0)
    elif nm < 645:
        g = (645 - nm) / 65.0
        return (1.0, g, 0.0)
    elif nm <= 780:
        return (1.0, 0.0, 0.0)
    else:
        return (0.8, 0.0, 0.0)


def plot_spot_diagrams(system: LayeredLensSystem,
                       lam_list=None,
                       title: str = "Spot Diagrams"):
    """
    Spot diagrams at the PV plane for each wavelength in lam_list.
    """
    if lam_list is None:
        lam_list = LAMBDA_GRID

    rays_in = make_input_bundle()
    n_lam   = len(lam_list)
    ncols   = min(5, n_lam)
    nrows   = int(np.ceil(n_lam / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    for idx, lam in enumerate(lam_list):
        ax  = axes[idx // ncols][idx % ncols]
        col = _wavelength_to_rgb(lam)
        rays_out = system.trace_bundle(rays_in, lam)
        if rays_out:
            pts = np.array([r.o for r in rays_out])
            ax.scatter(pts[:, 0] * 1e6, pts[:, 1] * 1e6,
                       s=10, color=col, alpha=0.8)
            rms = rms_spot_radius(rays_out)
            ax.set_title(f"{lam*1e9:.0f} nm\nRMS={rms*1e6:.1f} µm", fontsize=8)
        else:
            ax.set_title(f"{lam*1e9:.0f} nm\n(no rays)", fontsize=8)
            ax.text(0.5, 0.5, "×", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=20)

        ax.set_aspect("equal")
        ax.axhline(0, lw=0.5, color="k", alpha=0.3)
        ax.axvline(0, lw=0.5, color="k", alpha=0.3)
        ax.set_xlabel("x [µm]", fontsize=7)
        ax.set_ylabel("y [µm]", fontsize=7)

    # Hide unused subplots
    for idx in range(n_lam, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def plot_focus_shift(system: LayeredLensSystem,
                     lam_list=None,
                     title: str = "Chromatic Focus Shift"):
    """
    Show how the best-focus z and RMS spot size vary with wavelength
    (useful for diagnosing chromatic aberration before optimisation).
    """
    if lam_list is None:
        lam_list = LAMBDA_GRID

    z_nom  = system.z_target
    z_scan = np.linspace(z_nom * 0.5, z_nom * 2.0, 200)

    focus_z   = []
    focus_rms = []

    for lam in lam_list:
        zf, rms = system.find_paraxial_focus(lam, z_search=z_scan)
        focus_z.append(zf * 1e3)        # → mm
        focus_rms.append(rms * 1e6)     # → µm

    lam_nm = np.array(lam_list) * 1e9
    colors  = [_wavelength_to_rgb(l) for l in lam_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=13)

    for i, (lam, zf, rms, c) in enumerate(zip(lam_nm, focus_z, focus_rms, colors)):
        ax1.scatter(lam, zf,  color=c, s=40, zorder=3)
        ax2.scatter(lam, rms, color=c, s=40, zorder=3)

    ax1.plot(lam_nm, focus_z,  "k-", lw=0.8, alpha=0.5)
    ax2.plot(lam_nm, focus_rms,"k-", lw=0.8, alpha=0.5)

    ax1.set_xlabel("Wavelength [nm]");  ax1.set_ylabel("Best-focus z [mm]")
    ax2.set_xlabel("Wavelength [nm]");  ax2.set_ylabel("RMS spot at best-focus [µm]")
    ax1.set_title("Longitudinal chromatic aberration")
    ax2.set_title("Spot size at best focus")

    for ax in (ax1, ax2):
        ax.axvspan(500, 900, alpha=0.08, color="green", label="Peak PV band")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rms_vs_wavelength(system: LayeredLensSystem,
                           lam_list=None,
                           label: str = "system",
                           ax=None,
                           color="blue"):
    """
    Plot RMS spot radius vs wavelength at the system's fixed z_target.
    Returns the axes.
    """
    if lam_list is None:
        lam_list = LAMBDA_GRID

    rays_in = make_input_bundle()
    rms_vals = []
    for lam in lam_list:
        rays_out = system.trace_bundle(rays_in, lam)
        rms_vals.append(rms_spot_radius(rays_out) * 1e6 if rays_out else None)

    lam_nm = np.array(lam_list) * 1e9

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    valid = [(l, r) for l, r in zip(lam_nm, rms_vals) if r is not None]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", ms=4, label=label, color=color)

    ax.axvspan(500, 900, alpha=0.08, color="green", label="Peak PV band")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("RMS spot radius [µm]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_optimized_results(opt_result: dict,
                           initial_params=None,
                           mat1_init=None,
                           mat2_init=None):
    """
    Compare before/after optimisation: spot diagrams + RMS vs λ.
    """
    system_opt = opt_result["system"]
    mat1       = opt_result["mat1"]
    mat2       = opt_result["mat2"]

    fig_spot = plot_spot_diagrams(
        system_opt,
        title=f"Optimised Spot Diagrams ({mat1.name}/{mat2.name})")

    fig_rms, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"RMS Spot vs Wavelength – {mat1.name}/{mat2.name}")

    if initial_params is not None:
        m1 = mat1_init or mat1
        m2 = mat2_init or mat2
        sys_init = build_2layer_system(initial_params, mat1=m1, mat2=m2)
        plot_rms_vs_wavelength(sys_init, label="Initial", ax=ax, color="red")

    plot_rms_vs_wavelength(system_opt, label="Optimised", ax=ax, color="blue")
    plt.tight_layout()

    return fig_spot, fig_rms


# ─────────────────────────────────────────────────────────────────────────────
# 8. MATERIAL SURVEY  (coarse combinatorial screening)
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATE_MATERIALS = {
    "N-BK7":    Material.BK7(),
    "F2":       Material.F2(),
    "SF11":     Material.SF11(),
    "N-LASF9":  Material.LASF9(),
    "N-FK51A":  Material.FK51A(),
}


def material_survey(x0=None, verbose=True) -> list:
    """
    Evaluate concentration_cost for every (mat1, mat2) pair and return
    a ranked list of (cost, mat1_name, mat2_name).
    """
    if x0 is None:
        x0 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])

    names  = list(CANDIDATE_MATERIALS.keys())
    scores = []

    for n1_name in names:
        for n2_name in names:
            if n1_name == n2_name:
                continue
            m1 = CANDIDATE_MATERIALS[n1_name]
            m2 = CANDIDATE_MATERIALS[n2_name]
            cost = concentration_cost(
                x0, lambda p: build_2layer_system(p, mat1=m1, mat2=m2))
            scores.append((cost, n1_name, n2_name))

    scores.sort(key=lambda x: x[0])

    if verbose:
        print("\nMaterial survey (initial geometry):")
        print(f"  {'Cost':>12}  {'mat1':>10}  {'mat2':>10}")
        for cost, n1, n2 in scores[:10]:
            print(f"  {cost:12.4e}  {n1:>10}  {n2:>10}")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 9. CSV PARAMETER LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_params_from_csv(csv_path: str) -> dict:
    """
    Load simulation parameters from a two-column CSV file.

    Expected columns (additional columns such as 'unit' and 'description'
    are silently ignored):
        parameter  – parameter name (string key)
        value      – parameter value

    Recognised parameters
    ---------------------
    Lens geometry (values in millimetres, converted to metres internally):
        R0_mm    – entry surface radius of curvature
        R1_mm    – mid surface radius of curvature
        R2_mm    – exit surface radius of curvature
        t1_mm    – thickness of layer 1
        t2_mm    – thickness of layer 2
        f_mm     – target focal length

    Materials (must match a key in CANDIDATE_MATERIALS):
        mat1     – entry layer material name  (e.g. N-BK7)
        mat2     – exit  layer material name  (e.g. F2)

    Ray bundle:
        ray_radius_mm – aperture half-width  (default 1.0 mm)
        n_rays        – approximate number of rays (default 25)

    Wavelength grid:
        lam_start_nm  – first wavelength in nm (default 400)
        lam_end_nm    – last  wavelength in nm (default 1100)
        lam_steps     – number of wavelength samples (default 15)

    Optimiser:
        opt_method    – scipy.optimize method name (default Nelder-Mead)
        opt_maxiter   – maximum iterations (default 1500)

    Returns
    -------
    dict  – keys as listed above, values already cast to float / int.
    """
    float_keys = {
        "R0_mm", "R1_mm", "R2_mm", "t1_mm", "t2_mm", "f_mm",
        "ray_radius_mm", "lam_start_nm", "lam_end_nm",
    }
    int_keys = {"n_rays", "lam_steps", "opt_maxiter"}

    params: dict = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "parameter" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV '{csv_path}' must contain a 'parameter' column.")
        if "value" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV '{csv_path}' must contain a 'value' column.")
        for row in reader:
            key = row["parameter"].strip()
            val = row["value"].strip()
            if not key or not val:
                continue
            if key in float_keys:
                params[key] = float(val)
            elif key in int_keys:
                params[key] = int(val)
            else:
                params[key] = val   # str (material names, method names, …)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PV RAY-OPTICS ENGINE  (400–1100 nm)")
    print("=" * 60)

    # ── CLI argument: optional CSV parameter file ─────────────────────────
    parser = argparse.ArgumentParser(
        description="PV Concentrator Ray-Optics Engine")
    parser.add_argument(
        "--params", metavar="CSV", default=None,
        help="Path to a CSV parameter file (see params_template.csv).")
    args, _ = parser.parse_known_args()

    # ── Load parameters (CSV overrides built-in defaults) ─────────────────
    p: dict = {}
    if args.params:
        print(f"Loading parameters from: {args.params}")
        p = load_params_from_csv(args.params)

    R0_mm        = p.get("R0_mm",         50.0)
    R1_mm        = p.get("R1_mm",        -30.0)
    R2_mm        = p.get("R2_mm",        -50.0)
    t1_mm        = p.get("t1_mm",          5.0)
    t2_mm        = p.get("t2_mm",          5.0)
    f_mm         = p.get("f_mm",          50.0)
    mat1_name    = p.get("mat1",        "N-BK7")
    mat2_name    = p.get("mat2",           "F2")
    ray_radius   = p.get("ray_radius_mm",  1.0) * 1e-3   # mm → m
    n_rays       = p.get("n_rays",          25)
    lam_start    = p.get("lam_start_nm",  400.0) * 1e-9  # nm → m
    lam_end      = p.get("lam_end_nm",   1100.0) * 1e-9
    lam_steps    = p.get("lam_steps",       15)
    opt_method   = p.get("opt_method",  "Nelder-Mead")
    opt_maxiter  = p.get("opt_maxiter",   1500)

    x0 = np.array([R0_mm, R1_mm, R2_mm, t1_mm, t2_mm, f_mm]) * 1e-3
    lam_grid = np.linspace(lam_start, lam_end, lam_steps)

    if mat1_name not in CANDIDATE_MATERIALS:
        raise ValueError(
            f"Unknown mat1 '{mat1_name}'. "
            f"Choose from: {list(CANDIDATE_MATERIALS)}")
    if mat2_name not in CANDIDATE_MATERIALS:
        raise ValueError(
            f"Unknown mat2 '{mat2_name}'. "
            f"Choose from: {list(CANDIDATE_MATERIALS)}")

    init_mat1 = CANDIDATE_MATERIALS[mat1_name]
    init_mat2 = CANDIDATE_MATERIALS[mat2_name]

    print(f"\nParameters in use:")
    print(f"  Initial geometry (mm) : R0={R0_mm:+.1f}  R1={R1_mm:+.1f}  "
          f"R2={R2_mm:+.1f}  t1={t1_mm:.1f}  t2={t2_mm:.1f}  f={f_mm:.1f}")
    print(f"  Initial materials     : {mat1_name} / {mat2_name}")
    print(f"  Ray bundle            : radius={ray_radius*1e3:.2f} mm, "
          f"n_rays≈{n_rays}")
    print(f"  Wavelength grid       : {lam_start*1e9:.0f}–{lam_end*1e9:.0f} nm, "
          f"{lam_steps} points")
    print(f"  Optimiser             : method={opt_method}, "
          f"maxiter={opt_maxiter}")

    # ── Step 0: unit tests ────────────────────────────────────────────────
    run_tests()

    # ── Step 1: single-wavelength focus scan (unoptimised doublet) ────────
    print("Building initial 2-layer lens system ...")
    sys0 = build_2layer_system(x0, mat1=init_mat1, mat2=init_mat2)

    print("Plotting chromatic focus shift (initial system) ...")
    fig_shift = plot_focus_shift(
        sys0, lam_list=lam_grid,
        title=f"Chromatic Focus Shift – Initial {mat1_name}/{mat2_name}")
    fig_shift.savefig("focus_shift_initial.png", dpi=150)
    print("  Saved: focus_shift_initial.png")

    fig_spot0 = plot_spot_diagrams(
        sys0, lam_list=lam_grid,
        title=f"Spot Diagrams – Initial {mat1_name}/{mat2_name}")
    fig_spot0.savefig("spots_initial.png", dpi=150)
    print("  Saved: spots_initial.png")

    # ── Step 2: material survey ───────────────────────────────────────────
    scores = material_survey(x0)
    best_cost, best_m1_name, best_m2_name = scores[0]
    best_m1 = CANDIDATE_MATERIALS[best_m1_name]
    best_m2 = CANDIDATE_MATERIALS[best_m2_name]
    print(f"\nBest material pair from survey: {best_m1_name} / {best_m2_name}"
          f"  (cost={best_cost:.4e})")

    # ── Step 3: optimise best pair ────────────────────────────────────────
    opt = optimize_lens(
        mat1=best_m1, mat2=best_m2, x0=x0,
        method=opt_method, maxiter=opt_maxiter)

    # ── Step 4: visualise optimised results ───────────────────────────────
    fig_spot_opt, fig_rms = plot_optimized_results(
        opt, initial_params=x0,
        mat1_init=best_m1, mat2_init=best_m2)

    fig_spot_opt.savefig("spots_optimised.png", dpi=150)
    fig_rms.savefig("rms_comparison.png",       dpi=150)
    print("  Saved: spots_optimised.png")
    print("  Saved: rms_comparison.png")

    # ── Step 5: print final design ────────────────────────────────────────
    R0, R1, R2, t1, t2, f = opt["params"]
    print("\n── Optimised Design Parameters ──────────────────────────")
    print(f"  Material 1 (entry layer) : {opt['mat1'].name}")
    print(f"  Material 2 (exit  layer) : {opt['mat2'].name}")
    print(f"  R0 (entry surface)       : {R0*1e3:+.3f} mm")
    print(f"  R1 (mid   surface)       : {R1*1e3:+.3f} mm")
    print(f"  R2 (exit  surface)       : {R2*1e3:+.3f} mm")
    print(f"  Thickness layer 1        : {t1*1e3:.3f} mm")
    print(f"  Thickness layer 2        : {t2*1e3:.3f} mm")
    print(f"  Target focal length f    : {f*1e3:.3f} mm")
    print(f"  Final cost               : {opt['cost']:.4e}")

    # Print per-wavelength RMS summary
    print("\n── Per-wavelength RMS at PV plane ───────────────────────")
    print(f"  {'λ [nm]':>8}  {'weight':>6}  {'RMS [µm]':>10}")
    rays_in = make_input_bundle(radius=ray_radius, n_rays=n_rays)
    for lam in lam_grid:
        rays_out = opt["system"].trace_bundle(rays_in, lam)
        rms = rms_spot_radius(rays_out) * 1e6 if rays_out else float("nan")
        w   = pv_weight(lam)
        print(f"  {lam*1e9:8.1f}  {w:6.1f}  {rms:10.3f}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
