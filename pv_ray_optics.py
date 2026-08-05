"""
PV Concentrator Ray-Optics Engine
==================================
Covers 400–1100 nm (silicon PV absorption range).

Structure
---------
1. Physics primitives  : ray_sphere_intersect(), vector_snell()
2. Objects             : Material, Ray, Surface, LayeredLensSystem
3. Illumination        : make_input_bundle(), make_ray_fan()
4. Metrics             : rms_spot_radius(), concentration_cost()
5. System builder      : build_layered_system() / build_nlayer_system()
                         (N layers), build_2layer_system() (legacy 2-layer)
6. Optimizer           : optimize_lens()  (N-layer capable)
6b. Auto-design        : generate_material_combinations(), auto_design()
                         (joint material + geometry search)
7. Unit tests          : run_tests()
8. Visualization       : plot_spot_diagrams(), plot_focus_shift(),
                         plot_optimized_results()
"""

import csv
import argparse
import itertools
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

    # A ray legitimately starting exactly on the surface (t == 0, e.g. the
    # on-axis ray of an input bundle seeded at z=0 against a surface whose
    # own vertex is at z=0 by convention) must be accepted, not treated as
    # "behind the ray" — only a genuinely negative near root falls back to
    # the farther one (e.g. a ray already inside the sphere).
    if t < -1e-9:                      # try the farther root
        t = (-b + sqrt_disc) / (2.0 * a)
        if t <= 1e-9:
            return None, None, None
    else:
        t = max(t, 0.0)

    p = O + t * D
    n = (p - C) / np.linalg.norm(p - C)
    return t, p, n


def ray_fresnel_intersect(origin, direction, center, radius, groove_pitch):
    """
    Intersect a ray with a Fresnel-faceted approximation of a sphere.

    Models a Fresnel lens the standard "thin facet" way used in geometric
    (non-wave-optics) ray tracing: the physical surface is collapsed onto
    the flat vertex plane (real groove depths are tiny compared to the rest
    of the system), split into concentric annular zones of radial width
    `groove_pitch`. Within each zone the ray sees a single flat facet whose
    NORMAL matches the parent sphere's normal at that zone's center radius
    — the facet doesn't reproduce the sphere's shape, but it reproduces the
    parent sphere's local ray-bending direction, which is what a groove
    aims to do. Fewer/coarser zones (larger groove_pitch) means more
    faceting error, mirroring the real manufacturing tradeoff.

    Parameters
    ----------
    origin, direction : (3,) arrays – ray origin / unit direction
    center             : (3,) array – parent sphere's center (same
                          convention as ray_sphere_intersect / Surface)
    radius             : float – parent sphere's signed radius
    groove_pitch       : float – radial width of each zone [m]

    Returns
    -------
    t : float or None
    p : (3,) array or None – point on the flat vertex plane
    n : (3,) array or None – parent-sphere normal at the zone's center radius
    """
    O = np.asarray(origin,    dtype=float)
    D = np.asarray(direction, dtype=float)
    C = np.asarray(center,    dtype=float)
    R = float(radius)

    if abs(D[2]) < 1e-12:
        return None, None, None

    z0 = C[2] - R  # vertex z (same relationship build_layered_system uses)
    t = (z0 - O[2]) / D[2]
    # Accept a ray legitimately starting exactly on the flat facet plane
    # (t == 0) — see ray_sphere_intersect() for why only a genuinely
    # negative t is rejected as "behind the ray".
    if t < -1e-9:
        return None, None, None
    t = max(t, 0.0)

    p = O + t * D
    rho = float(np.hypot(p[0], p[1]))
    zone = np.floor(rho / groove_pitch)
    rho_ref = (zone + 0.5) * groove_pitch

    if rho > 1e-12:
        ux, uy = p[0] / rho, p[1] / rho
    else:
        ux, uy = 0.0, 0.0

    inside = max(R * R - rho_ref * rho_ref, 0.0)
    sign = 1.0 if R >= 0.0 else -1.0
    z_ref = C[2] - sign * np.sqrt(inside)
    point_ref = np.array([rho_ref * ux, rho_ref * uy, z_ref])

    n = (point_ref - C) / abs(R)
    return t, p, n


def fresnel_transmittance(n1, n2, cos_i, cos_t):
    """
    Unpolarized-light Fresnel power transmittance for a single interface —
    averages the s- and p-polarization reflectances (sunlight is
    unpolarized, so this codebase doesn't track polarization state at all).

    cos_i, cos_t : cosines of the incidence / refraction angles (>= 0);
                   scalar or array, matching n1/n2's usage site.
    n1, n2       : refractive indices of the incidence / transmission media

    Returns
    -------
    T : power transmittance, 1 - R (no absorption modeled)
    """
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    R = 0.5 * (rs**2 + rp**2)
    return 1.0 - R


def _snell_core(k_in, n_hat, n1, n2):
    """
    Shared implementation behind vector_snell() / vector_snell_T() — refract
    via the vector form of Snell's law and compute the Fresnel transmittance
    for the same interface in one pass (they need the same cos_i/cos_t).

    Returns (k_out, T, ok). ok=False (k_out=None, T=0.0) on TIR or a
    degenerate direction.
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
        return None, 0.0, False            # total internal reflection

    cos_t = np.sqrt(k_sq)
    k_out = eta * k_in + (eta * ci - cos_t) * n_hat
    norm  = np.linalg.norm(k_out)
    if norm < 1e-12:
        return None, 0.0, False

    T = fresnel_transmittance(n1, n2, ci, cos_t)
    return k_out / norm, T, True


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
    k_out, _T, ok = _snell_core(k_in, n_hat, n1, n2)
    return k_out if ok else None


def vector_snell_T(k_in, n_hat, n1, n2):
    """
    Like vector_snell(), but also returns the Fresnel power transmittance
    of this interface (fraction of incident power that continues on with
    the refracted ray; the rest is reflected, not absorbed).

    Returns (k_out, T), or (None, 0.0) on total internal reflection.
    """
    k_out, T, ok = _snell_core(k_in, n_hat, n1, n2)
    return (k_out, T) if ok else (None, 0.0)


# ── 1.1  Vectorized (batch) counterparts ───────────────────────────────────
# Same physics as ray_sphere_intersect() / vector_snell() above, but operate
# on (N,3) arrays of rays at once instead of one Ray object at a time. Used
# internally by LayeredLensSystem.trace_bundle() to avoid per-ray Python
# object overhead; the scalar functions above remain the source of truth
# that these are tested against (see run_tests() T10).

def _ray_sphere_intersect_batch(O, D, center, radius):
    """
    Vectorized ray-sphere intersection.

    O, D   : (N,3) arrays – ray origins / unit directions
    center : (3,) array
    radius : float (signed)

    Returns
    -------
    p    : (N,3) – intersection point (only meaningful where hit=True)
    n    : (N,3) – outward unit normal (only meaningful where hit=True)
    hit  : (N,) bool – whether a valid forward intersection exists
    """
    oc = O - center
    a = np.einsum("ij,ij->i", D, D)
    b = 2.0 * np.einsum("ij,ij->i", D, oc)
    c = np.einsum("ij,ij->i", oc, oc) - radius * radius
    disc = b * b - 4.0 * a * c

    hit = disc >= 0.0
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[hit] = np.sqrt(disc[hit])

    t_near = np.full_like(disc, np.nan)
    t_near[hit] = (-b[hit] - sqrt_disc[hit]) / (2.0 * a[hit])

    # A ray legitimately starting exactly on the surface (t == 0) must be
    # accepted, not treated as "behind the ray" — see the scalar
    # ray_sphere_intersect() for why. Only fall back to the farther root
    # when the near one is genuinely negative.
    near_ok = hit & (t_near >= -1e-9)
    t_near_clamped = np.where(near_ok, np.maximum(t_near, 0.0), np.nan)

    far_case = hit & ~near_ok
    t_far = np.full_like(disc, np.nan)
    t_far[far_case] = (-b[far_case] + sqrt_disc[far_case]) / (2.0 * a[far_case])
    far_ok = far_case & (t_far > 1e-9)

    t = np.where(near_ok, t_near_clamped, t_far)
    hit = near_ok | far_ok

    p = O + t[:, None] * D
    diff = p - center
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    norm_safe = np.where(norm < 1e-12, 1.0, norm)
    n = diff / norm_safe

    return p, n, hit


def _fresnel_intersect_batch(O, D, center, radius, groove_pitch):
    """
    Vectorized counterpart of ray_fresnel_intersect() — see that function's
    docstring for the physical model. Same (p, n, hit) contract as
    _ray_sphere_intersect_batch().
    """
    Dz = D[:, 2]
    hit = np.abs(Dz) >= 1e-12
    Dz_safe = np.where(hit, Dz, 1.0)

    z0 = center[2] - radius
    t = (z0 - O[:, 2]) / Dz_safe
    # Accept t == 0 (ray starting exactly on the facet plane) — see
    # ray_sphere_intersect() for why only genuinely negative t is rejected.
    hit = hit & (t > -1e-9)
    t = np.maximum(t, 0.0)

    p = O + t[:, None] * D
    rho = np.hypot(p[:, 0], p[:, 1])
    zone = np.floor(rho / groove_pitch)
    rho_ref = (zone + 0.5) * groove_pitch

    rho_safe = np.where(rho > 1e-12, rho, 1.0)
    ux = np.where(rho > 1e-12, p[:, 0] / rho_safe, 0.0)
    uy = np.where(rho > 1e-12, p[:, 1] / rho_safe, 0.0)

    sign = 1.0 if radius >= 0.0 else -1.0
    inside = np.maximum(radius * radius - rho_ref * rho_ref, 0.0)
    z_ref = center[2] - sign * np.sqrt(inside)

    point_ref = np.stack([rho_ref * ux, rho_ref * uy, z_ref], axis=1)
    n = (point_ref - center) / abs(radius)

    return p, n, hit


def _vector_snell_batch(k_in, n_hat, n1, n2):
    """
    Vectorized vector-form Snell's law, plus the per-ray Fresnel power
    transmittance of the interface (see fresnel_transmittance()).

    k_in, n_hat : (N,3) arrays
    n1, n2      : scalars (index of incidence / transmission medium)

    Returns
    -------
    k_out : (N,3) – refracted directions (only meaningful where ok=True)
    T     : (N,)  – Fresnel power transmittance (only meaningful where ok=True)
    ok    : (N,) bool – False where total internal reflection occurs
    """
    eta = n1 / n2
    ci = -np.einsum("ij,ij->i", k_in, n_hat)

    flip = ci < 0.0
    n_hat = np.where(flip[:, None], -n_hat, n_hat)
    ci = np.where(flip, -ci, ci)

    k_sq = 1.0 - eta**2 * (1.0 - ci**2)
    ok = k_sq >= 0.0

    sqrt_k = np.zeros_like(k_sq)
    sqrt_k[ok] = np.sqrt(k_sq[ok])

    k_out = eta * k_in + (eta * ci - sqrt_k)[:, None] * n_hat
    norm = np.linalg.norm(k_out, axis=1)
    ok = ok & (norm > 1e-12)
    norm_safe = np.where(norm < 1e-12, 1.0, norm)
    k_out = k_out / norm_safe[:, None]

    T = np.zeros_like(ci)
    T[ok] = fresnel_transmittance(n1, n2, ci[ok], sqrt_k[ok])

    return k_out, T, ok


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
    o     : (3,) ndarray – current origin
    d     : (3,) ndarray – unit direction
    power : float – fraction of incident power still carried by this ray,
                    accumulated as the product of each interface's Fresnel
                    transmittance (see Surface.refract). 1.0 = no losses yet.
    """

    def __init__(self, origin, direction, power: float = 1.0):
        self.o = np.array(origin,    dtype=float)
        self.d = np.array(direction, dtype=float)
        norm   = np.linalg.norm(self.d)
        if norm < 1e-12:
            raise ValueError("Ray direction must be non-zero.")
        self.d /= norm
        self.power = float(power)

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
        return Ray(self.o.copy(), self.d.copy(), self.power)

    def __repr__(self):
        return f"Ray(o={self.o}, d={self.d}, power={self.power:.4f})"


# ── 2.3  Surface ──────────────────────────────────────────────────────────────

class Surface:
    """
    Refracting surface separating two optical media — either a continuous
    sphere, or a Fresnel-faceted approximation of one (see
    ray_fresnel_intersect() for the physical model).

    Parameters
    ----------
    center       : (3,) – parent sphere's centre (defines vertex position
                   and z-location; a Fresnel surface still has one, since
                   it's a faceted approximation of this same sphere)
    radius       : float – signed radius of curvature
                     > 0  →  centre of curvature is to the right (+z)
                     < 0  →  centre of curvature is to the left  (−z)
                     ±inf →  flat surface (treated as very large sphere)
    mat_left     : Material on the –z side  (None → AIR)
    mat_right    : Material on the +z side  (None → AIR)
    surface_type : "spherical" (default) or "fresnel"
    groove_pitch : float – radial zone width [m], required if
                   surface_type == "fresnel"
    """

    def __init__(self, center, radius, mat_left=None, mat_right=None,
                 surface_type="spherical", groove_pitch=None):
        self.center    = np.array(center, dtype=float)
        self.R         = float(radius)
        self.mat_left  = mat_left  or AIR
        self.mat_right = mat_right or AIR
        if surface_type not in ("spherical", "fresnel"):
            raise ValueError(f"Unknown surface_type '{surface_type}'.")
        if surface_type == "fresnel" and not groove_pitch:
            raise ValueError("groove_pitch is required for a fresnel surface_type.")
        self.surface_type = surface_type
        self.groove_pitch = groove_pitch

    # ── Intersection ──────────────────────────────────────────────────────
    def intersect(self, ray: Ray):
        """
        Returns (point, normal) of the nearest intersection, or (None, None).
        Normal is oriented to point from left medium toward right medium.
        """
        if self.surface_type == "fresnel":
            t, p, n = ray_fresnel_intersect(ray.o, ray.d, self.center, self.R, self.groove_pitch)
        else:
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

        k_out, T = vector_snell_T(ray.d, n, n1, n2)
        if k_out is None:
            return None     # TIR

        return Ray(p, k_out, power=ray.power * T)

    # ── Vectorized refraction (batch of rays) ───────────────────────────────
    def refract_batch(self, O, D, power, alive, lam, going_forward=True):
        """
        Batch counterpart of refract(). O, D are (N,3) arrays of current
        origins/directions; power is an (N,) array of accumulated Fresnel
        transmittance so far; alive is an (N,) bool mask of rays still in
        play. Rays that miss, or hit but TIR, become not-alive; their
        O/D/power are left unchanged (never used again once dead).

        Returns (O_new, D_new, power_new, alive_new).
        """
        if self.surface_type == "fresnel":
            p, n, hit = _fresnel_intersect_batch(O, D, self.center, self.R, self.groove_pitch)
        else:
            p, n, hit = _ray_sphere_intersect_batch(O, D, self.center, self.R)
        still = alive & hit

        if going_forward:
            n1 = self.mat_left.n(lam)
            n2 = self.mat_right.n(lam)
        else:
            n1 = self.mat_right.n(lam)
            n2 = self.mat_left.n(lam)
            n = -n

        k_out, T, refr_ok = _vector_snell_batch(D, n, n1, n2)
        still = still & refr_ok

        O_new = np.where(still[:, None], p, O)
        D_new = np.where(still[:, None], k_out, D)
        power_new = np.where(still, power * T, power)
        return O_new, D_new, power_new, still


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
        Trace a list of rays at wavelength *lam*, all at once (vectorized
        over numpy arrays rather than looping trace_ray() per ray).

        Returns list of output Rays, in input order, skipping any ray that
        missed a surface, hit total internal reflection, or went parallel
        to the target plane.
        """
        if not rays:
            return []

        O = np.array([r.o for r in rays], dtype=float)
        D = np.array([r.d for r in rays], dtype=float)
        power = np.array([r.power for r in rays], dtype=float)
        alive = np.ones(len(rays), dtype=bool)

        for surf in self.surfaces:
            O, D, power, alive = surf.refract_batch(O, D, power, alive, lam, going_forward=going_forward)

        Dz = D[:, 2]
        can_propagate = alive & (np.abs(Dz) >= 1e-12)
        Dz_safe = np.where(Dz == 0.0, 1.0, Dz)
        t = (self.z_target - O[:, 2]) / Dz_safe
        O_final = O + t[:, None] * D

        return [
            Ray(O_final[i], D[i], power=power[i])
            for i in range(len(rays))
            if can_propagate[i]
        ]

    def trace_ray_path(self, ray: Ray, lam: float, going_forward: bool = True):
        """
        Trace a single ray and record every point along its path — used for
        drawing an optical-layout / ray-fan diagram, not for bulk metrics
        (see trace_bundle() for the vectorized, metrics-oriented version).

        Returns (points, alive, power):
            points : list of (3,) arrays — [origin, surface_1_hit, ...,
                     surface_N_hit, final_point]. If the ray is lost partway
                     through (miss/TIR), the list stops at the last point
                     reached (no final point on the target plane).
            alive  : True if the ray reached the target plane.
            power  : accumulated Fresnel transmittance up to wherever the
                     ray got to (1.0 = no losses yet; meaningless as a
                     "power delivered" number when alive=False, since TIR
                     redirects the ray rather than merely attenuating it).
        """
        points = [ray.o.copy()]
        cur = ray
        for surf in self.surfaces:
            nxt = surf.refract(cur, lam, going_forward=going_forward)
            if nxt is None:
                return points, False, cur.power
            points.append(nxt.o.copy())
            cur = nxt

        try:
            final = cur.copy().propagate_to_z(self.z_target)
        except ValueError:
            return points, False, cur.power
        points.append(final.o.copy())
        return points, True, final.power

    def find_paraxial_focus(self, lam: float, z_search=None, rays_in=None):
        """
        Scan z to find the z-plane that minimises RMS spot radius for lam.
        Returns (z_focus, rms_at_focus).
        """
        if z_search is None:
            z_search = np.linspace(
                self.z_target * 0.5, self.z_target * 2.0, 300)
        if rays_in is None:
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

def make_input_bundle(radius: float = 1e-3, n_rays: int = 25, tilt_deg: float = 0.0) -> list:
    """
    Create a square grid of paraxial rays over a circular aperture
    approximation, all traveling in the same direction.

    Parameters
    ----------
    radius   : aperture half-width [m]
    n_rays   : approximate total number of rays (nearest perfect square used)
    tilt_deg : incidence angle [deg] off the optical axis, tilted in the x-z
               plane (0.0 = normal incidence, the historical default). Used
               to sweep off-axis incidence for acceptance-angle analysis —
               see acceptance_angle_scan().

    Returns
    -------
    list[Ray]
    """
    side = int(np.round(np.sqrt(n_rays)))
    xs   = np.linspace(-radius, radius, side)
    ys   = np.linspace(-radius, radius, side)
    theta = np.radians(tilt_deg)
    direction = [np.sin(theta), 0.0, np.cos(theta)]
    rays = []
    for x in xs:
        for y in ys:
            rays.append(Ray(origin=[x, y, 0.0], direction=direction))
    return rays


def make_ray_fan(radius: float = 1e-3, n_rays: int = 9, y: float = 0.0) -> list:
    """
    Create a 1-D fan of paraxial rays spread along x at a fixed y (default
    the y=0 meridional plane), for optical-layout / ray-fan diagrams —
    as opposed to make_input_bundle()'s 2-D grid, which is meant for spot
    size / RMS metrics.

    Parameters
    ----------
    radius : aperture half-width [m]
    n_rays : number of rays in the fan
    y      : fixed y-coordinate [m] for every ray (0.0 = meridional plane)

    Returns
    -------
    list[Ray]
    """
    xs = np.linspace(-radius, radius, max(int(n_rays), 2))
    return [Ray(origin=[x, y, 0.0], direction=[0.0, 0.0, 1.0]) for x in xs]


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


def optical_efficiency(rays_out, n_rays_in: int) -> float:
    """
    Fraction of incident optical power that reaches the target plane,
    accounting for Fresnel reflection losses at every interface (but not
    where it lands — see power_within_radius() for that). Rays lost to a
    miss or TIR contribute 0.

    Parameters
    ----------
    rays_out  : list[Ray] – system.trace_bundle() output (already reflects
                each surviving ray's accumulated .power)
    n_rays_in : size of the illumination bundle that was traced (the
                denominator — lost rays count as 0, not as absent)

    Returns
    -------
    float in [0, 1]
    """
    if n_rays_in == 0:
        return 0.0
    return float(sum(r.power for r in rays_out) / n_rays_in)


def power_within_radius(rays_out, radius: float, n_rays_in: int) -> float:
    """
    Like optical_efficiency(), but only counts power from rays landing
    within `radius` of the optical axis at the target plane — i.e. power
    actually collected by a PV cell of that radius, not just power that
    reached the target plane somewhere. Used for acceptance-angle / CAP
    analysis (see acceptance_angle_scan()).
    """
    if n_rays_in == 0 or not rays_out:
        return 0.0
    total = 0.0
    for r in rays_out:
        if np.hypot(r.o[0], r.o[1]) <= radius:
            total += r.power
    return total / n_rays_in


def pv_weight(lam: float) -> float:
    """
    Spectral weight: 2× for the peak Si PV response band (500–900 nm),
    1× elsewhere.
    """
    return 2.0 if 500e-9 <= lam <= 900e-9 else 1.0


# Default wavelength grid: 15 points from 400 nm to 1100 nm
LAMBDA_GRID = np.linspace(400e-9, 1100e-9, 15)


def concentration_cost(params, system_builder, lam_list=None, rays_in=None) -> float:
    """
    Weighted sum of RMS² spot sizes across the wavelength grid.

    Parameters
    ----------
    params         : 1-D array of design variables
    system_builder : callable(params) → LayeredLensSystem
    lam_list       : wavelength grid [m]; defaults to LAMBDA_GRID
    rays_in        : list[Ray] illumination bundle; defaults to make_input_bundle()

    Returns
    -------
    float – cost (lower is better)
    """
    if lam_list is None:
        lam_list = LAMBDA_GRID
    if rays_in is None:
        rays_in = make_input_bundle()

    system   = system_builder(params)
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
# 3b. ACCEPTANCE ANGLE / CAP
# ─────────────────────────────────────────────────────────────────────────────
#
# A CPV concentrator's tolerance to tracking/pointing error is as important
# as its on-axis spot size — a design with a tiny RMS spot but a razor-thin
# acceptance angle is impractical to actually track. The standard figure of
# merit combining both is the concentration-acceptance product (CAP):
#
#   CAP = sqrt(C_geo) * sin(theta_accept)
#
# where C_geo is the geometric concentration ratio (aperture area / cell
# area) and theta_accept is the incidence angle at which collected power
# (within the cell, Fresnel-loss-weighted) drops to 90% of its on-axis
# value. Reference: Victoria et al., "The concentrator photovoltaics
# module: A key focus area for optics research", Advanced Photonics 3(1),
# 2021 — the CAP is defined there as the standard cross-technology CPV
# optical figure of merit.

def acceptance_angle_scan(system: LayeredLensSystem,
                          lam: float,
                          cell_radius: float,
                          aperture_radius: float,
                          n_rays: int = 25,
                          theta_max_deg: float = 3.0,
                          n_theta: int = 16):
    """
    Sweep incidence angle from 0 to theta_max_deg and, at each angle,
    compute the fraction of incident power collected within `cell_radius`
    of the axis at the target plane.

    Returns
    -------
    dict with keys:
        'thetas_deg'     : (n_theta,) array of scanned angles
        'collected'      : (n_theta,) array of raw collected-power fractions
        'collected_norm' : 'collected' normalized so collected_norm[0] == 1.0
                           (i.e. relative to the on-axis value)
    """
    thetas = np.linspace(0.0, theta_max_deg, n_theta)
    collected = np.zeros(n_theta)

    for i, theta in enumerate(thetas):
        rays_in = make_input_bundle(radius=aperture_radius, n_rays=n_rays, tilt_deg=theta)
        rays_out = system.trace_bundle(rays_in, lam)
        collected[i] = power_within_radius(rays_out, cell_radius, len(rays_in))

    norm = collected[0] if collected[0] > 1e-12 else 1.0
    return {
        "thetas_deg": thetas,
        "collected": collected,
        "collected_norm": collected / norm,
    }


def find_acceptance_angle(thetas_deg, normalized_collected, threshold: float = 0.9):
    """
    First angle (linearly interpolated) at which normalized collected power
    drops below `threshold` (default 0.9, the standard CPV definition).
    Returns None if it never drops below threshold within the scanned range.
    """
    below = np.where(np.asarray(normalized_collected) < threshold)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return 0.0
    x0, x1 = thetas_deg[i - 1], thetas_deg[i]
    y0, y1 = normalized_collected[i - 1], normalized_collected[i]
    frac = (threshold - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def concentration_acceptance_product(system: LayeredLensSystem,
                                     lam: float,
                                     cell_radius: float,
                                     aperture_radius: float,
                                     n_rays: int = 25,
                                     theta_max_deg: float = 3.0,
                                     n_theta: int = 16,
                                     threshold: float = 0.9) -> dict:
    """
    Compute the concentration-acceptance product (CAP) for a system —
    see the module note above this section.

    Returns
    -------
    dict with keys 'C_geo', 'theta_accept_deg' (None if not found within the
    scanned range), 'CAP' (None if theta_accept_deg is None), plus the raw
    'thetas_deg' / 'collected' / 'collected_norm' arrays from the scan.
    """
    scan = acceptance_angle_scan(
        system, lam, cell_radius, aperture_radius,
        n_rays=n_rays, theta_max_deg=theta_max_deg, n_theta=n_theta,
    )
    theta_accept = find_acceptance_angle(scan["thetas_deg"], scan["collected_norm"], threshold)
    c_geo = (aperture_radius / cell_radius) ** 2

    cap = None
    if theta_accept is not None:
        cap = float(np.sqrt(c_geo) * np.sin(np.radians(theta_accept)))

    return {
        "C_geo": c_geo,
        "theta_accept_deg": theta_accept,
        "CAP": cap,
        **scan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SYSTEM BUILDER  (N-layer stack, air on both ends)
# ─────────────────────────────────────────────────────────────────────────────
#
# Layout for n layers (all distances in metres, z increases to the right):
#
#   z=0        z=t1       z=t1+t2            z=sum(t)
#   |<- air ->|<- mat1 ->|<- mat2 ->| ... |<- matN ->|<--- air --->|
#  S0         S1         S2               S(n-1)    Sn
#
#   S0     : entry surface (air | mat1),   centre at (0,0, z0 + R0)
#   S1..   : internal surfaces (mat_i | mat_{i+1})
#   Sn     : exit surface (matN | air),    centre at (0,0, zn + Rn)
#
# Flat parameter layout used by the optimizer: params = [R0..Rn, t1..tn, f]
#   R0..Rn (n+1 values) : radii of curvature [m]
#   t1..tn (n values)   : thickness of each material layer [m]
#   f                   : target focal length / PV-plane z-position [m]

MIN_LAYER_THICKNESS = 0.5e-3  # guard: keep layer thicknesses positive and sensible


def build_layered_system(radii, thicknesses, materials, f,
                         surface_types=None, groove_pitches=None) -> LayeredLensSystem:
    """
    Build an N-layer layered lens system (air on both ends).

    Parameters
    ----------
    radii          : sequence of N+1 signed radii of curvature [m], one per
                     surface (entry, N-1 internal, exit)
    thicknesses    : sequence of N thicknesses [m], one per material layer
    materials      : sequence of N Material objects, one per layer
    f              : target focal length / PV-plane z-position [m]
    surface_types  : sequence of N+1 "spherical"/"fresnel" strings, one per
                     surface; defaults to all "spherical" (unchanged
                     behaviour). Fixed/chosen by the caller, not something
                     the optimizer searches over — see optimize_lens().
    groove_pitches : sequence of N+1 groove pitches [m] (or None where not
                     applicable), required alongside surface_types wherever
                     it says "fresnel"
    """
    radii = np.asarray(radii, dtype=float)
    n_layers = len(materials)
    n_surfaces = n_layers + 1
    if len(radii) != n_surfaces:
        raise ValueError(
            f"Expected {n_surfaces} radii for {n_layers} layers, got {len(radii)}.")
    if len(thicknesses) != n_layers:
        raise ValueError(
            f"Expected {n_layers} thicknesses for {n_layers} layers, got {len(thicknesses)}.")
    if surface_types is None:
        surface_types = ["spherical"] * n_surfaces
    if groove_pitches is None:
        groove_pitches = [None] * n_surfaces

    thicknesses = [max(float(t), MIN_LAYER_THICKNESS) for t in thicknesses]

    surfaces = []
    z = 0.0
    mat_left = AIR
    for i in range(n_layers):
        mat_right = materials[i]
        center = np.array([0.0, 0.0, z + radii[i]])
        surfaces.append(Surface(center, radii[i], mat_left, mat_right,
                                surface_type=surface_types[i], groove_pitch=groove_pitches[i]))
        z += thicknesses[i]
        mat_left = mat_right

    # Exit surface back to air
    center_exit = np.array([0.0, 0.0, z + radii[n_layers]])
    surfaces.append(Surface(center_exit, radii[n_layers], mat_left, AIR,
                            surface_type=surface_types[n_layers], groove_pitch=groove_pitches[n_layers]))

    return LayeredLensSystem(surfaces, z_target=f)


def build_nlayer_system(params, materials, surface_types=None, groove_pitches=None) -> LayeredLensSystem:
    """
    Build a layered system from a flat parameter vector
    [R0..Rn, t1..tn, f] and a list of N Material objects. surface_types /
    groove_pitches are passed straight through to build_layered_system() —
    they're fixed configuration, not part of the optimizable params vector.
    """
    params = np.asarray(params, dtype=float)
    n_layers = len(materials)
    expected_len = 2 * n_layers + 2
    if len(params) != expected_len:
        raise ValueError(
            f"Expected {expected_len} params for {n_layers} layers "
            f"([R0..R{n_layers}, t1..t{n_layers}, f]), got {len(params)}.")

    radii = params[:n_layers + 1]
    thicknesses = params[n_layers + 1:2 * n_layers + 1]
    f = params[2 * n_layers + 1]

    return build_layered_system(radii, thicknesses, materials, f,
                                surface_types=surface_types, groove_pitches=groove_pitches)


def build_2layer_system(params,
                        mat1: Material = None,
                        mat2: Material = None,
                        surface_types=None,
                        groove_pitches=None) -> LayeredLensSystem:
    """
    Build a 3-surface, 2-material layered lens system.

    Default materials: N-BK7 (crown) + F2 (flint) → achromatic base.
    Thin wrapper around build_nlayer_system() kept for backward
    compatibility with existing callers (CLI, older scripts, tests).
    """
    if mat1 is None:
        mat1 = Material.BK7()
    if mat2 is None:
        mat2 = Material.F2()

    return build_nlayer_system(params, [mat1, mat2],
                               surface_types=surface_types, groove_pitches=groove_pitches)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

def optimize_lens(materials=None,
                  mat1: Material = None,
                  mat2: Material = None,
                  x0=None,
                  method: str = "Nelder-Mead",
                  maxiter: int = 1500,
                  lam_list=None,
                  rays_in=None,
                  surface_types=None,
                  groove_pitches=None,
                  progress_callback=None) -> dict:
    """
    Minimise concentration_cost for an N-layer lens stack.

    Parameters
    ----------
    materials         : list of N Material objects, one per layer. Preferred
                         interface for N != 2 layers.
    mat1, mat2        : materials for the two layers — legacy 2-layer
                         interface, kept for backward compatibility. Ignored
                         if `materials` is given.
    x0                : initial params [R0..Rn, t1..tn, f] (metres);
                         required whenever the layer count isn't 2 (the
                         2-layer case falls back to the historical default).
    method            : scipy.optimize method
    maxiter           : maximum optimiser iterations
    lam_list          : wavelength grid [m] used for the honest init/final
                         cost and (subsampled) for the fast in-loop cost;
                         defaults to LAMBDA_GRID
    rays_in           : illumination bundle used the same way; defaults to
                         make_input_bundle()
    surface_types     : per-surface "spherical"/"fresnel" list, passed
                         straight through to build_layered_system() — a
                         fixed choice, not something this optimizer searches
                         over (surface type is categorical, not a continuum
                         Nelder-Mead/Powell can move through). Defaults to
                         all-spherical.
    groove_pitches    : per-surface groove pitch [m], required wherever
                         surface_types says "fresnel"
    progress_callback : optional callable(iteration:int, maxiter:int),
                         invoked after every scipy.optimize iteration —
                         lets a caller (e.g. a UI) drive a progress bar.

    Returns
    -------
    dict with keys 'params', 'cost', 'result', 'system', 'materials'
    (plus, for backward compatibility, 'mat1'/'mat2' when there are
    exactly 2 layers).
    """
    if materials is None:
        if mat1 is None:
            mat1 = Material.BK7()
        if mat2 is None:
            mat2 = Material.F2()
        materials = [mat1, mat2]

    n_layers = len(materials)

    if x0 is None:
        if n_layers == 2:
            x0 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])
        else:
            raise ValueError(
                "x0 must be provided explicitly when len(materials) != 2.")

    if lam_list is None:
        lam_list = LAMBDA_GRID
    if rays_in is None:
        rays_in = make_input_bundle()

    def builder(p):
        return build_nlayer_system(p, materials, surface_types=surface_types,
                                   groove_pitches=groove_pitches)

    # ── Fast coarse cost (fewer rays + coarser λ grid) used during search ──
    # Subsampled from the caller's own grid/bundle, capped at ~7 wavelengths
    # and ~9 rays, so the search stays fast regardless of the full-res
    # settings used for the honest init/final cost below.
    lam_step  = max(1, len(lam_list) // 7)
    FAST_LAMS = lam_list[::lam_step]
    ray_step   = max(1, len(rays_in) // 9)
    fast_rays  = rays_in[::ray_step]

    def fast_cost(params):
        system = builder(params)
        E = 0.0
        for lam in FAST_LAMS:
            rays_out = system.trace_bundle(fast_rays, lam)
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
        if progress_callback is not None:
            progress_callback(n, maxiter)

    mat_names = " / ".join(m.name for m in materials)
    print(f"\nOptimising {mat_names} lens ({n_layers} layer(s)) ...")
    init_cost = concentration_cost(
        x0, builder, lam_list=lam_list, rays_in=rays_in)  # full-res for display only
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
    final_cost = concentration_cost(
        result.x, builder, lam_list=lam_list, rays_in=rays_in)
    print(f"  Iterations   : {_state['nit']}  (converged={result.success})")
    print(f"  Final cost   : {final_cost:.6e}")

    system = builder(result.x)
    out = {
        "params":    result.x,
        "cost":      final_cost,
        "result":    result,
        "system":    system,
        "materials": materials,
    }
    if n_layers == 2:
        # Legacy keys, kept for callers written against the 2-layer API.
        out["mat1"], out["mat2"] = materials[0], materials[1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6b. AUTO-DESIGN  (joint material + geometry search)
# ─────────────────────────────────────────────────────────────────────────────
#
# Jointly searches over material combinations *and* geometry, rather than
# optimizing geometry for a fixed material choice (optimize_lens) or
# screening materials at a fixed geometry (material_survey).
#
# Strategy:
#   - Below `full_optimize_threshold` combinations: fully optimize every
#     combination (cheap enough to just do the honest thing).
#   - Above it: successive halving. Every combination gets a small
#     iteration budget; the worse half is dropped; the survivors get a
#     bigger budget (warm-started from where they left off); repeat for
#     `halving_rounds` rounds. Whatever's left is trimmed to the best
#     `top_k_final` and given a full-resolution optimize_lens() run for an
#     honest final cost. No combination is judged on a single fixed-geometry
#     snapshot the way material_survey's screening pass is — every
#     candidate gets to move from its starting point before being cut.

def generate_material_combinations(n_layers: int, material_names=None):
    """
    Ordered N-tuples of material names, excluding any combination with two
    *adjacent* identical layers (that interface would do nothing optically —
    n1 == n2 there). Non-adjacent repeats (e.g. layer 1 and layer 3 the same
    in a 3-layer stack) are allowed.
    """
    if material_names is None:
        material_names = list(CANDIDATE_MATERIALS.keys())

    combos = []
    for combo in itertools.product(material_names, repeat=n_layers):
        if any(combo[i] == combo[i + 1] for i in range(len(combo) - 1)):
            continue
        combos.append(combo)
    return combos


def _halving_round_counts(n_combos, halving_rounds, top_k_final):
    """Number of combinations optimized in each successive-halving round."""
    counts = []
    n = n_combos
    for _ in range(halving_rounds):
        counts.append(n)
        n = max(top_k_final, n // 2)
    counts.append(min(n, top_k_final))
    return counts


def auto_design(n_layers: int,
                x0,
                material_names=None,
                lam_list=None,
                rays_in=None,
                method: str = "Nelder-Mead",
                full_optimize_threshold: int = 40,
                screen_maxiter: int = 30,
                halving_rounds: int = 3,
                top_k_final: int = 3,
                final_maxiter: int = 500,
                surface_types=None,
                groove_pitches=None,
                progress_callback=None) -> dict:
    """
    Jointly search material combinations and geometry for the best N-layer
    design (see module strategy note above).

    Parameters
    ----------
    n_layers                : number of layers to search over
    x0                      : starting geometry [R0..Rn, t1..tn, f] (metres),
                               used as the initial point for every combination
    material_names          : candidate material names; defaults to all of
                               CANDIDATE_MATERIALS
    lam_list, rays_in       : as in optimize_lens(); defaults to
                               LAMBDA_GRID / make_input_bundle()
    method                  : scipy.optimize method
    full_optimize_threshold : combinations at or below this count are fully
                               optimized individually instead of halved
    screen_maxiter          : iteration budget for round 1 of halving
                               (doubles each subsequent round)
    halving_rounds          : number of halving rounds before the final pass
    top_k_final             : how many survivors get a full-resolution
                               optimize_lens() run at the end
    final_maxiter           : iteration budget for the full-optimize path and
                               the final pass of the halving path
    surface_types           : per-surface "spherical"/"fresnel" list, fixed
                               for every combination searched (see
                               optimize_lens() — this isn't part of the
                               search space, only geometry and materials are)
    groove_pitches          : per-surface groove pitch [m], as above
    progress_callback       : optional callable(step:int, total:int, label:str)

    Returns
    -------
    dict with keys:
        'best'                     : optimize_lens()-style result dict for
                                      the winning combination
        'best_materials'           : list of material names for the winner
        'leaderboard'              : list of {'materials','cost','params',
                                      'efficiency'} dicts, best first
        'n_combinations_total'     : number of combinations considered
        'n_combinations_evaluated' : number of optimize_lens() calls made
        'used_halving'             : whether the halving path was taken
    """
    combos = generate_material_combinations(n_layers, material_names)
    n_combos = len(combos)
    if n_combos == 0:
        raise ValueError("No valid material combinations for this layer count.")

    if lam_list is None:
        lam_list = LAMBDA_GRID
    if rays_in is None:
        rays_in = make_input_bundle()

    def _optimize_combo(combo, x_start, maxiter):
        materials = [CANDIDATE_MATERIALS[name] for name in combo]
        return optimize_lens(materials=materials, x0=x_start, method=method,
                             maxiter=maxiter, lam_list=lam_list, rays_in=rays_in,
                             surface_types=surface_types, groove_pitches=groove_pitches)

    step = 0
    used_halving = n_combos > full_optimize_threshold

    if not used_halving:
        total = n_combos
        results = []
        for combo in combos:
            res = _optimize_combo(combo, x0, final_maxiter)
            results.append((combo, res))
            step += 1
            if progress_callback is not None:
                progress_callback(step, total, "/".join(combo))
        results.sort(key=lambda cr: cr[1]["cost"])
    else:
        total = sum(_halving_round_counts(n_combos, halving_rounds, top_k_final))
        history = {}
        survivors = list(combos)

        for r in range(halving_rounds):
            budget = screen_maxiter * (2 ** r)
            round_results = []
            for combo in survivors:
                x_start = history[combo]["params"] if combo in history else x0
                res = _optimize_combo(combo, x_start, budget)
                history[combo] = res
                round_results.append((combo, res["cost"]))
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total, f"round {r + 1}: " + "/".join(combo))
            round_results.sort(key=lambda cr: cr[1])
            keep_n = max(top_k_final, len(round_results) // 2)
            survivors = [combo for combo, _ in round_results[:keep_n]]

        finalists = sorted(survivors, key=lambda c: history[c]["cost"])[:top_k_final]
        results = []
        for combo in finalists:
            res = _optimize_combo(combo, history[combo]["params"], final_maxiter)
            results.append((combo, res))
            step += 1
            if progress_callback is not None:
                progress_callback(step, total, "final: " + "/".join(combo))
        results.sort(key=lambda cr: cr[1]["cost"])

    best_combo, best_res = results[0]
    # Report an efficiency figure alongside cost, so a low-RMS combo that
    # throws away a lot of light to Fresnel losses (e.g. a steep, high-index
    # surface) doesn't look strictly better than it really is — the RMS-based
    # cost that drove the search doesn't account for that at all.
    leaderboard = []
    for combo, res in results:
        eff_vals = [
            optical_efficiency(res["system"].trace_bundle(rays_in, lam), len(rays_in))
            for lam in lam_list
        ]
        leaderboard.append({
            "materials": list(combo),
            "cost": res["cost"],
            "params": res["params"],
            "efficiency": float(np.mean(eff_vals)) if eff_vals else 0.0,
        })

    return {
        "best": best_res,
        "best_materials": list(best_combo),
        "leaderboard": leaderboard,
        "n_combinations_total": n_combos,
        "n_combinations_evaluated": step,
        "used_halving": used_halving,
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

    # ── T10: vectorized trace_bundle matches scalar trace_ray (regression) ─
    sys10 = build_2layer_system(
        np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3]))
    test_rays = make_input_bundle(radius=3e-3, n_rays=49)  # wide enough to hit TIR/misses
    lam10 = 550e-9

    ref_points = []
    for r in test_rays:
        ref = sys10.trace_ray(r.copy(), lam10)
        if ref is not None:
            ref_points.append(ref.o)

    n_ref_alive = 0
    max_diff_single = 0.0
    for r in test_rays:
        ref = sys10.trace_ray(r.copy(), lam10)
        got = sys10.trace_bundle([r.copy()], lam10)
        ref_alive = ref is not None
        got_alive = len(got) == 1
        assert ref_alive == got_alive, (
            "FAIL [T10]: scalar/vectorized disagree on whether a ray survives")
        if ref_alive:
            n_ref_alive += 1
            max_diff_single = max(max_diff_single, np.max(np.abs(got[0].o - ref.o)))

    assert n_ref_alive > 0, "FAIL [T10]: sanity check, no rays survived tracing"
    _assert_close(max_diff_single, 0.0, tol=1e-9,
                  label=f"T10 single-ray-batch matches scalar ({n_ref_alive} rays)")

    batch_out = sys10.trace_bundle([r.copy() for r in test_rays], lam10)
    assert len(batch_out) == len(ref_points), (
        "FAIL [T10b]: full-batch survivor count differs from scalar reference")
    max_diff_batch = max(
        np.max(np.abs(b.o - ref_o)) for b, ref_o in zip(batch_out, ref_points)
    )
    _assert_close(max_diff_batch, 0.0, tol=1e-9,
                  label=f"T10b full-batch matches scalar ({len(ref_points)} rays)")

    # ── T11: build_nlayer_system(n=2) == build_2layer_system (regression) ──
    p6 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])
    mat1_11, mat2_11 = Material.BK7(), Material.F2()
    sys_old = build_2layer_system(p6, mat1=mat1_11, mat2=mat2_11)
    sys_new = build_nlayer_system(p6, [mat1_11, mat2_11])
    assert len(sys_old.surfaces) == len(sys_new.surfaces) == 3, (
        "FAIL [T11]: surface count mismatch")
    for i, (s_old, s_new) in enumerate(zip(sys_old.surfaces, sys_new.surfaces)):
        _assert_close(s_old.R, s_new.R, tol=1e-15, label=f"T11 surface {i} radius")
        _assert_close(np.max(np.abs(s_old.center - s_new.center)), 0.0, tol=1e-15,
                      label=f"T11 surface {i} center")
        assert s_old.mat_left.name == s_new.mat_left.name, f"FAIL [T11]: surface {i} mat_left"
        assert s_old.mat_right.name == s_new.mat_right.name, f"FAIL [T11]: surface {i} mat_right"
    _assert_close(sys_old.z_target, sys_new.z_target, tol=1e-15, label="T11 z_target")

    # ── T12: trace_ray_path is consistent with trace_ray ───────────────────
    ray12 = Ray([0.5e-3, 0.3e-3, 0.0], [0.0, 0.0, 1.0])
    pts12, alive12, power12 = sys_new.trace_ray_path(ray12, 550e-9)
    ref12 = sys_new.trace_ray(ray12.copy(), 550e-9)
    assert alive12 == (ref12 is not None), "FAIL [T12]: alive flag disagrees with trace_ray"
    assert alive12, "FAIL [T12]: sanity check, expected ray to survive"
    assert len(pts12) == len(sys_new.surfaces) + 2, (
        "FAIL [T12]: expected one point per surface plus origin and final point")
    _assert_close(np.max(np.abs(pts12[-1] - ref12.o)), 0.0, tol=1e-12,
                  label="T12 trace_ray_path final point matches trace_ray")
    _assert_close(power12, ref12.power, tol=1e-12,
                  label="T12 trace_ray_path power matches trace_ray")

    # ── T13: 3-layer system builds and traces successfully (sanity) ────────
    p3 = np.array([50e-3, -25e-3, -40e-3, -60e-3, 4e-3, 3e-3, 4e-3, 55e-3])
    mats3 = [Material.BK7(), Material.SF11(), Material.FK51A()]
    sys3 = build_nlayer_system(p3, mats3)
    assert len(sys3.surfaces) == 4, "FAIL [T13]: expected 4 surfaces for 3 layers"
    fan3 = make_ray_fan(radius=2e-3, n_rays=9)
    assert len(fan3) == 9, "FAIL [T13]: make_ray_fan returned wrong ray count"
    out3 = sys3.trace_bundle(fan3, 550e-9)
    assert len(out3) > 0, "FAIL [T13]: expected at least some rays to survive a 3-layer trace"
    print("  PASS [T13 3-layer build/trace sanity]")

    # ── T14: generate_material_combinations excludes adjacent duplicates ───
    names5 = list(CANDIDATE_MATERIALS.keys())
    combos2 = generate_material_combinations(2, names5)
    assert len(combos2) == len(names5) * (len(names5) - 1), (
        "FAIL [T14]: expected all ordered pairs except same-material ones")
    assert all(c[0] != c[1] for c in combos2), "FAIL [T14]: found an adjacent-duplicate pair"
    combos3 = generate_material_combinations(3, ["A", "B"])
    # A-B-A and B-A-B are the only 3-tuples over 2 names with no adjacent repeat
    assert set(combos3) == {("A", "B", "A"), ("B", "A", "B")}, (
        "FAIL [T14]: unexpected 3-layer combinations over 2 materials")
    print(f"  PASS [T14 material combination generation ({len(combos2)} 2-layer pairs)]")

    # ── T15: auto_design full-optimize path (small combo count) ────────────
    x0_15 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])
    small_names = ["N-BK7", "F2"]  # -> exactly 2 valid combos for n_layers=2
    ad15 = auto_design(
        n_layers=2, x0=x0_15, material_names=small_names,
        full_optimize_threshold=40, final_maxiter=25,
    )
    assert ad15["used_halving"] is False, "FAIL [T15]: expected the full-optimize path"
    assert ad15["n_combinations_total"] == 2, "FAIL [T15]: expected exactly 2 combinations"
    assert len(ad15["leaderboard"]) == 2, "FAIL [T15]: leaderboard should have one entry per combo"
    assert np.isfinite(ad15["best"]["cost"]), "FAIL [T15]: best cost should be finite"
    assert ad15["leaderboard"][0]["cost"] <= ad15["leaderboard"][1]["cost"], (
        "FAIL [T15]: leaderboard should be sorted best-first")
    assert all(0.0 < e["efficiency"] <= 1.0 for e in ad15["leaderboard"]), (
        "FAIL [T15]: leaderboard efficiency should be a plausible fraction")
    print("  PASS [T15 auto_design full-optimize path]")

    # ── T16: auto_design successive-halving path (forced via low threshold) ─
    ad16 = auto_design(
        n_layers=2, x0=x0_15, material_names=names5,
        full_optimize_threshold=0,  # force halving even though 20 combos is small
        screen_maxiter=5, halving_rounds=2, top_k_final=2, final_maxiter=20,
    )
    assert ad16["used_halving"] is True, "FAIL [T16]: expected the halving path"
    assert ad16["n_combinations_total"] == len(combos2), (
        "FAIL [T16]: combination count should match generate_material_combinations")
    assert len(ad16["leaderboard"]) == 2, "FAIL [T16]: expected top_k_final=2 finalists"
    assert np.isfinite(ad16["best"]["cost"]), "FAIL [T16]: best cost should be finite"
    print(f"  PASS [T16 auto_design halving path "
          f"({ad16['n_combinations_evaluated']} optimize_lens calls)]")

    # ── T17: Fresnel transmittance at normal incidence matches ((n1-n2)/(n1+n2))² ─
    n1_17, n2_17 = 1.0, 1.5168  # air -> BK7
    T17 = fresnel_transmittance(n1_17, n2_17, 1.0, 1.0)  # normal incidence: cos_i=cos_t=1
    R_expected = ((n1_17 - n2_17) / (n1_17 + n2_17)) ** 2
    _assert_close(T17, 1.0 - R_expected, tol=1e-12, label="T17 normal-incidence Fresnel transmittance")

    # ── T18: identical media (n1==n2) transmit fully with an unbent ray ────
    k_out18, T18 = vector_snell_T([0.1, 0.0, 0.995], [0.0, 0.0, 1.0], 1.5, 1.5)
    assert k_out18 is not None, "FAIL [T18]: expected no TIR for identical media"
    _assert_close(T18, 1.0, tol=1e-12, label="T18 identical-media transmittance == 1")
    _assert_close(np.max(np.abs(np.asarray(k_out18) - np.array([0.1, 0.0, 0.995]) /
                  np.linalg.norm([0.1, 0.0, 0.995]))), 0.0, tol=1e-12,
                  label="T18 identical-media ray direction unchanged")

    # ── T19: batch power tracking matches scalar reference (regression) ────
    max_power_diff = 0.0
    n_checked = 0
    for r in test_rays:
        ref = sys10.trace_ray(r.copy(), lam10)
        got = sys10.trace_bundle([r.copy()], lam10)
        if ref is not None:
            assert len(got) == 1
            max_power_diff = max(max_power_diff, abs(got[0].power - ref.power))
            n_checked += 1
    assert n_checked > 0, "FAIL [T19]: sanity check, no rays survived tracing"
    assert all(0.0 < r.power <= 1.0 for r in sys10.trace_bundle(test_rays, lam10)), (
        "FAIL [T19]: surviving ray power should be a plausible fraction")
    _assert_close(max_power_diff, 0.0, tol=1e-9,
                  label=f"T19 vectorized power matches scalar ({n_checked} rays)")

    # ── T20: acceptance-angle scan / CAP — structural sanity ───────────────
    scan20 = acceptance_angle_scan(
        sys10, lam10, cell_radius=0.2e-3, aperture_radius=1e-3,
        n_rays=16, theta_max_deg=2.0, n_theta=9,
    )
    assert np.all(np.diff(scan20["thetas_deg"]) > 0), "FAIL [T20]: angles should be increasing"
    _assert_close(scan20["collected_norm"][0], 1.0, tol=1e-9,
                  label="T20 on-axis normalized collection == 1.0")
    cap20 = concentration_acceptance_product(
        sys10, lam10, cell_radius=0.2e-3, aperture_radius=1e-3,
        n_rays=16, theta_max_deg=2.0, n_theta=9,
    )
    assert cap20["C_geo"] > 0, "FAIL [T20]: C_geo should be positive"
    if cap20["theta_accept_deg"] is not None:
        assert 0.0 <= cap20["theta_accept_deg"] <= 2.0, "FAIL [T20]: acceptance angle out of scan range"
        assert cap20["CAP"] is not None and cap20["CAP"] > 0, "FAIL [T20]: CAP should be positive when found"
    print("  PASS [T20 acceptance-angle scan / CAP structural sanity]")

    # ── T21: Fresnel facet matches parent sphere exactly at zone centers ───
    R21, pitch21 = 50e-3, 0.2e-3
    center21 = np.array([0.0, 0.0, R21])  # vertex at z=0
    glass21 = Material.BK7()
    sph21 = Surface(center21, R21, AIR, glass21, surface_type="spherical")
    fres21 = Surface(center21, R21, AIR, glass21, surface_type="fresnel", groove_pitch=pitch21)
    lam21 = 550e-9

    max_zone_center_diff = 0.0
    for zone_idx in range(4):
        x21 = (zone_idx + 0.5) * pitch21  # exactly a zone center
        ray21 = Ray([x21, 0.0, -1e-3], [0.0, 0.0, 1.0])
        r_sph21 = sph21.refract(ray21.copy(), lam21)
        r_fres21 = fres21.refract(ray21.copy(), lam21)
        assert r_sph21 is not None and r_fres21 is not None, (
            "FAIL [T21]: expected both surfaces to refract this ray")
        max_zone_center_diff = max(max_zone_center_diff,
                                   np.max(np.abs(r_sph21.d - r_fres21.d)))
    _assert_close(max_zone_center_diff, 0.0, tol=1e-9,
                  label="T21 Fresnel facet direction matches parent sphere at zone centers")

    # Off zone-center: still a real refraction (not a miss), and close to the
    # parent sphere's direction (faceting error, not a different physics model)
    ray21b = Ray([0.55e-3, 0.0, -1e-3], [0.0, 0.0, 1.0])
    r_sph21b = sph21.refract(ray21b.copy(), lam21)
    r_fres21b = fres21.refract(ray21b.copy(), lam21)
    assert r_sph21b is not None and r_fres21b is not None
    off_center_diff = np.max(np.abs(r_sph21b.d - r_fres21b.d))
    assert 0.0 < off_center_diff < 1e-2, (
        "FAIL [T21b]: off-zone-center faceting error should be small but nonzero")
    print(f"  PASS [T21b off-zone-center faceting error is small ({off_center_diff:.2e})]")

    # ── T22: vectorized Fresnel batch matches scalar reference ─────────────
    # Rays must start *before* the surface (fres21's vertex is at z=0, same
    # as make_ray_fan()'s default origin) or every ray starts exactly on it.
    fan22 = [Ray(r.o - np.array([0.0, 0.0, 1e-3]), r.d) for r in make_ray_fan(radius=1e-3, n_rays=25)]
    max_diff22 = 0.0
    n_checked22 = 0
    for r in fan22:
        ref22 = fres21.refract(r.copy(), lam21)
        batch_O = np.array([r.o])
        batch_D = np.array([r.d])
        batch_power = np.array([r.power])
        batch_alive = np.array([True])
        O_new, D_new, power_new, alive_new = fres21.refract_batch(
            batch_O, batch_D, batch_power, batch_alive, lam21)
        ref_alive = ref22 is not None
        assert bool(alive_new[0]) == ref_alive, (
            "FAIL [T22]: scalar/vectorized Fresnel disagree on survival")
        if ref_alive:
            n_checked22 += 1
            max_diff22 = max(max_diff22, np.max(np.abs(D_new[0] - ref22.d)))
    assert n_checked22 > 0, "FAIL [T22]: sanity check, no rays survived"
    _assert_close(max_diff22, 0.0, tol=1e-9,
                  label=f"T22 vectorized Fresnel matches scalar ({n_checked22} rays)")

    # ── T23: a Fresnel-surfaced system builds, traces, and optimizes ───────
    p23 = np.array([50e-3, -30e-3, -50e-3, 5e-3, 5e-3, 50e-3])
    sys23 = build_2layer_system(
        p23, surface_types=["fresnel", "spherical", "spherical"],
        groove_pitches=[0.2e-3, None, None],
    )
    assert sys23.surfaces[0].surface_type == "fresnel"
    out23 = sys23.trace_bundle(make_input_bundle(radius=1e-3, n_rays=25), 550e-9)
    assert len(out23) > 0, "FAIL [T23]: expected at least some rays to survive"

    opt23 = optimize_lens(
        x0=p23, maxiter=15,
        surface_types=["fresnel", "spherical", "spherical"],
        groove_pitches=[0.2e-3, None, None],
    )
    assert opt23["system"].surfaces[0].surface_type == "fresnel", (
        "FAIL [T23]: optimizer's returned system should keep the Fresnel surface type")
    assert np.isfinite(opt23["cost"]), "FAIL [T23]: optimizer cost should be finite"
    print("  PASS [T23 Fresnel-surfaced system builds/traces/optimizes]")

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
                     rays_in=None,
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
        zf, rms = system.find_paraxial_focus(lam, z_search=z_scan, rays_in=rays_in)
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


def material_survey(x0=None, verbose=True, lam_list=None, rays_in=None,
                    surface_types=None, groove_pitches=None) -> list:
    """
    Evaluate concentration_cost for every (mat1, mat2) pair and return
    a ranked list of (cost, mat1_name, mat2_name).

    lam_list / rays_in default to LAMBDA_GRID / make_input_bundle() when
    omitted, matching the previous fixed-grid behaviour. surface_types /
    groove_pitches (fixed for every pair screened) default to all-spherical.
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
                x0, lambda p: build_2layer_system(
                    p, mat1=m1, mat2=m2,
                    surface_types=surface_types, groove_pitches=groove_pitches),
                lam_list=lam_list, rays_in=rays_in)
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

    rays_in = make_input_bundle(radius=ray_radius, n_rays=n_rays)

    # ── Step 1: single-wavelength focus scan (unoptimised doublet) ────────
    print("Building initial 2-layer lens system ...")
    sys0 = build_2layer_system(x0, mat1=init_mat1, mat2=init_mat2)

    print("Plotting chromatic focus shift (initial system) ...")
    fig_shift = plot_focus_shift(
        sys0, lam_list=lam_grid, rays_in=rays_in,
        title=f"Chromatic Focus Shift – Initial {mat1_name}/{mat2_name}")
    fig_shift.savefig("focus_shift_initial.png", dpi=150)
    print("  Saved: focus_shift_initial.png")

    fig_spot0 = plot_spot_diagrams(
        sys0, lam_list=lam_grid,
        title=f"Spot Diagrams – Initial {mat1_name}/{mat2_name}")
    fig_spot0.savefig("spots_initial.png", dpi=150)
    print("  Saved: spots_initial.png")

    # ── Step 2: material survey ───────────────────────────────────────────
    scores = material_survey(x0, lam_list=lam_grid, rays_in=rays_in)
    best_cost, best_m1_name, best_m2_name = scores[0]
    best_m1 = CANDIDATE_MATERIALS[best_m1_name]
    best_m2 = CANDIDATE_MATERIALS[best_m2_name]
    print(f"\nBest material pair from survey: {best_m1_name} / {best_m2_name}"
          f"  (cost={best_cost:.4e})")

    # ── Step 3: optimise best pair ────────────────────────────────────────
    opt = optimize_lens(
        mat1=best_m1, mat2=best_m2, x0=x0,
        method=opt_method, maxiter=opt_maxiter,
        lam_list=lam_grid, rays_in=rays_in)

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
    for lam in lam_grid:
        rays_out = opt["system"].trace_bundle(rays_in, lam)
        rms = rms_spot_radius(rays_out) * 1e6 if rays_out else float("nan")
        w   = pv_weight(lam)
        print(f"  {lam*1e9:8.1f}  {w:6.1f}  {rms:10.3f}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
