import io
import contextlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pv_ray_optics as optics


st.set_page_config(
    page_title="PV Ray-Optics Engine",
    page_icon="🔬",
    layout="wide",
)

MAX_LAYERS = 4
# Length MAX_LAYERS+1 so there's always a sensible default for the exit radius too.
DEFAULT_RADII_MM = [50.0, -30.0, -50.0, -70.0, -90.0]
DEFAULT_THICKNESS_MM = 5.0
DEFAULT_MATERIALS_CYCLE = ["N-BK7", "F2"]
DEFAULT_GROOVE_PITCH_MM = 0.2

DEFAULTS = {
    "n_layers": 2,
    "f_mm": 50.0,
    "ray_radius_mm": 1.0,
    "cell_radius_mm": 0.2,
    "n_rays": 25,
    "lam_start_nm": 400.0,
    "lam_end_nm": 1100.0,
    "lam_steps": 15,
    "opt_method": "Nelder-Mead",
    "opt_maxiter": 500,
}

RMS_COLOR = "#4C78A8"
INITIAL_COLOR = "#E45756"
OPTIMIZED_COLOR = "#4C78A8"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: values <-> physics objects
# ─────────────────────────────────────────────────────────────────────────────

def make_params(values):
    radii = np.array(values["radii_mm"], dtype=float) * 1e-3
    thicknesses = np.array(values["thicknesses_mm"], dtype=float) * 1e-3
    f = values["f_mm"] * 1e-3
    return np.concatenate([radii, thicknesses, [f]])


def get_materials(values):
    return [optics.CANDIDATE_MATERIALS[name] for name in values["materials"]]


def get_lambda_grid(values):
    return np.linspace(
        values["lam_start_nm"] * 1e-9,
        values["lam_end_nm"] * 1e-9,
        int(values["lam_steps"]),
    )


def get_rays_in(values):
    return optics.make_input_bundle(
        radius=values["ray_radius_mm"] * 1e-3,
        n_rays=int(values["n_rays"]),
    )


def get_groove_pitches_m(values):
    return [(p * 1e-3 if p is not None else None) for p in values["groove_pitches_mm"]]


def _wavelength_color(lam):
    r, g, b = optics._wavelength_to_rgb(lam)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _wavelength_cell_style(nm_value):
    """Background-color a table cell by its spectral color, with readable
    text — a quick-glance color legend baked right into the results table."""
    r, g, b = optics._wavelength_to_rgb(nm_value * 1e-9)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = "white" if luminance < 0.6 else "black"
    return f"background-color: rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}); color: {text_color}"


def _axis_id(idx, letter):
    n = idx + 1
    return letter if n == 1 else f"{letter}{n}"


def make_system(values):
    # Deliberately NOT cached: building a handful of Surface objects is
    # cheap, and st.cache_resource shares the exact same live object across
    # reruns/sessions rather than a copy. That's a real hazard here — if the
    # physics classes it's built from (Ray/Surface/...) change in a later
    # deploy but this function's own body doesn't, Streamlit has no way to
    # know the cached object is now stale (its source-hash-based cache
    # invalidation only sees *this* function's code, not build_layered_system
    # or the classes underneath it). A stale cached system then produces
    # Ray objects from the old class definition, silently missing whatever
    # attributes were added later. The actually expensive step (ray tracing)
    # is cached separately in trace_all_wavelengths(), whose own body does
    # change whenever its logic does, so it self-invalidates correctly.
    materials = get_materials(values)
    radii = np.array(values["radii_mm"], dtype=float) * 1e-3
    thicknesses = np.array(values["thicknesses_mm"], dtype=float) * 1e-3
    return optics.build_layered_system(
        radii, thicknesses, materials, values["f_mm"] * 1e-3,
        surface_types=values["surface_types"], groove_pitches=get_groove_pitches_m(values),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached tracing
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def trace_all_wavelengths(values):
    """
    Trace the full ray bundle across the full wavelength grid once and cache
    the numeric results. per_wavelength_metrics / plot_rms_plotly /
    plot_spots_plotly all read from this instead of re-tracing separately.
    """
    system = make_system(values)
    rays_in = get_rays_in(values)
    lam_grid = get_lambda_grid(values)
    cell_radius = values["cell_radius_mm"] * 1e-3

    rms_um = np.full(len(lam_grid), np.nan)
    n_rays_out = np.zeros(len(lam_grid), dtype=int)
    efficiency = np.zeros(len(lam_grid))
    collection_efficiency = np.zeros(len(lam_grid))
    spots_um = []

    for i, lam in enumerate(lam_grid):
        rays_out = system.trace_bundle(rays_in, lam)
        n_rays_out[i] = len(rays_out)
        # "Optical efficiency" only asks whether a ray reached the target
        # plane at all (Fresnel losses aside); it says nothing about WHERE
        # it landed. A spot bigger than the actual cell can look great on
        # that number while mostly missing the cell — "collection
        # efficiency" is the number that actually answers that.
        efficiency[i] = optics.optical_efficiency(rays_out, len(rays_in))
        collection_efficiency[i] = optics.power_within_radius(rays_out, cell_radius, len(rays_in))
        if rays_out:
            pts = np.array([r.o for r in rays_out])[:, :2] * 1e6
            rms_um[i] = optics.rms_spot_radius(rays_out) * 1e6
        else:
            pts = np.empty((0, 2))
        spots_um.append(pts)

    return {
        "lam_m": lam_grid,
        "rms_um": rms_um,
        "n_rays_out": n_rays_out,
        "efficiency": efficiency,
        "collection_efficiency": collection_efficiency,
        "spots_um": spots_um,
    }


def per_wavelength_metrics(values):
    trace = trace_all_wavelengths(values)
    lam_grid = trace["lam_m"]
    rms_um = trace["rms_um"]
    n_rays_out = trace["n_rays_out"]
    efficiency = trace["efficiency"]
    collection_efficiency = trace["collection_efficiency"]

    rows = []
    weighted_cost = 0.0

    for lam, rms, n_out, eff, coll_eff in zip(
        lam_grid, rms_um, n_rays_out, efficiency, collection_efficiency
    ):
        weight = optics.pv_weight(lam)
        if np.isfinite(rms):
            weighted_cost += weight * (rms * 1e-6) ** 2
        rows.append(
            {
                "Wavelength (nm)": lam * 1e9,
                "PV weight": weight,
                "RMS spot radius (µm)": rms,
                "Optical efficiency": eff,
                "Collection efficiency (on cell)": coll_eff,
                "Rays reaching PV": int(n_out),
            }
        )

    return pd.DataFrame(rows), weighted_cost


def _fresnel_zone_profile(R, cx, cz, aperture_half, pitch):
    """
    Sawtooth cross-section for a Fresnel surface, for drawing only. Each
    zone's facet is a straight segment whose slope matches the parent
    sphere's tangent at that zone's center radius, resetting to the flat
    vertex plane at each zone boundary — a schematic (not literal-depth)
    rendering, consistent with how ray_fresnel_intersect() itself treats
    each zone as flat with a matching normal rather than modeling the
    physical groove depth.
    """
    z0 = cz - R
    sign = 1.0 if R >= 0 else -1.0
    n_zones = max(int(np.ceil(aperture_half / pitch)), 1)

    xs, zs = [], []
    for k in range(n_zones):
        rho_start = k * pitch
        if rho_start >= aperture_half:
            break
        rho_end = min((k + 1) * pitch, aperture_half)
        rho_ref = min((k + 0.5) * pitch, aperture_half)
        inside = max(R * R - rho_ref * rho_ref, 1e-12)
        slope = sign * rho_ref / np.sqrt(inside)
        z_end = z0 + slope * (rho_end - rho_start)

        xs += [cx + rho_start, cx + rho_end, None]  # +x side
        zs += [z0, z_end, None]
        xs += [cx - rho_start, cx - rho_end, None]  # -x side (mirror)
        zs += [z0, z_end, None]

    return xs, zs


@st.cache_data(show_spinner=False)
def compute_optical_layout(values, n_fan_rays=7, lam_nm=550.0):
    """
    Geometry (surface profile points) + a small ray fan traced through the
    full path (not just the final PV-plane point), for the optical-layout
    diagram.
    """
    system = make_system(values)
    aperture = values["ray_radius_mm"] * 1e-3

    fan = optics.make_ray_fan(radius=aperture, n_rays=n_fan_rays)
    paths = []
    for ray in fan:
        pts, alive, power = system.trace_ray_path(ray, lam_nm * 1e-9)
        paths.append((np.array(pts), alive, power))

    surfaces = []
    for surf in system.surfaces:
        R = surf.R
        cx, cz = surf.center[0], surf.center[2]
        half_ap = min(aperture * 1.3, abs(R) * 0.97) if R != 0 else aperture * 1.3
        if surf.surface_type == "fresnel":
            xs, zs = _fresnel_zone_profile(R, cx, cz, half_ap, surf.groove_pitch)
        else:
            xs_arr = np.linspace(-half_ap, half_ap, 60) + cx
            sign = 1.0 if R >= 0 else -1.0
            zs_arr = cz - sign * np.sqrt(np.maximum(R ** 2 - (xs_arr - cx) ** 2, 0.0))
            xs, zs = xs_arr, zs_arr
        surfaces.append((xs, zs, surf.surface_type))

    return {
        "paths": paths,
        "surfaces": surfaces,
        "z_target": system.z_target,
        "aperture": aperture,
        "cell_radius": values["cell_radius_mm"] * 1e-3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_rms_plotly(series):
    """series: list of (values_dict, label, color) tuples."""
    fig = go.Figure()
    for values, label, color in series:
        trace = trace_all_wavelengths(values)
        marker_colors = [_wavelength_color(lam) for lam in trace["lam_m"]]
        fig.add_trace(go.Scatter(
            x=trace["lam_m"] * 1e9,
            y=trace["rms_um"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=9, color=marker_colors, line=dict(width=1, color="rgba(0,0,0,0.35)")),
            hovertemplate=f"%{{x:.0f}} nm<br>RMS = %{{y:.3f}} µm<extra>{label}</extra>",
        ))
    fig.add_vrect(x0=500, x1=900, fillcolor="green", opacity=0.08, line_width=0,
                  annotation_text="Peak PV band", annotation_position="top left")
    fig.update_layout(
        title="RMS Spot Radius at PV Plane",
        xaxis_title="Wavelength (nm)",
        yaxis_title="RMS spot radius (µm)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


def plot_spots_plotly(values):
    trace = trace_all_wavelengths(values)
    lam_grid = trace["lam_m"]
    rms_um = trace["rms_um"]
    spots_um = trace["spots_um"]
    cell_radius_um = values["cell_radius_mm"] * 1e3

    n_lam = len(lam_grid)
    ncols = min(5, n_lam)
    nrows = int(np.ceil(n_lam / ncols))

    titles = [
        f"{lam * 1e9:.0f} nm — RMS {rms:.2f} µm" if pts.shape[0] > 0
        else f"{lam * 1e9:.0f} nm — no rays"
        for lam, rms, pts in zip(lam_grid, rms_um, spots_um)
    ]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                         horizontal_spacing=0.05, vertical_spacing=0.12)

    for idx, (lam, pts) in enumerate(zip(lam_grid, spots_um)):
        row, col = idx // ncols + 1, idx % ncols + 1
        color = _wavelength_color(lam)
        if pts.shape[0] > 0:
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0], y=pts[:, 1], mode="markers",
                    marker=dict(color=color, size=5, opacity=0.85),
                    hovertemplate="x=%{x:.2f} µm<br>y=%{y:.2f} µm<extra></extra>",
                    showlegend=False,
                ),
                row=row, col=col,
            )
        # PV cell boundary, so it's visually obvious whether the spot fits
        # inside the actual cell or spills outside it.
        fig.add_shape(
            type="circle",
            x0=-cell_radius_um, y0=-cell_radius_um, x1=cell_radius_um, y1=cell_radius_um,
            line=dict(color="black", dash="dot", width=1.5),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="x (µm)", zeroline=True, row=row, col=col)
        fig.update_yaxes(
            title_text="y (µm)", zeroline=True,
            scaleanchor=_axis_id(idx, "x"), scaleratio=1,
            row=row, col=col,
        )

    fig.update_layout(
        title_text="Spot Diagrams at PV Plane (dotted circle = PV cell boundary)",
        showlegend=False,
        height=280 * nrows,
        margin=dict(t=80, b=40),
    )
    return fig


def _scale_with_gaps(seq, factor):
    """Like seq * factor, but seq may be a plain list containing None line-break
    markers (as built by _fresnel_zone_profile) instead of a numpy array."""
    return [v * factor if v is not None else None for v in seq]


def plot_optical_layout_plotly(values, n_fan_rays=7, lam_nm=550.0):
    data = compute_optical_layout(values, n_fan_rays=n_fan_rays, lam_nm=lam_nm)
    fig = go.Figure()

    surface_colors = {"spherical": "#4C78A8", "fresnel": "#F58518"}
    legend_shown = set()
    for xs, zs, surf_type in data["surfaces"]:
        is_fresnel = surf_type == "fresnel"
        x_plot = _scale_with_gaps(zs, 1e3) if is_fresnel else zs * 1e3
        y_plot = _scale_with_gaps(xs, 1e3) if is_fresnel else xs * 1e3
        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot, mode="lines",
            line=dict(color=surface_colors[surf_type], width=2),
            hoverinfo="skip",
            name=surf_type.capitalize(),
            legendgroup=surf_type,
            showlegend=surf_type not in legend_shown,
        ))
        legend_shown.add(surf_type)

    n_paths = len(data["paths"])
    for i, (pts, alive, power) in enumerate(data["paths"]):
        frac = i / max(n_paths - 1, 1)
        color = f"rgb({int(255 * frac)},80,{int(255 * (1 - frac))})"
        fig.add_trace(go.Scatter(
            x=pts[:, 2] * 1e3, y=pts[:, 0] * 1e3, mode="lines",
            line=dict(color=color, width=1.3, dash="solid" if alive else "dot"),
            opacity=max(0.2, power),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_vline(
        x=data["z_target"] * 1e3, line=dict(color="red", dash="dot", width=1.5),
        annotation_text="PV plane", annotation_position="top",
    )

    # The lens surface curves already show the full aperture width (at z=0,
    # y spans +/- ray_radius_mm) — this adds the actual PV cell width at the
    # target plane as a distinct marker, so aperture / converging spot /
    # cell are all visible on the same axes at once.
    z_target_mm = data["z_target"] * 1e3
    cell_radius_mm = data["cell_radius"] * 1e3
    fig.add_trace(go.Scatter(
        x=[z_target_mm, z_target_mm], y=[-cell_radius_mm, cell_radius_mm],
        mode="lines+markers",
        line=dict(color="white", width=5),
        marker=dict(size=10, symbol="line-ew", line=dict(width=2, color="white")),
        name="PV cell width",
        hovertemplate=f"PV cell: ±{cell_radius_mm:.3f} mm<extra>PV cell width</extra>",
    ))

    fig.update_layout(
        title=f"Optical Layout — ray fan @ {lam_nm:.0f} nm "
              "(dotted = lost ray, fainter = more Fresnel loss)",
        xaxis_title="z (mm)",
        yaxis_title="x — radial distance from optical axis (mm)",
        margin=dict(t=60, b=40),
    )
    return fig


def _spherical_z_of_rho(R, cz, rho):
    sign = 1.0 if R >= 0 else -1.0
    return cz - sign * np.sqrt(np.maximum(R ** 2 - rho ** 2, 0.0))


def _fresnel_z_of_rho(R, cz, rho, aperture_half, pitch):
    """Vectorized radial counterpart of _fresnel_zone_profile()'s per-zone
    linear approximation — evaluates the same schematic sawtooth at
    arbitrary radii instead of only at zone boundaries, so it can be
    revolved into a surface mesh."""
    z0 = cz - R
    sign = 1.0 if R >= 0 else -1.0
    rho = np.clip(rho, 0.0, aperture_half)
    k = np.floor(rho / pitch)
    rho_start = k * pitch
    rho_ref = np.minimum((k + 0.5) * pitch, aperture_half)
    inside = np.maximum(R * R - rho_ref ** 2, 1e-12)
    slope = sign * rho_ref / np.sqrt(inside)
    return z0 + slope * (rho - rho_start)


@st.cache_data(show_spinner=False)
def compute_optical_layout_3d(values, n_radii=4, n_azimuth=12, lam_nm=550.0,
                               mesh_rho=24, mesh_theta=48):
    """
    3-D counterpart of compute_optical_layout(): a starburst ray bundle
    traced through the full path, plus each surface revolved around the
    optical axis into a mesh — valid because every surface here is built
    on-axis (center x = y = 0), see build_layered_system().
    """
    system = make_system(values)
    aperture = values["ray_radius_mm"] * 1e-3

    bundle = optics.make_ray_starburst(radius=aperture, n_radii=n_radii, n_azimuth=n_azimuth)
    paths = []
    for ray in bundle:
        pts, alive, power = system.trace_ray_path(ray, lam_nm * 1e-9)
        paths.append((np.array(pts), alive, power))

    surfaces = []
    for surf in system.surfaces:
        R = surf.R
        cz = surf.center[2]
        half_ap = min(aperture * 1.3, abs(R) * 0.97) if R != 0 else aperture * 1.3
        rho_grid = np.linspace(0.0, half_ap, mesh_rho)
        theta_grid = np.linspace(0.0, 2 * np.pi, mesh_theta)
        RHO, THETA = np.meshgrid(rho_grid, theta_grid)
        if surf.surface_type == "fresnel":
            Z = _fresnel_z_of_rho(R, cz, RHO, half_ap, surf.groove_pitch)
        else:
            Z = _spherical_z_of_rho(R, cz, RHO)
        X = RHO * np.cos(THETA)
        Y = RHO * np.sin(THETA)
        surfaces.append((X, Y, Z, surf.surface_type))

    return {
        "paths": paths,
        "surfaces": surfaces,
        "z_target": system.z_target,
        "aperture": aperture,
        "cell_radius": values["cell_radius_mm"] * 1e-3,
    }


def plot_optical_layout_3d_plotly(values, n_radii=4, n_azimuth=12, lam_nm=550.0):
    data = compute_optical_layout_3d(values, n_radii=n_radii, n_azimuth=n_azimuth, lam_nm=lam_nm)
    fig = go.Figure()

    surface_colors = {"spherical": "#4C78A8", "fresnel": "#F58518"}
    legend_shown = set()
    for X, Y, Z, surf_type in data["surfaces"]:
        color = surface_colors[surf_type]
        fig.add_trace(go.Surface(
            x=X * 1e3, y=Y * 1e3, z=Z * 1e3,
            colorscale=[[0, color], [1, color]], showscale=False, opacity=0.5,
            name=surf_type.capitalize(), showlegend=surf_type not in legend_shown,
            hoverinfo="skip",
        ))
        legend_shown.add(surf_type)

    n_paths = len(data["paths"])
    for i, (pts, alive, power) in enumerate(data["paths"]):
        if pts.shape[0] < 2:
            continue
        frac = i / max(n_paths - 1, 1)
        color = f"rgb({int(255 * frac)},80,{int(255 * (1 - frac))})"
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0] * 1e3, y=pts[:, 1] * 1e3, z=pts[:, 2] * 1e3,
            mode="lines",
            line=dict(color=color, width=3, dash="solid" if alive else "dot"),
            opacity=max(0.2, power),
            hoverinfo="skip", showlegend=False,
        ))

    theta_full = np.linspace(0.0, 2 * np.pi, 100)
    cell_radius_mm = data["cell_radius"] * 1e3
    z_target_mm = data["z_target"] * 1e3
    fig.add_trace(go.Scatter3d(
        x=cell_radius_mm * np.cos(theta_full),
        y=cell_radius_mm * np.sin(theta_full),
        z=np.full_like(theta_full, z_target_mm),
        mode="lines",
        line=dict(color="white", width=5),
        name="PV cell boundary",
        hovertemplate=f"PV cell radius: {cell_radius_mm:.3f} mm<extra>PV cell boundary</extra>",
    ))

    fig.update_layout(
        title=f"3-D Optical Layout — ray bundle @ {lam_nm:.0f} nm "
              "(dotted = lost ray, fainter = more Fresnel loss)",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z — optical axis (mm)",
            aspectmode="cube",
        ),
        margin=dict(t=60, b=40),
        height=700,
    )
    return fig


def plot_focus_shift_plotly(system, lam_list, rays_in):
    z_nom = system.z_target
    z_scan = np.linspace(z_nom * 0.5, z_nom * 2.0, 200)

    focus_z, focus_rms = [], []
    for lam in lam_list:
        zf, rms = system.find_paraxial_focus(lam, z_search=z_scan, rays_in=rays_in)
        focus_z.append(zf * 1e3)
        focus_rms.append(rms * 1e6)

    lam_nm = np.array(lam_list) * 1e9
    colors = [_wavelength_color(l) for l in lam_list]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Longitudinal chromatic aberration", "Spot size at best focus"),
    )
    fig.add_trace(go.Scatter(
        x=lam_nm, y=focus_z, mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="rgba(120,120,120,0.5)"),
        hovertemplate="%{x:.0f} nm<br>z = %{y:.3f} mm<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lam_nm, y=focus_rms, mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="rgba(120,120,120,0.5)"),
        hovertemplate="%{x:.0f} nm<br>RMS = %{y:.3f} µm<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    for c in (1, 2):
        fig.add_vrect(x0=500, x1=900, fillcolor="green", opacity=0.08, line_width=0, row=1, col=c)

    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="Best-focus z (mm)", row=1, col=1)
    fig.update_yaxes(title_text="RMS spot at best focus (µm)", row=1, col=2)
    fig.update_layout(title_text="Chromatic Focus Shift", height=420, margin=dict(t=80, b=40))
    return fig


@st.cache_data(show_spinner=False)
def compute_acceptance_scan(values, lam_nm=550.0, theta_max_deg=3.0, n_theta=16, threshold=0.9):
    system = make_system(values)
    return optics.concentration_acceptance_product(
        system, lam_nm * 1e-9,
        cell_radius=values["cell_radius_mm"] * 1e-3,
        aperture_radius=values["ray_radius_mm"] * 1e-3,
        n_rays=int(values["n_rays"]),
        theta_max_deg=theta_max_deg, n_theta=n_theta, threshold=threshold,
    )


def plot_acceptance_plotly(scan):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scan["thetas_deg"], y=scan["collected_norm"], mode="lines+markers",
        line=dict(color=RMS_COLOR, width=2),
        hovertemplate="%{x:.3f}°<br>%{y:.1%} of on-axis<extra></extra>",
        name="Collected power",
    ))
    fig.add_hline(y=0.9, line=dict(color="red", dash="dot"), annotation_text="90% threshold")
    if scan["theta_accept_deg"] is not None:
        fig.add_vline(
            x=scan["theta_accept_deg"], line=dict(color="green", dash="dot"),
            annotation_text=f"acceptance = {scan['theta_accept_deg']:.3f}°",
        )
    title = f"Angular Acceptance — C_geo={scan['C_geo']:.1f}×"
    if scan["CAP"] is not None:
        title += f", CAP={scan['CAP']:.3f}"
    fig.update_layout(
        title=title,
        xaxis_title="Incidence angle (deg)",
        yaxis_title="Collected power (fraction of on-axis)",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(t=60, b=40),
    )
    return fig


def render_acceptance_expander(values, key_prefix):
    """
    Reusable "Angular acceptance (CAP)" expander — used identically in the
    Analysis, Optimize, and Auto-design tabs, each with their own `values`
    and a unique `key_prefix` (Streamlit needs unique widget/chart keys
    across tabs, see the plotly_chart duplicate-ID note elsewhere).
    """
    with st.expander("Angular acceptance (CAP)"):
        st.caption(
            "Concentration-Acceptance Product: sweeps incidence angle and finds "
            "where collected power (Fresnel-loss-weighted, within the PV cell "
            "radius) drops to 90% of its on-axis value — the standard CPV "
            "figure of merit combining concentration and angular tolerance."
        )
        if st.button("Calculate angular acceptance", type="secondary", key=f"{key_prefix}_accept_btn"):
            with st.spinner("Scanning incidence angle..."):
                scan = compute_acceptance_scan(values)
            c1, c2, c3 = st.columns(3)
            c1.metric("Geometric concentration", f"{scan['C_geo']:.1f}×")
            c2.metric(
                "Acceptance angle (90%)",
                f"{scan['theta_accept_deg']:.3f}°" if scan["theta_accept_deg"] is not None else "> scan range",
            )
            c3.metric("CAP", f"{scan['CAP']:.3f}" if scan["CAP"] is not None else "—")
            st.plotly_chart(plot_acceptance_plotly(scan), use_container_width=True, key=f"{key_prefix}_accept_fig")


@st.cache_data(show_spinner=False)
def compute_uniformity_scan(values, n_bins=6):
    system = make_system(values)
    rays_in = get_rays_in(values)
    lam_grid = get_lambda_grid(values)
    cell_radius = values["cell_radius_mm"] * 1e-3

    peak_to_avg = np.full(len(lam_grid), np.nan)
    cv = np.full(len(lam_grid), np.nan)
    for i, lam in enumerate(lam_grid):
        rays_out = system.trace_bundle(rays_in, lam)
        u = optics.irradiance_uniformity(rays_out, cell_radius, n_bins=n_bins)
        if u["peak_to_avg"] is not None:
            peak_to_avg[i] = u["peak_to_avg"]
            cv[i] = u["cv"]
    return {"lam_m": lam_grid, "peak_to_avg": peak_to_avg, "cv": cv}


def plot_uniformity_plotly(scan):
    lam_nm = scan["lam_m"] * 1e9
    colors = [_wavelength_color(lam) for lam in scan["lam_m"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Peak-to-average irradiance", "Coefficient of variation"),
    )
    fig.add_trace(go.Scatter(
        x=lam_nm, y=scan["peak_to_avg"], mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="rgba(120,120,120,0.5)"),
        hovertemplate="%{x:.0f} nm<br>peak/avg = %{y:.2f}×<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lam_nm, y=scan["cv"], mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="rgba(120,120,120,0.5)"),
        hovertemplate="%{x:.0f} nm<br>CV = %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color="green", dash="dot"), row=1, col=1,
                  annotation_text="perfectly uniform")
    fig.add_hline(y=0.0, line=dict(color="green", dash="dot"), row=1, col=2,
                  annotation_text="perfectly uniform")

    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="Peak / average irradiance", row=1, col=1)
    fig.update_yaxes(title_text="Coefficient of variation", row=1, col=2)
    fig.update_layout(title_text="Irradiance Uniformity on PV Cell", height=420, margin=dict(t=80, b=40))
    return fig


def render_uniformity_expander(values, key_prefix):
    """
    Reusable "Irradiance uniformity" expander — same pattern as
    render_acceptance_expander(), see its docstring.
    """
    with st.expander("Irradiance uniformity"):
        st.caption(
            "RMS spot radius alone doesn't say whether the cell is evenly lit "
            "or has a bright hotspot with a dim rim — multi-junction cells are "
            "sensitive to that (non-uniform illumination causes current "
            "mismatch between sub-cells). Radially binned within the PV cell "
            "radius; peak/average = 1.0 and CV = 0.0 both mean perfectly even. "
            "With few rays, bins can be noisy — increase the sidebar's ray "
            "count for a more reliable estimate."
        )
        if st.button("Calculate uniformity", type="secondary", key=f"{key_prefix}_unif_btn"):
            with st.spinner("Binning irradiance across the spectrum..."):
                scan = compute_uniformity_scan(values)
            finite = np.isfinite(scan["peak_to_avg"])
            c1, c2 = st.columns(2)
            c1.metric(
                "Mean peak/average",
                f"{np.nanmean(scan['peak_to_avg']):.2f}×" if finite.any() else "—",
            )
            c2.metric(
                "Mean CV",
                f"{np.nanmean(scan['cv']):.2f}" if finite.any() else "—",
            )
            st.plotly_chart(plot_uniformity_plotly(scan), use_container_width=True, key=f"{key_prefix}_unif_fig")


@st.cache_data(show_spinner=False)
def compute_tolerance_sensitivity(values, delta_mm=0.01):
    materials = get_materials(values)
    surface_types = values["surface_types"]
    groove_pitches = get_groove_pitches_m(values)

    def builder(p):
        return optics.build_nlayer_system(
            p, materials, surface_types=surface_types, groove_pitches=groove_pitches)

    return optics.tolerance_sensitivity(
        make_params(values), builder,
        lam_list=get_lambda_grid(values),
        rays_in=get_rays_in(values),
        param_names=optics.default_param_names(values["n_layers"]),
        delta=delta_mm * 1e-3,
    )


def plot_tolerance_plotly(sensitivity, delta_mm):
    order = sorted(range(len(sensitivity)),
                   key=lambda i: sensitivity[i]["rms_sensitivity_um_per_mm"])
    names = [sensitivity[i]["parameter"] for i in order]
    vals = [sensitivity[i]["rms_sensitivity_um_per_mm"] for i in order]

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color=RMS_COLOR)))
    fig.update_layout(
        title=f"RMS Spot Sensitivity per Parameter (±{delta_mm:g} mm perturbation)",
        xaxis_title="Δ RMS spot radius (µm) per mm of parameter error",
        yaxis_title="Parameter",
        margin=dict(t=60, b=40),
    )
    return fig


def render_tolerancing_expander(values, key_prefix):
    """
    Reusable "Tolerancing" expander — same pattern as
    render_acceptance_expander(), see its docstring.
    """
    with st.expander("Tolerancing (sensitivity analysis)"):
        st.caption(
            "One-at-a-time sensitivity: perturbs each geometry parameter by "
            "±delta (holding everything else fixed) and reports how much RMS "
            "spot size moves — a design that's optically best but collapses "
            "with a tiny manufacturing error isn't actually the best design "
            "to build. Higher sensitivity means that dimension needs tighter "
            "control. This doesn't cover decenter/tilt or simultaneous "
            "multi-parameter error, only one-at-a-time axis-aligned error."
        )
        delta_mm = st.number_input(
            "Perturbation size (mm)", min_value=0.001, value=0.01, step=0.005,
            format="%.4f", key=f"{key_prefix}_tol_delta",
        )
        if st.button("Run sensitivity analysis", type="secondary", key=f"{key_prefix}_tol_btn"):
            with st.spinner("Perturbing each parameter..."):
                sensitivity = compute_tolerance_sensitivity(values, delta_mm=delta_mm)

            df = pd.DataFrame([
                {
                    "Parameter": e["parameter"],
                    "Nominal (mm)": e["nominal"] * 1e3,
                    f"RMS @ -{delta_mm:g}mm (µm)": e["rms_minus_um"],
                    "RMS nominal (µm)": e["rms_nominal_um"],
                    f"RMS @ +{delta_mm:g}mm (µm)": e["rms_plus_um"],
                    "Sensitivity (µm/mm)": e["rms_sensitivity_um_per_mm"],
                }
                for e in sensitivity
            ]).sort_values("Sensitivity (µm/mm)", ascending=False)
            st.dataframe(
                df.style.format({
                    "Nominal (mm)": "{:+.4f}",
                    f"RMS @ -{delta_mm:g}mm (µm)": "{:.3f}",
                    "RMS nominal (µm)": "{:.3f}",
                    f"RMS @ +{delta_mm:g}mm (µm)": "{:.3f}",
                    "Sensitivity (µm/mm)": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.plotly_chart(
                plot_tolerance_plotly(sensitivity, delta_mm),
                use_container_width=True, key=f"{key_prefix}_tol_fig",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def flatten_values_for_export(values):
    rows = [{"parameter": "n_layers", "value": values["n_layers"]}]
    for i, r in enumerate(values["radii_mm"]):
        rows.append({"parameter": f"R{i}_mm", "value": r})
        rows.append({"parameter": f"R{i}_type", "value": values["surface_types"][i]})
        if values["groove_pitches_mm"][i] is not None:
            rows.append({"parameter": f"R{i}_pitch_mm", "value": values["groove_pitches_mm"][i]})
    for i, t in enumerate(values["thicknesses_mm"]):
        rows.append({"parameter": f"t{i + 1}_mm", "value": t})
    for i, m in enumerate(values["materials"]):
        rows.append({"parameter": f"mat{i + 1}", "value": m})
    for key in ("f_mm", "ray_radius_mm", "cell_radius_mm", "n_rays", "lam_start_nm",
                "lam_end_nm", "lam_steps", "opt_method", "opt_maxiter"):
        rows.append({"parameter": key, "value": values[key]})
    return rows


def params_csv(values):
    return pd.DataFrame(flatten_values_for_export(values)).to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

st.title("PV Concentrator Ray-Optics Engine")
st.caption(
    "Interactive analysis, material screening, and optimization of an "
    "N-layer refractive concentrator lens."
)

# Apply any pending "copy optimized params to sidebar" request *before* the
# sidebar widgets below are instantiated — Streamlit forbids writing to
# st.session_state[key] for a widget's key after that widget has already
# rendered in the current run, so the request is staged here instead.
if "_apply_params" in st.session_state:
    for _key, _val in st.session_state.pop("_apply_params").items():
        st.session_state[_key] = _val

def _render_surface_type_widget(key_prefix, label):
    """Spherical/Fresnel selector for one surface, plus a groove-pitch field
    that only appears when Fresnel is selected. Returns (type, pitch_mm)."""
    surf_type = st.selectbox(
        f"{label} type", ["Spherical", "Fresnel"], key=f"{key_prefix}_type",
    )
    if surf_type == "Fresnel":
        pitch = st.number_input(
            f"{label} groove pitch (mm)", min_value=0.01,
            value=DEFAULT_GROOVE_PITCH_MM, step=0.05, format="%.3f", key=f"{key_prefix}_pitch",
            help="Radial width of each Fresnel zone — smaller pitch means "
                 "more/finer grooves, closer to the smooth (spherical) curve; "
                 "larger pitch means fewer, coarser grooves and more faceting error.",
        )
    else:
        pitch = None
    return surf_type.lower(), pitch


with st.sidebar:
    st.header("Lens geometry")
    n_layers = st.number_input(
        "Number of layers",
        min_value=1,
        max_value=MAX_LAYERS,
        value=DEFAULTS["n_layers"],
        step=1,
        key="n_layers",
        help="Reacts immediately (outside the form below) so field count "
             "updates without needing to click Apply.",
    )

    with st.form("controls_form"):
        material_names_list = list(optics.CANDIDATE_MATERIALS.keys())
        radii_mm, thicknesses_mm, materials_sel = [], [], []
        surface_types, groove_pitches_mm = [], []

        for i in range(n_layers):
            st.markdown(f"**Layer {i + 1}**")
            r_label = "R0 — entry curvature (mm)" if i == 0 else f"R{i} — internal curvature (mm)"
            r_default = DEFAULT_RADII_MM[i] if i < len(DEFAULT_RADII_MM) else DEFAULT_RADII_MM[-1]
            r = st.number_input(r_label, value=r_default, step=1.0, format="%.3f", key=f"R{i}_mm")
            radii_mm.append(r)
            surf_type, pitch = _render_surface_type_widget(f"R{i}", f"R{i}")
            surface_types.append(surf_type)
            groove_pitches_mm.append(pitch)

            t = st.number_input(
                f"Layer {i + 1} thickness (mm)", min_value=0.5,
                value=DEFAULT_THICKNESS_MM, step=0.25, format="%.3f", key=f"t{i + 1}_mm",
            )
            thicknesses_mm.append(t)

            mat_default = DEFAULT_MATERIALS_CYCLE[i % len(DEFAULT_MATERIALS_CYCLE)]
            m = st.selectbox(
                f"Layer {i + 1} material", material_names_list,
                index=material_names_list.index(mat_default), key=f"mat{i + 1}",
            )
            materials_sel.append(m)

        r_exit_default = DEFAULT_RADII_MM[n_layers] if n_layers < len(DEFAULT_RADII_MM) else DEFAULT_RADII_MM[-1]
        r_exit = st.number_input(
            f"R{n_layers} — exit curvature (mm)", value=r_exit_default,
            step=1.0, format="%.3f", key=f"R{n_layers}_mm",
        )
        radii_mm.append(r_exit)
        surf_type, pitch = _render_surface_type_widget(f"R{n_layers}", f"R{n_layers}")
        surface_types.append(surf_type)
        groove_pitches_mm.append(pitch)

        f_mm = st.number_input(
            "PV plane / target focal length (mm)", min_value=1.0,
            value=DEFAULTS["f_mm"], step=1.0, format="%.3f", key="f_mm",
        )

        st.header("Ray bundle")
        ray_radius_mm = st.number_input(
            "Aperture half-width (mm)", min_value=0.01,
            value=DEFAULTS["ray_radius_mm"], step=0.05, format="%.3f", key="ray_radius_mm",
        )
        n_rays = st.select_slider(
            "Approximate number of rays",
            options=[9, 16, 25, 36, 49, 64, 81, 100],
            value=DEFAULTS["n_rays"], key="n_rays",
        )

        st.header("PV cell")
        cell_radius_mm = st.number_input(
            "Cell radius (mm)", min_value=0.001,
            value=DEFAULTS["cell_radius_mm"], step=0.05, format="%.3f", key="cell_radius_mm",
            help="Physical PV cell size at the target plane — used for the "
                 "angular-acceptance / concentration-acceptance-product (CAP) "
                 "analysis, distinct from the lens aperture above.",
        )

        st.header("Spectrum")
        lam_start_nm = st.number_input(
            "Start wavelength (nm)", min_value=300.0, max_value=1500.0,
            value=DEFAULTS["lam_start_nm"], step=10.0, key="lam_start_nm",
        )
        lam_end_nm = st.number_input(
            "End wavelength (nm)", min_value=300.0, max_value=1500.0,
            value=DEFAULTS["lam_end_nm"], step=10.0, key="lam_end_nm",
        )
        lam_steps = st.slider(
            "Wavelength samples", min_value=3, max_value=31,
            value=DEFAULTS["lam_steps"], step=2, key="lam_steps",
        )

        st.header("Optimization")
        opt_method = st.selectbox("Optimizer", ["Nelder-Mead", "Powell"], index=0, key="opt_method")
        opt_maxiter = st.slider(
            "Maximum iterations", min_value=50, max_value=2000,
            value=DEFAULTS["opt_maxiter"], step=50, key="opt_maxiter",
        )

        st.form_submit_button("Apply settings", type="primary", use_container_width=True)

    st.caption(
        "Changes take effect after clicking **Apply settings** — dragging "
        "sliders no longer re-traces the whole system on every step."
    )

values = {
    "n_layers": n_layers,
    "radii_mm": radii_mm,
    "surface_types": surface_types,
    "groove_pitches_mm": groove_pitches_mm,
    "thicknesses_mm": thicknesses_mm,
    "materials": materials_sel,
    "f_mm": f_mm,
    "ray_radius_mm": ray_radius_mm,
    "cell_radius_mm": cell_radius_mm,
    "n_rays": n_rays,
    "lam_start_nm": lam_start_nm,
    "lam_end_nm": lam_end_nm,
    "lam_steps": lam_steps,
    "opt_method": opt_method,
    "opt_maxiter": opt_maxiter,
}

if values["lam_end_nm"] <= values["lam_start_nm"]:
    st.error("The end wavelength must be greater than the start wavelength.")
    st.stop()

if values["n_layers"] == 2 and values["materials"][0] == values["materials"][1]:
    st.warning(
        "The material survey excludes pairs with identical materials. "
        "Direct analysis remains valid."
    )

system = make_system(values)
metrics_df, current_cost = per_wavelength_metrics(values)
materials = get_materials(values)

tab_analysis, tab_3d, tab_survey, tab_optimize, tab_auto, tab_export = st.tabs(
    ["Analysis", "3-D view", "Material survey", "Optimize", "Auto-design", "Export"]
)

with tab_analysis:
    cols = st.columns(len(materials) + 4)
    for i, mat in enumerate(materials):
        cols[i].metric(f"Layer {i + 1}", mat.name)
    cols[-4].metric("Weighted cost", f"{current_cost:.3e}")
    cols[-3].metric(
        "Mean RMS spot radius",
        f"{metrics_df['RMS spot radius (µm)'].mean():.2f} µm",
    )
    cols[-2].metric(
        "Mean optical efficiency",
        f"{metrics_df['Optical efficiency'].mean():.1%}",
        help="Fraction of incident power that reaches the target plane "
             "somewhere (Fresnel losses only) — doesn't check whether it "
             "actually landed on the cell. See collection efficiency for that.",
    )
    cols[-1].metric(
        "Mean collection efficiency",
        f"{metrics_df['Collection efficiency (on cell)'].mean():.1%}",
        help="Fraction of incident power landing within the sidebar's PV "
             "cell radius — the number that actually answers 'how much of "
             "the collected light lands on the cell'.",
    )

    st.subheader("Optical layout")
    st.plotly_chart(plot_optical_layout_plotly(values), use_container_width=True, key="layout_analysis")

    left, right = st.columns([1.1, 1])

    with left:
        st.plotly_chart(
            plot_rms_plotly([(values, "Current design", RMS_COLOR)]),
            use_container_width=True,
            key="rms_analysis",
        )

    with right:
        st.subheader("Per-wavelength results")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Wavelength (nm)": "{:.1f}",
                    "PV weight": "{:.1f}",
                    "RMS spot radius (µm)": "{:.3f}",
                    "Optical efficiency": "{:.1%}",
                    "Collection efficiency (on cell)": "{:.1%}",
                    "Rays reaching PV": "{:.0f}",
                }
            ).map(_wavelength_cell_style, subset=["Wavelength (nm)"]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Spot diagrams")
    st.plotly_chart(plot_spots_plotly(values), use_container_width=True, key="spots_analysis")

    with st.expander("Chromatic focus shift"):
        if st.button("Calculate focus shift", type="secondary"):
            with st.spinner("Scanning best-focus plane across wavelengths..."):
                fig_focus = plot_focus_shift_plotly(
                    system, get_lambda_grid(values), get_rays_in(values),
                )
            st.plotly_chart(fig_focus, use_container_width=True, key="focus_shift_analysis")

    render_acceptance_expander(values, "analysis")
    render_uniformity_expander(values, "analysis")
    render_tolerancing_expander(values, "analysis")

with tab_3d:
    st.caption(
        "Revolves the lens surfaces around the optical axis and traces a "
        "starburst ray bundle (several radii × azimuthal angles) through "
        "the full 3-D path — valid since every surface here is centered on "
        "the axis. Drag to rotate, scroll to zoom."
    )
    c1, c2, c3 = st.columns(3)
    n_radii_3d = c1.slider("Radial rings", min_value=1, max_value=8, value=4, key="viz3d_n_radii")
    n_azimuth_3d = c2.slider("Azimuthal rays per ring", min_value=4, max_value=24, value=12, key="viz3d_n_azimuth")
    lam_nm_3d = c3.number_input(
        "Wavelength (nm)", min_value=values["lam_start_nm"], max_value=values["lam_end_nm"],
        value=float(np.clip(550.0, values["lam_start_nm"], values["lam_end_nm"])),
        step=10.0, key="viz3d_lam_nm",
    )
    st.plotly_chart(
        plot_optical_layout_3d_plotly(values, n_radii=n_radii_3d, n_azimuth=n_azimuth_3d, lam_nm=lam_nm_3d),
        use_container_width=True, key="layout_3d",
    )

with tab_survey:
    if values["n_layers"] != 2:
        st.info(
            "Material survey compares ordered pairs of materials for a "
            "2-layer design. Set **Number of layers** to 2 in the sidebar "
            "to use it."
        )
    else:
        st.write(
            "This screens every ordered pair of distinct catalog materials at the "
            "current geometry, using the ray bundle and wavelength grid configured "
            "in the sidebar. It is a screening step, not an optimized comparison."
        )

        if st.button("Run material survey", type="primary"):
            with st.spinner("Evaluating material pairs..."):
                scores = optics.material_survey(
                    make_params(values),
                    verbose=False,
                    lam_list=get_lambda_grid(values),
                    rays_in=get_rays_in(values),
                    surface_types=values["surface_types"],
                    groove_pitches=get_groove_pitches_m(values),
                )

            survey_df = pd.DataFrame(
                scores,
                columns=["Cost", "Layer 1 material", "Layer 2 material"],
            )
            survey_df["Cost"] = survey_df["Cost"].map(lambda x: f"{x:.4e}")
            st.dataframe(survey_df, use_container_width=True, hide_index=True)

with tab_optimize:
    st.warning(
        "Optimization can be computationally expensive. Start with 200–500 "
        "iterations, then increase only after confirming the design behavior. "
        "More layers mean more free parameters, which the gradient-free "
        "Nelder-Mead/Powell optimizers handle less reliably — keep an eye on "
        "whether the run actually converges."
    )
    st.caption(
        "Uses the ray bundle and wavelength grid configured in the sidebar "
        "for the honest cost; a coarser subsample of both is used internally "
        "during the search for speed."
    )
    optimize_pitch = st.checkbox(
        "Optimize groove pitch for any Fresnel surface", value=True, key="opt_pitch_toggle",
        help="Search each Fresnel surface's groove pitch alongside geometry "
             "instead of holding it fixed at the sidebar's value.",
    )

    if st.button("Optimize lens", type="primary"):
        progress_bar = st.progress(0, text="Optimizing... iteration 0")
        console_box = st.empty()

        def _report_progress(n, maxiter):
            progress_bar.progress(
                min(n / maxiter, 1.0),
                text=f"Optimizing... iteration {n}/{maxiter}",
            )

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = optics.optimize_lens(
                materials=materials,
                x0=make_params(values),
                method=values["opt_method"],
                maxiter=int(values["opt_maxiter"]),
                lam_list=get_lambda_grid(values),
                rays_in=get_rays_in(values),
                surface_types=values["surface_types"],
                groove_pitches=get_groove_pitches_m(values),
                optimize_groove_pitch=optimize_pitch,
                progress_callback=_report_progress,
            )

        progress_bar.progress(1.0, text="Optimization complete.")
        console_box.code(capture.getvalue(), language="text")

        n_layers_r = values["n_layers"]
        params = result["params"]
        # Persisted in session_state (rather than kept as plain locals) so the
        # results section below survives the rerun triggered by the "apply to
        # sidebar" button, instead of vanishing because `st.button` is only
        # True on the run where it was actually clicked.
        st.session_state["opt_result"] = {
            "n_layers": n_layers_r,
            "radii_mm": (params[:n_layers_r + 1] * 1e3).tolist(),
            "thicknesses_mm": (params[n_layers_r + 1:2 * n_layers_r + 1] * 1e3).tolist(),
            "f_mm": float(params[2 * n_layers_r + 1] * 1e3),
            "materials": list(values["materials"]),
            "groove_pitches_mm": [p * 1e3 if p is not None else None for p in result["groove_pitches"]],
            "cost": result["cost"],
            "base_values": dict(values),
        }

    opt_state = st.session_state.get("opt_result")
    if opt_state is not None:
        n_layers_r = opt_state["n_layers"]
        radii_mm_r = opt_state["radii_mm"]
        thicknesses_mm_r = opt_state["thicknesses_mm"]
        f_mm_r = opt_state["f_mm"]

        base_values = opt_state["base_values"]
        opt_values = dict(base_values)
        opt_values["n_layers"] = n_layers_r
        opt_values["radii_mm"] = radii_mm_r
        opt_values["thicknesses_mm"] = thicknesses_mm_r
        opt_values["f_mm"] = f_mm_r
        opt_values["materials"] = opt_state["materials"]
        opt_values["groove_pitches_mm"] = opt_state["groove_pitches_mm"]
        opt_trace = trace_all_wavelengths(opt_values)
        opt_efficiency = opt_trace["efficiency"].mean()
        opt_collection = opt_trace["collection_efficiency"].mean()

        cols = st.columns(len(opt_state["materials"]) + 3)
        for i, mname in enumerate(opt_state["materials"]):
            cols[i].metric(f"Layer {i + 1}", mname)
        cols[-3].metric("Final cost", f"{opt_state['cost']:.4e}")
        cols[-2].metric("Mean optical efficiency", f"{opt_efficiency:.1%}")
        cols[-1].metric("Mean collection efficiency", f"{opt_collection:.1%}")

        param_rows = (
            [{"Parameter": f"R{i}", "Value (mm)": r} for i, r in enumerate(radii_mm_r)]
            + [{"Parameter": f"t{i + 1}", "Value (mm)": t} for i, t in enumerate(thicknesses_mm_r)]
            + [{"Parameter": "f", "Value (mm)": f_mm_r}]
        )
        st.dataframe(
            pd.DataFrame(param_rows).style.format({"Value (mm)": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Apply optimized parameters to sidebar"):
            pending = {"n_layers": n_layers_r, "f_mm": f_mm_r}
            for i, r in enumerate(radii_mm_r):
                pending[f"R{i}_mm"] = float(r)
            for i, t in enumerate(thicknesses_mm_r):
                pending[f"t{i + 1}_mm"] = float(t)
            for i, p in enumerate(opt_state["groove_pitches_mm"]):
                if p is not None:
                    pending[f"R{i}_pitch"] = float(p)
            st.session_state["_apply_params"] = pending
            st.rerun()

        st.plotly_chart(
            plot_rms_plotly([
                (base_values, "Initial", INITIAL_COLOR),
                (opt_values, "Optimized", OPTIMIZED_COLOR),
            ]),
            use_container_width=True,
            key="rms_optimize",
        )

        st.subheader("Optimized optical layout")
        st.plotly_chart(plot_optical_layout_plotly(opt_values), use_container_width=True, key="layout_optimize")

        st.subheader("Optimized spot diagrams")
        st.plotly_chart(plot_spots_plotly(opt_values), use_container_width=True, key="spots_optimize")

        render_acceptance_expander(opt_values, "optimize")
        render_uniformity_expander(opt_values, "optimize")
        render_tolerancing_expander(opt_values, "optimize")

with tab_auto:
    st.write(
        f"Jointly searches material combinations *and* geometry for the best "
        f"{values['n_layers']}-layer design, using the current sidebar "
        "geometry as the starting point for every combination."
    )
    st.caption(
        "Uses the ray bundle, wavelength grid, and optimizer method configured "
        "in the sidebar; the sidebar's \"Maximum iterations\" is reused as the "
        "final-resolution iteration budget below."
    )

    n_material_combos = len(optics.generate_material_combinations(values["n_layers"]))

    with st.expander("Advanced search settings"):
        full_optimize_threshold = st.number_input(
            "Full-optimize threshold — combinations at or below this count are "
            "fully optimized individually; above it, successive halving is used",
            min_value=1, value=40, step=1, key="auto_full_threshold",
        )
        screen_maxiter = st.number_input(
            "Halving round-1 iteration budget (doubles each round)",
            min_value=5, value=30, step=5, key="auto_screen_maxiter",
        )
        halving_rounds = st.number_input(
            "Number of halving rounds", min_value=1, max_value=6, value=3, step=1,
            key="auto_halving_rounds",
        )
        top_k_final = st.number_input(
            "Finalists given a full-resolution pass", min_value=1, value=3, step=1,
            key="auto_top_k",
        )
        optimize_pitch = st.checkbox(
            "Optimize groove pitch for any Fresnel surface", value=True, key="auto_pitch_toggle",
            help="Search each Fresnel surface's groove pitch alongside geometry "
                 "instead of holding it fixed at the sidebar's value.",
        )
        search_surface_types = st.checkbox(
            "Search surface types too (spherical vs. Fresnel per surface)",
            value=False, key="auto_search_surf_types",
            help=f"Also tries every spherical/Fresnel assignment across all "
                 f"{values['n_layers'] + 1} surfaces (2^{values['n_layers'] + 1} = "
                 f"{2 ** (values['n_layers'] + 1)} of them), crossed with the "
                 "material search — a large multiplier on top of it, so this "
                 "run will take proportionally longer. Off by default; when "
                 "off, every combination uses the sidebar's current per-surface "
                 "spherical/Fresnel choice.",
        )
        max_workers = st.number_input(
            "Parallel workers", min_value=1, max_value=16, value=1, step=1,
            key="auto_max_workers",
            help="Combinations within a round run concurrently on this many "
                 "threads. Benchmarking found this workload is dominated by "
                 "per-call Python overhead rather than large numpy array "
                 "work, so threads consistently measured ~15-20% *slower* "
                 "than sequential here — defaults to 1 for that reason. Only "
                 "raise it if you've verified it actually helps for your "
                 "own settings (ray count, iteration budget, etc.).",
        )

    n_combos = n_material_combos * (2 ** (values["n_layers"] + 1) if search_surface_types else 1)
    will_halve = n_combos > full_optimize_threshold
    st.caption(
        f"{n_combos} candidate combination(s) for {values['n_layers']} "
        f"layer(s) — " + ("will use successive halving." if will_halve
                           else "will fully optimize every combination.")
    )

    if st.button("Run auto-design", type="primary"):
        progress_bar = st.progress(0, text="Starting search...")

        def _report_auto_progress(step, total, label):
            progress_bar.progress(min(step / total, 1.0), text=f"{step}/{total} — {label}")

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            ad_result = optics.auto_design(
                n_layers=values["n_layers"],
                x0=make_params(values),
                lam_list=get_lambda_grid(values),
                rays_in=get_rays_in(values),
                method=values["opt_method"],
                surface_types=values["surface_types"],
                groove_pitches=get_groove_pitches_m(values),
                optimize_groove_pitch=optimize_pitch,
                search_surface_types=search_surface_types,
                full_optimize_threshold=int(full_optimize_threshold),
                screen_maxiter=int(screen_maxiter),
                halving_rounds=int(halving_rounds),
                top_k_final=int(top_k_final),
                final_maxiter=int(values["opt_maxiter"]),
                max_workers=int(max_workers),
                progress_callback=_report_auto_progress,
            )

        progress_bar.progress(1.0, text="Auto-design complete.")

        n_layers_r = values["n_layers"]
        best_params = ad_result["best"]["params"]
        best_pitches = ad_result["best"]["groove_pitches"]
        # Persisted in session_state so the results section below survives the
        # rerun triggered by the "apply to sidebar" button (see the Optimize
        # tab above for the same pattern and why it's needed).
        st.session_state["auto_result"] = {
            "n_layers": n_layers_r,
            "radii_mm": (best_params[:n_layers_r + 1] * 1e3).tolist(),
            "thicknesses_mm": (best_params[n_layers_r + 1:2 * n_layers_r + 1] * 1e3).tolist(),
            "f_mm": float(best_params[2 * n_layers_r + 1] * 1e3),
            "materials": ad_result["best_materials"],
            "surface_types": ad_result["best_surface_types"],
            "groove_pitches_mm": [p * 1e3 if p is not None else None for p in best_pitches],
            "search_surface_types": search_surface_types,
            "cost": ad_result["best"]["cost"],
            "leaderboard": ad_result["leaderboard"],
            "n_combinations_total": ad_result["n_combinations_total"],
            "n_combinations_evaluated": ad_result["n_combinations_evaluated"],
            "used_halving": ad_result["used_halving"],
            "base_values": dict(values),
            "console": capture.getvalue(),
        }

    auto_state = st.session_state.get("auto_result")
    if auto_state is not None:
        n_layers_r = auto_state["n_layers"]
        radii_mm_r = auto_state["radii_mm"]
        thicknesses_mm_r = auto_state["thicknesses_mm"]
        f_mm_r = auto_state["f_mm"]

        base_values = auto_state["base_values"]
        best_values = dict(base_values)
        best_values["n_layers"] = n_layers_r
        best_values["radii_mm"] = radii_mm_r
        best_values["thicknesses_mm"] = thicknesses_mm_r
        best_values["f_mm"] = f_mm_r
        best_values["materials"] = auto_state["materials"]
        # The winner's own surface types/pitches — NOT necessarily the same
        # as the current sidebar, since search_surface_types may have picked
        # a different spherical/Fresnel assignment than what's shown there.
        best_values["surface_types"] = auto_state["surface_types"]
        best_values["groove_pitches_mm"] = auto_state["groove_pitches_mm"]
        best_collection = trace_all_wavelengths(best_values)["collection_efficiency"].mean()

        surf_summary = "/".join(t[0].upper() for t in auto_state["surface_types"])
        st.success(
            f"Best design: {' / '.join(auto_state['materials'])}"
            + (f" — surfaces [{surf_summary}]" if auto_state["search_surface_types"] else "")
            + f" — searched {auto_state['n_combinations_evaluated']} of "
            f"{auto_state['n_combinations_total']} combination(s)"
            + (" via successive halving." if auto_state["used_halving"] else " (fully optimized).")
        )

        # leaderboard[0] is always the winning combo (results are sorted
        # best-first before the leaderboard is built), so its efficiency can
        # be reused directly instead of re-tracing.
        best_efficiency = auto_state["leaderboard"][0]["efficiency"]

        cols = st.columns(len(auto_state["materials"]) + 3)
        for i, mname in enumerate(auto_state["materials"]):
            cols[i].metric(f"Layer {i + 1}", mname)
        cols[-3].metric("Best cost", f"{auto_state['cost']:.4e}")
        cols[-2].metric("Optical efficiency", f"{best_efficiency:.1%}")
        cols[-1].metric("Collection efficiency", f"{best_collection:.1%}")

        param_rows = (
            [
                {
                    "Parameter": f"R{i}",
                    "Value (mm)": r,
                    "Type": auto_state["surface_types"][i],
                    "Pitch (mm)": auto_state["groove_pitches_mm"][i],
                }
                for i, r in enumerate(radii_mm_r)
            ]
            + [{"Parameter": f"t{i + 1}", "Value (mm)": t, "Type": "", "Pitch (mm)": None}
               for i, t in enumerate(thicknesses_mm_r)]
            + [{"Parameter": "f", "Value (mm)": f_mm_r, "Type": "", "Pitch (mm)": None}]
        )
        st.dataframe(
            pd.DataFrame(param_rows).style.format(
                {"Value (mm)": "{:+.4f}", "Pitch (mm)": "{:.4f}"}, na_rep=""
            ),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Apply best design to sidebar"):
            pending = {"n_layers": n_layers_r, "f_mm": f_mm_r}
            for i, r in enumerate(radii_mm_r):
                pending[f"R{i}_mm"] = float(r)
                pending[f"R{i}_type"] = auto_state["surface_types"][i].capitalize()
                if auto_state["groove_pitches_mm"][i] is not None:
                    pending[f"R{i}_pitch"] = float(auto_state["groove_pitches_mm"][i])
            for i, t in enumerate(thicknesses_mm_r):
                pending[f"t{i + 1}_mm"] = float(t)
            for i, mname in enumerate(auto_state["materials"]):
                pending[f"mat{i + 1}"] = mname
            st.session_state["_apply_params"] = pending
            st.rerun()

        st.subheader("Leaderboard")
        leaderboard_rows = []
        for entry in auto_state["leaderboard"]:
            row = {"Materials": " / ".join(entry["materials"])}
            if auto_state["search_surface_types"]:
                row["Surfaces"] = "/".join(t[0].upper() for t in entry["surface_types"])
            row["Cost"] = entry["cost"]
            row["Optical efficiency"] = entry["efficiency"]
            leaderboard_rows.append(row)
        st.dataframe(
            pd.DataFrame(leaderboard_rows).style.format(
                {"Cost": "{:.4e}", "Optical efficiency": "{:.1%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Best design — optical layout")
        st.plotly_chart(plot_optical_layout_plotly(best_values), use_container_width=True, key="layout_auto")

        st.subheader("Best design — RMS vs wavelength")
        st.plotly_chart(
            plot_rms_plotly([(best_values, "Best design", OPTIMIZED_COLOR)]),
            use_container_width=True, key="rms_auto",
        )

        st.subheader("Best design — spot diagrams")
        st.plotly_chart(plot_spots_plotly(best_values), use_container_width=True, key="spots_auto")

        render_acceptance_expander(best_values, "auto")
        render_uniformity_expander(best_values, "auto")
        render_tolerancing_expander(best_values, "auto")

        with st.expander("Console output"):
            st.code(auto_state["console"][-5000:], language="text")

with tab_export:
    st.download_button(
        "Download parameter CSV",
        data=params_csv(values),
        file_name="pv_ray_optics_parameters.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download per-wavelength results",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="pv_ray_optics_metrics.csv",
        mime="text/csv",
    )

    st.code(
        "streamlit run streamlit_app.py",
        language="bash",
    )
