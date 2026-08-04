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

DEFAULTS = {
    "n_layers": 2,
    "f_mm": 50.0,
    "ray_radius_mm": 1.0,
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


def _wavelength_color(lam):
    r, g, b = optics._wavelength_to_rgb(lam)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _axis_id(idx, letter):
    n = idx + 1
    return letter if n == 1 else f"{letter}{n}"


@st.cache_resource(show_spinner=False)
def _build_system(radii_mm, thicknesses_mm, material_names, f_mm):
    materials = [optics.CANDIDATE_MATERIALS[name] for name in material_names]
    radii = np.array(radii_mm, dtype=float) * 1e-3
    thicknesses = np.array(thicknesses_mm, dtype=float) * 1e-3
    return optics.build_layered_system(radii, thicknesses, materials, f_mm * 1e-3)


def make_system(values):
    # Cached on geometry + materials only, so tweaking ray/spectrum controls
    # (which don't affect the system itself) never forces a rebuild.
    return _build_system(
        tuple(values["radii_mm"]),
        tuple(values["thicknesses_mm"]),
        tuple(values["materials"]),
        values["f_mm"],
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

    rms_um = np.full(len(lam_grid), np.nan)
    n_rays_out = np.zeros(len(lam_grid), dtype=int)
    spots_um = []

    for i, lam in enumerate(lam_grid):
        rays_out = system.trace_bundle(rays_in, lam)
        n_rays_out[i] = len(rays_out)
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
        "spots_um": spots_um,
    }


def per_wavelength_metrics(values):
    trace = trace_all_wavelengths(values)
    lam_grid = trace["lam_m"]
    rms_um = trace["rms_um"]
    n_rays_out = trace["n_rays_out"]

    rows = []
    weighted_cost = 0.0

    for lam, rms, n_out in zip(lam_grid, rms_um, n_rays_out):
        weight = optics.pv_weight(lam)
        if np.isfinite(rms):
            weighted_cost += weight * (rms * 1e-6) ** 2
        rows.append(
            {
                "Wavelength (nm)": lam * 1e9,
                "PV weight": weight,
                "RMS spot radius (µm)": rms,
                "Rays reaching PV": int(n_out),
            }
        )

    return pd.DataFrame(rows), weighted_cost


@st.cache_data(show_spinner=False)
def compute_optical_layout(values, n_fan_rays=7, lam_nm=550.0):
    """
    Geometry (surface arc points) + a small ray fan traced through the full
    path (not just the final PV-plane point), for the optical-layout diagram.
    """
    system = make_system(values)
    aperture = values["ray_radius_mm"] * 1e-3

    fan = optics.make_ray_fan(radius=aperture, n_rays=n_fan_rays)
    paths = []
    for ray in fan:
        pts, alive = system.trace_ray_path(ray, lam_nm * 1e-9)
        paths.append((np.array(pts), alive))

    surfaces = []
    for surf in system.surfaces:
        R = surf.R
        cx, cz = surf.center[0], surf.center[2]
        half_ap = min(aperture * 1.3, abs(R) * 0.97) if R != 0 else aperture * 1.3
        xs = np.linspace(-half_ap, half_ap, 60) + cx
        sign = 1.0 if R >= 0 else -1.0
        zs = cz - sign * np.sqrt(np.maximum(R ** 2 - (xs - cx) ** 2, 0.0))
        surfaces.append((xs, zs))

    return {"paths": paths, "surfaces": surfaces, "z_target": system.z_target}


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_rms_plotly(series):
    """series: list of (values_dict, label, color) tuples."""
    fig = go.Figure()
    for values, label, color in series:
        trace = trace_all_wavelengths(values)
        fig.add_trace(go.Scatter(
            x=trace["lam_m"] * 1e9,
            y=trace["rms_um"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
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
        fig.update_xaxes(title_text="x (µm)", zeroline=True, row=row, col=col)
        fig.update_yaxes(
            title_text="y (µm)", zeroline=True,
            scaleanchor=_axis_id(idx, "x"), scaleratio=1,
            row=row, col=col,
        )

    fig.update_layout(
        title_text="Spot Diagrams at PV Plane",
        showlegend=False,
        height=280 * nrows,
        margin=dict(t=80, b=40),
    )
    return fig


def plot_optical_layout_plotly(values, n_fan_rays=7, lam_nm=550.0):
    data = compute_optical_layout(values, n_fan_rays=n_fan_rays, lam_nm=lam_nm)
    fig = go.Figure()

    for xs, zs in data["surfaces"]:
        fig.add_trace(go.Scatter(
            x=zs * 1e3, y=xs * 1e3, mode="lines",
            line=dict(color="#4C78A8", width=2),
            hoverinfo="skip", showlegend=False,
        ))

    n_paths = len(data["paths"])
    for i, (pts, alive) in enumerate(data["paths"]):
        frac = i / max(n_paths - 1, 1)
        color = f"rgb({int(255 * frac)},80,{int(255 * (1 - frac))})"
        fig.add_trace(go.Scatter(
            x=pts[:, 2] * 1e3, y=pts[:, 0] * 1e3, mode="lines",
            line=dict(color=color, width=1.3, dash="solid" if alive else "dot"),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_vline(
        x=data["z_target"] * 1e3, line=dict(color="red", dash="dot", width=1.5),
        annotation_text="PV plane", annotation_position="top",
    )
    fig.update_layout(
        title=f"Optical Layout — ray fan @ {lam_nm:.0f} nm (dotted = lost ray)",
        xaxis_title="z (mm)",
        yaxis_title="x (mm)",
        margin=dict(t=60, b=40),
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


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def flatten_values_for_export(values):
    rows = [{"parameter": "n_layers", "value": values["n_layers"]}]
    for i, r in enumerate(values["radii_mm"]):
        rows.append({"parameter": f"R{i}_mm", "value": r})
    for i, t in enumerate(values["thicknesses_mm"]):
        rows.append({"parameter": f"t{i + 1}_mm", "value": t})
    for i, m in enumerate(values["materials"]):
        rows.append({"parameter": f"mat{i + 1}", "value": m})
    for key in ("f_mm", "ray_radius_mm", "n_rays", "lam_start_nm",
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

        for i in range(n_layers):
            st.markdown(f"**Layer {i + 1}**")
            r_label = "R0 — entry curvature (mm)" if i == 0 else f"R{i} — internal curvature (mm)"
            r_default = DEFAULT_RADII_MM[i] if i < len(DEFAULT_RADII_MM) else DEFAULT_RADII_MM[-1]
            r = st.number_input(r_label, value=r_default, step=1.0, format="%.3f", key=f"R{i}_mm")
            radii_mm.append(r)

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
    "thicknesses_mm": thicknesses_mm,
    "materials": materials_sel,
    "f_mm": f_mm,
    "ray_radius_mm": ray_radius_mm,
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

tab_analysis, tab_survey, tab_optimize, tab_auto, tab_export = st.tabs(
    ["Analysis", "Material survey", "Optimize", "Auto-design", "Export"]
)

with tab_analysis:
    cols = st.columns(len(materials) + 2)
    for i, mat in enumerate(materials):
        cols[i].metric(f"Layer {i + 1}", mat.name)
    cols[-2].metric("Weighted cost", f"{current_cost:.3e}")
    cols[-1].metric(
        "Mean RMS spot radius",
        f"{metrics_df['RMS spot radius (µm)'].mean():.2f} µm",
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
                    "Rays reaching PV": "{:.0f}",
                }
            ),
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
            "cost": result["cost"],
            "base_values": dict(values),
        }

    opt_state = st.session_state.get("opt_result")
    if opt_state is not None:
        n_layers_r = opt_state["n_layers"]
        radii_mm_r = opt_state["radii_mm"]
        thicknesses_mm_r = opt_state["thicknesses_mm"]
        f_mm_r = opt_state["f_mm"]

        cols = st.columns(len(opt_state["materials"]) + 1)
        for i, mname in enumerate(opt_state["materials"]):
            cols[i].metric(f"Layer {i + 1}", mname)
        cols[-1].metric("Final cost", f"{opt_state['cost']:.4e}")

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
            st.session_state["_apply_params"] = pending
            st.rerun()

        base_values = opt_state["base_values"]
        opt_values = dict(base_values)
        opt_values["n_layers"] = n_layers_r
        opt_values["radii_mm"] = radii_mm_r
        opt_values["thicknesses_mm"] = thicknesses_mm_r
        opt_values["f_mm"] = f_mm_r
        opt_values["materials"] = opt_state["materials"]

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

    all_combos = optics.generate_material_combinations(values["n_layers"])
    n_combos = len(all_combos)

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

    will_halve = n_combos > full_optimize_threshold
    st.caption(
        f"{n_combos} candidate material combination(s) for {values['n_layers']} "
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
                full_optimize_threshold=int(full_optimize_threshold),
                screen_maxiter=int(screen_maxiter),
                halving_rounds=int(halving_rounds),
                top_k_final=int(top_k_final),
                final_maxiter=int(values["opt_maxiter"]),
                progress_callback=_report_auto_progress,
            )

        progress_bar.progress(1.0, text="Auto-design complete.")

        n_layers_r = values["n_layers"]
        best_params = ad_result["best"]["params"]
        # Persisted in session_state so the results section below survives the
        # rerun triggered by the "apply to sidebar" button (see the Optimize
        # tab above for the same pattern and why it's needed).
        st.session_state["auto_result"] = {
            "n_layers": n_layers_r,
            "radii_mm": (best_params[:n_layers_r + 1] * 1e3).tolist(),
            "thicknesses_mm": (best_params[n_layers_r + 1:2 * n_layers_r + 1] * 1e3).tolist(),
            "f_mm": float(best_params[2 * n_layers_r + 1] * 1e3),
            "materials": ad_result["best_materials"],
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

        st.success(
            f"Best design: {' / '.join(auto_state['materials'])} — searched "
            f"{auto_state['n_combinations_evaluated']} of "
            f"{auto_state['n_combinations_total']} combination(s)"
            + (" via successive halving." if auto_state["used_halving"] else " (fully optimized).")
        )

        cols = st.columns(len(auto_state["materials"]) + 1)
        for i, mname in enumerate(auto_state["materials"]):
            cols[i].metric(f"Layer {i + 1}", mname)
        cols[-1].metric("Best cost", f"{auto_state['cost']:.4e}")

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

        if st.button("Apply best design to sidebar"):
            pending = {"n_layers": n_layers_r, "f_mm": f_mm_r}
            for i, r in enumerate(radii_mm_r):
                pending[f"R{i}_mm"] = float(r)
            for i, t in enumerate(thicknesses_mm_r):
                pending[f"t{i + 1}_mm"] = float(t)
            for i, mname in enumerate(auto_state["materials"]):
                pending[f"mat{i + 1}"] = mname
            st.session_state["_apply_params"] = pending
            st.rerun()

        st.subheader("Leaderboard")
        leaderboard_df = pd.DataFrame([
            {"Materials": " / ".join(entry["materials"]), "Cost": entry["cost"]}
            for entry in auto_state["leaderboard"]
        ])
        leaderboard_df["Cost"] = leaderboard_df["Cost"].map(lambda x: f"{x:.4e}")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        base_values = auto_state["base_values"]
        best_values = dict(base_values)
        best_values["n_layers"] = n_layers_r
        best_values["radii_mm"] = radii_mm_r
        best_values["thicknesses_mm"] = thicknesses_mm_r
        best_values["f_mm"] = f_mm_r
        best_values["materials"] = auto_state["materials"]

        st.subheader("Best design — optical layout")
        st.plotly_chart(plot_optical_layout_plotly(best_values), use_container_width=True, key="layout_auto")

        st.subheader("Best design — RMS vs wavelength")
        st.plotly_chart(
            plot_rms_plotly([(best_values, "Best design", OPTIMIZED_COLOR)]),
            use_container_width=True, key="rms_auto",
        )

        st.subheader("Best design — spot diagrams")
        st.plotly_chart(plot_spots_plotly(best_values), use_container_width=True, key="spots_auto")

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
