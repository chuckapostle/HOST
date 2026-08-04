import io
import contextlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import pv_ray_optics as optics


st.set_page_config(
    page_title="PV Ray-Optics Engine",
    page_icon="🔬",
    layout="wide",
)

DEFAULTS = {
    "R0_mm": 50.0,
    "R1_mm": -30.0,
    "R2_mm": -50.0,
    "t1_mm": 5.0,
    "t2_mm": 5.0,
    "f_mm": 50.0,
    "mat1": "N-BK7",
    "mat2": "F2",
    "ray_radius_mm": 1.0,
    "n_rays": 25,
    "lam_start_nm": 400.0,
    "lam_end_nm": 1100.0,
    "lam_steps": 15,
    "opt_method": "Nelder-Mead",
    "opt_maxiter": 500,
}


def make_params(values):
    return np.array(
        [
            values["R0_mm"],
            values["R1_mm"],
            values["R2_mm"],
            values["t1_mm"],
            values["t2_mm"],
            values["f_mm"],
        ],
        dtype=float,
    ) * 1e-3


def get_materials(values):
    return (
        optics.CANDIDATE_MATERIALS[values["mat1"]],
        optics.CANDIDATE_MATERIALS[values["mat2"]],
    )


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


@st.cache_resource(show_spinner=False)
def _build_system(R0_mm, R1_mm, R2_mm, t1_mm, t2_mm, f_mm, mat1_name, mat2_name):
    mat1 = optics.CANDIDATE_MATERIALS[mat1_name]
    mat2 = optics.CANDIDATE_MATERIALS[mat2_name]
    params = np.array(
        [R0_mm, R1_mm, R2_mm, t1_mm, t2_mm, f_mm], dtype=float
    ) * 1e-3
    return optics.build_2layer_system(params, mat1=mat1, mat2=mat2)


def make_system(values):
    # Cached on geometry + materials only, so tweaking ray/spectrum controls
    # (which don't affect the system itself) never forces a rebuild.
    return _build_system(
        values["R0_mm"], values["R1_mm"], values["R2_mm"],
        values["t1_mm"], values["t2_mm"], values["f_mm"],
        values["mat1"], values["mat2"],
    )


@st.cache_data(show_spinner=False)
def trace_all_wavelengths(values):
    """
    Trace the full ray bundle across the full wavelength grid once and cache
    the numeric results. per_wavelength_metrics / plot_rms_custom /
    plot_spots_custom all read from this instead of re-tracing separately.
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


def plot_rms_custom(values, label="Current design", color="tab:blue"):
    trace = trace_all_wavelengths(values)
    lam_nm = trace["lam_m"] * 1e9
    rms_um = trace["rms_um"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        lam_nm,
        rms_um,
        marker="o",
        linewidth=1.8,
        label=label,
        color=color,
    )
    ax.axvspan(500, 900, alpha=0.10, color="green", label="Peak PV band")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("RMS spot radius (µm)")
    ax.set_title("RMS Spot Radius at PV Plane")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_spots_custom(values):
    trace = trace_all_wavelengths(values)
    lam_grid = trace["lam_m"]
    rms_um = trace["rms_um"]
    spots_um = trace["spots_um"]

    n_lam = len(lam_grid)
    ncols = min(5, n_lam)
    nrows = int(np.ceil(n_lam / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False,
    )
    fig.suptitle("Spot Diagrams at PV Plane", fontsize=14)

    for idx, lam in enumerate(lam_grid):
        ax = axes[idx // ncols][idx % ncols]
        pts = spots_um[idx]
        color = optics._wavelength_to_rgb(lam)

        if pts.shape[0] > 0:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=20,
                color=color,
                alpha=0.85,
            )
            ax.set_title(f"{lam * 1e9:.0f} nm\nRMS={rms_um[idx]:.2f} µm")
        else:
            ax.set_title(f"{lam * 1e9:.0f} nm\nNo rays")
            ax.text(
                0.5,
                0.5,
                "×",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=24,
            )

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    for idx in range(n_lam, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def params_csv(values):
    rows = [{"parameter": key, "value": value} for key, value in values.items()]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


st.title("PV Concentrator Ray-Optics Engine")
st.caption(
    "Interactive analysis, material screening, and optimization of a "
    "two-layer refractive concentrator lens."
)

# Apply any pending "copy optimized params to sidebar" request *before* the
# sidebar widgets below are instantiated — Streamlit forbids writing to
# st.session_state[key] for a widget's key after that widget has already
# rendered in the current run, so the request is staged here instead.
if "_apply_params" in st.session_state:
    for _key, _val in st.session_state.pop("_apply_params").items():
        st.session_state[_key] = _val

with st.sidebar:
    with st.form("controls_form"):
        st.header("Lens geometry")

        values = {}
        values["R0_mm"] = st.number_input(
            "R0 — entry curvature (mm)",
            value=DEFAULTS["R0_mm"],
            step=1.0,
            format="%.3f",
            key="R0_mm",
        )
        values["R1_mm"] = st.number_input(
            "R1 — internal curvature (mm)",
            value=DEFAULTS["R1_mm"],
            step=1.0,
            format="%.3f",
            key="R1_mm",
        )
        values["R2_mm"] = st.number_input(
            "R2 — exit curvature (mm)",
            value=DEFAULTS["R2_mm"],
            step=1.0,
            format="%.3f",
            key="R2_mm",
        )
        values["t1_mm"] = st.number_input(
            "Layer 1 thickness (mm)",
            min_value=0.5,
            value=DEFAULTS["t1_mm"],
            step=0.25,
            format="%.3f",
            key="t1_mm",
        )
        values["t2_mm"] = st.number_input(
            "Layer 2 thickness (mm)",
            min_value=0.5,
            value=DEFAULTS["t2_mm"],
            step=0.25,
            format="%.3f",
            key="t2_mm",
        )
        values["f_mm"] = st.number_input(
            "PV plane / target focal length (mm)",
            min_value=1.0,
            value=DEFAULTS["f_mm"],
            step=1.0,
            format="%.3f",
            key="f_mm",
        )

        st.header("Materials")
        material_names = list(optics.CANDIDATE_MATERIALS.keys())
        values["mat1"] = st.selectbox(
            "Layer 1 / entry material",
            material_names,
            index=material_names.index(DEFAULTS["mat1"]),
            key="mat1",
        )
        values["mat2"] = st.selectbox(
            "Layer 2 / exit material",
            material_names,
            index=material_names.index(DEFAULTS["mat2"]),
            key="mat2",
        )

        st.header("Ray bundle")
        values["ray_radius_mm"] = st.number_input(
            "Aperture half-width (mm)",
            min_value=0.01,
            value=DEFAULTS["ray_radius_mm"],
            step=0.05,
            format="%.3f",
            key="ray_radius_mm",
        )
        values["n_rays"] = st.select_slider(
            "Approximate number of rays",
            options=[9, 16, 25, 36, 49, 64, 81, 100],
            value=DEFAULTS["n_rays"],
            key="n_rays",
        )

        st.header("Spectrum")
        values["lam_start_nm"] = st.number_input(
            "Start wavelength (nm)",
            min_value=300.0,
            max_value=1500.0,
            value=DEFAULTS["lam_start_nm"],
            step=10.0,
            key="lam_start_nm",
        )
        values["lam_end_nm"] = st.number_input(
            "End wavelength (nm)",
            min_value=300.0,
            max_value=1500.0,
            value=DEFAULTS["lam_end_nm"],
            step=10.0,
            key="lam_end_nm",
        )
        values["lam_steps"] = st.slider(
            "Wavelength samples",
            min_value=3,
            max_value=31,
            value=DEFAULTS["lam_steps"],
            step=2,
            key="lam_steps",
        )

        st.header("Optimization")
        values["opt_method"] = st.selectbox(
            "Optimizer",
            ["Nelder-Mead", "Powell"],
            index=0,
            key="opt_method",
        )
        values["opt_maxiter"] = st.slider(
            "Maximum iterations",
            min_value=50,
            max_value=2000,
            value=DEFAULTS["opt_maxiter"],
            step=50,
            key="opt_maxiter",
        )

        st.form_submit_button("Apply settings", type="primary", use_container_width=True)

    st.caption(
        "Changes take effect after clicking **Apply settings** — dragging "
        "sliders no longer re-traces the whole system on every step."
    )

if values["lam_end_nm"] <= values["lam_start_nm"]:
    st.error("The end wavelength must be greater than the start wavelength.")
    st.stop()

if values["mat1"] == values["mat2"]:
    st.warning(
        "The material survey excludes pairs with identical materials. "
        "Direct analysis remains valid."
    )

system = make_system(values)
metrics_df, current_cost = per_wavelength_metrics(values)
mat1, mat2 = get_materials(values)

tab_analysis, tab_survey, tab_optimize, tab_export = st.tabs(
    ["Analysis", "Material survey", "Optimize", "Export"]
)

with tab_analysis:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Layer 1", mat1.name)
    col2.metric("Layer 2", mat2.name)
    col3.metric("Weighted cost", f"{current_cost:.3e}")
    col4.metric(
        "Mean RMS spot radius",
        f"{metrics_df['RMS spot radius (µm)'].mean():.2f} µm",
    )

    left, right = st.columns([1.1, 1])

    with left:
        rms_fig = plot_rms_custom(values)
        st.pyplot(rms_fig)
        plt.close(rms_fig)

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
    spots_fig = plot_spots_custom(values)
    st.pyplot(spots_fig)
    plt.close(spots_fig)

    with st.expander("Chromatic focus shift"):
        if st.button("Calculate focus shift", type="secondary"):
            with st.spinner("Scanning best-focus plane across wavelengths..."):
                fig_focus = optics.plot_focus_shift(
                    system,
                    lam_list=get_lambda_grid(values),
                    title=f"Chromatic Focus Shift — {mat1.name}/{mat2.name}",
                )
            st.pyplot(fig_focus)
            plt.close(fig_focus)

with tab_survey:
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
        "iterations, then increase only after confirming the design behavior."
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
                mat1=mat1,
                mat2=mat2,
                x0=make_params(values),
                method=values["opt_method"],
                maxiter=int(values["opt_maxiter"]),
                lam_list=get_lambda_grid(values),
                rays_in=get_rays_in(values),
                progress_callback=_report_progress,
            )

        progress_bar.progress(1.0, text="Optimization complete.")
        console_box.code(capture.getvalue(), language="text")

        # Persisted in session_state (rather than kept as plain locals) so the
        # results section below survives the rerun triggered by the "apply to
        # sidebar" button, instead of vanishing because `st.button` is only
        # True on the run where it was actually clicked.
        st.session_state["opt_result"] = {
            "params_mm": (result["params"] * 1e3).tolist(),
            "mat1_name": result["mat1"].name,
            "mat2_name": result["mat2"].name,
            "cost": result["cost"],
            "base_values": dict(values),
        }

    opt_state = st.session_state.get("opt_result")
    if opt_state is not None:
        opt_params_mm = opt_state["params_mm"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimized material 1", opt_state["mat1_name"])
        c2.metric("Optimized material 2", opt_state["mat2_name"])
        c3.metric("Final cost", f"{opt_state['cost']:.4e}")

        parameter_df = pd.DataFrame(
            {
                "Parameter": ["R0", "R1", "R2", "t1", "t2", "f"],
                "Value (mm)": opt_params_mm,
            }
        )
        st.dataframe(
            parameter_df.style.format({"Value (mm)": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Apply optimized parameters to sidebar"):
            st.session_state["_apply_params"] = dict(zip(
                ["R0_mm", "R1_mm", "R2_mm", "t1_mm", "t2_mm", "f_mm"],
                [float(v) for v in opt_params_mm],
            ))
            st.rerun()

        base_values = opt_state["base_values"]
        opt_values = dict(base_values)
        (
            opt_values["R0_mm"],
            opt_values["R1_mm"],
            opt_values["R2_mm"],
            opt_values["t1_mm"],
            opt_values["t2_mm"],
            opt_values["f_mm"],
        ) = opt_params_mm

        initial_fig = plot_rms_custom(
            base_values,
            label="Initial",
            color="tab:red",
        )
        ax = initial_fig.axes[0]
        opt_fig = plot_rms_custom(
            opt_values,
            label="Optimized",
            color="tab:blue",
        )
        opt_ax = opt_fig.axes[0]

        for line in opt_ax.lines:
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color="tab:blue",
                marker="o",
                label="Optimized",
            )
        plt.close(opt_fig)

        ax.set_title("Initial vs Optimized RMS Spot Radius")
        ax.legend()
        st.pyplot(initial_fig)
        plt.close(initial_fig)

        st.subheader("Optimized spot diagrams")
        opt_spots_fig = plot_spots_custom(opt_values)
        st.pyplot(opt_spots_fig)
        plt.close(opt_spots_fig)

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