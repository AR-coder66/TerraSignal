"""
TerraSignal - AI-Powered Satellite Mission Decision System
Streamlit Operations Console
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

from data import (
    ObservationRequest,
    get_scenario_observations,
    SCENARIO_REGISTRY,
)
from decision_engine import (
    PlanningWeights,
    OptimizationResult,
    optimize_observation_plan,
)
from evaluation import run_greedy_baseline, compare_plans
from agent_readiness import SatelliteDecisionAgentInterface


# =========================================================================
# Page Setup & Styling
# =========================================================================
st.set_page_config(
    page_title="TerraSignal | Earth-Observation Mission Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Mission-Control CSS
AEROSPACE_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background & headers */
    .stApp {
        background-color: #070b14;
        color: #e2e8f0;
    }

    /* Simulation Mode Header Badge */
    .sim-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 171, 0, 0.12);
        border: 1px solid rgba(255, 171, 0, 0.4);
        color: #ffab00;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
        padding: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    /* Mission Metric Cards */
    .telemetry-card {
        background: #0d1527;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 14px 16px;
        transition: border-color 0.2s ease;
    }
    .telemetry-card:hover {
        border-color: #00d2ff;
    }
    .telemetry-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }
    .telemetry-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d2ff;
        font-family: 'JetBrains Mono', monospace;
    }
    .telemetry-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Target Status Tags */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.70rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .status-selected {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-deferred {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .status-infeasible {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Sidebar Aerospace Control Panel */
    [data-testid="stSidebar"] {
        background-color: #0a0f1d !important;
        border-right: 1px solid #1e293b;
    }
    
    .sidebar-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 18px;
        margin-bottom: 8px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
    }

    /* Streamlit button custom styling */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        border: 1px solid #38bdf8;
        border-radius: 4px;
        padding: 8px 16px;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
        border-color: #7dd3fc;
    }

    /* Callout & Notes */
    .callout-box {
        background: #0f172a;
        border-left: 3px solid #00d2ff;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
        font-size: 0.82rem;
        line-height: 1.45;
        color: #cbd5e1;
    }
</style>
"""
st.markdown(AEROSPACE_CSS, unsafe_allow_html=True)


# =========================================================================
# State Initialization (Preserve state across Streamlit reruns)
# =========================================================================
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "plan_result" not in st.session_state:
    st.session_state.plan_result = None
if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None
if "comparison_report" not in st.session_state:
    st.session_state.comparison_report = None


# =========================================================================
# Sidebar: Live Mission Controls
# =========================================================================
with st.sidebar:
    st.markdown("<div class='sim-badge'>● SIMULATION MODE ACTIVE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>1. Scenario Selection</div>", unsafe_allow_html=True)
    
    scenario_choice = st.selectbox(
        "Mission Scenario",
        options=["Wildfire Response", "Flood Mapping", "Agricultural Survey"],
        index=0,
        help="Select the tactical Earth-observation domain. Each provides distinct, reproducible target requests.",
    )

    st.markdown("<div class='sidebar-section-title'>2. Orbital Pass Budgets</div>", unsafe_allow_html=True)
    
    time_budget = st.slider(
        "Available Imaging Time (min)",
        min_value=4.0,
        max_value=30.0,
        value=9.0,
        step=0.5,
        help="Satellite dwell, slew, and sensor integration time available in this orbital opportunity window.",
    )

    storage_budget = st.slider(
        "Available Solid-State Storage (GB)",
        min_value=2.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Remaining onboard high-speed storage buffer before next ground station downlink pass.",
    )

    cloud_cover = st.slider(
        "Scenario Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Simulated regional cloud fraction. Optical sensors degrade sharply above their threshold; SAR/Radar remains robust.",
    )

    st.markdown("<div class='sidebar-section-title'>3. Planning Objective Weights</div>", unsafe_allow_html=True)
    
    w_urgency = st.slider(
        "Urgency Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Priority for immediate life safety, active hazards, and fast-moving phenomena.",
    )
    
    w_value = st.slider(
        "Scientific / Tactical Value",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Priority for high-resolution intelligence, model baseline generation, and structural integrity.",
    )

    w_time_sens = st.slider(
        "Time-Sensitivity Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Criticality of capturing this target during current orbital pass versus deferring to next orbit.",
    )

    run_analysis = st.button("RUN MISSION ANALYSIS", use_container_width=True)

    st.markdown(
        "<div class='callout-box' style='margin-top: 20px; font-size: 0.72rem;'>"
        "<strong>Architectural Note:</strong><br>"
        "Calculations run locally via multi-constrained 0-1 knapsack optimization. "
        "No live spacecraft telecommand or telemetry link is claimed."
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# Execution Trigger & Calculation
# =========================================================================
requests = get_scenario_observations(scenario_choice)
weights = PlanningWeights(
    urgency_weight=w_urgency,
    value_weight=w_value,
    time_sensitivity_weight=w_time_sens,
)

# Auto-run initial analysis on first load or when user clicks run
if run_analysis or (not st.session_state.has_run):
    opt_result = optimize_observation_plan(
        requests=requests,
        available_time_minutes=time_budget,
        available_storage_gb=storage_budget,
        scenario_cloud_cover=cloud_cover,
        scenario_name=scenario_choice,
        weights=weights,
    )
    base_result = run_greedy_baseline(
        requests=requests,
        available_time_minutes=time_budget,
        available_storage_gb=storage_budget,
        scenario_cloud_cover=cloud_cover,
        weights=weights,
    )
    comparison = compare_plans(opt_result, base_result)

    st.session_state.has_run = True
    st.session_state.plan_result = opt_result
    st.session_state.baseline_result = base_result
    st.session_state.comparison_report = comparison

plan: OptimizationResult = st.session_state.plan_result
baseline = st.session_state.baseline_result
report = st.session_state.comparison_report


# =========================================================================
# Header & Mission Overview
# =========================================================================
col_head_left, col_head_right = st.columns([3, 1])
with col_head_left:
    st.markdown(
        "<div class='brand-title'>TERRASIGNAL <span style='font-size: 0.95rem; color: #38bdf8; font-weight: 400;'>| SATELLITE MISSION DECISION SYSTEM</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='brand-subtitle'>Constrained Earth-Observation Scheduling & Resource Allocation &bull; Scenario: <strong>{scenario_choice}</strong></div>",
        unsafe_allow_html=True,
    )

with col_head_right:
    st.markdown(
        "<div style='text-align: right;'>"
        "<span class='sim-badge'>● DETERMINISTIC SIMULATOR</span><br>"
        "<span style='font-size: 0.70rem; color: #64748b; font-family: monospace;'>ENGINE: Knapsack Global Solver</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# Telemetry Overview Strip
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Total Targets</div>"
        f"<div class='telemetry-value'>{len(requests)}</div>"
        f"<div class='telemetry-sub'>Candidate Requests</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Selected</div>"
        f"<div class='telemetry-value' style='color:#10b981;'>{len(plan.selected_targets)}</div>"
        f"<div class='telemetry-sub'>Scheduled for pass</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Deferred</div>"
        f"<div class='telemetry-value' style='color:#eab308;'>{len(plan.deferred_targets)}</div>"
        f"<div class='telemetry-sub'>Budget constrained</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Infeasible</div>"
        f"<div class='telemetry-value' style='color:#ef4444;'>{len(plan.infeasible_targets)}</div>"
        f"<div class='telemetry-sub'>Cloud obstructed</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Pass Imaging Time</div>"
        f"<div class='telemetry-value'>{plan.time_used_minutes:.1f}<span style='font-size:0.8rem; color:#94a3b8;'>/{time_budget:.0f}m</span></div>"
        f"<div class='telemetry-sub'>{plan.time_utilization_pct:.1f}% Pass Allocated</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with m6:
    st.markdown(
        f"<div class='telemetry-card'>"
        f"<div class='telemetry-label'>Storage Buffer</div>"
        f"<div class='telemetry-value'>{plan.storage_used_gb:.1f}<span style='font-size:0.8rem; color:#94a3b8;'>/{storage_budget:.0f}G</span></div>"
        f"<div class='telemetry-sub'>{plan.storage_utilization_pct:.1f}% Buffer Allocated</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# =========================================================================
# Main Tabs: Operational View, Map & Trade-offs, Evaluation, AI Agent Architecture
# =========================================================================
tab_ops, tab_viz, tab_eval, tab_arch = st.tabs([
    "📋 MISSION SCHEDULE & EXPLANATIONS",
    "🛰️ RESOURCE ALLOCATION & TARGET MAP",
    "⚖️ OPTIMIZER VS GREEDY EVALUATION",
    "🤖 STRANDS DECIDER & BEDROCK READINESS",
])


# -------------------------------------------------------------------------
# Tab 1: Mission Schedule & Explanations
# -------------------------------------------------------------------------
with tab_ops:
    col_t1_l, col_t1_r = st.columns([2, 1])

    with col_t1_l:
        st.markdown("#### Primary Tactical Observation Schedule")
        
        # Build comprehensive table data
        rows = []
        for t in plan.selected_targets:
            rows.append({
                "Status": "SELECTED",
                "ID": t.request.target_id,
                "Target Name": t.request.name,
                "Sensor": t.request.sensor_type,
                "Duration (m)": f"{t.request.duration_minutes:.1f}",
                "Storage (GB)": f"{t.request.storage_gb:.1f}",
                "Score": f"{t.target_value:.2f}",
                "Urgency": f"{t.request.urgency:.1f}",
                "Cloud Feasibility": f"{t.feasibility_score * 100:.0f}%",
                "Explanation": plan.per_target_explanations.get(t.request.target_id, ""),
            })

        for t in plan.deferred_targets:
            rows.append({
                "Status": "DEFERRED",
                "ID": t.request.target_id,
                "Target Name": t.request.name,
                "Sensor": t.request.sensor_type,
                "Duration (m)": f"{t.request.duration_minutes:.1f}",
                "Storage (GB)": f"{t.request.storage_gb:.1f}",
                "Score": f"{t.target_value:.2f}",
                "Urgency": f"{t.request.urgency:.1f}",
                "Cloud Feasibility": f"{t.feasibility_score * 100:.0f}%",
                "Explanation": plan.per_target_explanations.get(t.request.target_id, ""),
            })

        for t in plan.infeasible_targets:
            rows.append({
                "Status": "INFEASIBLE",
                "ID": t.request.target_id,
                "Target Name": t.request.name,
                "Sensor": t.request.sensor_type,
                "Duration (m)": f"{t.request.duration_minutes:.1f}",
                "Storage (GB)": f"{t.request.storage_gb:.1f}",
                "Score": f"{t.target_value:.2f}",
                "Urgency": f"{t.request.urgency:.1f}",
                "Cloud Feasibility": "0% (Obstructed)",
                "Explanation": plan.per_target_explanations.get(t.request.target_id, ""),
            })

        df_display = pd.DataFrame(rows)
        st.dataframe(
            df_display,
            column_config={
                "Status": st.column_config.TextColumn(width="small"),
                "ID": st.column_config.TextColumn(width="small"),
                "Target Name": st.column_config.TextColumn(width="medium"),
                "Explanation": st.column_config.TextColumn(width="large"),
            },
            hide_index=True,
            use_container_width=True,
        )

    with col_t1_r:
        st.markdown("#### Mission Decision Summary")
        
        st.markdown(
            f"""
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 14px;'>
                <div style='font-family: monospace; font-size: 0.72rem; color: #38bdf8; margin-bottom: 8px;'>
                    PASS OPTIMIZATION OUTCOME
                </div>
                <div style='font-size: 0.95rem; margin-bottom: 12px;'>
                    Total Selected Value: <strong style='color:#00d2ff; font-family: monospace;'>{plan.total_value_selected:.2f} pts</strong>
                </div>
                <div style='font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;'>
                    &bull; <strong>Mathematical Status:</strong> Globally optimal subset verified via exhaustive enumeration.<br>
                    &bull; <strong>Feasible Candidates:</strong> {len(plan.selected_targets) + len(plan.deferred_targets)} of {len(requests)} targets satisfy cloud criteria.<br>
                    &bull; <strong>Resource Saturation:</strong> Time margin remaining: {plan.time_remaining_minutes:.1f}m | Storage buffer remaining: {plan.storage_remaining_gb:.1f}GB.<br>
                </div>
                <hr style='border: 0; border-top: 1px solid #1e293b; margin: 12px 0;'>
                <div style='font-size: 0.75rem; color: #94a3b8;'>
                    <strong>Normalized Weights:</strong><br>
                    Urgency: {plan.weights.urgency_weight:.2f} &bull; Operational Value: {plan.weights.value_weight:.2f} &bull; Time-Sensitivity: {plan.weights.time_sensitivity_weight:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Optimization Notes")
        for note in plan.summary_notes:
            st.markdown(f"<div class='callout-box'>{note}</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# Tab 2: Resource Allocation & Target Map
# -------------------------------------------------------------------------
with tab_viz:
    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.markdown("#### Resource Utilization Gauges")

        # Plotly bullet/gauge charts for time & storage
        fig_gauges = go.Figure()

        # Time Bar
        fig_gauges.add_trace(go.Bar(
            y=["Imaging Time (min)"],
            x=[plan.time_used_minutes],
            name="Used",
            orientation='h',
            marker=dict(color="#00d2ff"),
            text=f"{plan.time_used_minutes:.1f}m ({plan.time_utilization_pct:.1f}%)",
            textposition="inside",
        ))
        fig_gauges.add_trace(go.Bar(
            y=["Imaging Time (min)"],
            x=[plan.time_remaining_minutes],
            name="Remaining",
            orientation='h',
            marker=dict(color="#1e293b"),
            text=f"{plan.time_remaining_minutes:.1f}m",
            textposition="inside",
        ))

        # Storage Bar
        fig_gauges.add_trace(go.Bar(
            y=["Storage (GB)"],
            x=[plan.storage_used_gb],
            name="Used",
            orientation='h',
            marker=dict(color="#10b981"),
            text=f"{plan.storage_used_gb:.1f}GB ({plan.storage_utilization_pct:.1f}%)",
            textposition="inside",
            showlegend=False,
        ))
        fig_gauges.add_trace(go.Bar(
            y=["Storage (GB)"],
            x=[plan.storage_remaining_gb],
            name="Remaining",
            orientation='h',
            marker=dict(color="#1e293b"),
            text=f"{plan.storage_remaining_gb:.1f}GB",
            textposition="inside",
            showlegend=False,
        ))

        fig_gauges.update_layout(
            barmode='stack',
            height=200,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="#070b14",
            plot_bgcolor="#070b14",
            font=dict(color="#cbd5e1", family="JetBrains Mono"),
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_gauges, use_container_width=True)

        st.markdown("#### Cloud Feasibility vs Sensitivity")
        
        # Scatter plot of feasibility vs cloud threshold
        scatter_data = []
        for t in plan.selected_targets:
            scatter_data.append({
                "Target": t.request.target_id,
                "Name": t.request.name,
                "Cloud Threshold (%)": t.request.max_acceptable_cloud,
                "Feasibility Score": t.feasibility_score * 100,
                "Status": "Selected",
                "Size": t.target_value * 2,
            })
        for t in plan.deferred_targets:
            scatter_data.append({
                "Target": t.request.target_id,
                "Name": t.request.name,
                "Cloud Threshold (%)": t.request.max_acceptable_cloud,
                "Feasibility Score": t.feasibility_score * 100,
                "Status": "Deferred",
                "Size": t.target_value * 2,
            })
        for t in plan.infeasible_targets:
            scatter_data.append({
                "Target": t.request.target_id,
                "Name": t.request.name,
                "Cloud Threshold (%)": t.request.max_acceptable_cloud,
                "Feasibility Score": 0.0,
                "Status": "Infeasible",
                "Size": t.target_value * 2,
            })

        df_scatter = pd.DataFrame(scatter_data)
        fig_scatter = px.scatter(
            df_scatter,
            x="Cloud Threshold (%)",
            y="Feasibility Score",
            color="Status",
            size="Size",
            hover_name="Name",
            color_discrete_map={"Selected": "#10b981", "Deferred": "#eab308", "Infeasible": "#ef4444"},
            labels={"Cloud Threshold (%)": "Max Acceptable Cloud Cover (%)", "Feasibility Score": "Feasibility Metric (%)"},
        )
        fig_scatter.add_vline(x=cloud_cover, line_dash="dash", line_color="#ffab00", annotation_text=f"Current Cloud ({cloud_cover:.0f}%)")
        fig_scatter.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#070b14",
            plot_bgcolor="#0d1527",
            font=dict(color="#cbd5e1", family="Inter"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_v2:
        st.markdown("#### Tactical Geographic Observation Map")
        
        map_points = []
        for t in plan.selected_targets:
            map_points.append({
                "lat": t.request.target_lat,
                "lon": t.request.target_lon,
                "name": f"[{t.request.target_id}] {t.request.name}",
                "status": "Selected",
                "color": "#10b981",
                "size": 18,
            })
        for t in plan.deferred_targets:
            map_points.append({
                "lat": t.request.target_lat,
                "lon": t.request.target_lon,
                "name": f"[{t.request.target_id}] {t.request.name}",
                "status": "Deferred",
                "color": "#eab308",
                "size": 14,
            })
        for t in plan.infeasible_targets:
            map_points.append({
                "lat": t.request.target_lat,
                "lon": t.request.target_lon,
                "name": f"[{t.request.target_id}] {t.request.name}",
                "status": "Infeasible",
                "color": "#ef4444",
                "size": 12,
            })

        df_map = pd.DataFrame(map_points)
        fig_map = px.scatter_geo(
            df_map,
            lat="lat",
            lon="lon",
            color="status",
            hover_name="name",
            color_discrete_map={"Selected": "#10b981", "Deferred": "#eab308", "Infeasible": "#ef4444"},
            size="size",
            projection="natural earth",
        )
        fig_map.update_geos(
            bgcolor="#070b14",
            lakecolor="#0d1527",
            landcolor="#141e33",
            showocean=True,
            oceancolor="#070b14",
            showcountries=True,
            countrycolor="#334155",
            coastlinecolor="#475569",
            fitbounds="locations",
        )
        fig_map.update_layout(
            height=490,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#070b14",
            font=dict(color="#cbd5e1", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)


# -------------------------------------------------------------------------
# Tab 3: Evaluation & Baseline Comparison
# -------------------------------------------------------------------------
with tab_eval:
    st.markdown("#### Algorithmic Comparison: Global Knapsack Optimizer vs. Greedy Heuristic")
    st.markdown(
        "<div class='callout-box'>"
        "<strong>Benchmarking Methodology:</strong> Both methods evaluate the identical candidate targets, "
        "normalized valuation weights, and hard dual-resource budgets (imaging time & storage). "
        "The <em>Greedy Baseline</em> sequentially schedules targets ranked by value-per-resource density, "
        "while the <em>TerraSignal Engine</em> computes the mathematically global optimum via exhaustive subset search."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='telemetry-card'>"
            f"<div class='telemetry-label'>Optimizer Total Value</div>"
            f"<div class='telemetry-value' style='color:#00d2ff;'>{plan.total_value_selected:.2f}</div>"
            f"<div class='telemetry-sub'>{len(plan.selected_targets)} targets scheduled</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='telemetry-card'>"
            f"<div class='telemetry-label'>Greedy Baseline Value</div>"
            f"<div class='telemetry-value' style='color:#94a3b8;'>{baseline.total_value_selected:.2f}</div>"
            f"<div class='telemetry-sub'>{len(baseline.selected_targets)} targets scheduled</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        color_lift = "#10b981" if report.value_difference > 0 else "#64748b"
        st.markdown(
            f"<div class='telemetry-card'>"
            f"<div class='telemetry-label'>Objective Value Lift</div>"
            f"<div class='telemetry-value' style='color:{color_lift};'>+{report.value_difference:.2f}</div>"
            f"<div class='telemetry-sub'>+{report.value_lift_pct:.1f}% relative improvement</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='telemetry-card'>"
            f"<div class='telemetry-label'>Portfolio Convergence</div>"
            f"<div class='telemetry-value' style='font-size:1.15rem; color:{'#10b981' if report.is_identical else '#ffab00'};'>"
            f"{'CONVERGED' if report.is_identical else 'DIVERGED'}</div>"
            f"<div class='telemetry-sub'>{'Identical target selection' if report.is_identical else 'Optimizer found better combo'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    col_bench_l, col_bench_r = st.columns([1, 1])

    with col_bench_l:
        st.markdown("##### Detailed Metric Comparison")
        comp_df = pd.DataFrame([
            {
                "Metric": "Total Target Value",
                "TerraSignal Optimizer": f"{plan.total_value_selected:.2f}",
                "Greedy Baseline": f"{baseline.total_value_selected:.2f}",
                "Delta": f"{'+' if report.value_difference >= 0 else ''}{report.value_difference:.2f}",
            },
            {
                "Metric": "Selected Targets Count",
                "TerraSignal Optimizer": f"{len(plan.selected_targets)}",
                "Greedy Baseline": f"{len(baseline.selected_targets)}",
                "Delta": f"{'+' if report.targets_count_diff >= 0 else ''}{report.targets_count_diff}",
            },
            {
                "Metric": "Pass Time Used",
                "TerraSignal Optimizer": f"{plan.time_used_minutes:.1f} min ({plan.time_utilization_pct:.1f}%)",
                "Greedy Baseline": f"{baseline.time_used_minutes:.1f} min ({baseline.time_utilization_pct:.1f}%)",
                "Delta": f"{report.time_util_diff_pct:+.1f}%",
            },
            {
                "Metric": "Storage Consumed",
                "TerraSignal Optimizer": f"{plan.storage_used_gb:.1f} GB ({plan.storage_utilization_pct:.1f}%)",
                "Greedy Baseline": f"{baseline.storage_used_gb:.1f} GB ({baseline.storage_utilization_pct:.1f}%)",
                "Delta": f"{report.storage_util_diff_pct:+.1f}%",
            },
            {
                "Metric": "Mathematical Optimality",
                "TerraSignal Optimizer": "Guaranteed Global Optimum",
                "Greedy Baseline": "Heuristic (No Guarantee)",
                "Delta": "Proven Advantage",
            },
        ])
        st.table(comp_df)

    with col_bench_r:
        st.markdown("##### Algorithmic Takeaway")
        st.markdown(
            f"""
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 16px;'>
                <div style='font-size: 0.88rem; color: #f1f5f9; line-height: 1.6;'>
                    {report.summary}
                </div>
                <hr style='border: 0; border-top: 1px solid #1e293b; margin: 12px 0;'>
                <div style='font-size: 0.78rem; color: #94a3b8;'>
                    <strong>Why Greedy Baselines Fail in Aerospace Scheduling:</strong><br>
                    Greedy heuristics commit irrevocably to early high-density requests. When payloads have 
                    differing aspect ratios between imaging duration and storage consumption, greedy selection 
                    frequently leaves stranded resource fragments that cannot fit remaining high-priority targets. 
                    Exhaustive search explores the full combinatorial envelope to discover optimal packing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------------
# Tab 4: Strands Decider 2B & Amazon Bedrock Readiness
# -------------------------------------------------------------------------
with tab_arch:
    st.markdown("#### Genuine AI Agent Architecture & Future Cloud Integration")
    
    st.markdown(
        """
        TerraSignal separates **agentic strategy formulation** from **deterministic optimization** 
        and **natural language flight briefing**. This guarantees that AI reasoning cannot violate physical orbital 
        or payload constraints.
        """
    )

    st.markdown(
        """
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0;'>
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;'>
                <div style='color: #38bdf8; font-family: monospace; font-size: 0.72rem; font-weight: bold;'>STAGE 1: INPUTS</div>
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;'>User orbital budgets, environmental cloud metrics, and tactical scenario targets.</div>
            </div>
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;'>
                <div style='color: #00d2ff; font-family: monospace; font-size: 0.72rem; font-weight: bold;'>STAGE 2: STRANDS DECIDER 2B</div>
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;'>Small-parameter decision model evaluates tactical tension and policy weights.</div>
            </div>
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;'>
                <div style='color: #10b981; font-family: monospace; font-size: 0.72rem; font-weight: bold;'>STAGE 3: DECISION ENGINE</div>
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;'>Mathematical knapsack solver guarantees budget adherence and global optimum.</div>
            </div>
            <div style='background: #0d1527; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;'>
                <div style='color: #eab308; font-family: monospace; font-size: 0.72rem; font-weight: bold;'>STAGE 4: AMAZON BEDROCK</div>
                <div style='font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;'>Grounded LLM synthesizes executive flight briefing from verified mathematical facts.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("##### 1. Optimizer Callable Tool Specification (Strands Decider 2B / Agent SDK)")
        st.caption("Schema exposed to the AI agent runtime as a deterministic callable tool:")
        tool_spec = SatelliteDecisionAgentInterface.format_optimizer_tool_spec()
        st.json(tool_spec)

    with col_a2:
        st.markdown("##### 2. Grounded Mission Briefing Payload (Amazon Bedrock Readiness)")
        st.caption("Verified mathematical payload generated from current schedule (anti-hallucination contract):")
        bedrock_payload = SatelliteDecisionAgentInterface.prepare_bedrock_grounded_payload(plan)
        st.json(bedrock_payload.grounded_plan_summary)

    st.markdown("---")
    st.markdown("##### Remaining Work Required for Full AWS & Decider Deployment")
    st.markdown(
        """
        - **Amazon Bedrock IAM & Boto3 Integration:** Provision IAM execution role with `bedrock:InvokeModel` permissions and configure Boto3 client targeting `anthropic.claude-3-sonnet` or `amazon.titan-text-express`.
        - **Strands Decider 2B Host Deployment:** Deploy Decider 2B on an Amazon SageMaker real-time endpoint or local ONNX runtime with structured JSON decoding.
        - **Ground Station Pass Event Bridge:** Connect plan output to automated AWS IoT Ground Station pass scheduling triggers.
        """
    )
