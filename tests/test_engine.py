"""
TerraSignal - Automated Test Suite

Verifies:
- Imaging-time and storage limits are never exceeded.
- Cloud-infeasible targets are never selected.
- Optimizer value is >= greedy baseline value for identical inputs.
- Changing mission constraints changes the resulting plan appropriately.
- Different mission types use their own appropriate synthetic datasets.
- Zero feasible targets and oversized targets are handled gracefully.
- Invalid inputs raise expected errors.
- Explanations match actual selection and feasibility results.
- Tie-breaking is deterministic.
- Known dataset tests confirm the mathematical global optimum.
"""

import pytest
from data import (
    ObservationRequest,
    get_scenario_observations,
    WILDFIRE_SCENARIO,
    FLOOD_SCENARIO,
    AGRICULTURAL_SCENARIO,
)
from decision_engine import (
    PlanningWeights,
    calculate_target_value,
    evaluate_feasibility,
    optimize_observation_plan,
)
from evaluation import run_greedy_baseline, compare_plans
from agent_readiness import SatelliteDecisionAgentInterface


# =========================================================================
# 1. Dataset Verification
# =========================================================================

def test_scenario_datasets_distinct_and_sized():
    """Verify each mission scenario has at least 8 reproducible, distinct targets."""
    for name, dataset in [
        ("Wildfire Response", WILDFIRE_SCENARIO),
        ("Flood Mapping", FLOOD_SCENARIO),
        ("Agricultural Survey", AGRICULTURAL_SCENARIO),
    ]:
        assert len(dataset) >= 8, f"{name} must have at least 8 targets"
        ids = [req.target_id for req in dataset]
        assert len(ids) == len(set(ids)), f"Duplicate target IDs found in {name}"
        
        for req in dataset:
            assert 1.0 <= req.urgency <= 10.0
            assert 1.0 <= req.operational_value <= 10.0
            assert 1.0 <= req.time_sensitivity <= 10.0
            assert req.duration_minutes > 0.0
            assert req.storage_gb > 0.0
            assert 0.0 <= req.cloud_sensitivity <= 1.0
            assert 0.0 <= req.max_acceptable_cloud <= 100.0


def test_scenario_retrieval_and_error():
    """Verify proper retrieval and unknown scenario rejection."""
    wf = get_scenario_observations("Wildfire Response")
    assert len(wf) == 8
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario_observations("Martian Reconnaissance")


# =========================================================================
# 2. Feasibility Evaluation
# =========================================================================

def test_cloud_feasibility_logic():
    """Verify cloud cover threshold and sensitivity score."""
    req = ObservationRequest(
        target_id="TEST-01",
        name="Test Sensor",
        category="Test",
        urgency=8.0,
        operational_value=8.0,
        time_sensitivity=8.0,
        duration_minutes=2.0,
        storage_gb=1.0,
        sensor_type="Optical",
        cloud_sensitivity=0.5,
        max_acceptable_cloud=30.0,
        target_lat=0.0,
        target_lon=0.0,
        description="Feasibility test",
    )
    # Clear sky
    is_feas, score, reason = evaluate_feasibility(req, 0.0)
    assert is_feas is True
    assert score == 1.0
    assert reason is None

    # Within envelope (20% cloud cover <= 30% threshold)
    is_feas, score, reason = evaluate_feasibility(req, 20.0)
    assert is_feas is True
    assert 0.0 < score < 1.0
    assert reason is None

    # Exceeds threshold (35% cloud cover > 30% threshold)
    is_feas, score, reason = evaluate_feasibility(req, 35.0)
    assert is_feas is False
    assert score == 0.0
    assert "exceeds sensor threshold" in reason


def test_cloud_cover_bounds_validation():
    """Verify invalid cloud cover triggers ValueError."""
    req = WILDFIRE_SCENARIO[0]
    with pytest.raises(ValueError):
        evaluate_feasibility(req, -5.0)
    with pytest.raises(ValueError):
        evaluate_feasibility(req, 105.0)


# =========================================================================
# 3. Target Valuation and Normalization
# =========================================================================

def test_target_value_normalization():
    """Verify weights are normalized and score stays on 1-10 scale."""
    req = ObservationRequest(
        target_id="TEST-02",
        name="Valuation Test",
        category="Test",
        urgency=10.0,
        operational_value=6.0,
        time_sensitivity=8.0,
        duration_minutes=2.0,
        storage_gb=1.0,
        sensor_type="Optical",
        cloud_sensitivity=0.5,
        max_acceptable_cloud=50.0,
        target_lat=0.0,
        target_lon=0.0,
        description="Valuation test",
    )
    # Weights sum to 2.0 -> will be normalized to 0.5, 0.25, 0.25
    weights = PlanningWeights(urgency_weight=1.0, value_weight=0.5, time_sensitivity_weight=0.5)
    score = calculate_target_value(req, weights)
    # 0.5 * 10 + 0.25 * 6 + 0.25 * 8 = 5.0 + 1.5 + 2.0 = 8.5
    assert score == 8.5

    # Test all zero weights rejection
    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        PlanningWeights(0, 0, 0).normalized()


# =========================================================================
# 4. Optimization Engine Constraints & Safety
# =========================================================================

def test_initial_wildfire_scenario_constraints():
    """
    Test initial configuration: Wildfire Response, 9 minutes, 5 GB, 20% cloud cover.
    Verify hard limits are strictly respected.
    """
    plan = optimize_observation_plan(
        requests=WILDFIRE_SCENARIO,
        available_time_minutes=9.0,
        available_storage_gb=5.0,
        scenario_cloud_cover=20.0,
        scenario_name="Wildfire Response",
    )
    assert plan.time_used_minutes <= 9.0 + 1e-9
    assert plan.storage_used_gb <= 5.0 + 1e-9
    assert plan.total_value_selected > 0
    assert len(plan.selected_targets) > 0

    # Ensure no cloud-infeasible target is selected
    for target in plan.selected_targets:
        assert target.is_feasible is True
        assert target.request.max_acceptable_cloud >= 20.0


def test_resource_limits_never_exceeded_under_tight_budget():
    """Verify resource constraints hold under extreme budgets."""
    for time_lim in [1.0, 3.0, 5.5, 9.0, 15.0]:
        for stor_lim in [0.8, 1.5, 3.2, 5.0, 10.0]:
            plan = optimize_observation_plan(
                requests=WILDFIRE_SCENARIO,
                available_time_minutes=time_lim,
                available_storage_gb=stor_lim,
                scenario_cloud_cover=25.0,
            )
            assert plan.time_used_minutes <= time_lim + 1e-9
            assert plan.storage_used_gb <= stor_lim + 1e-9


def test_cloud_infeasible_targets_never_selected():
    """At 100% cloud cover, radar/SAR targets might survive if threshold allows, but optical must be excluded."""
    plan = optimize_observation_plan(
        requests=FLOOD_SCENARIO,
        available_time_minutes=20.0,
        available_storage_gb=15.0,
        scenario_cloud_cover=82.0,  # Only SAR targets (FL-01, FL-03 have threshold 85%)
    )
    for t in plan.selected_targets:
        assert t.request.max_acceptable_cloud >= 82.0
    for t in plan.infeasible_targets:
        assert t.request.max_acceptable_cloud < 82.0


def test_zero_feasible_targets_handled_gracefully():
    """At 100% cloud cover when all thresholds are below 100%, plan handles 0 feasible targets without error."""
    plan = optimize_observation_plan(
        requests=WILDFIRE_SCENARIO,
        available_time_minutes=15.0,
        available_storage_gb=10.0,
        scenario_cloud_cover=95.0,  # All WF targets have max_acceptable_cloud <= 60%
    )
    assert len(plan.selected_targets) == 0
    assert len(plan.deferred_targets) == 0
    assert len(plan.infeasible_targets) == len(WILDFIRE_SCENARIO)
    assert plan.total_value_selected == 0.0
    assert plan.time_used_minutes == 0.0
    assert plan.storage_used_gb == 0.0


def test_individually_oversized_targets_handled():
    """Budget smaller than any single target returns 0 selected targets gracefully."""
    plan = optimize_observation_plan(
        requests=WILDFIRE_SCENARIO,
        available_time_minutes=0.5,  # Shortest WF target is 1.8m
        available_storage_gb=0.2,   # Smallest storage is 0.9GB
        scenario_cloud_cover=10.0,
    )
    assert len(plan.selected_targets) == 0
    assert len(plan.deferred_targets) == len(WILDFIRE_SCENARIO)
    assert plan.total_value_selected == 0.0


def test_invalid_planning_inputs_rejected():
    """Verify negative times or negative storage raise ValueError."""
    with pytest.raises(ValueError):
        optimize_observation_plan(WILDFIRE_SCENARIO, available_time_minutes=-1.0, available_storage_gb=5.0, scenario_cloud_cover=20.0)
    with pytest.raises(ValueError):
        optimize_observation_plan(WILDFIRE_SCENARIO, available_time_minutes=9.0, available_storage_gb=-2.0, scenario_cloud_cover=20.0)
    with pytest.raises(ValueError):
        optimize_observation_plan(WILDFIRE_SCENARIO, available_time_minutes=9.0, available_storage_gb=5.0, scenario_cloud_cover=120.0)


# =========================================================================
# 5. Mathematical Global Optimum Verification on Known Dataset
# =========================================================================

def test_knapsack_global_optimum_vs_greedy():
    """
    Construct a classic multi-choice knapsack dilemma where the greedy heuristic fails:
    Target A: Value 10, Duration 6m, Storage 3GB (Density: high single target)
    Target B: Value 7,  Duration 3m, Storage 1.5GB
    Target C: Value 7,  Duration 3m, Storage 1.5GB
    Total available: 6m, 3GB.
    
    Greedy takes Target A first (Value = 10, using 6m/3GB).
    Optimal takes B + C (Value = 14, using 6m/3GB).
    """
    synthetic_targets = [
        ObservationRequest(
            target_id="OPT-A",
            name="Large High Single",
            category="Test",
            urgency=10.0,
            operational_value=10.0,
            time_sensitivity=10.0,
            duration_minutes=6.0,
            storage_gb=3.0,
            sensor_type="Radar",
            cloud_sensitivity=0.1,
            max_acceptable_cloud=90.0,
            target_lat=0.0,
            target_lon=0.0,
            description="High value large item",
        ),
        ObservationRequest(
            target_id="OPT-B",
            name="Synergistic Item 1",
            category="Test",
            urgency=7.0,
            operational_value=7.0,
            time_sensitivity=7.0,
            duration_minutes=3.0,
            storage_gb=1.5,
            sensor_type="Radar",
            cloud_sensitivity=0.1,
            max_acceptable_cloud=90.0,
            target_lat=0.0,
            target_lon=0.0,
            description="Item B",
        ),
        ObservationRequest(
            target_id="OPT-C",
            name="Synergistic Item 2",
            category="Test",
            urgency=7.0,
            operational_value=7.0,
            time_sensitivity=7.0,
            duration_minutes=3.0,
            storage_gb=1.5,
            sensor_type="Radar",
            cloud_sensitivity=0.1,
            max_acceptable_cloud=90.0,
            target_lat=0.0,
            target_lon=0.0,
            description="Item C",
        ),
    ]

    opt_plan = optimize_observation_plan(
        requests=synthetic_targets,
        available_time_minutes=6.0,
        available_storage_gb=3.0,
        scenario_cloud_cover=10.0,
    )
    assert opt_plan.total_value_selected == 14.0
    selected_ids = {t.request.target_id for t in opt_plan.selected_targets}
    assert selected_ids == {"OPT-B", "OPT-C"}

    # Run greedy baseline
    greedy_plan = run_greedy_baseline(
        requests=synthetic_targets,
        available_time_minutes=6.0,
        available_storage_gb=3.0,
        scenario_cloud_cover=10.0,
    )
    # Optimizer must outperform or tie greedy baseline
    assert opt_plan.total_value_selected >= greedy_plan.total_value_selected
    report = compare_plans(opt_plan, greedy_plan)
    assert report.value_difference >= 0.0


def test_optimizer_never_worse_than_greedy_on_all_scenarios():
    """
    Verify across multiple scenarios and parameter variations that optimizer value >= greedy value.
    """
    for scenario_name in ["Wildfire Response", "Flood Mapping", "Agricultural Survey"]:
        requests = get_scenario_observations(scenario_name)
        for (t_lim, s_lim, cloud) in [(5.0, 3.0, 15.0), (9.0, 5.0, 20.0), (14.0, 8.0, 35.0)]:
            opt = optimize_observation_plan(requests, t_lim, s_lim, cloud, scenario_name)
            base = run_greedy_baseline(requests, t_lim, s_lim, cloud)
            assert opt.total_value_selected >= base.total_value_selected - 1e-7, (
                f"Optimizer ({opt.total_value_selected}) failed to meet or exceed greedy baseline "
                f"({base.total_value_selected}) on {scenario_name}"
            )


# =========================================================================
# 6. Sensitivity, Explanations & Tie-Breaking
# =========================================================================

def test_changing_constraints_changes_plan():
    """Changing available time from 4 to 20 minutes expands selected portfolio."""
    plan_tight = optimize_observation_plan(WILDFIRE_SCENARIO, available_time_minutes=4.0, available_storage_gb=2.0, scenario_cloud_cover=15.0)
    plan_abundant = optimize_observation_plan(WILDFIRE_SCENARIO, available_time_minutes=20.0, available_storage_gb=15.0, scenario_cloud_cover=15.0)

    assert len(plan_abundant.selected_targets) > len(plan_tight.selected_targets)
    assert plan_abundant.total_value_selected > plan_tight.total_value_selected


def test_explanations_consistency():
    """Every target must receive a consistent explanation reflecting its status."""
    plan = optimize_observation_plan(
        WILDFIRE_SCENARIO,
        available_time_minutes=9.0,
        available_storage_gb=5.0,
        scenario_cloud_cover=20.0,
    )
    assert len(plan.per_target_explanations) == len(WILDFIRE_SCENARIO)
    for t in plan.selected_targets:
        assert plan.per_target_explanations[t.request.target_id].startswith("SELECTED:")
    for t in plan.deferred_targets:
        assert plan.per_target_explanations[t.request.target_id].startswith("DEFERRED:")
    for t in plan.infeasible_targets:
        assert plan.per_target_explanations[t.request.target_id].startswith("INFEASIBLE:")


def test_deterministic_tie_breaking():
    """Repeated runs with identical inputs produce bit-for-bit identical schedules."""
    plan1 = optimize_observation_plan(WILDFIRE_SCENARIO, 9.0, 5.0, 20.0)
    plan2 = optimize_observation_plan(WILDFIRE_SCENARIO, 9.0, 5.0, 20.0)
    ids1 = [t.request.target_id for t in plan1.selected_targets]
    ids2 = [t.request.target_id for t in plan2.selected_targets]
    assert ids1 == ids2
    assert plan1.total_value_selected == plan2.total_value_selected


# =========================================================================
# 7. Agent Architecture Contracts & Bedrock Payload
# =========================================================================

def test_agent_tool_schema_and_bedrock_payload():
    """Test schema and payload generation for Strands Decider and Bedrock readiness."""
    tool_spec = SatelliteDecisionAgentInterface.format_optimizer_tool_spec()
    assert tool_spec["name"] == "optimize_satellite_pass"
    assert "required" in tool_spec["parameters"]

    plan = optimize_observation_plan(WILDFIRE_SCENARIO, 9.0, 5.0, 20.0, scenario_name="Wildfire Response")
    prompt_payload = SatelliteDecisionAgentInterface.prepare_bedrock_grounded_payload(plan)
    assert "grounded_plan_summary" in prompt_payload.__dict__
    assert prompt_payload.grounded_plan_summary["scenario"] == "Wildfire Response"
    assert prompt_payload.grounded_plan_summary["resource_budget"]["time_available_min"] == 9.0
    assert "Flight Operations Copilot" in prompt_payload.system_prompt
