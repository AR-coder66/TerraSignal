"""
TerraSignal - Decision Engine

Constrained optimization and feasibility evaluation module for satellite observation scheduling.
This module is fully independent of Streamlit and can be invoked programmatically or by an AI agent tool.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from itertools import combinations
from data import ObservationRequest


@dataclass(frozen=True)
class PlanningWeights:
    """
    Weights for target valuation scoring.
    Will be normalized so that w_urgency + w_value + w_time = 1.0.
    """
    urgency_weight: float = 0.40
    value_weight: float = 0.35
    time_sensitivity_weight: float = 0.25

    def normalized(self) -> "PlanningWeights":
        total = self.urgency_weight + self.value_weight + self.time_sensitivity_weight
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        return PlanningWeights(
            urgency_weight=self.urgency_weight / total,
            value_weight=self.value_weight / total,
            time_sensitivity_weight=self.time_sensitivity_weight / total,
        )


@dataclass
class EvaluatedTarget:
    """
    Intermediate representation of an observation target with feasibility and valuation results.
    """
    request: ObservationRequest
    target_value: float
    is_feasible: bool
    feasibility_score: float  # 0.0 to 1.0 (estimated feasibility under current cloud cover)
    infeasible_reason: Optional[str] = None


@dataclass
class OptimizationResult:
    """
    Structured outcome of the constrained optimization planning process.
    """
    scenario_name: str
    available_time_minutes: float
    available_storage_gb: float
    scenario_cloud_cover: float
    weights: PlanningWeights

    selected_targets: List[EvaluatedTarget]
    deferred_targets: List[EvaluatedTarget]
    infeasible_targets: List[EvaluatedTarget]

    total_value_selected: float
    time_used_minutes: float
    storage_used_gb: float
    time_remaining_minutes: float
    storage_remaining_gb: float
    time_utilization_pct: float
    storage_utilization_pct: float

    optimal_found: bool
    optimization_method: str
    per_target_explanations: Dict[str, str]
    summary_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "available_time_minutes": self.available_time_minutes,
            "available_storage_gb": self.available_storage_gb,
            "scenario_cloud_cover": self.scenario_cloud_cover,
            "total_value_selected": round(self.total_value_selected, 4),
            "time_used_minutes": round(self.time_used_minutes, 2),
            "storage_used_gb": round(self.storage_used_gb, 2),
            "time_remaining_minutes": round(self.time_remaining_minutes, 2),
            "storage_remaining_gb": round(self.storage_remaining_gb, 2),
            "time_utilization_pct": round(self.time_utilization_pct, 2),
            "storage_utilization_pct": round(self.storage_utilization_pct, 2),
            "selected_count": len(self.selected_targets),
            "deferred_count": len(self.deferred_targets),
            "infeasible_count": len(self.infeasible_targets),
            "optimal_found": self.optimal_found,
            "optimization_method": self.optimization_method,
            "per_target_explanations": self.per_target_explanations,
            "summary_notes": self.summary_notes,
        }


def calculate_target_value(request: ObservationRequest, weights: PlanningWeights) -> float:
    """
    Calculate documented target-value score:
        Score = (w_u * urgency + w_v * operational_value + w_t * time_sensitivity)
    With normalized weights, the resulting score resides on the same 1.0 - 10.0 scale as inputs.
    """
    nw = weights.normalized()
    score = (
        nw.urgency_weight * request.urgency +
        nw.value_weight * request.operational_value +
        nw.time_sensitivity_weight * request.time_sensitivity
    )
    return round(score, 4)


def evaluate_feasibility(request: ObservationRequest, scenario_cloud_cover: float) -> Tuple[bool, float, Optional[str]]:
    """
    Evaluate whether a target is feasible under simulated cloud conditions.
    
    Feasibility rule:
    - If scenario_cloud_cover > request.max_acceptable_cloud, target is infeasible.
    - Simulated feasibility score decreases linearly as cloud cover approaches max_acceptable_cloud:
        feasibility_score = max(0.0, 1.0 - (scenario_cloud_cover / max_acceptable_cloud) * request.cloud_sensitivity)
    """
    if scenario_cloud_cover < 0.0 or scenario_cloud_cover > 100.0:
        raise ValueError(f"Cloud cover must be between 0 and 100%, got {scenario_cloud_cover}")

    if scenario_cloud_cover > request.max_acceptable_cloud:
        reason = (
            f"Simulated cloud cover ({scenario_cloud_cover:.1f}%) exceeds sensor threshold "
            f"({request.max_acceptable_cloud:.1f}%) for {request.sensor_type} payload."
        )
        return False, 0.0, reason

    # Within acceptable envelope
    ratio = scenario_cloud_cover / max(request.max_acceptable_cloud, 1.0)
    feasibility_score = round(max(0.05, 1.0 - (ratio * request.cloud_sensitivity)), 4)
    return True, feasibility_score, None


def optimize_observation_plan(
    requests: List[ObservationRequest],
    available_time_minutes: float,
    available_storage_gb: float,
    scenario_cloud_cover: float,
    scenario_name: str = "Mission",
    weights: Optional[PlanningWeights] = None,
) -> OptimizationResult:
    """
    Constrained Optimization Engine:
    
    1. Evaluates cloud feasibility for all observation requests.
    2. Calculates normalized target-value scores.
    3. Finds the combination of feasible targets that maximizes total target value
       subject to:
           sum(duration_minutes) <= available_time_minutes
           sum(storage_gb) <= available_storage_gb
    4. Deterministic tie-breaking:
       - Higher total value
       - Higher total urgency
       - Lower total time consumption
       - Lower total storage consumption
       - Lexicographical target ID ordering
    5. Formulates structured per-target explanations distinguishing infeasible, selected, and resource-deferred targets.
    """
    # Validation
    if available_time_minutes < 0:
        raise ValueError("Available imaging time cannot be negative.")
    if available_storage_gb < 0:
        raise ValueError("Available storage cannot be negative.")
    if scenario_cloud_cover < 0 or scenario_cloud_cover > 100:
        raise ValueError("Scenario cloud cover must be between 0 and 100%.")

    if weights is None:
        weights = PlanningWeights()

    # Step 1 & 2: Feasibility & Valuation
    evaluated: List[EvaluatedTarget] = []
    feasible_targets: List[EvaluatedTarget] = []
    infeasible_targets: List[EvaluatedTarget] = []

    for req in requests:
        is_feas, feas_score, inf_reason = evaluate_feasibility(req, scenario_cloud_cover)
        val = calculate_target_value(req, weights)
        target = EvaluatedTarget(
            request=req,
            target_value=val,
            is_feasible=is_feas,
            feasibility_score=feas_score,
            infeasible_reason=inf_reason,
        )
        evaluated.append(target)
        if is_feas:
            feasible_targets.append(target)
        else:
            infeasible_targets.append(target)

    # Step 3: Optimal Subset Selection via Exhaustive Enumeration
    # The search space is bounded (N <= 20 in tactical passes, initial datasets N=8 -> 2^8 = 256 subsets).
    # Exhaustive enumeration guarantees mathematical global optimum for the multiconstrained 0-1 knapsack problem.
    n_feasible = len(feasible_targets)
    best_subset: Tuple[EvaluatedTarget, ...] = ()
    best_value: float = -1.0
    best_urgency: float = -1.0
    best_time: float = float("inf")
    best_storage: float = float("inf")
    best_ids_str: str = ""

    # Include empty set as valid baseline (0 value, 0 resources used)
    all_subsets = []
    for r in range(n_feasible + 1):
        for combo in combinations(feasible_targets, r):
            all_subsets.append(combo)

    for subset in all_subsets:
        tot_time = sum(t.request.duration_minutes for t in subset)
        tot_storage = sum(t.request.storage_gb for t in subset)

        # Check multi-resource budget feasibility
        # Numerical tolerance of 1e-9 to prevent floating point inaccuracies
        if tot_time <= available_time_minutes + 1e-9 and tot_storage <= available_storage_gb + 1e-9:
            tot_val = sum(t.target_value for t in subset)
            tot_urg = sum(t.request.urgency for t in subset)
            ids_str = ",".join(sorted(t.request.target_id for t in subset))

            # Multi-level deterministic tie-breaking
            is_better = False
            if tot_val > best_value + 1e-7:
                is_better = True
            elif abs(tot_val - best_value) <= 1e-7:
                if tot_urg > best_urgency + 1e-7:
                    is_better = True
                elif abs(tot_urg - best_urgency) <= 1e-7:
                    if tot_time < best_time - 1e-7:
                        is_better = True
                    elif abs(tot_time - best_time) <= 1e-7:
                        if tot_storage < best_storage - 1e-7:
                            is_better = True
                        elif abs(tot_storage - best_storage) <= 1e-7:
                            if ids_str < best_ids_str:
                                is_better = True

            if is_better:
                best_value = tot_val
                best_urgency = tot_urg
                best_time = tot_time
                best_storage = tot_storage
                best_ids_str = ids_str
                best_subset = subset

    selected_list = list(best_subset)
    # Sort selected by value descending for display
    selected_list.sort(key=lambda t: (t.target_value, t.request.urgency), reverse=True)
    selected_ids = {t.request.target_id for t in selected_list}

    deferred_targets = [t for t in feasible_targets if t.request.target_id not in selected_ids]
    # Sort deferred by value descending
    deferred_targets.sort(key=lambda t: (t.target_value, t.request.urgency), reverse=True)

    # Step 4: Metric Accounting
    time_used = sum(t.request.duration_minutes for t in selected_list)
    storage_used = sum(t.request.storage_gb for t in selected_list)
    total_val = sum(t.target_value for t in selected_list)

    time_rem = max(0.0, available_time_minutes - time_used)
    storage_rem = max(0.0, available_storage_gb - storage_used)

    time_util = (time_used / available_time_minutes * 100.0) if available_time_minutes > 0 else 0.0
    storage_util = (storage_used / available_storage_gb * 100.0) if available_storage_gb > 0 else 0.0

    # Step 5: Decision Explanations
    explanations: Dict[str, str] = {}
    for t in selected_list:
        explanations[t.request.target_id] = (
            f"SELECTED: High composite value ({t.target_value:.2f}) and fits within combined "
            f"imaging budget ({t.request.duration_minutes:.1f}m / {t.request.storage_gb:.1f}GB)."
        )

    for t in deferred_targets:
        # Check whether the individual target could have fit into the remaining margin
        fits_time = t.request.duration_minutes <= (time_rem + 1e-9)
        fits_storage = t.request.storage_gb <= (storage_rem + 1e-9)
        
        if not fits_time and not fits_storage:
            reason = (
                f"DEFERRED: Target requires {t.request.duration_minutes:.1f}m imaging time and "
                f"{t.request.storage_gb:.1f}GB storage, exceeding remaining margins "
                f"({time_rem:.1f}m / {storage_rem:.1f}GB)."
            )
        elif not fits_time:
            reason = (
                f"DEFERRED: Insufficient imaging time remaining ({time_rem:.1f}m available, "
                f"requires {t.request.duration_minutes:.1f}m)."
            )
        elif not fits_storage:
            reason = (
                f"DEFERRED: Insufficient onboard payload storage remaining ({storage_rem:.1f}GB available, "
                f"requires {t.request.storage_gb:.1f}GB)."
            )
        else:
            # Fits in remaining budget, but omitted because another subset delivered higher total value
            reason = (
                f"DEFERRED: Target could fit individually, but competing target combinations "
                f"achieved a higher overall objective value within the dual budget."
            )
        explanations[t.request.target_id] = reason

    for t in infeasible_targets:
        explanations[t.request.target_id] = f"INFEASIBLE: {t.infeasible_reason}"

    # Step 6: Summary Notes
    summary_notes = []
    if len(feasible_targets) == 0:
        summary_notes.append("No observation requests met environmental cloud feasibility thresholds.")
    elif len(selected_list) == 0:
        summary_notes.append("Resource budgets are insufficient to image any individual feasible observation.")
    else:
        summary_notes.append(
            f"Optimized schedule allocated {time_used:.1f}m / {available_time_minutes:.1f}m imaging time "
            f"({time_util:.1f}%) and {storage_used:.1f}GB / {available_storage_gb:.1f}GB storage ({storage_util:.1f}%)."
        )
        if time_rem < 1.0 or storage_rem < 0.5:
            summary_notes.append("Satellite pass capacity is nearly saturated under the primary resource constraints.")

    return OptimizationResult(
        scenario_name=scenario_name,
        available_time_minutes=available_time_minutes,
        available_storage_gb=available_storage_gb,
        scenario_cloud_cover=scenario_cloud_cover,
        weights=weights.normalized(),
        selected_targets=selected_list,
        deferred_targets=deferred_targets,
        infeasible_targets=infeasible_targets,
        total_value_selected=total_val,
        time_used_minutes=time_used,
        storage_used_gb=storage_used,
        time_remaining_minutes=time_rem,
        storage_remaining_gb=storage_rem,
        time_utilization_pct=time_util,
        storage_utilization_pct=storage_util,
        optimal_found=True,
        optimization_method="Exhaustive Global Knapsack Optimizer",
        per_target_explanations=explanations,
        summary_notes=summary_notes,
    )
