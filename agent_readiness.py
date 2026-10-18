"""
TerraSignal - AI Agent Architecture & AWS Readiness Specification

This module defines the integration interfaces, schema, and contracts for integrating
TerraSignal's deterministic decision engine with:
1. Strands Decider 2B (agentic policy / strategy selection)
2. Amazon Bedrock (grounded natural language mission briefings)

Status: Architectural specification and local validation contracts.
No fake API calls or synthetic AWS credentials are executed.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
from decision_engine import OptimizationResult, PlanningWeights


@dataclass
class AgentStrategyInput:
    """
    Schema passed to Strands Decider 2B to select tactical mission priority weights.
    """
    scenario_name: str
    scenario_cloud_cover: float
    time_budget_minutes: float
    storage_budget_gb: float
    candidate_target_count: int
    operational_context: str


@dataclass
class AgentStrategyDecision:
    """
    Structured policy returned by Strands Decider 2B.
    """
    recommended_strategy_name: str
    justification: str
    weights: PlanningWeights
    risk_tolerance: str  # e.g., "CONSERVATIVE", "BALANCED", "AGGRESSIVE"


@dataclass
class BedrockBriefingPrompt:
    """
    Strict, hallucination-resistant payload structure for Amazon Bedrock Claude / Titan model.
    Contains grounded quantitative results from the deterministic optimization engine.
    """
    system_prompt: str
    grounded_plan_summary: Dict[str, Any]
    user_instructions: str


class SatelliteDecisionAgentInterface:
    """
    Interface definition for the genuine AI decision agent workflow.
    
    Workflow:
    1. User sets environment & constraints.
    2. [Future Decider 2B]: Evaluates tactical urgency vs long-term science and adjusts weights.
    3. [Local Tool]: optimize_observation_plan executes mathematical knapsack optimization.
    4. [Local Validator]: Verifies all hard constraints (time, storage, cloud limit) hold 100%.
    5. [Future Bedrock]: Generates executive verbal briefing strictly adhering to verified facts.
    """

    @staticmethod
    def format_optimizer_tool_spec() -> Dict[str, Any]:
        """
        Tool schema formatted for Anthropic / Bedrock / Strands Decider tool-use calling.
        """
        return {
            "name": "optimize_satellite_pass",
            "description": "Calculates the mathematically optimal subset of Earth-observation targets under time, storage, and cloud constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_name": {"type": "string", "enum": ["Wildfire Response", "Flood Mapping", "Agricultural Survey"]},
                    "available_time_minutes": {"type": "number", "minimum": 1.0, "maximum": 60.0},
                    "available_storage_gb": {"type": "number", "minimum": 0.5, "maximum": 100.0},
                    "scenario_cloud_cover": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    "urgency_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "value_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "time_sensitivity_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["scenario_name", "available_time_minutes", "available_storage_gb", "scenario_cloud_cover"],
            },
        }

    @staticmethod
    def prepare_bedrock_grounded_payload(plan: OptimizationResult) -> BedrockBriefingPrompt:
        """
        Prepares structured, verified context to prevent generative hallucination in Bedrock.
        """
        selected_summaries = [
            {
                "id": t.request.target_id,
                "name": t.request.name,
                "category": t.request.category,
                "score": t.target_value,
                "time_min": t.request.duration_minutes,
                "storage_gb": t.request.storage_gb,
            }
            for t in plan.selected_targets
        ]

        deferred_summaries = [
            {
                "id": t.request.target_id,
                "name": t.request.name,
                "reason": plan.per_target_explanations.get(t.request.target_id, "Deferred by optimizer"),
            }
            for t in plan.deferred_targets
        ]

        grounded_data = {
            "scenario": plan.scenario_name,
            "cloud_cover_pct": plan.scenario_cloud_cover,
            "resource_budget": {
                "time_available_min": plan.available_time_minutes,
                "time_used_min": round(plan.time_used_minutes, 2),
                "time_utilization_pct": round(plan.time_utilization_pct, 1),
                "storage_available_gb": plan.available_storage_gb,
                "storage_used_gb": round(plan.storage_used_gb, 2),
                "storage_utilization_pct": round(plan.storage_utilization_pct, 1),
            },
            "selected_targets": selected_summaries,
            "deferred_targets": deferred_summaries,
            "infeasible_targets_count": len(plan.infeasible_targets),
            "total_value_score": round(plan.total_value_selected, 2),
            "mathematically_optimal": plan.optimal_found,
        }

        system_instruction = (
            "You are an Aerospace Flight Operations Copilot. Synthesize an executive verbal mission briefing "
            "strictly using the grounded numerical parameters provided in the JSON payload. Do NOT fabricate additional "
            "targets, do NOT alter resource utilization statistics, and do NOT claim live satellite communications. "
            "State clearly that this pass plan was determined via deterministic multi-objective knapsack optimization."
        )

        return BedrockBriefingPrompt(
            system_prompt=system_instruction,
            grounded_plan_summary=grounded_data,
            user_instructions="Provide a concise 3-paragraph tactical pass briefing for the flight director.",
        )
