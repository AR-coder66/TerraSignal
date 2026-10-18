# TerraSignal — AI-Powered Satellite Mission Decision System

TerraSignal is a functional, interactive Earth-observation planning and mission decision-support system. It simulates how an intelligent agent or flight operations copilot prioritizes competing observation targets when satellite imaging dwell time, solid-state storage buffers, and atmospheric conditions (cloud cover) are strictly constrained.

> **Operational Boundary Notice:** TerraSignal is an operational simulation and planning decision-support prototype using deterministic synthetic mission models. It does **not** claim to command real orbiting spacecraft, receive live downlink telemetry, or directly execute ground control commands.

---

## 1. Problem Overview

Low-Earth Orbit (LEO) remote sensing satellites operate under severe tactical and physical constraints:
- **Tight Orbital Imaging Windows:** Sensor slewing, thermal settling, and line-of-sight dwell time over areas of interest are limited to minutes per pass.
- **Onboard Storage Bottlenecks:** Solid-state memory buffers must retain raw multispectral, hyperspectral, and SAR data until the next ground station downlink opportunity.
- **Environmental Obstructions:** Cloud cover severely degrades optical sensors while radar (SAR) payloads penetrate weather with minimal attenuation.
- **Competing Tactical Objectives:** Disaster response commanders, environmental scientists, and infrastructure managers compete for the same pass capacity.

TerraSignal resolves these trade-offs mathematically, providing transparent, accountable decision explanations for every target selected, deferred, or rejected.

---

## 2. Core Architecture & Modules

```
TerraSignal/
├── app.py                  # Aerospace-grade Streamlit interactive operations console
├── decision_engine.py      # Independent optimization engine (0-1 knapsack global solver)
├── data.py                 # Deterministic, reproducible synthetic mission scenarios
├── evaluation.py           # Greedy heuristic baseline & algorithmic benchmarking
├── agent_readiness.py      # Strands Decider 2B & Amazon Bedrock architectural contracts
├── tests/
│   └── test_engine.py      # Automated pytest verification suite (17 comprehensive tests)
├── .streamlit/
│   └── config.toml         # Aerospace dark theme configuration
├── requirements.txt        # Minimal pinned runtime dependencies
└── README.md               # System documentation & reproduction guide
```

---

## 3. Mathematical Foundations

### 3.1 Target Valuation Scoring
Target score is computed using user-defined or agent-selected weights normalized such that:
$$w_{\text{urgency}} + w_{\text{value}} + w_{\text{time}} = 1.0$$

$$\text{Target Score} = w_{\text{urgency}} \cdot \text{Urgency} + w_{\text{value}} \cdot \text{Value} + w_{\text{time}} \cdot \text{TimeSensitivity}$$

With inputs on a $[1.0, 10.0]$ scale, the composite valuation remains strictly on the $[1.0, 10.0]$ scale.

### 3.2 Environmental Cloud Feasibility Model
Each observation target defines a payload sensor type, a maximum acceptable cloud percentage ($C_{\max}$), and a cloud sensitivity factor ($s \in [0.0, 1.0]$):
- If $\text{CloudCover} > C_{\max}$: The target is deemed **INFEASIBLE** (e.g., optical imaging occluded by cloud layers).
- If $\text{CloudCover} \le C_{\max}$: Feasible with an attenuated quality score:
  $$\text{Feasibility Metric} = \max\left(0.05, 1.0 - \frac{\text{CloudCover}}{C_{\max}} \cdot s\right)$$

### 3.3 Constrained Optimization Engine
Finding the optimal subset of targets is formulated as a **Multi-Constrained 0-1 Knapsack Problem**:
$$\max \sum_{i \in \mathcal{F}} x_i \cdot \text{Score}_i$$
Subject to:
$$\sum_{i \in \mathcal{F}} x_i \cdot \text{Duration}_i \le T_{\text{available}}$$
$$\sum_{i \in \mathcal{F}} x_i \cdot \text{Storage}_i \le S_{\text{available}}$$
$$x_i \in \{0, 1\} \quad \forall i \in \mathcal{F}$$

Where $\mathcal{F}$ is the set of cloud-feasible targets. TerraSignal solves this via exhaustive combinatorial search over all $2^{|\mathcal{F}|}$ subsets, guaranteeing the global mathematical optimum.

#### Deterministic Multi-Level Tie-Breaking:
If multiple subsets yield identical objective values:
1. Higher cumulative urgency score.
2. Lower total imaging time consumption (higher operational margin).
3. Lower total onboard storage consumption.
4. Lexicographical ordering of sorted target IDs.

---

## 4. Algorithmic Baseline Comparison

TerraSignal evaluates the optimizer against a **Greedy Multi-Resource Efficiency Heuristic**:
1. Computes normalized dual-resource cost:
   $$\text{Cost}_i = \frac{\text{Duration}_i}{T_{\text{available}}} + \frac{\text{Storage}_i}{S_{\text{available}}}$$
2. Ranks feasible candidates descending by efficiency: $\frac{\text{Score}_i}{\text{Cost}_i}$.
3. Greedily allocates targets until capacity is exhausted.

**Takeaway:** The exhaustive optimizer's total value is mathematically proven to be $\ge$ the greedy baseline's value. In constrained scenarios with varying target footprints, greedy heuristics strand capacity fragments, whereas the global solver explores synergistic combinations to achieve superior objective values.

---

## 5. Synthetic Mission Scenarios

TerraSignal provides reproducible datasets across three realistic mission profiles (at least 8 distinct targets each):

1. **Wildfire Response (Default):** Fire perimeter mapping, hotspot assessment, population exposure, smoke plume analysis, burn-scar monitoring, evacuation corridors, infrastructure exposure, and vegetation stress.
2. **Flood Mapping:** River breach inundation, urban flood depth, drainage surge, transport lifeline surveys, agricultural runoff, drinking water plants, subsidence, and recession tracking.
3. **Agricultural Survey:** NDVI phenology, drought stress/evapotranspiration, pest outbreaks, L-band root-zone soil moisture, harvest yield forecasting, illegal aquifer tracing, tillage audits, and canopy biomass.

---

## 6. AI Agent Architecture & AWS Readiness

TerraSignal is architected to decouple reasoning from deterministic safety constraints:
1. **Inputs:** Mission planners configure pass budgets and environmental parameters.
2. **Agent Strategy (Future Strands Decider 2B):** A small decision model evaluates tactical context (e.g., life safety vs. long-term science) and selects planning weights.
3. **Deterministic Tool:** The local `optimize_satellite_pass` knapsack tool executes and returns verified assignments.
4. **Hard Constraint Validation:** All orbital budgets are checked; generative models are not permitted to alter numerical schedules.
5. **Mission Briefing (Future Amazon Bedrock):** Generative briefing prompt is constructed from grounded mathematical facts (`agent_readiness.py`) to eliminate hallucination.

### Status of Integrations:
- **Local Optimization Engine:** Fully implemented and verified.
- **Strands Decider 2B Tool Schema:** Fully defined and tested in [agent_readiness.py](file:///c:/Users/ars_6/Desktop/Projects/TerraSignal/agent_readiness.py).
- **Amazon Bedrock Grounded Contract:** Structured JSON payload pipeline implemented; AWS SDK / Boto3 credentials pending live AWS environment provisioning.

---

## 7. Setup & Execution Guide (PowerShell / Windows)

### 7.1 Prerequisites
- Python 3.13 (already present in workspace `.venv`)

### 7.2 Run Automated Test Suite
Execute all 17 unit and mathematical correctness tests:
```powershell
.\.venv\Scripts\python -m pytest -v
```

### 7.3 Launch the Interactive Operations Console
Launch Streamlit in your default browser:
```powershell
.\.venv\Scripts\streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## 8. Verified Test Matrix

The automated test suite (`tests/test_engine.py`) enforces:
1. `test_scenario_datasets_distinct_and_sized`: Verifies 8+ distinct targets per scenario.
2. `test_scenario_retrieval_and_error`: Rejects invalid mission scenario queries.
3. `test_cloud_feasibility_logic`: Confirms sensor threshold cutoffs and quality attenuation.
4. `test_cloud_cover_bounds_validation`: Validates $[0, 100]\%$ cloud bounds.
5. `test_target_value_normalization`: Confirms weights normalize and output is bounded.
6. `test_initial_wildfire_scenario_constraints`: Default 9 min, 5 GB, 20% cloud cover strictly holds.
7. `test_resource_limits_never_exceeded_under_tight_budget`: Validates boundary tightness across 25 budget pairs.
8. `test_cloud_infeasible_targets_never_selected`: Infeasible targets excluded under high cloud conditions.
9. `test_zero_feasible_targets_handled_gracefully`: Handles 100% cloud cover with 0 crashes.
10. `test_individually_oversized_targets_handled`: Zero-allocation handling when budget is smaller than any single item.
11. `test_invalid_planning_inputs_rejected`: Rejects negative times, storage, or out-of-bounds inputs.
12. `test_knapsack_global_optimum_vs_greedy`: Verifies optimizer surpasses greedy heuristic on known counter-example.
13. `test_optimizer_never_worse_than_greedy_on_all_scenarios`: Verifies $V_{\text{opt}} \ge V_{\text{greedy}}$ across all scenarios.
14. `test_changing_constraints_changes_plan`: Sensitivity verification for budget adjustments.
15. `test_explanations_consistency`: Ensures every target receives an accurate, human-readable reason.
16. `test_deterministic_tie_breaking`: Bit-for-bit repeatability across repeated runs.
17. `test_agent_tool_schema_and_bedrock_payload`: Validates Decider 2B tool schema and Bedrock grounded payload contracts.
