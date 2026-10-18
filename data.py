"""
TerraSignal - Mission Scenarios & Synthetic Observation Data

This module defines deterministic, reproducible observation requests for distinct
Earth-observation missions: Wildfire Response, Flood Mapping, and Agricultural Survey.
All values are deterministic simulation models, not physical ground telemetry.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class ObservationRequest:
    """
    Represents an Earth-observation acquisition request.
    
    Attributes:
        target_id: Unique identifier for the observation target.
        name: Descriptive operational name.
        category: Functional category or operational domain.
        urgency: Urgency score on a scale of 1.0 to 10.0 (higher = more time-critical).
        operational_value: Scientific or tactical value on a scale of 1.0 to 10.0.
        time_sensitivity: Criticality of observing within current pass (1.0 to 10.0).
        duration_minutes: Estimated satellite pointing and imaging duration in minutes.
        storage_gb: Estimated raw telemetry / payload storage consumption in GB.
        sensor_type: Primary sensor band (e.g., Optical, High-Res Optical, Thermal IR, SAR / Radar, Multispectral).
        cloud_sensitivity: Sensitivity coefficient to cloud degradation (0.0 = completely unaffected, 1.0 = highly vulnerable).
        max_acceptable_cloud: Maximum scenario cloud cover percentage before observation is declared infeasible.
        target_lat: Simulated latitude for mission map visualization.
        target_lon: Simulated longitude for mission map visualization.
        description: Brief operational description of the imaging objective.
    """
    target_id: str
    name: str
    category: str
    urgency: float
    operational_value: float
    time_sensitivity: float
    duration_minutes: float
    storage_gb: float
    sensor_type: str
    cloud_sensitivity: float
    max_acceptable_cloud: float
    target_lat: float
    target_lon: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name": self.name,
            "category": self.category,
            "urgency": self.urgency,
            "operational_value": self.operational_value,
            "time_sensitivity": self.time_sensitivity,
            "duration_minutes": self.duration_minutes,
            "storage_gb": self.storage_gb,
            "sensor_type": self.sensor_type,
            "cloud_sensitivity": self.cloud_sensitivity,
            "max_acceptable_cloud": self.max_acceptable_cloud,
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "description": self.description,
        }


# -------------------------------------------------------------------------
# Scenario 1: Wildfire Response (Default Scenario)
# Coordinates centered around Pacific Northwest / Northern California wildland interface
# -------------------------------------------------------------------------
WILDFIRE_SCENARIO: List[ObservationRequest] = [
    ObservationRequest(
        target_id="WF-01",
        name="Fire Perimeter Mapping",
        category="Tactical Response",
        urgency=9.5,
        operational_value=9.0,
        time_sensitivity=9.5,
        duration_minutes=2.5,
        storage_gb=1.4,
        sensor_type="Thermal IR / High-Res Optical",
        cloud_sensitivity=0.45,
        max_acceptable_cloud=35.0,
        target_lat=39.75,
        target_lon=-121.62,
        description="Active flame front delineation and containment perimeter mapping."
    ),
    ObservationRequest(
        target_id="WF-02",
        name="Hotspot Assessment",
        category="Thermal Detection",
        urgency=8.8,
        operational_value=8.2,
        time_sensitivity=8.5,
        duration_minutes=1.8,
        storage_gb=0.9,
        sensor_type="Thermal IR",
        cloud_sensitivity=0.25,
        max_acceptable_cloud=50.0,
        target_lat=39.91,
        target_lon=-121.48,
        description="Detection of sub-canopy thermal anomalies and smoldering ember zones."
    ),
    ObservationRequest(
        target_id="WF-03",
        name="Population Exposure Mapping",
        category="Life Safety",
        urgency=9.8,
        operational_value=9.5,
        time_sensitivity=9.8,
        duration_minutes=3.0,
        storage_gb=1.8,
        sensor_type="High-Res Optical",
        cloud_sensitivity=0.60,
        max_acceptable_cloud=25.0,
        target_lat=39.68,
        target_lon=-121.75,
        description="High-resolution survey of populated settlement boundaries under threat."
    ),
    ObservationRequest(
        target_id="WF-04",
        name="Smoke Plume Analysis",
        category="Atmospheric Tracking",
        urgency=6.5,
        operational_value=7.0,
        time_sensitivity=6.0,
        duration_minutes=2.0,
        storage_gb=1.1,
        sensor_type="Multispectral",
        cloud_sensitivity=0.30,
        max_acceptable_cloud=45.0,
        target_lat=40.12,
        target_lon=-121.20,
        description="Aerosol optical depth and atmospheric dispersion modeling for regional air quality."
    ),
    ObservationRequest(
        target_id="WF-05",
        name="Burn-Scar Monitoring",
        category="Post-Event Analysis",
        urgency=4.5,
        operational_value=6.8,
        time_sensitivity=4.0,
        duration_minutes=2.2,
        storage_gb=1.3,
        sensor_type="Multispectral",
        cloud_sensitivity=0.50,
        max_acceptable_cloud=30.0,
        target_lat=39.52,
        target_lon=-121.35,
        description="Normalized Burn Ratio (NBR) evaluation for vegetative recovery baseline."
    ),
    ObservationRequest(
        target_id="WF-06",
        name="Evacuation Corridor Assessment",
        category="Life Safety",
        urgency=9.2,
        operational_value=8.9,
        time_sensitivity=9.0,
        duration_minutes=2.4,
        storage_gb=1.5,
        sensor_type="High-Res Optical",
        cloud_sensitivity=0.55,
        max_acceptable_cloud=25.0,
        target_lat=39.82,
        target_lon=-121.85,
        description="Verification of highway passability and spot fire encroachment on egress routes."
    ),
    ObservationRequest(
        target_id="WF-07",
        name="Critical Infrastructure Exposure",
        category="Asset Protection",
        urgency=8.0,
        operational_value=8.4,
        time_sensitivity=7.5,
        duration_minutes=2.0,
        storage_gb=1.2,
        sensor_type="Thermal IR / SAR",
        cloud_sensitivity=0.15,
        max_acceptable_cloud=60.0,
        target_lat=39.60,
        target_lon=-121.55,
        description="High-voltage transmission substation and municipal water reservoir inspection."
    ),
    ObservationRequest(
        target_id="WF-08",
        name="Vegetation Stress Monitoring",
        category="Predictive Risk",
        urgency=5.0,
        operational_value=6.5,
        time_sensitivity=4.5,
        duration_minutes=2.6,
        storage_gb=1.6,
        sensor_type="Hyperspectral",
        cloud_sensitivity=0.50,
        max_acceptable_cloud=30.0,
        target_lat=40.25,
        target_lon=-121.90,
        description="Fuel moisture content and dry canopy index in adjacent unburned fuel beds."
    ),
]


# -------------------------------------------------------------------------
# Scenario 2: Flood Mapping
# Coordinates centered around Mississippi / Gulf Coast river basin
# -------------------------------------------------------------------------
FLOOD_SCENARIO: List[ObservationRequest] = [
    ObservationRequest(
        target_id="FL-01",
        name="River Breach & Inundation Extent",
        category="Tactical Response",
        urgency=9.6,
        operational_value=9.4,
        time_sensitivity=9.6,
        duration_minutes=2.8,
        storage_gb=1.7,
        sensor_type="SAR / Radar",
        cloud_sensitivity=0.05,
        max_acceptable_cloud=85.0,
        target_lat=29.95,
        target_lon=-90.07,
        description="Synthetic Aperture Radar delineation of levee breach overflow into residential basin."
    ),
    ObservationRequest(
        target_id="FL-02",
        name="Urban Flood Depth Mapping",
        category="Life Safety",
        urgency=9.1,
        operational_value=8.8,
        time_sensitivity=9.0,
        duration_minutes=2.2,
        storage_gb=1.5,
        sensor_type="High-Res Optical",
        cloud_sensitivity=0.70,
        max_acceptable_cloud=20.0,
        target_lat=30.03,
        target_lon=-90.15,
        description="High-resolution street-level water level estimation and stranded vehicle detection."
    ),
    ObservationRequest(
        target_id="FL-03",
        name="Drainage Basin Peak Surge",
        category="Hydrology",
        urgency=7.8,
        operational_value=8.0,
        time_sensitivity=7.5,
        duration_minutes=2.0,
        storage_gb=1.2,
        sensor_type="SAR / Radar",
        cloud_sensitivity=0.05,
        max_acceptable_cloud=85.0,
        target_lat=30.22,
        target_lon=-89.88,
        description="All-weather microwave observation of estuarine surge retention and tidal interaction."
    ),
    ObservationRequest(
        target_id="FL-04",
        name="Transportation Lifeline Survey",
        category="Asset Protection",
        urgency=8.9,
        operational_value=8.5,
        time_sensitivity=8.7,
        duration_minutes=2.5,
        storage_gb=1.6,
        sensor_type="High-Res Optical / SAR",
        cloud_sensitivity=0.35,
        max_acceptable_cloud=40.0,
        target_lat=29.85,
        target_lon=-90.30,
        description="Interstate causeway bridge abutments and rail causeway structural submergence."
    ),
    ObservationRequest(
        target_id="FL-05",
        name="Agricultural Runoff & Siltation",
        category="Environmental Impact",
        urgency=4.8,
        operational_value=6.2,
        time_sensitivity=4.0,
        duration_minutes=1.9,
        storage_gb=1.0,
        sensor_type="Multispectral",
        cloud_sensitivity=0.60,
        max_acceptable_cloud=25.0,
        target_lat=30.45,
        target_lon=-90.50,
        description="Turbidity plume tracking and topsoil erosion assessment across coastal flats."
    ),
    ObservationRequest(
        target_id="FL-06",
        name="Drinking Water Plant Perimeter",
        category="Critical Infrastructure",
        urgency=8.7,
        operational_value=8.9,
        time_sensitivity=8.4,
        duration_minutes=1.8,
        storage_gb=1.1,
        sensor_type="Optical / Thermal",
        cloud_sensitivity=0.55,
        max_acceptable_cloud=30.0,
        target_lat=30.10,
        target_lon=-90.25,
        description="Floodwall freeboard monitoring at regional water purification facility."
    ),
    ObservationRequest(
        target_id="FL-07",
        name="Subsidence & Saturated Ground",
        category="Geotechnical Risk",
        urgency=6.0,
        operational_value=7.2,
        time_sensitivity=5.5,
        duration_minutes=2.6,
        storage_gb=1.4,
        sensor_type="SAR Interferometry",
        cloud_sensitivity=0.10,
        max_acceptable_cloud=75.0,
        target_lat=29.70,
        target_lon=-89.95,
        description="Soil saturation index and ground displacement detection along flood barriers."
    ),
    ObservationRequest(
        target_id="FL-08",
        name="Post-Crest Recession Tracking",
        category="Recovery Operations",
        urgency=5.2,
        operational_value=6.7,
        time_sensitivity=4.8,
        duration_minutes=2.1,
        storage_gb=1.3,
        sensor_type="SAR / Radar",
        cloud_sensitivity=0.08,
        max_acceptable_cloud=80.0,
        target_lat=30.35,
        target_lon=-90.10,
        description="Floodwater drawdown rate and standing water pool monitoring in low-lying sectors."
    ),
]


# -------------------------------------------------------------------------
# Scenario 3: Agricultural Survey
# Coordinates centered around Central Valley / Midwest crop belt
# -------------------------------------------------------------------------
AGRICULTURAL_SCENARIO: List[ObservationRequest] = [
    ObservationRequest(
        target_id="AG-01",
        name="Crop Health & NDVI Phenology",
        category="Crop Monitoring",
        urgency=6.8,
        operational_value=9.2,
        time_sensitivity=7.0,
        duration_minutes=2.5,
        storage_gb=1.6,
        sensor_type="Multispectral (NDVI)",
        cloud_sensitivity=0.65,
        max_acceptable_cloud=20.0,
        target_lat=36.75,
        target_lon=-119.80,
        description="Calibrated vegetation index surveying canopy density across high-value orchard plots."
    ),
    ObservationRequest(
        target_id="AG-02",
        name="Drought Stress & Evapotranspiration",
        category="Water Management",
        urgency=7.5,
        operational_value=9.0,
        time_sensitivity=7.5,
        duration_minutes=2.2,
        storage_gb=1.3,
        sensor_type="Thermal IR",
        cloud_sensitivity=0.45,
        max_acceptable_cloud=35.0,
        target_lat=36.50,
        target_lon=-119.65,
        description="Surface temperature anomalies and plant water stress detection for irrigation efficiency."
    ),
    ObservationRequest(
        target_id="AG-03",
        name="Pest Infestation Outbreak Survey",
        category="Plant Pathology",
        urgency=8.5,
        operational_value=8.8,
        time_sensitivity=8.5,
        duration_minutes=2.8,
        storage_gb=1.9,
        sensor_type="High-Res Optical",
        cloud_sensitivity=0.70,
        max_acceptable_cloud=20.0,
        target_lat=36.90,
        target_lon=-120.05,
        description="Early necrotic spot detection and canopy defoliation spreading from field margins."
    ),
    ObservationRequest(
        target_id="AG-04",
        name="Soil Moisture Active Radar",
        category="Soil Science",
        urgency=6.0,
        operational_value=7.8,
        time_sensitivity=5.8,
        duration_minutes=2.0,
        storage_gb=1.1,
        sensor_type="L-Band SAR",
        cloud_sensitivity=0.05,
        max_acceptable_cloud=90.0,
        target_lat=36.35,
        target_lon=-119.50,
        description="Microwave dielectric profile measuring volumetric water content in root zone."
    ),
    ObservationRequest(
        target_id="AG-05",
        name="Harvest Readiness & Yield Forecast",
        category="Economic Forecast",
        urgency=5.8,
        operational_value=8.2,
        time_sensitivity=6.2,
        duration_minutes=2.4,
        storage_gb=1.5,
        sensor_type="Hyperspectral",
        cloud_sensitivity=0.55,
        max_acceptable_cloud=25.0,
        target_lat=37.10,
        target_lon=-120.25,
        description="Maturity stage index and seed head senescence estimation ahead of harvesting windows."
    ),
    ObservationRequest(
        target_id="AG-06",
        name="Illegal Aquifer Extraction Tracing",
        category="Regulatory Compliance",
        urgency=7.0,
        operational_value=7.6,
        time_sensitivity=6.0,
        duration_minutes=1.8,
        storage_gb=1.0,
        sensor_type="Thermal IR / Optical",
        cloud_sensitivity=0.40,
        max_acceptable_cloud=35.0,
        target_lat=36.20,
        target_lon=-119.30,
        description="Center-pivot irrigation nighttime thermal signatures against water rights ledger."
    ),
    ObservationRequest(
        target_id="AG-07",
        name="Cover Crop & Tillage Practice Audit",
        category="Sustainability",
        urgency=4.2,
        operational_value=6.5,
        time_sensitivity=3.5,
        duration_minutes=2.1,
        storage_gb=1.2,
        sensor_type="Multispectral",
        cloud_sensitivity=0.50,
        max_acceptable_cloud=30.0,
        target_lat=36.65,
        target_lon=-119.95,
        description="Verification of conservation tillage and carbon-retention cover crops for credit verification."
    ),
    ObservationRequest(
        target_id="AG-08",
        name="Canopy Height & Biomass Lidar Proxy",
        category="Structural Survey",
        urgency=5.0,
        operational_value=7.4,
        time_sensitivity=4.5,
        duration_minutes=2.7,
        storage_gb=1.7,
        sensor_type="Stereo Optical / SAR",
        cloud_sensitivity=0.30,
        max_acceptable_cloud=45.0,
        target_lat=37.25,
        target_lon=-120.40,
        description="Canopy surface height modeling and above-ground biomass quantification."
    ),
]


SCENARIO_REGISTRY: Dict[str, List[ObservationRequest]] = {
    "Wildfire Response": WILDFIRE_SCENARIO,
    "Flood Mapping": FLOOD_SCENARIO,
    "Agricultural Survey": AGRICULTURAL_SCENARIO,
}


def get_scenario_observations(scenario_name: str) -> List[ObservationRequest]:
    """Retrieve the observation dataset for a given scenario name."""
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid choices: {list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[scenario_name]
