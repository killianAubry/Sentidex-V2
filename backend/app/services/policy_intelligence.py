from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp
from typing import Any

from pydantic import BaseModel, Field


class CompanyInput(BaseModel):
    company_name: str = Field(min_length=1)
    ticker: str | None = None
    company_description: str | None = None


class CompanyUnderstanding(BaseModel):
    company_name: str
    industry: str
    sectors: list[str]
    products: list[str]
    operations_type: list[str]
    summary: str


class CommodityDependencyItem(BaseModel):
    name: str
    dependency_score: float


class CommodityDependency(BaseModel):
    commodities: list[CommodityDependencyItem]
    logistics_dependency_score: float


class GeographicLocation(BaseModel):
    type: str
    country: str
    region: str
    lat: float
    lon: float
    importance_score: float


class GeographicOperations(BaseModel):
    locations: list[GeographicLocation]


class PolicyItem(BaseModel):
    policy_id: str
    country: str
    policy_type: str
    title: str
    status: str
    relevance: str
    relevance_score: float
    source_weight: float
    confidence_score: float
    published_at: str


class PolicyScan(BaseModel):
    policies: list[PolicyItem]


class ImpactWindow(BaseModel):
    score: float
    severity: str


class ImpactScoring(BaseModel):
    short_term_impact: ImpactWindow
    medium_term_impact: ImpactWindow
    long_term_impact: ImpactWindow
    overall_confidence: float


class RegionImpact(BaseModel):
    location: str
    operations: str
    key_dependencies: list[str]
    relevant_policies: list[str]
    impact: str
    reason: str
    time_horizon: str


class OperationalReport(BaseModel):
    company_overview: dict[str, Any]
    operational_regions: list[RegionImpact]


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphState(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[GraphEdge]


class RefreshPlan(BaseModel):
    policies: str = "daily"
    commodity_dependencies: str = "weekly"
    geographic_checks: str = "monthly"


class PipelineResult(BaseModel):
    company_input: CompanyInput
    company_understanding: CompanyUnderstanding
    commodity_dependency: CommodityDependency
    geographic_operations: GeographicOperations
    country_policy: PolicyScan
    impact_scoring: ImpactScoring
    operational_report: OperationalReport
    graph: GraphState
    refresh_plan: RefreshPlan
    generated_at: str


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _severity(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _time_decay_days(days_since: int, half_life_days: int = 120) -> float:
    return 0.5 ** (days_since / max(half_life_days, 1))


@dataclass
class PolicyIntelligenceContext:
    company: CompanyInput
    understanding: CompanyUnderstanding | None = None
    commodity_dependency: CommodityDependency | None = None
    geo_ops: GeographicOperations | None = None
    policy_scan: PolicyScan | None = None
    impact_scoring: ImpactScoring | None = None
    report: OperationalReport | None = None


class CompanyUnderstandingAgent:
    INDUSTRY_MAP = {
        "semiconductor": ("Semiconductors", ["Technology", "Electronics"]),
        "electric": ("Automobiles", ["Transportation", "Energy Transition"]),
        "oil": ("Energy", ["Energy", "Industrial"]),
        "pharma": ("Pharmaceuticals", ["Healthcare", "Biotechnology"]),
        "retail": ("Retail", ["Consumer", "Ecommerce"]),
    }

    def run(self, payload: CompanyInput) -> CompanyUnderstanding:
        description = (payload.company_description or "").lower()
        industry = "Diversified Industrials"
        sectors = ["Industrials"]
        for keyword, mapped in self.INDUSTRY_MAP.items():
            if keyword in description:
                industry, sectors = mapped
                break

        products = self._extract_products(description)
        operations_type = self._detect_ops_type(description)
        summary = payload.company_description or (
            f"{payload.company_name} operates in {industry} with emphasis on {', '.join(sectors)} segments."
        )
        return CompanyUnderstanding(
            company_name=payload.company_name,
            industry=industry,
            sectors=sectors,
            products=products,
            operations_type=operations_type,
            summary=summary,
        )

    @staticmethod
    def _extract_products(description: str) -> list[str]:
        lookup = {
            "battery": "batteries",
            "chip": "semiconductors",
            "vehicle": "vehicles",
            "software": "software",
            "cloud": "cloud services",
            "drug": "therapeutics",
            "logistics": "logistics services",
            "refining": "refined fuels",
        }
        products = [value for key, value in lookup.items() if key in description]
        return products or ["core products/services not specified"]

    @staticmethod
    def _detect_ops_type(description: str) -> list[str]:
        ops: list[str] = []
        if any(k in description for k in ["manufact", "factory", "assembly"]):
            ops.append("production")
        if any(k in description for k in ["mine", "upstream", "extraction"]):
            ops.append("upstream")
        if any(k in description for k in ["retail", "distribution", "consumer"]):
            ops.append("downstream")
        if not ops:
            ops.extend(["upstream", "downstream"])
        return ops


class CommodityDependencyAgent:
    COMMODITY_KEYWORDS = {
        "oil": ["fuel", "oil", "petro", "shipping"],
        "natural gas": ["gas", "power", "thermal"],
        "semiconductors": ["chip", "semiconductor", "compute"],
        "lithium": ["battery", "lithium", "ev"],
        "copper": ["copper", "wiring", "electrical"],
        "aluminum": ["aluminum", "lightweight", "aerospace"],
        "agriculture": ["crop", "food", "agriculture", "fertilizer"],
    }

    def run(self, understanding: CompanyUnderstanding) -> CommodityDependency:
        text = f"{understanding.summary} {' '.join(understanding.products)} {' '.join(understanding.operations_type)}".lower()
        commodities: list[CommodityDependencyItem] = []
        for commodity, keys in self.COMMODITY_KEYWORDS.items():
            hits = sum(1 for key in keys if key in text)
            if hits == 0:
                continue
            dependency = _clamp_score(0.3 + (hits / len(keys)) * 0.7)
            commodities.append(CommodityDependencyItem(name=commodity, dependency_score=dependency))

        if not commodities:
            commodities = [CommodityDependencyItem(name="oil", dependency_score=0.35)]

        logistics_dependency = _clamp_score(0.4 + (0.08 * len(understanding.operations_type)) + (0.05 * len(commodities)))
        return CommodityDependency(commodities=commodities, logistics_dependency_score=logistics_dependency)


class GeographicOperationsAgent:
    COUNTRY_COORDS = {
        "United States": (37.0902, -95.7129),
        "China": (35.8617, 104.1954),
        "Taiwan": (23.6978, 120.9605),
        "Germany": (51.1657, 10.4515),
        "India": (20.5937, 78.9629),
        "Mexico": (23.6345, -102.5528),
    }

    def run(self, understanding: CompanyUnderstanding, dependencies: CommodityDependency) -> GeographicOperations:
        locations = [
            GeographicLocation(
                type="headquarters",
                country="United States",
                region="North America",
                lat=self.COUNTRY_COORDS["United States"][0],
                lon=self.COUNTRY_COORDS["United States"][1],
                importance_score=0.75,
            )
        ]

        if any(c.name == "semiconductors" for c in dependencies.commodities):
            locations.append(self._location("manufacturing", "Taiwan", "East Asia", 0.82))
        if any(c.name in {"lithium", "copper", "aluminum"} for c in dependencies.commodities):
            locations.append(self._location("supplier", "China", "East Asia", 0.69))
        if dependencies.logistics_dependency_score > 0.6:
            locations.append(self._location("distribution", "Germany", "Europe", 0.58))
            locations.append(self._location("distribution", "Mexico", "North America", 0.52))

        # Keep highest importance location per (type, country) as a simple dedupe rule.
        deduped: dict[tuple[str, str], GeographicLocation] = {}
        for loc in locations:
            key = (loc.type, loc.country)
            if key not in deduped or deduped[key].importance_score < loc.importance_score:
                deduped[key] = loc
        return GeographicOperations(locations=list(deduped.values()))

    def _location(self, loc_type: str, country: str, region: str, score: float) -> GeographicLocation:
        lat, lon = self.COUNTRY_COORDS.get(country, (0.0, 0.0))
        return GeographicLocation(type=loc_type, country=country, region=region, lat=lat, lon=lon, importance_score=score)


class CountryPolicyAgent:
    POLICY_LIBRARY = {
        "United States": [
            ("trade", "Proposed tariff updates on strategic imports", "active_discussion", "high"),
            ("tax", "Draft changes to manufacturing tax incentives", "proposed", "medium"),
        ],
        "Taiwan": [
            ("export_controls", "Debate on advanced chip export controls", "draft", "high"),
            ("labor", "Skilled labor mobility reform", "proposed", "medium"),
        ],
        "China": [
            ("industrial_subsidy", "Expanded EV battery subsidy consultation", "active_discussion", "high"),
            ("environment", "Tightened emissions compliance for heavy industry", "draft", "medium"),
        ],
        "Germany": [
            ("environment", "Carbon border mechanism implementation phase", "active", "medium"),
            ("labor", "Collective labor framework revision", "proposed", "low"),
        ],
        "Mexico": [
            ("trade", "Cross-border customs digitalization reform", "draft", "medium"),
            ("tax", "Border manufacturing VAT revision", "proposed", "medium"),
        ],
    }

    SOURCE_WEIGHT = {
        "trade": 0.93,
        "environment": 0.86,
        "export_controls": 0.92,
        "industrial_subsidy": 0.84,
        "labor": 0.73,
        "tax": 0.8,
    }

    RELEVANCE_SCORE = {"low": 0.35, "medium": 0.62, "high": 0.86}

    def run(self, understanding: CompanyUnderstanding, geo_ops: GeographicOperations) -> PolicyScan:
        policies: list[PolicyItem] = []
        today = datetime.now(timezone.utc)
        for loc in geo_ops.locations:
            templates = self.POLICY_LIBRARY.get(loc.country, [])
            for index, template in enumerate(templates):
                policy_type, title, status, relevance = template
                days_since = 20 + index * 35
                decay = _time_decay_days(days_since)
                relevance_score = self.RELEVANCE_SCORE[relevance]
                source_weight = self.SOURCE_WEIGHT.get(policy_type, 0.7)
                confidence = _clamp_score(0.45 + (0.4 * source_weight) + (0.15 * decay))
                if understanding.industry == "Semiconductors" and policy_type == "export_controls":
                    relevance_score = _clamp_score(relevance_score + 0.08)
                policies.append(
                    PolicyItem(
                        policy_id=f"{loc.country[:3].upper()}-{policy_type[:4].upper()}-{index+1}",
                        country=loc.country,
                        policy_type=policy_type,
                        title=title,
                        status=status,
                        relevance=relevance,
                        relevance_score=relevance_score,
                        source_weight=source_weight,
                        confidence_score=confidence,
                        published_at=(today - timedelta(days=days_since)).date().isoformat(),
                    )
                )

        return PolicyScan(policies=policies)


class ImpactScoringEngine:
    RELEVANCE_MULTIPLIER = {"low": 0.5, "medium": 0.75, "high": 1.0}

    def run(self, dependencies: CommodityDependency, geo_ops: GeographicOperations, policies: PolicyScan) -> ImpactScoring:
        commodity_dependency = sum(c.dependency_score for c in dependencies.commodities) / max(len(dependencies.commodities), 1)
        loc_by_country = {loc.country: loc.importance_score for loc in geo_ops.locations}

        base_scores: list[float] = []
        confidences: list[float] = []
        for policy in policies.policies:
            location_importance = loc_by_country.get(policy.country, 0.5)
            policy_relevance = policy.relevance_score * self.RELEVANCE_MULTIPLIER[policy.relevance]
            confidence = _clamp_score((policy.confidence_score + policy.source_weight) / 2)
            score = commodity_dependency * location_importance * policy_relevance * confidence
            base_scores.append(score)
            confidences.append(confidence)

        aggregate = _clamp_score(sum(base_scores) / max(len(base_scores), 1))
        overall_confidence = _clamp_score(sum(confidences) / max(len(confidences), 1))

        short_term = _clamp_score(aggregate * 1.1)
        medium_term = aggregate
        long_term = _clamp_score(aggregate * 0.85)

        return ImpactScoring(
            short_term_impact=ImpactWindow(score=short_term, severity=_severity(short_term)),
            medium_term_impact=ImpactWindow(score=medium_term, severity=_severity(medium_term)),
            long_term_impact=ImpactWindow(score=long_term, severity=_severity(long_term)),
            overall_confidence=overall_confidence,
        )


class OperationalReportGenerator:
    def run(
        self,
        understanding: CompanyUnderstanding,
        dependencies: CommodityDependency,
        geo_ops: GeographicOperations,
        policies: PolicyScan,
        impact: ImpactScoring,
    ) -> OperationalReport:
        policies_by_country: dict[str, list[PolicyItem]] = {}
        for policy in policies.policies:
            policies_by_country.setdefault(policy.country, []).append(policy)

        top_dependencies = [commodity.name for commodity in sorted(dependencies.commodities, key=lambda c: c.dependency_score, reverse=True)[:3]]

        regions: list[RegionImpact] = []
        for location in sorted(geo_ops.locations, key=lambda item: item.importance_score, reverse=True):
            country_policies = policies_by_country.get(location.country, [])
            max_relevance = max((item.relevance_score for item in country_policies), default=0.3)
            severity = _severity(max_relevance * location.importance_score)
            policy_titles = [item.title for item in country_policies[:3]]

            regions.append(
                RegionImpact(
                    location=location.country,
                    operations=f"{location.type} operations in {location.region}",
                    key_dependencies=top_dependencies,
                    relevant_policies=policy_titles,
                    impact=severity,
                    reason=(
                        f"{location.country} location has importance {location.importance_score} and policy relevance {round(max_relevance, 3)}"
                    ),
                    time_horizon=(
                        "0-6 months" if impact.short_term_impact.severity == "high" else "6-18 months"
                    ),
                )
            )

        overview = {
            "company_name": understanding.company_name,
            "industry": understanding.industry,
            "core_operations": understanding.operations_type,
            "key_dependencies": top_dependencies,
        }
        return OperationalReport(company_overview=overview, operational_regions=regions)


class GraphBuilder:
    def build(
        self,
        company: CompanyInput,
        dependencies: CommodityDependency,
        geo_ops: GeographicOperations,
        policy_scan: PolicyScan,
    ) -> GraphState:
        nodes: list[dict[str, Any]] = [{"id": f"company::{company.company_name}", "label": "Company", "name": company.company_name}]
        edges: list[GraphEdge] = []

        for commodity in dependencies.commodities:
            node_id = f"commodity::{commodity.name}"
            nodes.append({"id": node_id, "label": "Commodity", "name": commodity.name})
            edges.append(
                GraphEdge(
                    source=f"company::{company.company_name}",
                    target=node_id,
                    edge_type="depends_on",
                    attributes={"dependency_score": commodity.dependency_score},
                )
            )

        country_nodes: set[str] = set()
        for index, location in enumerate(geo_ops.locations, start=1):
            facility_id = f"facility::{company.company_name}::{index}"
            country_id = f"country::{location.country}"
            nodes.append(
                {
                    "id": facility_id,
                    "label": "Facility",
                    "name": f"{location.type.title()} - {location.country}",
                    "importance_score": location.importance_score,
                }
            )
            edges.append(GraphEdge(source=f"company::{company.company_name}", target=facility_id, edge_type="located_in"))
            edges.append(GraphEdge(source=facility_id, target=country_id, edge_type="located_in"))
            if country_id not in country_nodes:
                nodes.append({"id": country_id, "label": "Country", "name": location.country})
                country_nodes.add(country_id)

        for policy in policy_scan.policies:
            policy_id = f"policy::{policy.policy_id}"
            country_id = f"country::{policy.country}"
            nodes.append(
                {
                    "id": policy_id,
                    "label": "Policy",
                    "name": policy.title,
                    "policy_type": policy.policy_type,
                    "relevance_score": policy.relevance_score,
                }
            )
            edges.append(GraphEdge(source=country_id, target=policy_id, edge_type="regulated_by"))

        for commodity in dependencies.commodities:
            for location in geo_ops.locations:
                edges.append(
                    GraphEdge(
                        source=f"commodity::{commodity.name}",
                        target=f"country::{location.country}",
                        edge_type="exposed_to",
                        attributes={"exposure": _clamp_score(commodity.dependency_score * location.importance_score)},
                    )
                )
        return GraphState(nodes=nodes, edges=edges)


class PipelineOrchestrator:
    def __init__(self) -> None:
        self.company_agent = CompanyUnderstandingAgent()
        self.dependency_agent = CommodityDependencyAgent()
        self.geo_agent = GeographicOperationsAgent()
        self.policy_agent = CountryPolicyAgent()
        self.impact_engine = ImpactScoringEngine()
        self.report_generator = OperationalReportGenerator()
        self.graph_builder = GraphBuilder()

    def run(self, company_input: CompanyInput) -> PipelineResult:
        understanding = self.company_agent.run(company_input)
        dependencies = self.dependency_agent.run(understanding)
        geo_ops = self.geo_agent.run(understanding, dependencies)
        policy_scan = self.policy_agent.run(understanding, geo_ops)
        impacts = self.impact_engine.run(dependencies, geo_ops, policy_scan)
        report = self.report_generator.run(understanding, dependencies, geo_ops, policy_scan, impacts)
        graph = self.graph_builder.build(company_input, dependencies, geo_ops, policy_scan)

        return PipelineResult(
            company_input=company_input,
            company_understanding=understanding,
            commodity_dependency=dependencies,
            geographic_operations=geo_ops,
            country_policy=policy_scan,
            impact_scoring=impacts,
            operational_report=report,
            graph=graph,
            refresh_plan=RefreshPlan(),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


class RefreshCoordinator:
    PERIODS = {
        "daily": timedelta(days=1),
        "weekly": timedelta(days=7),
        "monthly": timedelta(days=30),
    }

    def should_refresh(self, last_run_at: datetime, cadence: str) -> bool:
        period = self.PERIODS.get(cadence)
        if period is None:
            raise ValueError(f"Unknown cadence: {cadence}")
        return datetime.now(timezone.utc) - last_run_at >= period

    def next_refresh_at(self, last_run_at: datetime, cadence: str) -> datetime:
        period = self.PERIODS.get(cadence)
        if period is None:
            raise ValueError(f"Unknown cadence: {cadence}")
        return last_run_at + period


class InMemoryPipelineStore:
    def __init__(self) -> None:
        self._results: dict[str, PipelineResult] = {}

    def save(self, result: PipelineResult) -> None:
        self._results[result.company_input.company_name.lower()] = result

    def get(self, company_name: str) -> PipelineResult | None:
        return self._results.get(company_name.lower())

    def get_risk_summary(self, company_name: str) -> dict[str, Any] | None:
        result = self.get(company_name)
        if not result:
            return None
        return {
            "company_name": result.company_input.company_name,
            "short_term": result.impact_scoring.short_term_impact.model_dump(),
            "medium_term": result.impact_scoring.medium_term_impact.model_dump(),
            "long_term": result.impact_scoring.long_term_impact.model_dump(),
            "overall_confidence": result.impact_scoring.overall_confidence,
        }
