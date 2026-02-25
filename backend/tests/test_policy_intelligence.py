from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.policy_intelligence import CompanyInput, PipelineOrchestrator


def test_pipeline_outputs_expected_sections():
    orchestrator = PipelineOrchestrator()
    payload = CompanyInput(
        company_name="Acme Mobility",
        ticker="ACME",
        company_description="Acme manufactures electric vehicles with battery and chip intensive production and global logistics.",
    )

    result = orchestrator.run(payload)

    assert result.company_understanding.industry == "Automobiles"
    assert len(result.commodity_dependency.commodities) >= 2
    assert result.impact_scoring.short_term_impact.score >= result.impact_scoring.long_term_impact.score
    assert any(node["label"] == "Policy" for node in result.graph.nodes)


def test_all_scores_are_in_range():
    orchestrator = PipelineOrchestrator()
    payload = CompanyInput(company_name="Generic Co", company_description="A diversified manufacturing and distribution business.")
    result = orchestrator.run(payload)

    for commodity in result.commodity_dependency.commodities:
        assert 0 <= commodity.dependency_score <= 1

    assert 0 <= result.commodity_dependency.logistics_dependency_score <= 1
    assert 0 <= result.impact_scoring.short_term_impact.score <= 1
    assert 0 <= result.impact_scoring.medium_term_impact.score <= 1
    assert 0 <= result.impact_scoring.long_term_impact.score <= 1
    assert 0 <= result.impact_scoring.overall_confidence <= 1
