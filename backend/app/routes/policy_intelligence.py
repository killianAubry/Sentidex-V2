from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.policy_intelligence import CompanyInput, InMemoryPipelineStore, PipelineOrchestrator

router = APIRouter(tags=["policy-intelligence"])

orchestrator = PipelineOrchestrator()
store = InMemoryPipelineStore()


@router.post("/analyze-company")
def analyze_company(payload: CompanyInput) -> dict:
    result = orchestrator.run(payload)
    store.save(result)
    return result.model_dump()


@router.get("/get-company-risk")
def get_company_risk(company_name: str = Query(..., min_length=1)) -> dict:
    risk = store.get_risk_summary(company_name)
    if not risk:
        raise HTTPException(status_code=404, detail="Company not found. Run /analyze-company first.")
    return risk


@router.get("/get-company-locations")
def get_company_locations(company_name: str = Query(..., min_length=1)) -> dict:
    result = store.get(company_name)
    if not result:
        raise HTTPException(status_code=404, detail="Company not found. Run /analyze-company first.")
    return {
        "company_name": result.company_input.company_name,
        "locations": [location.model_dump() for location in result.geographic_operations.locations],
    }


@router.get("/get-policy-impact")
def get_policy_impact(company_name: str = Query(..., min_length=1), country: str | None = None) -> dict:
    result = store.get(company_name)
    if not result:
        raise HTTPException(status_code=404, detail="Company not found. Run /analyze-company first.")

    policies = result.country_policy.policies
    if country:
        policies = [policy for policy in policies if policy.country.lower() == country.lower()]

    return {
        "company_name": result.company_input.company_name,
        "country": country,
        "policies": [policy.model_dump() for policy in policies],
    }


@router.get("/get-commodity-exposure")
def get_commodity_exposure(company_name: str = Query(..., min_length=1)) -> dict:
    result = store.get(company_name)
    if not result:
        raise HTTPException(status_code=404, detail="Company not found. Run /analyze-company first.")

    commodities = sorted(
        [commodity.model_dump() for commodity in result.commodity_dependency.commodities],
        key=lambda item: item["dependency_score"],
        reverse=True,
    )
    return {
        "company_name": result.company_input.company_name,
        "logistics_dependency_score": result.commodity_dependency.logistics_dependency_score,
        "commodities": commodities,
    }
