"""Admin endpoints for provider API keys and the indexing model (/api/settings/ai)."""
from typing import Callable, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app_settings import (
    PAGE_CONTENT_CHARS_RANGE,
    PROVIDERS,
    TOP_PAGES_RANGE,
    AppSettings,
    SettingsError,
    provider_of,
)
from model_catalog import InvalidKeyError, ModelCatalog, ProviderError


class ProviderKeyRequest(BaseModel):
    apiKey: str = Field(min_length=1, max_length=500)


class AiSettingsPatch(BaseModel):
    indexingMode: Optional[Literal["local", "llm"]] = None
    indexingModel: Optional[str] = Field(default=None, max_length=200)


def create_ai_settings_router(settings: AppSettings, catalog: ModelCatalog, admin_dependency: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/settings/ai", dependencies=[Depends(admin_dependency)])

    def require_provider(provider: str) -> None:
        if provider not in PROVIDERS:
            raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    @router.get("")
    def get_ai_settings():
        return settings.view()

    @router.patch("")
    def patch_ai_settings(req: AiSettingsPatch):
        try:
            settings.update_indexing(mode=req.indexingMode, model=req.indexingModel)
        except SettingsError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return settings.view()

    @router.get("/models")
    def get_models(refresh: bool = False):
        keys = {provider: settings.effective_key(provider) for provider in PROVIDERS}
        return catalog.list_all({provider: key for provider, key in keys.items() if key}, refresh=refresh)

    @router.put("/providers/{provider}")
    def put_provider_key(provider: str, req: ProviderKeyRequest):
        require_provider(provider)
        api_key = req.apiKey.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required.")
        try:
            catalog.validate_key(provider, api_key)
        except InvalidKeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        settings.set_provider_key(provider, api_key)
        catalog.invalidate(provider)
        return settings.provider_status(provider)

    @router.delete("/providers/{provider}")
    def delete_provider_key(provider: str):
        require_provider(provider)
        settings.delete_provider_key(provider)
        catalog.invalidate(provider)
        return settings.provider_status(provider)

    return router


class RetrievalSettingsPatch(BaseModel):
    topPages: Optional[int] = Field(default=None, ge=TOP_PAGES_RANGE[0], le=TOP_PAGES_RANGE[1])
    reranker: Optional[Literal["off", "bge"]] = None
    answerMode: Optional[Literal["extractive", "llm"]] = None
    pageContentChars: Optional[int] = Field(default=None, ge=PAGE_CONTENT_CHARS_RANGE[0], le=PAGE_CONTENT_CHARS_RANGE[1])


def create_retrieval_settings_router(
    settings: AppSettings,
    admin_dependency: Callable,
    reranker_status: Callable[[], tuple[bool, Optional[str]]],
) -> APIRouter:
    """Admin endpoints for how queries retrieve and answer (/api/settings/retrieval)."""
    router = APIRouter(prefix="/api/settings/retrieval", dependencies=[Depends(admin_dependency)])

    def view() -> dict:
        bge_ok, bge_reason = reranker_status()
        _, model = settings.get_indexing()
        provider = provider_of(model or "")
        llm_ok = bool(provider and settings.effective_key(provider))
        return {
            **settings.retrieval_view(),
            "available": {
                "reranker": {"bge": bge_ok, "reason": bge_reason},
                "llmAnswers": {
                    "enabled": llm_ok,
                    "model": model,
                    "reason": None if llm_ok else "Set up a model with an API key in AI / LLM first.",
                },
            },
        }

    @router.get("")
    def get_retrieval_settings():
        return view()

    @router.patch("")
    def patch_retrieval_settings(req: RetrievalSettingsPatch):
        if req.reranker == "bge":
            bge_ok, bge_reason = reranker_status()
            if not bge_ok:
                raise HTTPException(status_code=400, detail=f"BGE re-ranking can't run here: {bge_reason}")
        try:
            settings.update_retrieval(
                top_pages=req.topPages,
                reranker=req.reranker,
                answer_mode=req.answerMode,
                page_content_chars=req.pageContentChars,
            )
        except SettingsError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return view()

    return router
