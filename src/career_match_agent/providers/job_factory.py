from collections.abc import Sequence

from career_match_agent.core.config import Settings
from career_match_agent.providers.adzuna import AdzunaJobProvider
from career_match_agent.providers.arbeitnow import ArbeitnowJobProvider
from career_match_agent.providers.base import JobProvider
from career_match_agent.providers.composite import CompositeJobProvider
from career_match_agent.providers.jooble import JoobleJobProvider


def create_job_provider(settings: Settings, *, provider_names: Sequence[str]) -> JobProvider:
    """Create one or more configured job providers."""
    providers: list[JobProvider] = []

    for provider_name in provider_names:
        if provider_name == "arbeitnow":
            providers.append(ArbeitnowJobProvider(base_url=settings.arbeitnow_base_url,
                                                  timeout_seconds=(settings.arbeitnow_timeout_seconds),
                                                  maximum_pages=(settings.arbeitnow_max_pages),
                                                  user_agent=settings.http_user_agent))

        elif provider_name == "adzuna":
            if not settings.adzuna_app_id:
                raise RuntimeError("Adzuna is selected but CAREER_MATCH_ADZUNA_APP_ID is not configured.")

            if not settings.adzuna_app_key:
                raise RuntimeError("Adzuna is selected but CAREER_MATCH_ADZUNA_APP_KEY is not configured.")

            providers.append(AdzunaJobProvider(base_url=settings.adzuna_base_url,
                                               country=settings.adzuna_country,
                                               app_id=settings.adzuna_app_id,
                                               app_key=settings.adzuna_app_key,
                                               timeout_seconds=(settings.adzuna_timeout_seconds),
                                               results_per_page=(settings.adzuna_results_per_page),
                                               maximum_requests=(settings.adzuna_max_requests_per_search),
                                               user_agent=settings.http_user_agent))

        elif provider_name == "jooble":
            if not settings.jooble_api_key:
                raise RuntimeError("Jooble is selected but CAREER_MATCH_JOOBLE_API_KEY is not configured.")

            providers.append(JoobleJobProvider(base_url=settings.jooble_base_url,
                                               api_key=settings.jooble_api_key,
                                               default_location=(settings.jooble_default_location),
                                               timeout_seconds=(settings.jooble_timeout_seconds),
                                               results_per_page=(settings.jooble_results_per_page),
                                               maximum_requests=(settings.jooble_max_requests_per_search),
                                               user_agent=settings.http_user_agent))

        else:
            raise RuntimeError(f"Unsupported job provider: {provider_name}.")

    if not providers:
        raise RuntimeError("At least one job provider must be selected.")

    if len(providers) == 1:
        return providers[0]

    return CompositeJobProvider(providers)
