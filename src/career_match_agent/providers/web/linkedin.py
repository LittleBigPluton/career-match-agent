import re
from typing import Any

from career_match_agent.models.web_job import WebJobSource
from career_match_agent.providers.web.base import (
    BaseWebJobParser,
    DomSelectorConfiguration
)


class LinkedInWebJobParser(BaseWebJobParser):
    """Parse user-supplied LinkedIn job pages."""
    source = WebJobSource.LINKEDIN
    parser_version = "linkedin-web-parser-v1"
    supported_domains = ("linkedin.com",)
    selectors = DomSelectorConfiguration(title=("h1.top-card-layout__title", ".job-details-jobs-unified-top-card__job-title h1", ".jobs-unified-top-card__job-title h1", "h1"),
                                         company=(".topcard__org-name-link", ".top-card-layout__card a.topcard__flavor--black-link",
                                                 ".job-details-jobs-unified-top-card__company-name", ".jobs-unified-top-card__company-name"),
                                         location=(".topcard__flavor--bullet", ".job-details-jobs-unified-top-card__primary-description-container", ".jobs-unified-top-card__bullet"),
                                         description=( ".show-more-less-html__markup", ".jobs-description-content__text", ".jobs-box__html-content", ".jobs-description__content"))

    def extract_external_id(self, *, source_url: str, payload: dict[str, Any] | None) -> str:
        match = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)", source_url, flags=re.IGNORECASE)

        if match:
            return match.group(1)

        return super().extract_external_id(source_url=source_url, payload=payload)
