import re
from typing import Any

from career_match_agent.models.web_job import WebJobSource
from career_match_agent.providers.web.base import (
    BaseWebJobParser,
    DomSelectorConfiguration,
)


class StepStoneWebJobParser(BaseWebJobParser):
    """Parse user-supplied StepStone job pages."""
    source = WebJobSource.STEPSTONE
    parser_version = "stepstone-web-parser-v1"
    supported_domains = ("stepstone.de", "stepstone.com", "stepstone.at", "stepstone.be", "stepstone.nl")
    selectors = DomSelectorConfiguration(title=("[data-at='header-job-title']", "[data-testid='job-title']", "h1"),
                                         company=("[data-at='header-company-name']", "[data-testid='company-name']"),
                                         location=("[data-at='header-location']", "[data-testid='location']"),
                                         description=("[data-at='job-ad-content']", "[data-testid='job-ad-content']", "article"))

    def extract_external_id(self, *, source_url: str, payload: dict[str, Any] | None) -> str:
        match = re.search(r"--(\d+)(?:-inline)?\.html", source_url, flags=re.IGNORECASE)

        if match:
            return match.group(1)

        return super().extract_external_id(source_url=source_url, payload=payload)
