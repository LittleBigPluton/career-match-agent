from typing import Any
from urllib.parse import (
    parse_qs,
    urlparse,
)

from career_match_agent.models.web_job import WebJobSource
from career_match_agent.providers.web.base import (
    BaseWebJobParser,
    DomSelectorConfiguration,
)


class GlassdoorWebJobParser(BaseWebJobParser):
    """Parse user-supplied Glassdoor job pages."""
    source = WebJobSource.GLASSDOOR
    parser_version = "glassdoor-web-parser-v1"
    supported_domains = ("glassdoor.com", "glassdoor.de", "glassdoor.co.uk")
    selectors = DomSelectorConfiguration(title=("[data-test='job-title']", "[data-testid='job-title']", "h1"),
                                         company=("[data-test='employer-name']", "[data-testid='employer-name']", ".EmployerProfile_employerName__Xemli"),
                                         location=("[data-test='location']", "[data-testid='location']"),
                                         description=("[data-test='jobDescriptionContent']", "[data-testid='job-description']", ".JobDetails_jobDescription__uW_fK"))

    def extract_external_id(self, *, source_url: str, payload: dict[str, Any] | None) -> str:
        query = parse_qs(urlparse(source_url).query)
        for key in ("jobListingId", "jl"):
            values = query.get(key)

            if values and values[0]:
                return values[0]

        return super().extract_external_id(source_url=source_url, payload=payload)
