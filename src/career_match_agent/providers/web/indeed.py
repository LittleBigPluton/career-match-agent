from typing import Any
from urllib.parse import (
    parse_qs,
    urlparse,
)

from career_match_agent.models.web_job import WebJobSource
from career_match_agent.providers.web.base import (
    BaseWebJobParser,
    DomSelectorConfiguration
)


class IndeedWebJobParser(BaseWebJobParser):
    """Parse user-supplied Indeed job pages."""
    source = WebJobSource.INDEED
    parser_version = "indeed-web-parser-v1"
    supported_domains = ("indeed.com", "indeed.de", "indeed.co.uk", "indeed.fr")
    selectors = DomSelectorConfiguration(title=("[data-testid='jobsearch-JobInfoHeader-title']", "h1.jobsearch-JobInfoHeader-title", "h1"),
                                         company=("[data-testid='inlineHeader-companyName']", "[data-company-name='true']",
                                                  ".jobsearch-InlineCompanyRating-companyHeader"),
                                         location=("[data-testid='job-location']", ".jobsearch-JobInfoHeader-subtitle div"),
                                         description=("#jobDescriptionText", ".jobsearch-jobDescriptionText"))

    def extract_external_id(self, *, source_url: str, payload: dict[str, Any] | None) -> str:
        query = parse_qs(urlparse(source_url).query)

        job_key = query.get("jk")
        if job_key and job_key[0]:
            return job_key[0]

        return super().extract_external_id(source_url=source_url, payload=payload)
