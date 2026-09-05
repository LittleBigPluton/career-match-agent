import json

import pytest

from career_match_agent.models.web_job import (
    WebJobExtractionStrategy,
    WebJobSource
)
from career_match_agent.providers.web.base import WebJobDocument
from career_match_agent.providers.web.glassdoor import GlassdoorWebJobParser
from career_match_agent.providers.web.indeed import IndeedWebJobParser
from career_match_agent.providers.web.linkedin import LinkedInWebJobParser
from career_match_agent.providers.web.stepstone import StepStoneWebJobParser
from career_match_agent.providers.web.base import UnsupportedWebJobSourceError
from career_match_agent.providers.web.registry import create_default_web_job_parser_registry

def create_job_html() -> str:
    payload = {"@context": "https://schema.org",
               "@type": "JobPosting",
               "identifier": {"@type": "PropertyValue", "value": "structured-123"},
               "title": ("Machine Learning Engineer"),
               "description": ("<p>Develop Python and PyTorch machine-learning systems.</p>"),
               "hiringOrganization": {"@type": "Organization", "name": "Example AI GmbH",},
               "jobLocation": {"@type": "Place", "address": {"@type": "PostalAddress", "addressLocality": "Berlin", "addressCountry": "DE"}},
               "employmentType": "FULL_TIME",
               "datePosted": "2026-08-20",
               "skills": ["Python", "PyTorch"]}

    return f"""
    <html>
      <head>
        <script type="application/ld+json">
          {json.dumps(payload)}
        </script>
      </head>
      <body></body>
    </html>
    """


@pytest.mark.parametrize(("parser", "source_url", "expected_source"),
                         [(LinkedInWebJobParser(), "https://www.linkedin.com/jobs/view/1234567890/", WebJobSource.LINKEDIN),
                          (IndeedWebJobParser(),"https://de.indeed.com/viewjob?jk=indeed123", WebJobSource.INDEED),
                          (StepStoneWebJobParser(), "https://www.stepstone.de/stellenangebote--example--123456-inline.html", WebJobSource.STEPSTONE),
                          (GlassdoorWebJobParser(), "https://www.glassdoor.com/partner/jobListing.htm?jobListingId=987654", WebJobSource.GLASSDOOR)])

def test_jsonld_parsing_across_sources(parser, source_url, expected_source) -> None:
    result = parser.parse(WebJobDocument(source_url=source_url, html=create_job_html(), content_sha256="0" * 64))
    assert parser.source == expected_source
    assert (result.strategy == WebJobExtractionStrategy.JSON_LD)
    assert result.job.title == ("Machine Learning Engineer")
    assert result.job.company == ("Example AI GmbH")
    assert result.job.location == ("Berlin, DE")
    assert "Python" in result.job.tags

def test_linkedin_dom_fallback() -> None:
    html = """
    <html>
      <body>
        <h1 class="top-card-layout__title">
          Junior AI Engineer
        </h1>

        <a class="topcard__org-name-link">
          Example AI GmbH
        </a>

        <span class="topcard__flavor--bullet">
          Berlin, Germany
        </span>

        <div class="show-more-less-html__markup">
          Build Python AI services.
          Full-time role.
        </div>
      </body>
    </html>
    """
    parser = LinkedInWebJobParser()
    result = parser.parse(WebJobDocument(source_url=("https://www.linkedin.com/jobs/view/1234567890/"), html=html, content_sha256="0" * 64))
    assert (result.strategy == WebJobExtractionStrategy.DOM_FALLBACK)
    assert result.job.title == ("Junior AI Engineer")
    assert result.job.company == ("Example AI GmbH")
    assert result.job.external_id == ("1234567890")

def test_registry_selects_indeed() -> None:
    registry = (create_default_web_job_parser_registry())
    parser = registry.get_parser("https://de.indeed.com/viewjob?jk=123")
    assert parser.source == (WebJobSource.INDEED)

def test_registry_rejects_unknown_source() -> None:
    registry = (create_default_web_job_parser_registry())
    with pytest.raises(UnsupportedWebJobSourceError):
        registry.get_parser("https://example.com/job/123")
