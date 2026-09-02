from career_match_agent.providers.web.base import (
    ParsedWebJob,
    UnsupportedWebJobSourceError,
    WebJobDocument,
    WebJobParser
)
from career_match_agent.providers.web.glassdoor import GlassdoorWebJobParser
from career_match_agent.providers.web.indeed import IndeedWebJobParser
from career_match_agent.providers.web.linkedin import LinkedInWebJobParser
from career_match_agent.providers.web.stepstone import StepStoneWebJobParser


class WebJobParserRegistry:
    """Resolve job-page parsers by source URL."""

    def __init__(self, parsers: list[WebJobParser]) -> None:
        self.parsers = parsers

    def get_parser(self, source_url: str) -> WebJobParser:
        """Return the parser supporting a URL."""
        for parser in self.parsers:
            if parser.supports_url(source_url):
                return parser

        raise UnsupportedWebJobSourceError(f"No web-job parser supports the supplied URL: {source_url}")

    def parse(self, document: WebJobDocument) -> tuple[WebJobParser, ParsedWebJob]:
        """Resolve and execute the correct parser."""
        parser = self.get_parser(document.source_url)
        return (parser, parser.parse(document))


def create_default_web_job_parser_registry() -> WebJobParserRegistry:
    """Create the configured built-in parser set."""
    return WebJobParserRegistry(
        parsers=[LinkedInWebJobParser(), IndeedWebJobParser(), StepStoneWebJobParser(), GlassdoorWebJobParser()])
