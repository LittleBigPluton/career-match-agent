import json
from bs4 import BeautifulSoup
from career_match_agent.providers.web.common import extract_jobposting_jsonld


def test_extract_jobposting_from_jsonld() -> None:
    payload = {"@context": "https://schema.org",
               "@type": "JobPosting",
               "title": "Machine Learning Engineer",
               "description": ("<p>Build machine-learning models.</p>"),
               "hiringOrganization": {"@type": "Organization", "name": "Example AI GmbH"}}

    html = f"""
    <html>
      <head>
        <script type="application/ld+json">
          {json.dumps(payload)}
        </script>
      </head>
      <body></body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser",)
    jobs, warnings = (extract_jobposting_jsonld(soup))
    assert len(jobs) == 1
    assert jobs[0]["title"] == ("Machine Learning Engineer")
    assert warnings == []


def test_extract_jobposting_from_graph() -> None:
    payload = {"@context": "https://schema.org", "@graph": [{"@type": "Organization", "name": "Example AI GmbH"}, {"@type": "JobPosting", "title": "Data Scientist"}]}
    html = f"""
                <script type="application/ld+json">
                {json.dumps(payload)}
                </script>
            """
    soup = BeautifulSoup(html, "html.parser")
    jobs, _ = (extract_jobposting_jsonld(soup))
    assert len(jobs) == 1
    assert jobs[0]["title"] == ("Data Scientist")


def test_invalid_jsonld_is_reported_as_warning() -> None:
    html = """
              <script type="application/ld+json">
              {invalid json
              </script>
           """
    soup = BeautifulSoup(html, "html.parser")
    jobs, warnings = (extract_jobposting_jsonld(soup))

    assert jobs == []
    assert len(warnings) == 1
