import re

from career_match_agent.models.candidate import (
    EmploymentType,
    JobPreferences,
    SeniorityLevel,
    WorkMode
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def contains_term(text: str, term: str) -> bool:
    """Return whether a standalone preference term occurs in text."""
    pattern = rf"(?<!\w){re.escape(term.casefold())}(?!\w)"
    return re.search(pattern, text.casefold()) is not None

def merge_unique(existing: list[str], explicit: list[str]) -> list[str]:
    """Merge strings while preserving order and removing duplicates."""
    return list(dict.fromkeys([*existing, *explicit]))

# ---------------------------------------------------------------------------
# Work modes
# ---------------------------------------------------------------------------
WORK_MODE_TERMS: dict[WorkMode, tuple[str, ...]] = {WorkMode.REMOTE: ("remote",
                                                                      "fully remote",
                                                                      "work from home"),
                                                    WorkMode.HYBRID: ("hybrid",
                                                                      "hybrid work"),
                                                    WorkMode.ON_SITE: ("on-site",
                                                                       "onsite",
                                                                       "on site",
                                                                       "office based",
                                                                       "office-based")}


def detect_explicit_work_modes(preference_text: str) -> list[WorkMode]:
    """Detect work modes explicitly requested by the user."""
    detected: list[WorkMode] = []
    for mode, terms in WORK_MODE_TERMS.items():
        if any(contains_term(preference_text, term) for term in terms):
            detected.append(mode)
    return detected


def preserve_explicit_work_modes(*, preference_text: str, preferences: JobPreferences) -> JobPreferences:
    """
    Keep only work modes explicitly supported by the user text.

    This prevents the LLM from inventing remote/hybrid/on-site
    constraints.
    """
    explicit_modes = (detect_explicit_work_modes(preference_text))
    return preferences.model_copy(update={"work_modes": explicit_modes})


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
KNOWN_LOCATION_ALIASES: dict[str, str] = {"germany": "Germany",
                                          "deutschland": "Germany",
                                          "berlin": "Berlin",
                                          "munich": "Munich",
                                          "münchen": "Munich",
                                          "hamburg": "Hamburg",
                                          "cologne": "Cologne",
                                          "köln": "Cologne",
                                          "nuremberg": "Nuremberg",
                                          "nürnberg": "Nuremberg",
                                          "regensburg": "Regensburg"}


def detect_explicit_locations(preference_text: str) -> list[str]:
    """Detect explicitly named supported locations."""
    locations: list[str] = []
    for alias, canonical in (KNOWN_LOCATION_ALIASES.items()):
        if not contains_term(preference_text, alias):
            continue

        if canonical not in locations:
            locations.append(canonical)

    return locations


def preserve_explicit_locations(*, preference_text: str, preferences: JobPreferences) -> JobPreferences:
    """Restore locations explicitly stated by the user."""
    explicit_locations = (detect_explicit_locations(preference_text))
    merged_locations = merge_unique(preferences.locations, explicit_locations)
    return preferences.model_copy(update={"locations": merged_locations})


# ---------------------------------------------------------------------------
# Employment types
# ---------------------------------------------------------------------------
EMPLOYMENT_TYPE_TERMS: dict[EmploymentType, tuple[str, ...]] = {EmploymentType.FULL_TIME: ("full time",
                                                                                           "full-time",
                                                                                           "fulltime",
                                                                                           "vollzeit"),
                                                                EmploymentType.PART_TIME: ("part time",
                                                                                           "part-time",
                                                                                           "parttime",
                                                                                           "teilzeit"),
                                                                EmploymentType.INTERNSHIP: ("intern",
                                                                                            "internship",
                                                                                            "praktikum",
                                                                                            "praktikant",
                                                                                            "praktikantin"),
                                                                EmploymentType.CONTRACT: ("contract",
                                                                                          "contractor",
                                                                                          "freelance",
                                                                                          "freelancer",
                                                                                          "temporary",
                                                                                          "fixed term",
                                                                                          "fixed-term",
                                                                                          "befristet")}


def detect_explicit_employment_types(preference_text: str) -> list[EmploymentType]:
    """Detect employment types explicitly requested by the user."""
    detected: list[EmploymentType] = []
    for employment_type, terms in EMPLOYMENT_TYPE_TERMS.items():
        if any(contains_term(preference_text, term) for term in terms):
            detected.append(employment_type)

    return detected


def preserve_explicit_employment_types(*, preference_text: str, preferences: JobPreferences) -> JobPreferences:
    """Keep employment types explicitly supported by user text."""
    explicit_types = detect_explicit_employment_types(preference_text)
    return preferences.model_copy(update={"employment_types": explicit_types})


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------
SENIORITY_TERMS: dict[SeniorityLevel, tuple[str, ...]] = {SeniorityLevel.SENIOR: ("senior",
                                                                                  "sr",
                                                                                  "staff",
                                                                                  "principal",
                                                                                  "lead",
                                                                                  "head of",
                                                                                  "director"),
                                                          SeniorityLevel.INTERNSHIP: ("intern",
                                                                                      "internship",
                                                                                      "praktikum",
                                                                                      "praktikant"),
                                                          SeniorityLevel.ENTRY_LEVEL: ("entry level",
                                                                                       "entry-level",
                                                                                       "graduate",
                                                                                       "trainee",
                                                                                       "recent graduate"),
                                                          SeniorityLevel.JUNIOR: ("junior",
                                                                                  "jr"),
                                                          SeniorityLevel.MID_LEVEL: ("mid level",
                                                                                     "mid-level",
                                                                                     "middle level",
                                                                                     "intermediate")}


def detect_explicit_seniority_levels(preference_text: str) -> list[SeniorityLevel]:
    """Detect seniority levels explicitly requested by the user."""
    detected: list[SeniorityLevel] = []
    for seniority, terms in (SENIORITY_TERMS.items()):
        if any(contains_term(preference_text, term) for term in terms):
            detected.append(seniority)

    return detected


def preserve_explicit_seniority_levels(*, preference_text: str, preferences: JobPreferences) -> JobPreferences:
    """Use seniority levels supported by explicit user wording."""
    explicit_levels = (detect_explicit_seniority_levels(preference_text))
    return preferences.model_copy(update={"seniority_levels": explicit_levels})


# ---------------------------------------------------------------------------
# Complete preference validation
# ---------------------------------------------------------------------------
def validate_explicit_preferences(*, preference_text: str, preferences: JobPreferences) -> JobPreferences:
    """
    Validate structured LLM preferences against explicit user input.

    The LLM performs semantic interpretation, while this layer protects
    explicit deterministic constraints.
    """
    validated = preserve_explicit_work_modes(preference_text=preference_text, preferences=preferences)
    validated = preserve_explicit_locations(preference_text=preference_text, preferences=validated)
    validated = preserve_explicit_employment_types(preference_text=preference_text, preferences=validated)
    validated = preserve_explicit_seniority_levels(preference_text=preference_text, preferences=validated)
    return validated
