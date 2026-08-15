import re
from collections.abc import Iterable

from career_match_agent.models.candidate import (
    CandidateProfile,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import DetectedLanguageRequirement
from career_match_agent.services.job_normalizer import normalize_for_matching


ROLE_ALIASES: dict[str, set[str]] = {"machine learning engineer": {"machine learning engineer",
                                                                    "ml engineer",
                                                                    "ai engineer",
                                                                    "artificial intelligence engineer",
                                                                    "machine learning developer",
                                                                    "applied ai engineer",
                                                                    "generative ai engineer",
                                                                    "genai engineer",
                                                                    "llm engineer"},

                                     "data scientist": {"data scientist",
                                                        "applied data scientist",
                                                        "analytics scientist"},

                                     "machine learning scientist": {"machine learning scientist",
                                                                    "ml scientist",
                                                                    "applied scientist",
                                                                    "applied machine learning scientist",
                                                                    "applied ml scientist"},

                                     "applied ml scientist": {"applied ml scientist",
                                                              "applied machine learning scientist",
                                                              "machine learning scientist",
                                                              "ml scientist",
                                                              "applied scientist"}}


SENIORITY_PATTERNS: list[tuple[SeniorityLevel, tuple[str, ...]]] = [(SeniorityLevel.SENIOR,(r"\bsenior\b",
                                                                                            r"\bsr\.?\b",
                                                                                            r"\bstaff\b",
                                                                                            r"\bprincipal\b",
                                                                                            r"\blead\b",
                                                                                            r"\bhead of\b",
                                                                                            r"\bdirector\b")),

                                                                    (SeniorityLevel.INTERNSHIP,(r"\bintern\b",
                                                                                                r"\binternship\b",
                                                                                                r"\bpraktikum\b",
                                                                                                r"\bpraktikant")),

                                                                    (SeniorityLevel.ENTRY_LEVEL,(r"\bentry level\b",
                                                                                                 r"\bentry-level\b",
                                                                                                 r"\bgraduate\b",
                                                                                                 r"\btrainee\b",
                                                                                                 r"\brecent graduate\b")),

                                                                    (SeniorityLevel.JUNIOR,(r"\bjunior\b",
                                                                                            r"\bjr\.?\b")),

                                                                    (SeniorityLevel.MID_LEVEL,(r"\bmid level\b",
                                                                                               r"\bmid-level\b",
                                                                                               r"\bmiddle level\b",
                                                                                               r"\bintermediate\b"))]


REMOTE_PATTERNS = (r"\bfully remote\b",
                   r"\bremote first\b",
                   r"\bremote-first\b",
                   r"\bwork from home\b",
                   r"\bhome office\b",
                   r"\bremote work\b")

HYBRID_PATTERNS = (r"\bhybrid\b",
                   r"\bhybrid working\b",
                   r"\bhybrides arbeiten\b",
                   r"\bhybrid work\b")

ON_SITE_PATTERNS = (r"\bon[ -]?site\b",
                    r"\boffice based\b",
                    r"\boffice-based\b",
                    r"\bwork from the office\b",
                    r"\bvor ort\b")


LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {"English": ("english", "englisch"),
                                                "German": ("german", "deutsch"),
                                                "French": ("french", "franzosisch", "französisch"),
                                                "Spanish": ("spanish", "spanisch"),
                                                "Turkish": ("turkish", "turkisch", "türkisch"),
                                                "Italian": ("italian", "italienisch")}


LANGUAGE_REQUIREMENT_MARKERS = ("required",
                                "mandatory",
                                "must have",
                                "essential",
                                "requirement",
                                "requirements",
                                "erforderlich",
                                "voraussetzung",
                                "vorausgesetzt",
                                "notwendig",
                                "kenntnisse",
                                "sprachkenntnisse",
                                "fluent",
                                "fluency",
                                "business fluent",
                                "professional proficiency",
                                "native",
                                "mother tongue",
                                "fliessend",
                                "fließend",
                                "verhandlungssicher",
                                "muttersprache")


ALTERNATIVE_LANGUAGE_MARKERS = ("such as", "one of", "either", "for example", "e.g.")

LEVEL_PATTERNS: list[tuple[str, str]] = [(r"\bC2\b", "C2"),
                                         (r"\bC1\b", "C1"),
                                         (r"\bB2\b", "B2"),
                                         (r"\bB1\b", "B1"),
                                         (r"\bA2\b", "A2"),
                                         (r"\bA1\b", "A1"),
                                         (r"\b(native|mother tongue|muttersprache)\b", "native"),
                                         (r"\b(business fluent|verhandlungssicher)\b", "C1"),
                                         (r"\b(fluent|fluency|fliessend|fließend)\b", "C1"),
                                         (r"\bprofessional proficiency\b", "B2")]


LANGUAGE_LEVEL_VALUES: dict[str, int] = {"A1": 1,
                                         "A2": 2,
                                         "B1": 3,
                                         "B2": 4,
                                         "C1": 5,
                                         "C2": 6,
                                         "NATIVE": 7}


def text_matches_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    """Return whether any regular expression matches the text."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def create_job_searchable_text(job: JobPosting) -> str:
    """Combine relevant job fields for deterministic matching."""
    return "\n".join([job.title, job.company, job.location or "", " ".join(job.tags), job.description])


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Match phrases using complete normalized tokens."""
    normalized_text = normalize_for_matching(text)
    normalized_phrase = normalize_for_matching(phrase)

    if not normalized_text or not normalized_phrase:
        return False

    text_tokens = normalized_text.split()
    phrase_tokens = normalized_phrase.split()

    if not phrase_tokens:
        return False

    text_token_set = set(text_tokens)
    return all(token in text_token_set for token in phrase_tokens)


def role_terms(role: str) -> set[str]:
    """Return all deterministic aliases for a role family."""
    normalized_role = normalize_for_matching(role)
    for canonical_role, aliases in ROLE_ALIASES.items():
        normalized_canonical = normalize_for_matching(canonical_role)
        normalized_aliases = {normalize_for_matching(alias) for alias in aliases}
        if (normalized_role == normalized_canonical or normalized_role in normalized_aliases):
            return {canonical_role, *aliases, role}

    return {role}

def detect_matching_roles(job: JobPosting, preferred_roles: list[str]) -> list[str]:
    """Return preferred roles compatible with the job title."""
    title_and_tags = " ".join([job.title, " ".join(job.tags)])
    matching_roles: list[str] = []
    for role in preferred_roles:
        if any(contains_normalized_phrase(title_and_tags, candidate_term) for candidate_term in role_terms(role)):
            matching_roles.append(role)

    return matching_roles


def detect_seniority(job: JobPosting) -> SeniorityLevel | None:
    title = job.title
    for seniority, patterns in SENIORITY_PATTERNS:
        if text_matches_any_pattern(title, patterns):
            return seniority

    return None


def detect_work_modes(job: JobPosting) -> list[WorkMode]:
    """Detect supported working modes from provider data and text."""
    searchable_text = create_job_searchable_text(job)
    detected_modes: list[WorkMode] = []
    if job.remote is True:
        detected_modes.append(WorkMode.REMOTE)

    if text_matches_any_pattern(searchable_text, HYBRID_PATTERNS):
        detected_modes.append(WorkMode.HYBRID)

    if text_matches_any_pattern(searchable_text, REMOTE_PATTERNS):
        if WorkMode.REMOTE not in detected_modes:
            detected_modes.append(WorkMode.REMOTE)

    if text_matches_any_pattern(searchable_text, ON_SITE_PATTERNS):
        detected_modes.append(WorkMode.ON_SITE)

    if not detected_modes and job.remote is False:
        detected_modes.append(WorkMode.ON_SITE)

    return list(dict.fromkeys(detected_modes))


def detect_language_level(text: str) -> str | None:
    """Detect the strongest explicit language level in text."""
    detected_levels: list[tuple[int, str]] = []
    for pattern, level in LEVEL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            value = LANGUAGE_LEVEL_VALUES[level.upper()]
            detected_levels.append((value, level))

    if not detected_levels:
        return None

    return max(detected_levels, key=lambda item: item[0])[1]

def detect_language_level_for_language(line: str, language_aliases: tuple[str, ...]) -> str | None:
    """Detect a proficiency level associated with one specific language."""
    for alias in language_aliases:
        escaped_alias = re.escape(alias)
        cefr_patterns = [rf"\b{escaped_alias}\b[^.;,]{{0,20}}\b(A1|A2|B1|B2|C1|C2)\b", rf"\b(A1|A2|B1|B2|C1|C2)\b[^.;,]{{0,20}}\b{escaped_alias}\b"]
        for pattern in cefr_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()

        qualitative_patterns: list[tuple[str, str | None]] = [(rf"\bnative(?:\s+speaker)?(?:\s+in)?\s+" rf"\b{escaped_alias}\b", "native",),
                                                              (rf"\bfluent(?:\s+in)?\s+" rf"\b{escaped_alias}\b", "C1",),
                                                              (rf"\bbusiness[- ]fluent(?:\s+in)?\s+" rf"\b{escaped_alias}\b", "C1",),
                                                              (rf"\bconversational(?:\s+in)?\s+" rf"\b{escaped_alias}\b", None)]

        for pattern, level in qualitative_patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return level

    return None

def detect_language_requirements(job: JobPosting) -> list[DetectedLanguageRequirement]:
    """Detect explicit language requirements from a job posting."""
    searchable_text = create_job_searchable_text(job)
    detected_requirements: list[DetectedLanguageRequirement] = []
    detected_languages: set[str] = set()
    for line in searchable_text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        normalized_line = normalize_for_matching(cleaned_line)
        contains_alternative_marker = any(normalize_for_matching(marker) in normalized_line for marker in ALTERNATIVE_LANGUAGE_MARKERS)
        if contains_alternative_marker:
            continue

        for (language, aliases) in LANGUAGE_ALIASES.items():
            if language in detected_languages:
                continue

            language_is_present = any(contains_normalized_phrase(cleaned_line, alias) for alias in aliases)
            if not language_is_present:
                continue

            has_requirement_marker = any(contains_normalized_phrase(normalized_line, marker) for marker in LANGUAGE_REQUIREMENT_MARKERS)
            minimum_level = (detect_language_level_for_language(line, aliases))
            if (not has_requirement_marker and minimum_level is None):
                continue

            detected_requirements.append(DetectedLanguageRequirement(language=language, minimum_level=minimum_level, evidence=cleaned_line[:500]))
            detected_languages.add(language)

    return detected_requirements


def normalize_language_name(value: str) -> str:
    """Normalize a candidate language name."""
    normalized_value = normalize_for_matching(value)
    for language, aliases in LANGUAGE_ALIASES.items():
        normalized_aliases = {normalize_for_matching(alias) for alias in aliases}
        if normalized_value in normalized_aliases:
            return language

    return value.strip().title()


def normalize_language_level(value: str | None) -> str | None:
    """Normalize a candidate proficiency description."""
    if value is None:
        return None

    normalized_value = normalize_for_matching(value)
    direct_levels = {"a1": "A1",
                     "a2": "A2",
                     "b1": "B1",
                     "b2": "B2",
                     "c1": "C1",
                     "c2": "C2",
                     "native": "native",
                     "mother tongue": "native",
                     "muttersprache": "native",
                     "fluent": "C1",
                     "business fluent": "C1",
                     "professional proficiency": "B2"}

    if normalized_value in direct_levels:
        return direct_levels[normalized_value]

    return detect_language_level(value)


def build_candidate_language_map(profile: CandidateProfile, preferred_languages: list[str]) -> dict[str, str | None]:
    """Combine CV languages and accepted job languages."""
    languages: dict[str, str | None] = {}
    for language_entry in profile.languages:
        language_name = normalize_language_name(language_entry.language)
        languages[language_name] = normalize_language_level(language_entry.proficiency)

    for language in preferred_languages:
        language_name = normalize_language_name(language)
        languages.setdefault(language_name, None)

    return languages

def language_level_satisfies(candidate_level: str, required_level: str) -> bool:
    """Compare normalized candidate and required levels."""
    candidate_value = LANGUAGE_LEVEL_VALUES[candidate_level.upper()]
    required_value = LANGUAGE_LEVEL_VALUES[required_level.upper()]
    return candidate_value >= required_value
