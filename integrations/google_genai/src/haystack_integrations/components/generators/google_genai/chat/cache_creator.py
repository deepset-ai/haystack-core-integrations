# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from google.genai import types
from haystack import logging
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.common.google_genai.utils import _get_client

logger = logging.getLogger(__name__)


@component
class GoogleGenAICacheCreator:
    """
    Creates a Gemini context cache so that repeated `GoogleGenAIChatGenerator` requests can reuse cached tokens.

    This context is stored server-side by Google and can be reused across multiple `GoogleGenAIChatGenerator`
    calls until it expires. The returned `cache_name` is the reference to be passed to the `GoogleGenAIChatGenerator`
    via:

        `generation_kwargs={"cached_content": cache_name}`

    Context caching has a minimum input size per model, see https://ai.google.dev/gemini-api/docs/caching

    Typical minimums:
        1024 tokens for Flash models
        4096 for Pro models.

    If your content is too small, the API returns INVALID_ARGUMENT

    This component uses the same authentication as `GoogleGenAIChatGenerator`: set `GOOGLE_API_KEY` or
    `GEMINI_API_KEY`, or use `api="vertex"` with Application Default Credentials or API key.

    The cache content as a minimum size, depending on the model you use.  If your content is too small, the API will
    return an error. Flash models have a lower minimum (1024 tokens), while Pro models typically require at least
    4096 tokens  for caching. See: https://ai.google.dev/gemini-api/docs/caching?lang=python

    ### Usage example

    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator, GoogleGenAICacheCreator
    from haystack.dataclasses.chat_message import ChatMessage

    MUSIC_THEORY_CONTEXT = '''
    You are a patient and precise music theory teacher. You answer only from this reference. Be concise but accurate.

    ## Core concepts

    **Pitch and intervals**
    - Notes are named A–G; with sharps/flats we get 12 distinct pitches per octave (chromatic scale).
    - An *interval* is the distance between two notes. *Semitone* = 1 half step (e.g. C–C♯). *Tone* = 2 half steps (e.g. C–D).
    - Common intervals: unison (0), minor 2nd (1 semitone), major 2nd (2), minor 3rd (3), major 3rd (4), perfect 4th (5), tritone (6), perfect 5th (7), minor 6th (8), major 6th (9), minor 7th (10), major 7th (11), octave (12).
    - *Inversion*: flip the interval (e.g. 4th up → 5th down). Inversion pairs: 2nd↔7th, 3rd↔6th, 4th↔5th; major↔minor, perfect stays perfect; dim↔aug.

    **Scales**
    - *Major scale*: W–W–H–W–W–W–H (e.g. C major: C D E F G A B C). Tonic, supertonic, mediant, subdominant, dominant, submediant, leading tone.
    - *Natural minor*: same notes as major from the 6th degree (e.g. A minor relative to C major): W–H–W–W–H–W–W.
    - *Harmonic minor*: natural minor with raised 7th (e.g. A B C D E F G♯ A). Creates a leading tone and an augmented 2nd between ♭6 and ♯7.
    - *Melodic minor*: ascending raises 6 and 7; descending often reverts to natural minor.
    - *Circle of fifths*: key signatures. Moving clockwise (C→G→D→A…): add a sharp (or remove a flat). Counter-clockwise: add a flat (or remove a sharp). Order of sharps: F C G D A E B. Order of flats: B E A D G C F.

    **Chords**
    - *Triads*: root, 3rd, 5th. Major (M3 + m3), minor (m3 + M3), diminished (m3 + m3), augmented (M3 + M3).
    - In *roman numerals*, uppercase = major, lowercase = minor, ° = diminished, + = augmented. In C: I = C, ii = Dm, iii = Em, IV = F, V = G, vi = Am, vii° = Bdim.
    - *Seventh chords*: add a 7th. Dominant 7 (V7): major triad + minor 7th (e.g. G7). Major 7 (e.g. Cmaj7): major triad + major 7th. Minor 7 (e.g. Dm7): minor triad + minor 7th. Half-diminished (e.g. Bm7♭5): dim triad + minor 7th.
    - *Cadences*: authentic (V→I), plagal (IV→I), half (any→V), deceptive (V→vi).

    **Rhythm and meter**
    - *Beat*: steady pulse. *Meter*: grouping of beats (e.g. 4/4 = four quarter notes per bar).
    - *Note values*: whole, half, quarter, eighth, sixteenth (and dotted variants = 1.5× duration).
    - *Time signatures*: top = beats per bar, bottom = note value that gets one beat (4 = quarter, 8 = eighth). Common: 4/4, 3/4, 6/8 (two groups of three eighths).
    - *Syncopation*: emphasis on off-beats or weak beats.

    **Modes and colour**
    - *Modes* are scales that use the same seven notes as major but start on a different degree. Ionian = major (C to C); Dorian (D to D in C major, ♭3 ♭7); Phrygian (E to E, ♭2 ♭3 ♭6 ♭7); Lydian (F to F, ♯4); Mixolydian (G to G, ♭7); Aeolian = natural minor (A to A); Locrian (B to B, ♭2 ♭3 ♭5 ♭6 ♭7).
    - *Transposition*: moving a phrase or piece to another key; keep the same intervals. *Modulation*: changing key within a piece; common pivot chords link the old and new keys.

    **Form and texture**
    - *Phrase*: short unit (often 4 or 8 bars). *Period*: two phrases (e.g. antecedent–consequent).
    - *Texture*: monophonic (one line), homophonic (melody + chords), polyphonic (several independent lines).

    Answer only from this reference. If the question is outside it, say so and suggest rephrasing.
    '''

    # Create cache with the music theory context (~1024 tokens)
    cache_creator = GoogleGenAICacheCreator(model="gemini-2.5-flash")
    result = cache_creator.run(
        contents=[MUSIC_THEORY_CONTEXT],
        system_instruction="You are a music theory teacher. Answer only from the cached reference. Be clear and concise.",
        display_name="music-theory-teacher",
        ttl="3600s",
    )
    cache_name = result["cache_name"]

    # Ask questions about music theory
    chat_gen = GoogleGenAIChatGenerator(model="gemini-2.5-flash")
    response = chat_gen.run(
        messages=[
            ChatMessage.from_user("What is a dominant seventh chord, and how does it resolve?")
        ],
        generation_kwargs={"cached_content": cache_name},
    )
    print(response)
    > {'replies': [
        ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, 
        _content=[TextContent(text='A dominant 7th chord (V7) is a major triad with an added minor 7th. For example, G7 is a dominant 7th chord.\n\nIt commonly resolves to the tonic (V→I) in an authentic cadence, or to the submediant (V→vi) in a deceptive cadence.')], 
        _name=None, 
        _meta={
            'model': 'gemini-2.5-flash', 'finish_reason': 'stop', 
            'usage': {
                prompt_tokens': 1175, 'completion_tokens': 68, 'total_tokens': 1504, 'thoughts_token_count': 261, 
                'cached_content_token_count': 1161, 'prompt_token_count': 1175, 'candidates_token_count': 68, 
                'total_token_count': 1504, 
                'cache_tokens_details': [{'modality': 'TEXT', 'token_count': 1161}], 
                'prompt_tokens_details': [{'modality': 'TEXT', 'token_count': 1175}]}
            })]
        }
    """  # noqa: E501, RUF002

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        model: str = "gemini-2.0-flash-001",
    ):
        """
        :param api_key: Google API key (defaults to env GOOGLE_API_KEY / GEMINI_API_KEY).
        :param api: "gemini" for Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project for Vertex AI (required for Vertex with ADC).
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g. "us-central1").
        :param model: Model to create the cache for (use explicit version, e.g. gemini-2.0-flash-001).
        """

        self._client = _get_client(
            api_key=api_key, api=api, vertex_ai_project=vertex_ai_project, vertex_ai_location=vertex_ai_location
        )
        self._api_key = api_key
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model = model

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self._api_key.to_dict(),
            api=self._api,
            vertex_ai_project=self._vertex_ai_project,
            vertex_ai_location=self._vertex_ai_location,
            model=self._model,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleGenAICacheCreator":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @staticmethod
    def _contents_to_config_parts(contents: list[str]) -> list[types.Content]:
        """Build Google GenAI Content list from a list of text strings."""
        if not contents:
            return []
        parts = [types.Part(text=text) for text in contents if text.strip()]
        if not parts:
            return []
        return [types.Content(role="user", parts=parts)]

    @component.output_types(cache_name=str, expire_time=str, total_token_count=int)
    def run(
        self,
        contents: list[str],
        *,
        system_instruction: str | None = None,
        display_name: str | None = None,
        ttl: str = "3600s",
    ) -> dict[str, Any]:
        """
        Create a context cache with the given text contents.

        :param contents: List of text strings to cache. Must be large enough for the model's minimum, typically 1024
            tokens for Flash, 4096 for Pro.
        :param system_instruction: Optional system instruction to cache with the content.
        :param display_name: Optional human-readable name for the cache.
        :param ttl: Time-to-live, e.g. "3600s" (1 hour) or "86400s" (24 hours).
        :returns:

            A dict, with the following keys:

            ```python
            {
                "cache_name": "example_cache_name",
                "expire_time": "2023-12-31T23:59:59Z",  # ISO string format
                "total_token_count": 2048
            }
            ```

            "cache_name" is the value to be passed to `GoogleGenAIChatGenerator` for cache reuse, in
                `generation_kwargs={"cached_content": cache_name}`
            "expire_time" is the time when the cache will expire, in ISO string format.
            "total_token_count" is the total number of tokens in the cached content,
        """
        if not contents:
            msg = "contents must be a non-empty list of strings. Received empty list."
            raise ValueError(msg)

        config_contents = GoogleGenAICacheCreator._contents_to_config_parts(contents)

        if not config_contents:
            msg = "contents produced no valid text parts to cache. Ensure contents is a list of non-empty strings."
            raise ValueError(msg)

        config_kwargs: dict[str, Any] = {
            "contents": config_contents,
            "display_name": display_name or "haystack-cache",
            "system_instruction": system_instruction.strip()
            if system_instruction and system_instruction.strip()
            else None,
            "ttl": ttl,
        }

        config = types.CreateCachedContentConfig(**config_kwargs)

        try:
            cache = self._client.caches.create(model=self._model, config=config)
        except Exception as e:
            msg = (
                f"Failed to create Google GenAI cache for model {self._model} with config {config_kwargs}. "
                f"Exception: {e}"
            )
            logger.error(msg, exc_info=True)
            raise

        cache_name = getattr(cache, "name", None) or str(cache.name)
        expire_time = getattr(cache, "expire_time", None)

        if expire_time is not None and hasattr(expire_time, "isoformat"):
            expire_time = expire_time.isoformat()

        usage = getattr(cache, "usage_metadata", None)
        total_token_count = getattr(usage, "total_token_count", None) if usage is not None else None

        return {
            "cache_name": cache_name,
            "expire_time": expire_time,
            "total_token_count": total_token_count,
        }
