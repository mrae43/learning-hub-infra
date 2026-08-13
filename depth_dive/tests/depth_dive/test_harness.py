"""Tests for the Depth Dive harness entrypoint (ADR-0020).

The assembly agent is mocked at the harness boundary (its own suite covers
grounding and search policy) so these tests exercise orchestration only:
transform -> framing -> assembly -> response assembly. The web-search provider
is the deterministic ``StubWebSearchClient``.
"""

import io
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import TypeAdapter

from core.clients import InMemoryEmbedder, MockCompletionProvider
from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.depth_dive import (
    HarnessBRequest,
    HarnessBResponse,
    InteractiveAnimation,
    Treatment,
)
from core.types.responses import CitedPassage
from core.types.retrieval_config import RetrievalConfig
from depth_dive import harness
from depth_dive.assembly.assembly_agent import AssemblyResult
from depth_dive.framing.framing_agent import FramingBrief
from depth_dive.generation.fallback_animation import build_fallback_animation
from depth_dive.harness import run_dive
from depth_dive.transform import PassageTransformError
from depth_dive.web_search.client import StubWebSearchClient, WebSearchResult
from depth_dive.web_search.wrapper import FALLBACK_NOTE

_VALID_TEXT = TextPassage(content="Attention is all you need.")
_captured_passage_adapter: TypeAdapter[CapturedPassage] = TypeAdapter(CapturedPassage)

_CONFIG = RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5)

_FALLBACK_ANIMATION = build_fallback_animation("test fallback")


def _completion() -> MockCompletionProvider:
    """A deterministic completion provider for the mocked-assembly tests."""
    return MockCompletionProvider("""{"output_type":"interactive_animation"}""")


@pytest.fixture
def patched_assembly(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``run_assembly`` to return an ungrounded result by default."""
    assembly = MagicMock(
        return_value=AssemblyResult(
            grounded=False,
            cited_passages=[],
            animation=_FALLBACK_ANIMATION,
        )
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)
    return assembly


def _dive(
    passage: CapturedPassage,
    *,
    web_search: StubWebSearchClient | None = None,
    completion_provider: MockCompletionProvider | None = None,
) -> HarnessBResponse:
    return run_dive(
        HarnessBRequest(captured_passage=passage),
        session=MagicMock(),
        embedder=InMemoryEmbedder(),
        config=_CONFIG,
        web_search=web_search or StubWebSearchClient(),
        completion_provider=completion_provider or _completion(),
    )


def _png_bytes() -> bytes:
    """Build a real PNG payload."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_image() -> ImagePassage:
    return ImagePassage(content=_png_bytes(), media_type="image/png")


def _valid_table() -> TablePassage:
    return TablePassage(rows=[["a", "1"], ["b", "2"]], headers=["l", "v"])


def test_run_dive_returns_hardcoded_animation(patched_assembly: MagicMock) -> None:
    """A valid text passage yields a HarnessBResponse with the demo animation."""
    response = _dive(_VALID_TEXT)
    assert isinstance(response, HarnessBResponse)
    assert isinstance(response.output, InteractiveAnimation)
    assert response.output.output_type == "interactive_animation"
    assert len(response.output.elements) > 0
    assert len(response.output.steps) > 0
    assert response.output.initial_state


def test_run_dive_mirrors_ungrounded_assembly_result(patched_assembly: MagicMock) -> None:
    """An ungrounded assembly result leaves the search flags off and cites nothing."""
    response = _dive(_VALID_TEXT)
    assert response.grounded is False
    assert response.cited_passages == []
    assert response.external_search_attempted is False
    assert response.external_search_failed is False
    assert response.external_search_note is None


def test_run_dive_mirrors_grounded_assembly_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grounded assembly result populates grounded and cited_passages."""
    passage_id = uuid4()
    assembly = MagicMock(
        return_value=AssemblyResult(
            grounded=True,
            cited_passages=[CitedPassage(chunk_id=passage_id, text="cited corpus chunk")],
            animation=_FALLBACK_ANIMATION,
        )
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)

    response = _dive(_VALID_TEXT)

    assert response.grounded is True
    assert len(response.cited_passages) == 1
    assert response.cited_passages[0].chunk_id == passage_id
    assert response.cited_passages[0].text == "cited corpus chunk"


def test_run_dive_mirrors_successful_search_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful search leaves the failed flag off and no note."""
    results = [
        WebSearchResult(title="Paper", url="https://example.com/paper", snippet="short quote")
    ]
    assembly = MagicMock(
        return_value=AssemblyResult(
            grounded=False,
            cited_passages=[],
            external_search_attempted=True,
            external_search_failed=False,
            external_search_results=results,
            animation=_FALLBACK_ANIMATION,
        )
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)

    response = _dive(_VALID_TEXT)

    assert response.external_search_attempted is True
    assert response.external_search_failed is False
    assert response.external_search_note is None


def test_run_dive_mirrors_failed_search_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """A double-failed search surfaces the failed flag and the user-facing note."""
    assembly = MagicMock(
        return_value=AssemblyResult(
            grounded=False,
            cited_passages=[],
            external_search_attempted=True,
            external_search_failed=True,
            external_search_note=FALLBACK_NOTE,
            animation=_FALLBACK_ANIMATION,
        )
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)

    response = _dive(_VALID_TEXT)

    assert response.external_search_attempted is True
    assert response.external_search_failed is True
    assert response.external_search_note == FALLBACK_NOTE


def test_run_dive_passes_web_search_client_to_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness hands the web-search provider to the assembly agent."""
    assembly = MagicMock(
        return_value=AssemblyResult(
            grounded=False,
            cited_passages=[],
            animation=_FALLBACK_ANIMATION,
        )
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)
    client = StubWebSearchClient()

    run_dive(
        HarnessBRequest(captured_passage=_VALID_TEXT),
        session=MagicMock(),
        embedder=InMemoryEmbedder(),
        config=_CONFIG,
        web_search=client,
        completion_provider=_completion(),
    )

    assert assembly.call_args.kwargs["web_search"] is client


def test_run_dive_passes_passage_and_brief_to_assembly(
    patched_assembly: MagicMock,
) -> None:
    """Assembly consumes the captured passage, the brief, and the providers."""
    session = MagicMock()
    embedder = InMemoryEmbedder()
    completion_provider = _completion()
    run_dive(
        HarnessBRequest(captured_passage=_VALID_TEXT),
        session=session,
        embedder=embedder,
        config=_CONFIG,
        web_search=StubWebSearchClient(),
        completion_provider=completion_provider,
    )

    patched_assembly.assert_called_once()
    call = patched_assembly.call_args
    assert call.args[0] == _VALID_TEXT
    assert isinstance(call.args[1], FramingBrief)
    assert call.kwargs["session"] is session
    assert call.kwargs["embedder"] is embedder
    assert call.kwargs["config"] is _CONFIG
    assert call.kwargs["completion_provider"] is completion_provider


def test_run_dive_recommends_and_applies_worked_example(patched_assembly: MagicMock) -> None:
    """The harness treats the passage as a worked_example; no routing overrides."""
    response = _dive(_VALID_TEXT)
    assert response.recommended_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.routing_note is None


def test_run_dive_mirrors_the_assembly_animation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness returns the assembly agent's generated scene graph as output."""
    first_animation = build_fallback_animation("first dive")
    assembly = MagicMock(
        return_value=AssemblyResult(grounded=False, cited_passages=[], animation=first_animation)
    )
    monkeypatch.setattr(harness, "run_assembly", assembly)

    response = _dive(_VALID_TEXT)

    assert response.output is first_animation
    assert response.output.model_dump() == first_animation.model_dump()


def test_run_dive_output_tracks_assembly_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different assembly animations yield different outputs (no hardcoded payload)."""
    first_animation = build_fallback_animation("first dive")
    second_animation = build_fallback_animation("second dive")
    monkeypatch.setattr(
        harness,
        "run_assembly",
        MagicMock(
            return_value=AssemblyResult(
                grounded=False,
                cited_passages=[],
                animation=first_animation,
            )
        ),
    )
    first = _dive(_VALID_TEXT)
    monkeypatch.setattr(
        harness,
        "run_assembly",
        MagicMock(
            return_value=AssemblyResult(
                grounded=False,
                cited_passages=[],
                animation=second_animation,
            )
        ),
    )
    second = _dive(TextPassage(content="A completely different passage."))

    assert first.output.model_dump() != second.output.model_dump()


def test_run_dive_validates_before_building(patched_assembly: MagicMock) -> None:
    """A passage violating text bounds is rejected, not silently built."""
    with pytest.raises(PassageTransformError):
        _dive(TextPassage(content=""))
    patched_assembly.assert_not_called()


def test_run_dive_accepts_image_passage(patched_assembly: MagicMock) -> None:
    """A valid image passage returns a valid HarnessBResponse."""
    response = _dive(_valid_image())
    assert isinstance(response, HarnessBResponse)
    assert response.output.output_type == "interactive_animation"


def test_run_dive_accepts_table_passage(patched_assembly: MagicMock) -> None:
    """A valid table passage returns a valid HarnessBResponse."""
    response = _dive(_valid_table())
    assert isinstance(response, HarnessBResponse)
    assert response.output.output_type == "interactive_animation"


def test_run_dive_accepts_code_passage(patched_assembly: MagicMock) -> None:
    """A valid code passage returns a valid HarnessBResponse."""
    response = _dive(CodePassage(content="def f():\n    return 1", language="python"))
    assert isinstance(response, HarnessBResponse)
    assert response.output.output_type == "interactive_animation"


def test_run_dive_rejects_invalid_image(patched_assembly: MagicMock) -> None:
    """An image that fails transform validation is rejected, not silently built."""
    with pytest.raises(PassageTransformError):
        _dive(ImagePassage(content=b"not an image", media_type="image/png"))
    patched_assembly.assert_not_called()


def test_run_dive_accepts_type_checked_union_payload(patched_assembly: MagicMock) -> None:
    """The union type parses request-shaped payloads before the harness runs."""
    passage = _captured_passage_adapter.validate_python(
        {"passage_type": "text", "content": "Typed union payload", "chunk_id": None}
    )
    response = _dive(passage)
    assert response.output.output_type == "interactive_animation"
