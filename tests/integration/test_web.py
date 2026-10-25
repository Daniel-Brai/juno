import httpx
import pytest
import respx

from juno.adapters.web import HttpFetcher
from juno.domain.values import ToolCall, WebPage
from juno.ports import FetchFailed
from juno.tools.permissions import Action, Mode, Permissions
from juno.tools.web import WebTools

pytestmark = pytest.mark.anyio

PAGE = """
<html><head><style>body { color: red }</style><script>track()</script></head>
<body><nav>Home | Docs</nav>
<main><h1>Install</h1><p>Run <code>uv add juno</code> then read the <a href="/guide">guide</a>.</p></main>
<footer>(c) somebody</footer></body></html>
"""


class TestHttpFetcher:
    @respx.mock
    async def test_html_becomes_markdown_without_page_chrome(self) -> None:
        respx.get("https://docs.example/start").mock(
            return_value=httpx.Response(301, headers={"Location": "https://docs.example/install"})
        )
        respx.get("https://docs.example/install").mock(return_value=httpx.Response(200, html=PAGE))

        page = await HttpFetcher().fetch("https://docs.example/start")

        assert page.url == "https://docs.example/install"
        assert page.text == "# Install\n\nRun `uv add juno` then read the [guide](/guide)."

    @respx.mock
    async def test_binary_and_oversized_responses_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        respx.get("https://x.example/logo.png").mock(
            return_value=httpx.Response(
                200, content=b"\x89PNG", headers={"content-type": "image/png"}
            )
        )
        respx.get("https://x.example/big.txt").mock(
            return_value=httpx.Response(200, text="x" * 2_000)
        )
        monkeypatch.setattr(HttpFetcher, "LIMIT_BYTES", 1_000)

        with pytest.raises(FetchFailed, match="image/png, not text"):
            await HttpFetcher().fetch("https://x.example/logo.png")

        with pytest.raises(FetchFailed, match="larger than"):
            await HttpFetcher().fetch("https://x.example/big.txt")


class Pages:
    def __init__(self, text: str, guides: frozenset[str] = frozenset()) -> None:
        self.text = text
        self.guides = guides
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> WebPage:
        self.fetched.append(url)

        if url.endswith("/llms.txt"):
            if url not in self.guides:
                raise FetchFailed(f"{url} answered 404")

            return WebPage(url=url, content_type="text/plain", text="# Docs\n- [Guide](/guide)")

        return WebPage(url=url, content_type="text/plain", text=self.text)


class TestWebTools:
    async def test_long_pages_are_cut_with_a_note_and_bad_urls_refused(self) -> None:
        [tool] = WebTools(Pages("a" * 1_200)).tools()

        cut = await tool.run(
            ToolCall(
                id="1", name="web_fetch", args={"url": "https://x.example", "max_chars": 1_000}
            )
        )
        refused = await tool.run(
            ToolCall(id="2", name="web_fetch", args={"url": "file:///etc/passwd"})
        )

        assert cut.content.endswith("… 200 more characters; raise max_chars to read them")
        assert refused.is_error and "http or https" in refused.content

    @pytest.mark.parametrize(
        ("mode", "action"), [(Mode.ASK, Action.ASK), (Mode.PLAN, Action.ALLOW)]
    )
    def test_fetching_asks_first_but_plan_mode_may_look_things_up(
        self, mode: Mode, action: Action
    ) -> None:
        call = ToolCall(id="1", name="web_fetch", args={"url": "https://x.example"})

        assert Permissions(mode=mode).decide(call, read_only=False)[0] is action

    async def test_a_sites_llms_txt_is_mentioned_once(self) -> None:
        pages = Pages("hello", guides=frozenset({"https://docs.example/llms.txt"}))
        [tool] = WebTools(pages).tools()

        first = await tool.run(
            ToolCall(id="1", name="web_fetch", args={"url": "https://docs.example/start"})
        )
        second = await tool.run(
            ToolCall(id="2", name="web_fetch", args={"url": "https://docs.example/other"})
        )
        elsewhere = await tool.run(
            ToolCall(id="3", name="web_fetch", args={"url": "https://plain.example/"})
        )

        assert first.content.endswith(
            "publishes https://docs.example/llms.txt, an index of its docs for models."
        )
        assert "llms.txt" not in second.content and "llms.txt" not in elsewhere.content
        assert pages.fetched.count("https://docs.example/llms.txt") == 1
