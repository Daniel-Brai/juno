from typing import ClassVar
from urllib.parse import urlsplit

from juno.domain.commands import FetchUrl
from juno.ports import FetchFailed, WebFetcher
from juno.tools.registry import Tool, ToolError


class WebTools:
    GUIDE: ClassVar[str] = "/llms.txt"
    GUIDES: ClassVar[tuple[str, ...]] = ("llms.txt", "llms-full.txt")

    def __init__(self, fetcher: WebFetcher) -> None:
        self._fetcher = fetcher
        self._guides: dict[str, bool] = {}

    def tools(self) -> list[Tool[FetchUrl]]:
        return [
            Tool(
                name="web_fetch",
                description=(
                    "Fetch a web page or text file and return it as Markdown, e.g. to read "
                    "documentation. Many sites publish /llms.txt, a plain index of their docs "
                    "written for models, and some /llms-full.txt with the full text; for "
                    "documentation, try those first. Needs the user's approval unless allowed."
                ),
                input_model=FetchUrl,
                handler=self.fetch,
                idempotent=True,
            )
        ]

    async def fetch(self, args: FetchUrl) -> str:
        try:
            page = await self._fetcher.fetch(args.url)
        except FetchFailed as error:
            raise ToolError(str(error)) from None

        text = page.text[: args.max_chars]
        more = len(page.text) - len(text)
        note = f"\n\n… {more} more characters; raise max_chars to read them" if more else ""
        return f"{page.url}\n\n{text}{note}{await self._guide_note(page.url)}"

    async def _guide_note(self, url: str) -> str:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"

        if parts.path.rsplit("/", 1)[-1] in self.GUIDES or origin in self._guides:
            return ""

        try:
            guide = await self._fetcher.fetch(origin + self.GUIDE)
        except FetchFailed:
            self._guides[origin] = False
            return ""

        found = guide.content_type.startswith("text/plain") or guide.content_type == "text/markdown"
        self._guides[origin] = found and bool(guide.text.strip())

        if not self._guides[origin]:
            return ""

        return f"\n\nThis site publishes {origin}{self.GUIDE}, an index of its docs for models."
