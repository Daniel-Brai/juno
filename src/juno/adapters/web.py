from typing import ClassVar

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from juno.domain.values import WebPage
from juno.ports import FetchFailed


class HttpFetcher:
    LIMIT_BYTES: ClassVar[int] = 2 * 1024**2
    NOISE: ClassVar[tuple[str, ...]] = (
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "svg",
        "iframe",
        "form",
    )
    TEXT: ClassVar[tuple[str, ...]] = ("text/", "application/json", "application/xml")
    AGENT: ClassVar[str] = "juno"
    TIMEOUT: ClassVar[httpx.Timeout] = httpx.Timeout(30, connect=10)

    async def fetch(self, url: str) -> WebPage:
        try:
            async with (
                httpx.AsyncClient(
                    follow_redirects=True, timeout=self.TIMEOUT, headers={"User-Agent": self.AGENT}
                ) as http,
                http.stream("GET", url) as response,
            ):
                response.raise_for_status()
                kind = response.headers.get("content-type", "").partition(";")[0].strip().lower()

                if not self._readable(kind):
                    raise FetchFailed(f"{url} is {kind or 'an unknown type'}, not text")

                body = await self._read(response)
                final = str(response.url)
        except httpx.HTTPStatusError as error:
            raise FetchFailed(f"{url} answered {error.response.status_code}") from error
        except httpx.HTTPError as error:
            raise FetchFailed(f"could not fetch {url}: {error}") from error

        text = body.decode(errors="replace")
        return WebPage(
            url=final, content_type=kind, text=self.markdown(text) if "html" in kind else text
        )

    @classmethod
    def markdown(cls, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(cls.NOISE):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.body or soup
        kept: list[str] = []

        for line in markdownify(str(main), heading_style="ATX").splitlines():
            if line.strip() or (kept and kept[-1]):
                kept.append(line.rstrip())

        return "\n".join(kept).strip()

    def _readable(self, kind: str) -> bool:
        return kind.startswith(self.TEXT) or kind.endswith(("+json", "+xml"))

    async def _read(self, response: httpx.Response) -> bytes:
        body = bytearray()

        async for chunk in response.aiter_bytes():
            body += chunk

            if len(body) > self.LIMIT_BYTES:
                raise FetchFailed(f"{response.url} is larger than {self.LIMIT_BYTES // 1024**2} MB")

        return bytes(body)
