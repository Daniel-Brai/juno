import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client

from juno.config import McpServer
from juno.domain.values import RemoteResult, RemoteTool


class McpServers:
    def __init__(self, servers: Mapping[str, McpServer]) -> None:
        self._servers = dict(servers)
        self._clients: dict[str, Client[Any]] = {}
        self._stack = AsyncExitStack()

    async def connect(self) -> list[str]:
        problems: list[str] = []

        for name, server in self._servers.items():
            try:
                self._clients[name] = await self._stack.enter_async_context(
                    Client({"mcpServers": {name: self.spec(server)}})
                )
            except Exception as error:
                problems.append(f"MCP server {name} did not start: {error}")

        return problems

    async def tools(self) -> list[RemoteTool]:
        found: list[RemoteTool] = []

        for name, client in self._clients.items():
            for tool in await client.list_tools():
                found.append(
                    RemoteTool(
                        server=name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=dict(tool.input_schema),
                    )
                )

        return found

    async def call(self, server: str, name: str, args: dict[str, Any], /) -> RemoteResult:
        result = await self._clients[server].call_tool(name, args, raise_on_error=False)
        texts = [str(getattr(block, "text", "")) for block in result.content]
        return RemoteResult(
            text="\n".join(text for text in texts if text), is_error=result.is_error
        )

    async def close(self) -> None:
        await self._stack.aclose()

    @staticmethod
    def spec(server: McpServer) -> dict[str, Any]:
        if server.url is not None:
            return {"url": server.url}

        spec: dict[str, Any] = {"command": server.command, "args": list(server.args)}

        if server.env:
            values = {key: os.path.expandvars(value) for key, value in server.env.items()}
            spec["env"] = {**os.environ, **values}

        return spec
