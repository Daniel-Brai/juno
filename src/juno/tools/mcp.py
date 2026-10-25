from collections.abc import Sequence
from typing import ClassVar

from juno.domain.commands import RemoteArgs
from juno.domain.values import RemoteTool
from juno.ports import McpGateway
from juno.tools.registry import Tool, ToolError


class McpTools:
    PREFIX: ClassVar[str] = "mcp"
    MAX_NAME: ClassVar[int] = 64

    def __init__(self, gateway: McpGateway, remote: Sequence[RemoteTool]) -> None:
        self._gateway = gateway
        self._remote = list(remote)

    def tools(self) -> list[Tool[RemoteArgs]]:
        return [self._tool(remote) for remote in self._remote]

    @classmethod
    def name_for(cls, remote: RemoteTool) -> str:
        name = f"{cls.PREFIX}__{cls._safe(remote.server)}__{cls._safe(remote.name)}"
        return name[: cls.MAX_NAME]

    def _tool(self, remote: RemoteTool) -> Tool[RemoteArgs]:
        async def call(args: RemoteArgs) -> str:
            result = await self._gateway.call(remote.server, remote.name, args.values)

            if result.is_error:
                raise ToolError(result.text or f"{remote.name} failed")

            return result.text or "(no output)"

        description = remote.description or f"{remote.name} from the {remote.server} MCP server"
        return Tool(
            name=self.name_for(remote),
            description=f"[{remote.server}] {description}",
            input_model=RemoteArgs,
            handler=call,
            schema=remote.input_schema or {"type": "object"},
        )

    @staticmethod
    def _safe(text: str) -> str:
        return "".join(char if char.isalnum() or char in "_-" else "_" for char in text)
