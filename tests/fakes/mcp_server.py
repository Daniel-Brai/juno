from mcp.server import MCPServer

server = MCPServer("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    server.run()
