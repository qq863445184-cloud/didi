from __future__ import annotations

import sys
from pathlib import Path

import anyio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.weather_mcp_server import create_weather_mcp_server


async def main_async() -> None:
    server = create_weather_mcp_server()

    tools = await server.list_tools()
    print("[weather mcp tools]")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    print("\n[weather_now]")
    result = await server.call_tool("weather_now", {"city": "北京"})
    print(_mcp_text(result))

    print("\n[weather_forecast]")
    result = await server.call_tool("weather_forecast", {"city": "上海", "days": 3})
    print(_mcp_text(result))


def _mcp_text(result) -> str:
    if isinstance(result, dict):
        return str(result)
    if isinstance(result, tuple) and result:
        return _mcp_text(result[0])
    return "\n".join(str(getattr(item, "text", item)) for item in result)


def main() -> None:
    anyio.run(main_async)


if __name__ == "__main__":
    main()
