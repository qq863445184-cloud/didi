from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.weather_mcp_server import create_weather_mcp_server


mcp = create_weather_mcp_server()


if __name__ == "__main__":
    mcp.run()
