# Weather MCP Server

This server exposes weather query tools through MCP. It uses Open-Meteo public
APIs by default, so no API key is required.

## Tools

- `weather_now(city, language="zh", temperature_unit="celsius")`
  - Resolve a city name and return current weather.
- `weather_forecast(city, days=3, language="zh", temperature_unit="celsius")`
  - Resolve a city name and return a 1-7 day forecast.

## Run Locally

From the project root:

```bash
python app/weather_mcp_server.py
```

Windows virtualenv example:

```powershell
.\.venv\Scripts\python.exe app\weather_mcp_server.py
```

## MCP Client Config

For a chapter demo, use the project-level server:

```json
{
  "command": "python",
  "args": ["app/weather_mcp_server.py"]
}
```

For platform-style publishing, use the package under `weather-mcp-server/`,
which follows the format shown in chapter 10.5.2:

```text
weather-mcp-server/
├── README.md
├── LICENSE
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── smithery.yaml
└── server.py
```

## Demo

From the project root:

```bash
python scripts/chapter10_weather_mcp_demo.py
```

Windows virtualenv example:

```powershell
.\.venv\Scripts\python.exe scripts\chapter10_weather_mcp_demo.py
```

## Publish Notes

- Transport: stdio by default through `FastMCP`.
- External APIs:
  - `https://geocoding-api.open-meteo.com/v1/search`
  - `https://api.open-meteo.com/v1/forecast`
- Secrets: none required by default.
- Tests use a fake HTTP client and do not require network access.
