# Weather MCP Server

A lightweight MCP server for querying current weather and short forecasts.
It uses Open-Meteo public APIs and does not require an API key.

## Tools

### `weather_now`

Get current weather for a city.

Parameters:

- `city` string, required. City name, such as `北京`, `Shanghai`, or `New York`.
- `language` string, optional, default `zh`. Language passed to Open-Meteo geocoding.
- `temperature_unit` string, optional, default `celsius`.

### `weather_forecast`

Get a 1-7 day weather forecast for a city.

Parameters:

- `city` string, required.
- `days` integer, optional, default `3`.
- `language` string, optional, default `zh`.
- `temperature_unit` string, optional, default `celsius`.

## Local Run

From this package directory:

```bash
python server.py
```

From the repository root:

```bash
python weather-mcp-server/server.py
```

## MCP Client Config

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather-mcp-server/server.py"]
    }
  }
}
```

## Smithery

This package includes `smithery.yaml` for platform-style publishing.

## Docker

Build from the repository root so the Dockerfile can copy the shared `app/`
implementation:

```bash
docker build -f weather-mcp-server/Dockerfile -t weather-mcp-server .
docker run --rm -i weather-mcp-server
```

## External APIs

- `https://geocoding-api.open-meteo.com/v1/search`
- `https://api.open-meteo.com/v1/forecast`

## License

MIT
