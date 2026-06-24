from __future__ import annotations

import anyio

from app.weather_mcp_server import (
    FORECAST_URL,
    GEOCODING_URL,
    create_weather_mcp_server,
    fetch_weather,
    format_current_weather,
    format_daily_forecast,
    resolve_location,
)


class FakeWeatherHttpClient:
    def __init__(self):
        self.calls = []

    def get_json(self, url, params):
        self.calls.append((url, params))
        if url == GEOCODING_URL:
            return {
                "results": [
                    {
                        "name": "北京",
                        "country": "中国",
                        "latitude": 39.9042,
                        "longitude": 116.4074,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            }
        if url == FORECAST_URL:
            return {
                "current": {
                    "temperature_2m": 26,
                    "relative_humidity_2m": 45,
                    "weather_code": 0,
                    "wind_speed_10m": 8,
                },
                "current_units": {"temperature_2m": "C", "wind_speed_10m": "km/h"},
                "daily": {
                    "time": ["2026-06-24", "2026-06-25"],
                    "weather_code": [0, 61],
                    "temperature_2m_max": [31, 27],
                    "temperature_2m_min": [21, 20],
                    "precipitation_probability_max": [10, 70],
                },
            }
        raise AssertionError(f"unexpected url: {url}")


def test_weather_location_and_forecast_helpers():
    client = FakeWeatherHttpClient()

    location = resolve_location("北京", client)
    weather = fetch_weather(location, client, forecast_days=2)

    assert location.name == "北京"
    assert "当前天气：晴" in format_current_weather(location, weather)
    assert "2026-06-25: 小雨" in format_daily_forecast(location, weather)


def test_weather_mcp_exposes_tools_and_calls_current_weather():
    async def scenario():
        server = create_weather_mcp_server(http_client=FakeWeatherHttpClient())

        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        result = await server.call_tool("weather_now", {"city": "北京"})

        assert {"weather_now", "weather_forecast"}.issubset(names)
        assert "北京, 中国 当前天气：晴" in _mcp_text(result)

    anyio.run(scenario)


def test_weather_mcp_calls_forecast_tool():
    async def scenario():
        server = create_weather_mcp_server(http_client=FakeWeatherHttpClient())

        result = await server.call_tool("weather_forecast", {"city": "北京", "days": 2})

        assert "北京, 中国 天气预报" in _mcp_text(result)
        assert "最高降水概率 70%" in _mcp_text(result)

    anyio.run(scenario)


def _mcp_text(result) -> str:
    if isinstance(result, dict):
        return str(result)
    if isinstance(result, tuple) and result:
        return _mcp_text(result[0])
    return "\n".join(str(getattr(item, "text", item)) for item in result)
