from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

from mcp.server.fastmcp import FastMCP


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class HttpClient(Protocol):
    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Return parsed JSON for a GET request."""


class UrlLibHttpClient:
    """Small stdlib HTTP client so the server has minimal publish-time dependencies."""

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        with urlopen(f"{url}?{query}", timeout=self.timeout) as response:  # nosec B310 - fixed public API URLs
            return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class Location:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str


def create_weather_mcp_server(http_client: HttpClient | None = None) -> FastMCP:
    """Create a publishable weather MCP server backed by Open-Meteo."""
    client = http_client or UrlLibHttpClient()
    mcp = FastMCP("weather-query-mcp")

    @mcp.tool(
        name="weather_now",
        description="Get current weather for a city using Open-Meteo. No API key is required.",
    )
    def weather_now(city: str, language: str = "zh", temperature_unit: str = "celsius") -> str:
        """Return current weather for a city."""
        location = resolve_location(city, client, language=language)
        weather = fetch_weather(location, client, forecast_days=1, temperature_unit=temperature_unit)
        return format_current_weather(location, weather)

    @mcp.tool(
        name="weather_forecast",
        description="Get a short weather forecast for a city using Open-Meteo.",
    )
    def weather_forecast(
        city: str,
        days: int = 3,
        language: str = "zh",
        temperature_unit: str = "celsius",
    ) -> str:
        """Return a daily forecast for a city."""
        days = max(1, min(int(days), 7))
        location = resolve_location(city, client, language=language)
        weather = fetch_weather(location, client, forecast_days=days, temperature_unit=temperature_unit)
        return format_daily_forecast(location, weather)

    return mcp


def resolve_location(city: str, http_client: HttpClient, language: str = "zh") -> Location:
    """Resolve city text to coordinates via Open-Meteo geocoding."""
    if not city.strip():
        raise ValueError("city must not be empty")

    payload = http_client.get_json(
        GEOCODING_URL,
        {
            "name": city.strip(),
            "count": 1,
            "language": language,
            "format": "json",
        },
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Cannot resolve city: {city}")

    first = results[0]
    return Location(
        name=str(first.get("name", city)),
        country=str(first.get("country", "")),
        latitude=float(first["latitude"]),
        longitude=float(first["longitude"]),
        timezone=str(first.get("timezone", "auto")),
    )


def fetch_weather(
    location: Location,
    http_client: HttpClient,
    *,
    forecast_days: int,
    temperature_unit: str = "celsius",
) -> dict[str, Any]:
    """Fetch current weather and daily forecast from Open-Meteo."""
    return http_client.get_json(
        FORECAST_URL,
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": location.timezone or "auto",
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": forecast_days,
            "temperature_unit": temperature_unit,
        },
    )


def format_current_weather(location: Location, weather: dict[str, Any]) -> str:
    current = weather.get("current") or {}
    units = weather.get("current_units") or {}
    temperature_unit = units.get("temperature_2m", "C")
    wind_unit = units.get("wind_speed_10m", "km/h")
    return (
        f"{location.name}, {location.country} 当前天气："
        f"{_weather_code_name(current.get('weather_code'))}，"
        f"气温 {current.get('temperature_2m')} {temperature_unit}，"
        f"湿度 {current.get('relative_humidity_2m')}%，"
        f"风速 {current.get('wind_speed_10m')} {wind_unit}。"
    )


def format_daily_forecast(location: Location, weather: dict[str, Any]) -> str:
    daily = weather.get("daily") or {}
    dates = daily.get("time") or []
    max_t = daily.get("temperature_2m_max") or []
    min_t = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    precipitation = daily.get("precipitation_probability_max") or []

    lines = [f"{location.name}, {location.country} 天气预报："]
    for index, date in enumerate(dates):
        lines.append(
            "- "
            f"{date}: {_weather_code_name(_at(codes, index))}，"
            f"{_at(min_t, index)}-{_at(max_t, index)} C，"
            f"最高降水概率 {_at(precipitation, index)}%。"
        )
    return "\n".join(lines)


def _at(items: list[Any], index: int) -> Any:
    return items[index] if index < len(items) else "N/A"


def _weather_code_name(code: Any) -> str:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "未知"

    mapping = {
        0: "晴",
        1: "大部晴朗",
        2: "局部多云",
        3: "阴",
        45: "雾",
        48: "雾凇",
        51: "小毛毛雨",
        53: "中等毛毛雨",
        55: "大毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "小阵雨",
        81: "中等阵雨",
        82: "强阵雨",
        95: "雷暴",
    }
    return mapping.get(value, f"天气代码 {value}")


mcp = create_weather_mcp_server()


if __name__ == "__main__":
    mcp.run()
