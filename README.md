# Weather MCP Server

这是一个用于 ModelScope MCP 广场发布的天气查询 MCP 服务。

服务代码位于：

```text
langgraph_minimal/weather-mcp-server/
```

## 服务配置信息

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["langgraph_minimal/weather-mcp-server/server.py"]
    }
  }
}
```

## 服务启动方式

```bash
python langgraph_minimal/weather-mcp-server/server.py
```

## MCP 工具

### `weather_now`

查询城市当前天气。

参数：

- `city`：城市名称，例如 `北京`、`上海`、`New York`
- `language`：可选，默认 `zh`
- `temperature_unit`：可选，默认 `celsius`

### `weather_forecast`

查询城市 1-7 天天气预报。

参数：

- `city`：城市名称
- `days`：预报天数，默认 `3`
- `language`：可选，默认 `zh`
- `temperature_unit`：可选，默认 `celsius`

## 外部依赖

本服务使用 Open-Meteo 公共 API：

- `https://geocoding-api.open-meteo.com/v1/search`
- `https://api.open-meteo.com/v1/forecast`

默认不需要 API Key。

## 本地验证

```bash
cd langgraph_minimal
python scripts/chapter10_weather_mcp_demo.py
```

## 发布包结构

```text
langgraph_minimal/weather-mcp-server/
├── README.md
├── LICENSE
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── smithery.yaml
└── server.py
```
