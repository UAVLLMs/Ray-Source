# API 接口说明

## 基本信息

- 服务入口：`api_server.py`
- 启动脚本：`./run_api.sh`
- 健康检查：`GET /health`
- 对话接口：`POST /chat`
- 默认端口：`8000`
- 默认认证：`Authorization: Bearer sk-datafountain-demo-0608`

## 启动

```bash
pip install -r requirements.txt
./run_api.sh
```

如需自定义监听地址：

```bash
HOST=127.0.0.1 PORT=8000 ./run_api.sh
```

## GET /health

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok",
  "engine_ready": true,
  "timeout_s": 50.0,
  "multimodal_timeout_s": 60.0,
  "auth_configured": true,
  "classifier_provider": "deepseek_binary_vote",
  "classifier_configured": true,
  "classifier_model": "deepseek-v4-flash"
}
```

## POST /chat

### 请求头

```text
Authorization: Bearer sk-datafountain-demo-0608
Content-Type: application/json
```

### 请求体

```json
{
  "question": "椅子的扶手使用一段时间后为什么会松动？",
  "images": [],
  "session_id": "demo",
  "stream": false
}
```

字段说明：

| 字段 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题，不能为空。 |
| `images` | array[string] | 否 | Base64 data URL 图片，格式如 `data:image/png;base64,...`，最多 3 张。 |
| `session_id` | string | 否 | 会话 ID；不传时服务端自动生成。 |
| `stream` | boolean | 否 | `false` 返回完整 JSON；`true` 通过 SSE 返回阶段事件、文本增量和最终答案。 |

## 流式回答

可向原 `/chat` 传入 `"stream": true`，也可直接请求 `/chat/stream`。响应类型为 `text/event-stream`：

```text
event: status
data: {"stage":"retrieve","message":"正在执行手册预检索"}

event: delta
data: {"text":"启动健身单车前"}

event: done
data: {"answer":"完整校验后答案","pics":[],"route":"tech"}
```

技术题的 `delta` 是主模型实时草稿；`done.answer` 是经过证据选择和格式化后的最终答案，客户端结束时应以它为准。

### 调用示例

```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H 'Authorization: Bearer sk-datafountain-demo-0608' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "椅子的扶手使用一段时间后为什么会松动？",
    "session_id": "demo"
  }'
```

### 成功响应

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "...",
    "session_id": "demo",
    "timestamp": 1780000000
  }
}
```

其中 `data.answer` 与离线 CSV 提交 `ret` 字段同源：

- 客服题：纯文本客服回答。
- 技术题：正文 + `<PIC>` 锚点 + 末尾图片数组字符串。

## 错误响应

常见错误：

| HTTP 状态码 | 场景 |
| ---: | --- |
| 401 | 缺少 Bearer Token 或 Token 错误。 |
| 422 | 请求字段校验失败，如问题为空、图片格式错误、图片过大。 |
| 500 | Agent 或上游模型内部错误。 |
| 504 | Agent 超时。 |

## 配置覆盖

默认配置写在 `config_runtime.py`。如需覆盖，可在启动前设置同名环境变量，例如：

```bash
SILICONFLOW_BASE_URL="https://your-openai-compatible-endpoint/v1" \
SILICONFLOW_API_KEY="sk-..." \
SILICONFLOW_MODEL="gpt-5.5" \
./run_api.sh
```

默认 embedding / rerank 使用硅基流动远程服务，无需本地启动额外服务。
