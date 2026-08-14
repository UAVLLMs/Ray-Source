# Chunk SQL 管理

当前 chunk 会同步到 `data/chunk-admin/chunks.sqlite`，用于自然语言只读查询。

原始 `retrieval_chunks.json`、`section_chunks.json` 和 `catalog.json` 仍是权威源。SQLite 只是管理镜像，不直接修改 JSON，也不绕过现有发布、备份和 BM25/Dense 索引重建流程。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/admin/chunk-sql/status` | 查看 SQLite 镜像状态 |
| POST | `/admin/chunk-sql/sync` | 从当前 JSON 同步镜像 |
| POST | `/admin/chunk-sql/plan` | 编译自然语言为白名单 SQL 并返回只读结果 |

3011 网关对应路径为 `/ragv6-api/chunk-sql/...`。

请求示例：

```json
{
  "instruction": "查询 Air Fryer 中带图片的警告 chunk",
  "limit": 10
}
```

支持手册、chunk 编号、标题或正文关键词、图片、警告、故障、步骤、部件、章节和数量统计等条件。危险指令会直接拒绝。当前版本不执行自然语言生成的 `UPDATE`、`DELETE` 或 DDL。
