# RAGV6 Chunk 数据库管理与一键切分

## 入口

- 管理页面：`http://127.0.0.1:3011/ragV6/chunk-manager/`
- 管理 API：`http://127.0.0.1:8014/admin/chunks/*`
- 命令行：`python chunk_pipeline.py <手册文件> --manual <规范名称>`

管理 API 复用 `/chat` 的 Bearer Token；浏览器只访问 Node 网关，由网关注入
Token。通过 `CHUNK_ADMIN_ENABLED=0` 可以完全关闭后端管理路由。

## 一键切分流程

1. 导入 Markdown、TXT、DOCX 或 PDF。
2. 解析 Markdown 标题树；纯文本采用保守标题推断。
3. 提取图片锚点并记录其字符位置。
4. 生成完整父章节 `section_chunks`。
5. 按段落、完整句子、步骤和目标长度生成 `retrieval_chunks`。
6. 自动识别警告、操作、故障、部件、规格、保修和图片标签。
7. 检查空块、父子关系、连续 ID、长度上限、短块和重复块。
8. 预览通过后事务式写入数据库；发布前自动生成完整备份。
9. 后台重建 BM25、FAISS 与元数据索引，成功后热切换线上引擎。

## 命令行示例

只生成预览：

```bash
python chunk_pipeline.py "待导入/智能咖啡机.md" \
  --manual "智能咖啡机手册" \
  --preview-json "preview.json"
```

发布并重建索引：

```bash
python chunk_pipeline.py "待导入/智能咖啡机.md" \
  --manual "智能咖啡机手册" \
  --publish \
  --rebuild-index
```

覆盖同名手册：

```bash
python chunk_pipeline.py "待导入/智能咖啡机-v2.md" \
  --manual "智能咖啡机手册" \
  --publish \
  --replace-existing \
  --rebuild-index
```

## 主要管理 API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/admin/chunks/status` | 数据库与索引状态 |
| GET | `/admin/chunks/manuals` | 手册清单 |
| GET | `/admin/chunks/manual/{name}` | 父章节和检索块详情 |
| POST | `/admin/chunks/preview` | 文件解析、切分和质量预览 |
| POST | `/admin/chunks/publish` | 备份并发布正式数据 |
| POST | `/admin/chunks/rebuild` | 后台重建并热切换索引 |
| GET | `/admin/chunks/jobs/{job_id}` | 查询任务进度 |
| GET | `/admin/chunks/backups` | 发布备份 |
| POST | `/admin/chunks/rollback` | 回滚备份 |
| POST | `/admin/chunks/search-test` | 只验证召回、不生成答案 |

## 数据一致性

管理程序不会只修改 `manual_sections`。一次正式发布会同步更新源 Markdown、
手册章节、父章节、检索块、catalog 和章节摘要。发布后索引状态变为
`pending_rebuild`，只有 BM25/FAISS 重建并热切换成功后才清空待发布标记。

如重建失败，旧的内存索引仍继续服务；数据文件保留“待重建”状态，管理界面
会明确显示，不会把“文件已写入”误报为“索引已上线”。
