# Raysource Mobile App

Raysource 移动端采用同源可安装 Web App（PWA）方案。应用与桌面网页共用问答、图片上传、历史会话、流式进度和手册图片接口，不维护第二套业务逻辑。

## 用户入口

```text
https://raysource.cloud
```

手机浏览器打开后，使用系统菜单中的“添加到主屏幕”或页面内“安装应用”按钮。安装后以独立窗口启动，应用名称为 Raysource，方向优先使用竖屏。

## 发布资源

| 资源 | 作用 |
|---|---|
| `public/ragv6-ui/manifest.json` | 应用名称、启动地址、显示模式和图标声明 |
| `public/ragv6-ui/service-worker.js` | 静态壳缓存和网络失败回退 |
| `public/ragv6-ui/ray-source-icon-192.png` | 192x192 安装图标 |
| `public/ragv6-ui/ray-source-icon-512.png` | 512x512 安装图标 |
| `public/ragv6-ui/index.html` | PWA 元数据、安装入口和 Service Worker 注册 |
| `public/ragv6-ui/styles.css` | 桌面和手机响应式布局、安全区适配 |

## 更新策略

修改静态资源后递增 `service-worker.js` 中的 `CACHE_NAME`，例如 `raysource-shell-v2` 改为 `raysource-shell-v3`。发布后重启网页网关并检查：

```text
GET /manifest.json
GET /service-worker.js
GET /ragv6/ray-source-icon-192.png
GET /ragv6/ray-source-icon-512.png
```

四个资源都必须返回 HTTP 200。问答接口、历史接口和图片接口不进入 Service Worker 离线缓存，避免展示过期答案或会话数据。

## 运行验证

```powershell
node --check ragv6-standalone\web-client\server.js
node --check ragv6-standalone\web-client\public\ragv6-ui\app.js
node --check ragv6-standalone\web-client\public\ragv6-ui\service-worker.js
```

移动端重点检查：

- 左侧咨询/产品抽屉可以打开和关闭。
- 右侧执行明细不会遮挡输入框。
- 输入框、图片上传和发送按钮在软键盘弹出时仍可操作。
- 回答中的手册图片和来源链接可以点击。
- 横向滚动宽度不超过视口宽度。
- 网络暂时断开时只回退静态应用壳，不伪造问答结果。

## 业务接口

前端统一使用 Raysource 公共命名空间：

```text
/raysource-api/chat
/raysource-api/progress
/raysource-api/model-profile
/raysource-api/model-profile/switch
/raysource-api/translate
/raysource-api/account/*
/raysource-api/feedback
```

旧路径仅作为兼容映射保留，不应在新的移动端代码中继续使用。
