# GPT 统一联网后端安装说明

## MaiBot 1.4.0：DeepSeek 兼容接口

**v0.2.3** 使用 MaiBot 原生的 `deepseek` 适配器。启动配置只接受内置适配器类型，桥接插件在本机提供 `/v1/tools/search` 和 `/v1/tools/fetch`，继续使用 GPT 搜索和网页正文提取。该配置可在原版主程序使用。[1.4.0 配置校验](https://github.com/Mai-with-u/MaiBot/blob/v1.4.0/src/web_search/config.py) · [1.4.0 DeepSeek 适配器](https://github.com/Mai-with-u/MaiBot/blob/v1.4.0/src/web_search/adapters/deepseek.py)

1. 更新桥接插件并安装 `requirements.txt` 中的依赖。
2. 用 `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'` 生成本机接口密钥。在桥接插件的 `config.toml` 合并以下设置，保留实际使用的搜索模型。

   ```toml
   [web]
   enabled = true
   model = "YOUR_VISIBLE_MODEL_SLUG"
   deepseek_port = 18765
   deepseek_api_key = "<本机随机密钥>"
   ```

3. 在实例 `config/model_config.toml` 替换旧的 `gpt_builtin` 服务表，填入同一密钥并保持选中该服务。

   ```toml
   [web.services.gpt_builtin]
   adapter = "deepseek"
   base_url = "http://127.0.0.1:18765"
   api_key = "<本机随机密钥>"
   timeout = 270.0

   [model_task_config.web_search]
   model_list = ["gpt_builtin"]
   ```

4. 启动 MaiBot。接口随桥接插件启动，只监听 `127.0.0.1`。端口可更换，两处配置必须一致。MaiBot 的 DeepSeek 适配器限制单条检索词为 **128 个字符**，网页正文通过独立的抓取接口读取。

### 旧版补丁方案

**v0.2.1** 为现有桥接插件新增公开 `search` / `fetch` API，并附带 MaiBot 统一联网后端补丁。
聊天中的 `web_search`、麦生活的新闻阅读与主动探索均可通过同一配置使用 GPT 内置搜索。
网页读取使用独立无凭据客户端，返回实际静态网页正文。

## 兼容范围

- 桥接插件保留原插件 ID、数据目录、OAuth 授权和 LLM Provider；新增 API 在 SDK 2.16.0 验证。
- 配套主程序源码基线为 **MaiBot 1.3.19**，提交 `c6429547af711555e3d4d1b8d04b413572cb616c`。
- 安装器逐文件核对基线哈希；目标存在其他改动时整体拒绝安装。版本号相同也应先执行 `--check`。
- 麦生活 **1.24.0+** 使用 `ctx.web`，无需修改其业务源码；本次核对版本为 1.24.6。
- 仍使用自有搜索链的旧麦生活或 MaiLifeSuite 不会自动切换，应先迁移到使用宿主 `ctx.web` 的版本。
- 本地源码、模拟上游与静态网页读取已验证；真实账号联网权限与完整 QQ 消息流须在部署环境验收。

## 1. 更新桥接插件

停止 MaiBot，将安装包解压到独立目录，从解压后的插件目录运行，使用 MaiBot 的 Python 环境：

```bash
python tools/install_update.py --target /实际/MaiBot/plugins/ChatGPT-MaiBot-connect-plugin
python -m pip install -r /实际/MaiBot/plugins/ChatGPT-MaiBot-connect-plugin/requirements.txt
```

首次安装可直接把插件目录放入 `plugins/` 后安装依赖。更新脚本保留原 `config.toml`、账号和 host ID。
本次新增 `beautifulsoup4` 依赖。主程序补丁单独安装，避免只更新插件后误以为主程序后端已接入。

## 2. 安装主程序后端

从解压后的插件目录运行，目标可以是 **MaiBot 源码目录或实例目录**：

```bash
python tools/install_host_backend.py --target /实际/MaiBot --check
python tools/install_host_backend.py --target /实际/MaiBot
```

源码布局更新 **7 个文件**；uv/pip 包安装布局只更新 **4 个 Python 后端文件**，跳过包中没有的面板源码与 README。安装器逐文件检查原始哈希并备份到主程序位置的 `.gpt-web-backups/`。
实例目录（如 `~/MaiBotInstance`）只保存配置、插件和数据，主程序可能位于 uv 工具环境的 `site-packages`。安装器会识别当前 `maibot` 启动器与虚拟环境；检测到多个可用环境时，用 `--python /实际/启动MaiBot的Python` 明确指定。实例中遗留的旧 `.venv` 不会被当作唯一依据。

包安装环境可能被多个实例共用，补丁作用于该环境；升级或重装 MaiBot 后需重新检查。原有 `config/`、插件目录和数据目录不在修改范围。发生写入失败时尝试自动恢复。
重复安装已应用的相同版本会跳过。校验失败时，根据 `host_backend/unified-web-backend.patch` 人工移植；不要强行覆盖不同版本源码。

## 3. 设置桥接插件

在桥接插件已有 `config.toml` 保留原 `[plugin]` 配置，新增：

```toml
[web]
enabled = true
model = "YOUR_VISIBLE_MODEL_SLUG"
fetch_timeout_seconds = 30
```

`model` 必须替换为同一授权账号的 `models` 命令输出。这个模型专门用于搜索，聊天模型可以继续使用原配置。

```bash
python -m chatgpt_maibot --data-dir /实际/MaiBot/data/plugins/local.chatgpt-maibot-connect models
```

专用搜索与 `[plugin].enable_web_search` 独立。通常保留后者为 `false`，由主程序决定何时调用联网工具。
开启 `[web]` 不会在插件加载时发起搜索，也不需要重新授权。

## 4. 分配统一联网服务

在 MaiBot 的 `config/model_config.toml` 合并以下配置。已有功能分配应修改原节，避免重复 TOML 表：

```toml
[web.services.gpt_builtin]
adapter = "plugin"
plugin_id = "local.chatgpt-maibot-connect"
search_api = "search"
fetch_api = "fetch"
api_version = "1"
timeout = 270.0

[model_task_config.web_search]
model_list = ["gpt_builtin"]
```

桥接插件请求默认总超时 240 秒，Host 示例预算为 270 秒。调用所在 Tool、定时任务或 API 组件的外层超时也应覆盖所需预算。
插件后端不填写 API Key、base_url 或 MCP server_ref，凭据继续由桥接插件管理。
重启 MaiBot 后，麦生活在其「联网」配置中开启所需的联网总开关、新闻或主动探索。

## 5. 验收

在插件目录执行独立检查，以下命令会发起一次真实搜索或网页请求：

```bash
python -m chatgpt_maibot --data-dir /实际/MaiBot/data/plugins/local.chatgpt-maibot-connect search --model YOUR_VISIBLE_MODEL_SLUG --query "Python 官方版本说明" --max-results 3
python -m chatgpt_maibot --data-dir /实际/MaiBot/data/plugins/local.chatgpt-maibot-connect fetch --url https://www.python.org/
```

搜索应返回完成的 search 操作、真实 URL、标题与摘要；摘要中的 `[GPT 生成摘要]` 标记予以保留。
网页读取应返回 `content_kind="page_text"` 和实际提取正文。未返回完整 `response.completed`、搜索失败、授权/用量失败均明确报错。

通过独立检查后，再验证聊天工具和麦生活的新闻/探索历史。麦生活只需已声明 `web.search` / `web.fetch`，无需读取桥接插件凭据。
主程序未进行功能分配时，插件 `ctx.web` 调用仍按原逻辑报错。

## WebUI 的使用

主程序 Python 后端补丁生效后，直接修改上述 TOML 即可使用。随包也提供新增的联网服务表单源码。
MaiBot 默认使用已安装的 `maibot-dashboard` 包；表单源码需要重新构建并选择本地面板，才会出现在浏览器中。
在 MaiBot 的 `dashboard/` 目录完成官方前端依赖安装与构建：

```bash
bun install --frozen-lockfile
bun run build
```

启动 MaiBot 的环境中设置 `MAIBOT_WEBUI_USE_LOCAL_DASHBOARD=1`，重启后可在模型设置的联网服务中选择「插件（ChatGPT 等）」。
无需构建面板也可以使用 TOML 配置完成接入。安装器不改动全局面板包或系统环境变量。

本次前端依赖安装超时，表单源码已通过 TSX 解析和格式检查，完整面板类型检查与构建未通过。建议先使用前述 TOML 接入；完整验证记录见 TESTING.md。

## 结果含义与读取范围

- 来源 URL 和标题来自托管搜索记录及引用，模型正文中的自由生成 URL 无法创建搜索结果。
- 逐来源摘要只在生成内容中的 URL 与真实来源严格匹配时采用，并明确标记为 GPT 生成；缺失时保留来源供正文读取。
- 发布时间未知时返回 `None`，获取时间另行保存；不会拿当前时间充当发布时间。
- `max_results` 为返回条数上限，不能保证搜索一定给出足量来源。
- 空结果与失败分开处理；失败不自动切换其他搜索服务或 API Key。
- 正文读取支持公开 80/443 端口的 HTML、纯文本及 Markdown，拒绝本机、内网、保留 IP、嵌入凭据和超大响应。
- 每次重定向重新检查地址，连接固定到已检查的公网 IP，并保留原域名的 Host、TLS SNI 和证书校验。
- 网页客户端不使用 OAuth、Cookie、环境代理或模型；需要浏览器执行脚本的页面、验证码及付费页面可能无法读取。

## 回滚

停止 MaiBot，用安装时输出的备份路径运行：

```bash
python tools/install_host_backend.py --target /实际/MaiBot --rollback /安装时输出的主程序位置/.gpt-web-backups/安装时生成的目录
```

回滚前同样检查当前文件哈希；安装后又有本地改动的文件会阻止整体回滚。
主程序源码回滚后，将联网功能分配恢复原服务，并移除 `gpt_builtin` 服务配置；安装器不改写实例配置。

## 一手资料

- [MaiBot 联网服务源码](https://github.com/Mai-with-u/MaiBot/tree/c6429547af711555e3d4d1b8d04b413572cb616c/src/web_search)。
- [MaiBot SDK 的 ctx.web 接口](https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/web.md)。
- [麦生活的宿主搜索封装](https://github.com/octmicy/Mai_life/blob/1514f62d8e5441c9a2ab52de158d7105eaa3b3b3/information/host_search.py)。
- [OpenAI 内置联网搜索](https://developers.openai.com/api/docs/guides/tools-web-search)。
- [ChatGPT 授权调用限制](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)。
