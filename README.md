# ChatGPT-MaiBot-connect-plugin

通过 **Sign in with ChatGPT** 将支持方案授权的 **ChatGPT** 账号接入 **MaiBot**。插件可以为聊天模型任务提供 Responses 接口，也可以让聊天和麦生活共用 GPT 内置搜索，并独立读取公开网页正文。

## 功能与兼容性

- **聊天模型** 支持文本、图片输入、MaiBot 函数工具及 JSON 输出。模型以账号实际可用的模型标识为准。
- **统一联网** 在 **MaiBot 1.4.0** 使用原生 `deepseek` 适配器加载配置。桥接插件在本机实现兼容接口，搜索请求由 GPT 内置搜索处理，网页正文由插件直接抓取。[MaiBot 1.4.0 DeepSeek 适配器](https://github.com/Mai-with-u/MaiBot/blob/v1.4.0/src/web_search/adapters/deepseek.py)
- 当前版本为 **0.2.3**。基础桥接要求 **Python 3.12+**、**MaiBot 1.3.14+** 和 **插件 SDK 2.16.0+**。联网配置按主程序版本区分，旧版补丁的适用范围见[统一联网安装说明](UNIFIED_WEB_BACKEND.md)。

## 安装与授权

1. 将项目复制到 MaiBot 的 `plugins/ChatGPT-MaiBot-connect-plugin/`。使用运行插件的 Python 环境安装依赖，插件 SDK 由 MaiBot Runner 提供。

   ```bash
   python -m pip install -r requirements.txt
   ```

2. 启动 MaiBot 一次，确认 Runner 为插件分配的数据目录。从插件目录运行以下命令，将示例路径替换为实际数据目录。在同一台电脑的浏览器中完成授权。

   ```bash
   plugin_data="/path/to/MaiBot/data/plugins/local.chatgpt-maibot-connect"
   python -m chatgpt_maibot --data-dir "$plugin_data" login
   python -m chatgpt_maibot --data-dir "$plugin_data" models
   ```

3. 如需将插件用于聊天模型任务，将 [模型配置示例](examples/model_config.fragment.toml) 的相关段落合并进现有的 `config/model_config.toml`，并将 `model_identifier` 替换为 `models` 返回的模型标识。聊天模型使用 `openai_chatgpt_plan_siwc`，账号授权由插件管理。

## 启用统一联网

联网默认关闭。按 [MaiBot 1.4.0 联网配置](UNIFIED_WEB_BACKEND.md) 填写插件配置中的 `[web]`，设置模型和本地端口，生成接口密钥。随后在 MaiBot 的 `config/model_config.toml` 中用同一端口和密钥配置 `gpt_builtin` 服务，并在 `[model_task_config.web_search]` 中选中它。

兼容接口随插件启动，只监听 `127.0.0.1`。MaiBot 的 DeepSeek 适配器将单条搜索词限制为 **128 个字符**。正文读取适用于能公开访问的静态网页。插件配置的完整示例见 [plugin-config.example.toml](examples/plugin-config.example.toml)。

如果还希望桥接模型在生成聊天回复时自行发起搜索，可在插件的 `[plugin]` 中设置 `enable_web_search = true`。

可用本地助手分别验证插件的搜索和正文读取。把示例模型换成授权后查到的标识。

```bash
python -m chatgpt_maibot --data-dir "$plugin_data" search --model YOUR_MODEL_SLUG --query "Python 官方文档"
python -m chatgpt_maibot --data-dir "$plugin_data" fetch --url https://docs.python.org/3/
```

## 数据与账号

- `accounts.json` 保存 OAuth 凭据，`host_id.txt` 标识本机。两者保存在 MaiBot 分配的插件数据目录。部署配置中的本地接口密钥也属于私有数据。项目的 `.gitignore` 已覆盖这些文件及本机配置、日志和备份。
- 聊天模型请求会将所需的对话历史、图片和工具结果发送到 OpenAI。搜索会发送查询词，网页读取使用单独的客户端直接访问网页。
- `status` 查看脱敏账号状态，`login --new` 添加账号或工作区，`logout` 撤销会话。每个命令均使用同一个 `--data-dir`。方案用量和已授权应用可在 [ChatGPT 用量设置](https://chatgpt.com/settings/usage) 查看。

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
python tools/check_release.py
```

完整复现步骤和验证范围见 [TESTING.md](TESTING.md)。本地一致性检查覆盖插件声明、依赖与配置示例。正式发布还需填写作者和仓库地址，确认插件 ID，再运行 `python tools/check_release.py --release`。

本项目采用 **MIT** 许可证，见 [LICENSE](LICENSE)。
