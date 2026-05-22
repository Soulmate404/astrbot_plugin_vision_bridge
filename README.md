# AstrBot Vision Bridge

给 AstrBot 注册两个 LLM Tool，让普通文本模型通过工具获得图像能力。

- `vision_analyze`：把对话图片、本地图片或本地目录里的图片交给多模态模型分析，再把文本转述返回给主模型。
- `image_generate`：把主模型给出的提示词交给文生图模型，生成图片并发回当前会话。

插件默认使用：

- 识图：OpenAI 兼容 `POST {base_url}/chat/completions`
- 文生图：阿里云 DashScope SDK `ImageGeneration.call`

## 适用场景

- 当前主模型不支持图片输入，但支持工具调用。
- 需要让机器人主动读取某个本地目录里的图片。
- 想把识图模型、文生图模型和主对话模型拆开配置。
- 识图使用 NewAPI、OneAPI、LiteLLM、OpenRouter 等 OpenAI 兼容服务。
- 文生图使用阿里云 DashScope / 百炼的 `wan2.7-image-pro`。

如果主模型完全不支持工具调用，它不能主动调用 `vision_analyze` / `image_generate`。这种情况需要另写消息拦截逻辑，在请求主模型前自动注入图片描述。

## 安装

把本目录放到 AstrBot 插件目录：

```text
data/plugins/astrbot_plugin_vision_bridge
```

然后重启 AstrBot。

依赖：

```bash
pip install httpx dashscope
```

如果 AstrBot 会自动读取插件目录下的 `requirements.txt`，也可以让 AstrBot 自动安装。

## 配置

在 AstrBot WebUI 的插件配置里填写。

### 通用识图配置

| 配置项 | 说明 | 示例 |
| --- | --- | --- |
| `api_key` | 多模态模型 API Key | `sk-...` |
| `base_url` | OpenAI 兼容接口地址 | `https://api.openai.com/v1` |
| `model` | 支持图片输入的模型 | `gpt-4o-mini` |
| `max_images` | 单次最多分析图片数 | `4` |
| `max_output_tokens` | 识图转述最大输出 token | `800` |
| `timeout` | HTTP 请求超时秒数 | `60` |
| `extra_body` | 追加到 `/chat/completions` 请求体的对象参数 | `{"temperature": 0.2}` |

### 本地图片读取配置

| 配置项 | 说明 |
| --- | --- |
| `allowed_local_dirs` | 允许工具读取的本地目录白名单 |
| `allow_any_local_path` | 是否允许读取任意本地路径 |
| `local_recursive` | 扫描目录时是否默认递归 |

建议配置 `allowed_local_dirs`，不要轻易打开 `allow_any_local_path`。  
原因是 LLM Tool 的参数可能由模型生成，开放任意路径会让模型有机会读取 AstrBot 进程可访问的图片文件。

示例：

```json
{
  "allowed_local_dirs": [
    "/home/soul/images",
    "/data/bot/images"
  ],
  "allow_any_local_path": false
}
```

### 文生图配置

| 配置项 | 说明 | 示例 |
| --- | --- | --- |
| `image_provider` | 文生图接口类型，默认 DashScope | `dashscope` |
| `image_api_key` | 阿里云 DashScope API Key，留空时复用 `api_key` | `sk-...` |
| `image_base_url` | DashScope API 地址 | `https://dashscope.aliyuncs.com/api/v1` |
| `image_model` | 文生图模型 | `wan2.7-image-pro` |
| `image_size` | 默认图片尺寸 | `2K` |
| `image_n` | 默认生成数量，DashScope 组图模式最多 12 | `4` |
| `image_timeout` | 文生图超时秒数 | `300` |
| `image_enable_sequential` | 启用 Wan 组图模式 | `true` |
| `image_download_results` | 下载 DashScope 临时 URL 到本地 | `true` |
| `image_output_dir` | 生成图片保存目录，留空使用插件数据目录下的 `images` | 留空 |
| `image_extra_body` | 追加到 DashScope 调用参数的对象 | `{"watermark": false}` |

国内站通常使用：

```text
https://dashscope.aliyuncs.com/api/v1
```

国际站可改为：

```text
https://dashscope-intl.aliyuncs.com/api/v1
```

## 启用工具

在聊天里执行：

```text
/tool ls
/tool on vision_analyze
/tool on image_generate
```

## 工具说明

### `vision_analyze`

让主模型分析图片，并返回文本描述。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 希望多模态模型重点分析什么 |
| `image_path` | string | 否 | 本地图片文件或目录 |
| `image_paths` | string[] | 否 | 多个本地图片文件或目录 |
| `recursive` | boolean | 否 | 目录是否递归扫描 |

行为：

- 如果传了 `image_path` / `image_paths`，优先读取本地图片。
- 如果没传本地路径，尝试从当前消息链里读取图片。
- 本地路径受 `allowed_local_dirs` 白名单限制。
- 对话消息里的图片如果是平台临时文件，不受本地白名单限制。

手动测试：

```text
/vision_analyze /home/soul/images/test.png 请转写图片里的文字
```

如果当前消息带图片，也可以直接：

```text
/vision_analyze
```

### `image_generate`

让主模型生成图片。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `prompt` | string | 是 | 文生图提示词 |
| `size` | string | 否 | 图片尺寸，例如 `2K`、`1K`、`1024*1024` |
| `n` | number | 否 | 生成数量，DashScope 组图模式最多 12 |
| `enable_sequential` | boolean | 否 | 是否启用 DashScope 组图模式 |

行为：

- DashScope 返回的图片 URL 通常 24 小时有效。
- 默认会把 DashScope 图片 URL 下载到 `image_output_dir`，再把本地图片发回当前会话。
- `image_timeout` 同时用于等待 DashScope 生图结果和下载结果图片。
- 如果下载失败，会退回发送原始 URL。
- 工具返回给主模型的是生成结果摘要，包括本地路径、URL 或模型文本。

手动测试：

```text
/image_generate 一张白底极简风格的机械键盘产品图
```

## DashScope 文生图调用

插件内部等价于下面这种调用：

```python
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

message = Message(
    role="user",
    content=[{"text": "电影感组图，记录同一只流浪橘猫，特征必须前后一致。"}],
)

rsp = ImageGeneration.call(
    model="wan2.7-image-pro",
    api_key="sk-...",
    messages=[message],
    enable_sequential=True,
    n=4,
    size="2K",
)
```

插件会从返回值的 `output.choices[].message.content[].image` 中提取图片 URL。

## OpenAI 兼容识图接口要求

识图请求体使用 `image_url` 格式：

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述图片内容"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "max_tokens": 800
}
```

## OpenAI 兼容文生图模式

如果你仍然想走 OpenAI 兼容的 `/images/generations`，把配置改成：

```json
{
  "image_provider": "openai",
  "image_base_url": "https://api.openai.com/v1",
  "image_model": "gpt-image-1",
  "image_size": "1024*1024",
  "image_n": 1
}
```

对应请求路径：

```text
POST {image_base_url}/images/generations
```

部分服务商只支持图片 URL，不支持 `data:image/...;base64,...`。这种情况下，当前插件的本地图片识别可能不可用，需要先把本地图片上传成公网或服务商可访问的 URL。

## 常见问题

### `/tool ls` 看不到工具

检查插件是否加载成功，依赖 `httpx` 和 `dashscope` 是否安装，AstrBot 日志里是否出现导入错误。

### 主模型不会主动调用工具

确认主模型支持 Function Calling / Tools，并且已经执行：

```text
/tool on vision_analyze
/tool on image_generate
```

如果模型不支持工具调用，只能手动命令触发，或者另写自动拦截注入逻辑。

### 本地图片提示不在白名单内

把图片所在目录加入 `allowed_local_dirs`，例如：

```json
{
  "allowed_local_dirs": ["/home/soul/images"]
}
```

### 文生图成功但没有发出图片

DashScope 返回的是临时图片 URL，插件默认会下载到本地再发送。  
如果下载失败，检查 AstrBot 服务器是否能访问 `dashscope-result-*.oss-*` 结果 URL。

### 图片发送失败

本地图片发送依赖 AstrBot 所用平台适配器支持 `Image.fromFileSystem`。  
如果平台不支持本地文件发送，可以让文生图服务返回 URL，或者把生成文件上传到可访问地址后再发送。

## 文件结构

```text
astrbot_plugin_vision_bridge/
├── main.py
├── _conf_schema.json
├── metadata.yaml
├── requirements.txt
└── README.md
```
