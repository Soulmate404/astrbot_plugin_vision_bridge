import asyncio
import base64
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable

import httpx
from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

try:
    import dashscope
    from dashscope.aigc.image_generation import ImageGeneration
    from dashscope.api_entities.dashscope_response import Message

    _DASHSCOPE_AVAILABLE = True
except ImportError:
    _DASHSCOPE_AVAILABLE = False
    dashscope = None  # type: ignore[assignment]
    ImageGeneration = None  # type: ignore[assignment]
    Message = None  # type: ignore[assignment]


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_DOWNLOAD_CONCURRENCY = 4


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class VisionBridgeTool(FunctionTool[AstrAgentContext]):
    plugin: Any = Field(default=None, exclude=True)
    name: str = "vision_analyze"
    description: str = (
        "Analyze images from the current conversation or from allowed local files/directories "
        "with a multimodal model, then return a concise text description."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to focus on when analyzing the image.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Optional local image file or directory path. If omitted, use images in the current message.",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional local image file or directory paths.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to scan local directories recursively.",
                },
            },
            "required": ["question"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult | str:
        question = str(kwargs.get("question") or self.plugin.default_question)
        image_paths = self.plugin.normalize_image_paths(
            kwargs.get("image_path"), kwargs.get("image_paths")
        )
        recursive = _as_bool(kwargs.get("recursive"), self.plugin.local_recursive)

        event = getattr(getattr(context, "context", None), "event", None)
        return await self.plugin.analyze_images(
            question=question,
            event=event,
            local_paths=image_paths,
            recursive=recursive,
        )


@dataclass
class ImageGenerateTool(FunctionTool[AstrAgentContext]):
    plugin: Any = Field(default=None, exclude=True)
    name: str = "image_generate"
    description: str = (
        "Generate an image from a text prompt with DashScope or an OpenAI-compatible image generation API. "
        "Return local file paths or URLs for the generated image."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The image generation prompt.",
                },
                "size": {
                    "type": "string",
                    "description": "Optional image size, such as 2K, 1K, 1024*1024, or 16:9.",
                },
                "n": {
                    "type": "number",
                    "description": "Optional number of images to generate.",
                },
                "enable_sequential": {
                    "type": "boolean",
                    "description": "Optional DashScope Wan sequential image-set mode.",
                },
            },
            "required": ["prompt"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult | str:
        prompt = str(kwargs.get("prompt") or "").strip()
        size = str(kwargs.get("size") or self.plugin.image_size).strip()
        n = int(kwargs.get("n") or self.plugin.image_n)
        enable_sequential = _as_bool(
            kwargs.get("enable_sequential"), self.plugin.image_enable_sequential
        )
        event = getattr(getattr(context, "context", None), "event", None)
        result = await self.plugin.generate_image(
            prompt=prompt, size=size, n=n, enable_sequential=enable_sequential
        )
        if result.get("error"):
            return result["error"]
        if event is not None:
            await self.plugin.send_generated_images(event, result)
        return result.get("summary", "图片生成完成。")


class VisionBridgePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.api_key = str(config.get("api_key", "")).strip()
        self.base_url = str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.model = str(config.get("model", "gpt-4o-mini")).strip()
        self.timeout = float(config.get("timeout", 60))
        self.max_images = int(config.get("max_images", 4))
        self.max_output_tokens = int(config.get("max_output_tokens", 800))
        self.default_question = str(
            config.get("default_question", "请客观描述图片内容，保留关键文字、物体、场景和可能与用户问题相关的细节。")
        )
        self.allowed_local_dirs = self._resolve_allowed_dirs(
            config.get("allowed_local_dirs", [])
        )
        self.allow_any_local_path = _as_bool(config.get("allow_any_local_path"), False)
        self.local_recursive = _as_bool(config.get("local_recursive"), False)
        self.extra_body = config.get("extra_body", {}) or {}

        self.image_provider = str(config.get("image_provider", "dashscope")).strip().lower()
        self.image_api_key = str(config.get("image_api_key", "")).strip() or self.api_key
        self.image_base_url = str(
            config.get("image_base_url", "") or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")
        self.image_model = str(config.get("image_model", "wan2.7-image-pro")).strip()
        self.image_size = str(config.get("image_size", "2K")).strip()
        self.image_n = int(config.get("image_n", 4))
        self.image_timeout = float(config.get("image_timeout", 300))
        self.image_enable_sequential = _as_bool(config.get("image_enable_sequential"), True)
        self.image_download_results = _as_bool(config.get("image_download_results"), True)
        default_image_output_dir = StarTools.get_data_dir() / "images"
        configured_image_output_dir = str(config.get("image_output_dir", "") or "").strip()
        self.image_output_dir = (
            Path(configured_image_output_dir).expanduser()
            if configured_image_output_dir
            else default_image_output_dir
        )
        self.image_extra_body = config.get("image_extra_body", {}) or {}

        self._http = None
        self._download_semaphore = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)

        self.context.add_llm_tools(VisionBridgeTool(plugin=self), ImageGenerateTool(plugin=self))
        logger.info("vision_analyze and image_generate tools registered.")

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._http

    async def terminate(self):
        if self._http is not None:
            await self._http.aclose()

    @filter.command("vision_analyze")
    async def vision_analyze_command(self, event: AstrMessageEvent):
        """手动测试图片转述。用法：/vision_analyze [本地图片或目录] [问题]"""
        text = (event.message_str or "").strip()
        if text.startswith("/vision_analyze"):
            text = text[len("/vision_analyze"):].strip()

        local_paths: list[str] = []
        question = self.default_question

        if text:
            first, _, rest = text.partition(" ")
            first = first.strip()
            rest = rest.strip()
            if self._looks_like_path(first):
                local_paths = [first]
                if rest:
                    question = rest
            else:
                question = text

        result = await self.analyze_images(
            question=question,
            event=event,
            local_paths=local_paths,
            recursive=self.local_recursive,
        )
        yield event.plain_result(str(result))

    @filter.command("image_generate")
    async def image_generate_command(self, event: AstrMessageEvent):
        """手动测试文生图。用法：/image_generate 一张赛博朋克风格的猫"""
        prompt = (event.message_str or "").strip()
        if prompt.startswith("/image_generate"):
            prompt = prompt[len("/image_generate") :].strip()
        result = await self.generate_image(
            prompt=prompt,
            size=self.image_size,
            n=self.image_n,
            enable_sequential=self.image_enable_sequential,
        )
        if result.get("error"):
            yield event.plain_result(result["error"])
            return
        paths = result.get("paths") or []
        urls = result.get("urls") or []
        for path in paths:
            yield event.image_result(path)
        if urls:
            yield event.plain_result("\n".join(urls))
        if not paths and not urls:
            yield event.plain_result("图片生成完成，但响应里没有可用的图片。")

    async def analyze_images(
        self,
        question: str,
        event: AstrMessageEvent | None = None,
        local_paths: list[str] | None = None,
        recursive: bool = False,
    ) -> str:
        if not self.api_key:
            return "vision_analyze 配置错误：请先在插件配置中填写多模态模型 API Key。"
        if not self.model:
            return "vision_analyze 配置错误：请先在插件配置中填写多模态模型名称。"

        image_refs: list[dict[str, Any]] = []
        errors: list[str] = []

        for path in local_paths or []:
            refs, path_errors = self._collect_local_image_refs(path, recursive=recursive)
            image_refs.extend(refs)
            errors.extend(path_errors)

        if not image_refs and event is not None:
            image_refs.extend(self._collect_event_image_refs(event))

        if not image_refs:
            detail = "\n".join(errors)
            return "没有找到可分析的图片。" + (f"\n{detail}" if detail else "")

        image_refs = image_refs[: self.max_images]
        try:
            answer = await self._call_vision_model(question, image_refs)
        except Exception as exc:
            logger.exception("vision_analyze failed.")
            return f"调用多模态模型失败：{exc}"

        if errors:
            return answer + "\n\n以下本地路径未被使用：\n" + "\n".join(errors)
        return answer

    async def generate_image(
        self,
        prompt: str,
        size: str | None = None,
        n: int | None = None,
        enable_sequential: bool | None = None,
    ) -> dict[str, Any]:
        if not prompt:
            return {"error": "image_generate 参数错误：prompt 不能为空。"}
        if not self.image_api_key:
            return {"error": "image_generate 配置错误：请先填写图片生成 API Key 或通用 api_key。"}
        if not self.image_model:
            return {"error": "image_generate 配置错误：请先填写图片生成模型名称。"}

        if self.image_provider == "dashscope":
            max_n = 12 if _as_bool(enable_sequential, self.image_enable_sequential) else 4
        else:
            max_n = 4
        n = max(1, min(int(n or self.image_n), max_n))
        if self.image_provider == "dashscope":
            return await self._generate_image_dashscope(
                prompt=prompt,
                size=size or self.image_size,
                n=n,
                enable_sequential=_as_bool(enable_sequential, self.image_enable_sequential),
            )
        return await self._generate_image_openai(prompt=prompt, size=size or self.image_size, n=n)

    async def _generate_image_openai(self, prompt: str, size: str, n: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.image_model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        if isinstance(self.image_extra_body, dict):
            body.update(self.image_extra_body)

        headers = {"Authorization": f"Bearer {self.image_api_key}", "Content-Type": "application/json"}
        try:
            resp = await self._ensure_http().post(
                f"{self.image_base_url}/images/generations",
                headers=headers,
                json=body,
                timeout=httpx.Timeout(self.image_timeout),
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.exception("image_generate failed.")
            return {"error": f"调用文生图模型失败：{exc}"}

        try:
            return self._store_generated_images(payload)
        except Exception as exc:
            logger.exception("Failed to store generated image.")
            return {"error": f"图片生成成功，但保存结果失败：{exc}"}

    async def _generate_image_dashscope(
        self, prompt: str, size: str, n: int, enable_sequential: bool
    ) -> dict[str, Any]:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_dashscope_image_generation,
                    prompt,
                    size,
                    n,
                    enable_sequential,
                ),
                timeout=self.image_timeout,
            )
        except asyncio.TimeoutError:
            return {"error": f"调用 DashScope 文生图超时：已等待 {self.image_timeout:g} 秒。"}
        except ModuleNotFoundError:
            return {"error": "调用 DashScope 文生图失败：请先安装 dashscope 依赖。"}
        except Exception as exc:
            logger.exception("dashscope image_generate failed.")
            return {"error": f"调用 DashScope 文生图失败：{exc}"}

        try:
            result = await self._store_dashscope_images(response)
        except Exception as exc:
            logger.exception("Failed to store DashScope generated image.")
            return {"error": f"DashScope 图片生成成功，但保存结果失败：{exc}"}
        return result

    def _call_dashscope_image_generation(
        self, prompt: str, size: str, n: int, enable_sequential: bool
    ) -> Any:
        if not _DASHSCOPE_AVAILABLE:
            raise ModuleNotFoundError("请先安装 dashscope 依赖。")

        dashscope.base_http_api_url = self.image_base_url
        message = Message(role="user", content=[{"text": prompt}])
        kwargs: dict[str, Any] = {
            "model": self.image_model,
            "api_key": self.image_api_key,
            "messages": [message],
            "enable_sequential": enable_sequential,
            "n": n,
            "size": size,
        }
        if isinstance(self.image_extra_body, dict):
            kwargs.update(self.image_extra_body)
        return ImageGeneration.call(**kwargs)

    async def send_generated_images(self, event: AstrMessageEvent, result: dict[str, Any]) -> None:
        chain = []
        for path in result.get("paths") or []:
            chain.append(Comp.Image.fromFileSystem(path))
        for url in result.get("urls") or []:
            chain.append(Comp.Image.fromURL(url))
        if not chain:
            return
        try:
            await self.context.send_message(event.unified_msg_origin, chain)
        except Exception:
            logger.exception("Failed to send generated images to conversation.")

    def normalize_image_paths(self, image_path: Any, image_paths: Any) -> list[str]:
        paths: list[str] = []
        if isinstance(image_path, str) and image_path.strip():
            paths.append(image_path.strip())
        if isinstance(image_paths, str) and image_paths.strip():
            paths.append(image_paths.strip())
        elif isinstance(image_paths, list):
            paths.extend(str(item).strip() for item in image_paths if str(item).strip())
        return paths

    def _store_generated_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data") or []
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        urls: list[str] = []
        revised_prompts: list[str] = []

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            if item.get("revised_prompt"):
                revised_prompts.append(str(item["revised_prompt"]))
            if item.get("url"):
                urls.append(str(item["url"]))
                continue
            b64_json = item.get("b64_json")
            if not b64_json:
                continue
            try:
                image_bytes = base64.b64decode(b64_json)
            except Exception:
                logger.exception("Failed to decode b64_json for item %d", index)
                continue
            path = self.image_output_dir / f"{int(time.time())}_{index}_{self._short_id(b64_json)}.png"
            path.write_bytes(image_bytes)
            paths.append(str(path))

        summary = "图片生成完成。"
        if paths:
            summary += "\n本地文件：\n" + "\n".join(paths)
        if urls:
            summary += "\n图片 URL：\n" + "\n".join(urls)
        if revised_prompts:
            summary += "\n修订后的提示词：\n" + "\n".join(revised_prompts)
        return {"paths": paths, "urls": urls, "revised_prompts": revised_prompts, "summary": summary}

    async def _store_dashscope_images(self, response: Any) -> dict[str, Any]:
        status_code = self._get_value(response, "status_code")
        if status_code and int(status_code) >= 400:
            code = self._get_value(response, "code") or ""
            message = self._get_value(response, "message") or response
            return {"error": f"DashScope 文生图失败：{code} {message}".strip()}

        output = self._get_value(response, "output") or {}
        choices = self._get_value(output, "choices") or []
        urls: list[str] = []
        texts: list[str] = []
        for choice in choices:
            message = self._get_value(choice, "message") or {}
            content = self._get_value(message, "content") or []
            for item in content:
                image = self._get_value(item, "image")
                text = self._get_value(item, "text")
                if image:
                    urls.append(str(image))
                if text:
                    texts.append(str(text))

        if not urls:
            output_text = self._get_value(output, "text")
            if output_text:
                texts.append(str(output_text))

        paths: list[str] = []
        kept_urls = urls
        if urls and self.image_download_results:
            paths, kept_urls = await self._download_generated_urls(urls)

        summary = "DashScope 图片生成完成。"
        if paths:
            summary += "\n本地文件：\n" + "\n".join(paths)
        if kept_urls:
            summary += "\n图片 URL：\n" + "\n".join(kept_urls)
        if texts:
            summary += "\n模型文本：\n" + "\n".join(texts)
        return {"paths": paths, "urls": kept_urls, "source_urls": urls, "texts": texts, "summary": summary}

    async def _download_generated_urls(self, urls: list[str]) -> tuple[list[str], list[str]]:
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        failed_urls: list[str] = []
        timestamp = int(time.time())

        async def _download_one(index: int, url: str):
            async with self._download_semaphore:
                try:
                    resp = await self._ensure_http().get(url, timeout=httpx.Timeout(self.image_timeout))
                    resp.raise_for_status()
                    ext = self._guess_image_ext(url, resp.headers.get("content-type", ""))
                    path = self.image_output_dir / f"{timestamp}_{index}_{self._short_id(url)}{ext}"
                    path.write_bytes(resp.content)
                    paths.append(str(path))
                except Exception:
                    logger.exception("Failed to download generated image: %s", url)
                    failed_urls.append(url)

        await asyncio.gather(*[_download_one(i, url) for i, url in enumerate(urls)])
        return paths, failed_urls

    def _guess_image_ext(self, url: str, content_type: str) -> str:
        ext = Path(urlparse(url).path).suffix.lower()
        if ext in IMAGE_SUFFIXES:
            return ext
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        return guessed or ".png"

    def _get_value(self, obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _short_id(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", value)
        return cleaned[:8] or "image"

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return bool(
            value.startswith(("/", "~/", "./", "../"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
        )

    def _resolve_allowed_dirs(self, values: Any) -> list[Path]:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        resolved = []
        for value in values:
            if not str(value).strip():
                continue
            resolved.append(Path(str(value)).expanduser().resolve())
        return resolved

    def _collect_event_image_refs(self, event: AstrMessageEvent) -> list[dict[str, Any]]:
        refs = []
        message_chain = getattr(getattr(event, "message_obj", None), "message", []) or []
        for comp in message_chain:
            if not self._is_image_component(comp):
                continue
            ref = self._component_to_image_ref(comp)
            if ref:
                refs.append(ref)
        return refs

    def _is_image_component(self, comp: Any) -> bool:
        comp_type = getattr(comp, "type", "")
        comp_name = comp.__class__.__name__.lower()
        return str(comp_type).lower().endswith("image") or comp_name == "image"

    def _component_to_image_ref(self, comp: Any) -> dict[str, Any] | None:
        for attr in ("url", "file", "path"):
            value = getattr(comp, attr, None)
            if not value:
                continue
            ref = self._value_to_image_ref(
                str(value), source=f"message.{attr}", enforce_local_allowlist=False
            )
            if ref:
                return ref
        return None

    def _collect_local_image_refs(
        self, raw_path: str, recursive: bool = False
    ) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            root = Path(raw_path).expanduser().resolve()
        except Exception as exc:
            return [], [f"{raw_path}: 路径无法解析：{exc}"]

        if not self._is_local_path_allowed(root):
            return [], [f"{raw_path}: 不在 allowed_local_dirs 白名单内"]
        if not root.exists():
            return [], [f"{raw_path}: 路径不存在"]

        candidates: Iterable[Path]
        if root.is_dir():
            iterator = root.rglob("*") if recursive else root.iterdir()
            candidates = (p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            candidates = [root]

        refs = []
        errors = []
        for path in candidates:
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                errors.append(f"{path}: 不是支持的图片类型")
                continue
            try:
                refs.append(self._file_to_data_url_ref(path))
            except Exception as exc:
                errors.append(f"{path}: 读取失败：{exc}")
        return refs, errors

    def _is_local_path_allowed(self, path: Path) -> bool:
        if self.allow_any_local_path:
            return True
        return any(path == allowed or allowed in path.parents for allowed in self.allowed_local_dirs)

    def _value_to_image_ref(
        self, value: str, source: str, enforce_local_allowlist: bool = True
    ) -> dict[str, Any] | None:
        if value.startswith(("http://", "https://", "data:image/")):
            return {"type": "image_url", "image_url": {"url": value}, "source": source}
        if value.startswith("file://"):
            value = value.removeprefix("file://")
        if value.startswith("base64://"):
            return {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + value.removeprefix("base64://")},
                "source": source,
            }

        path = Path(value).expanduser()
        if path.exists():
            try:
                resolved = path.resolve()
                if enforce_local_allowlist and not self._is_local_path_allowed(resolved):
                    logger.warning("Image path from message is not allowed: %s", resolved)
                    return None
                return self._file_to_data_url_ref(resolved)
            except Exception as exc:
                logger.warning("Failed to read image path from message %s: %s", value, exc)
        return None

    def _file_to_data_url_ref(self, path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
            "source": str(path),
        }

    async def _call_vision_model(self, question: str, image_refs: list[dict[str, Any]]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "你是图片转述工具。请根据用户问题分析图片，输出给另一个不支持多模态的语言模型使用。"
                    "回答要准确、简洁；如果图片里有文字、表格、代码或错误信息，请尽量完整转写。\n\n"
                    f"用户问题：{question}"
                ),
            }
        ]
        content.extend({"type": "image_url", "image_url": ref["image_url"]} for ref in image_refs)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_output_tokens,
        }
        if isinstance(self.extra_body, dict):
            body.update(self.extra_body)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = await self._ensure_http().post(f"{self.base_url}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        payload = resp.json()

        return payload["choices"][0]["message"]["content"].strip()
