"""让麦麦真正"看懂"说说/动态里的配图。

主程序自带的识图通道（`src/chat/image_system/image_manager.py`）跑在宿主进程里、插件进不去，
插件侧只能用 SDK 的 `llm.generate` 能力走**同一个 vlm 任务 + 同一套提示词**，让视觉模型把
图片转成文字描述。`qzone_api` 预留了 `set_image_manager()` 钩子，`on_load` 里挂上本类即可。

这里返回的字符串会被 `utils.py` 直接拼进说说内容再送给模型，所以成功时包成 `[图片：描述]`，
失败时退回 `[图片]`（与上游行为一致，不让调用方拿到空串）。

两处工程细节：
1. 大图先缩到长边 1280 再送模型。QQ 空间里常见的是整屏截图（base64 后 5~25 MB），
   不缩会撞上插件 RPC 的能力调用超时，表现为 `图片识别失败（task=vlm）:
   RPCError: [E_TIMEOUT] 请求 cap.call 超时`。
2. 识别失败自动重试一次，只在真的失败时多花一次调用。
"""

import base64
import asyncio
import hashlib
import io
from typing import Any

# 与主程序 prompts/zh-CN/image_description.prompt 保持一致
IMAGE_DESCRIPTION_PROMPT = (
    "请用中文详细描述这张图片的内容。如果有文字，请把文字描述概括出来，"
    "请留意其主题、直观感受，输出为一段平文本，最多100字，请注意不要分点，就输出一段文本"
)

_PLACEHOLDER = "[图片]"

# 超过这个字节数就先压缩再送模型（base64 后约放大 4/3）
_SHRINK_THRESHOLD_BYTES = 512 * 1024
_MAX_EDGE = 1280
_JPEG_QUALITY = 85

# 识图这次能力调用的 RPC 超时（毫秒）。插件运行时的默认上限是 30s，
# 而一张图走 vlm 实测可能要 70s+（模型排队/多模态编码），因此单独放宽。
# 注意：超时值由 SDK 的 call_capability(timeout_ms=...) 消费，不会进到 LLM 请求体。
_CALL_TIMEOUT_MS = 90_000

# 识别失败时的重试次数（不含首次）与退避间隔；能力调用超时不重试（再等一轮也多半还是超时）
_RETRY_TIMES = 1
_RETRY_DELAY = 0.5


def is_timeout_error(exc: BaseException) -> bool:
    """判断异常是否为能力调用/RPC 超时。"""

    text = str(exc)
    return (
        isinstance(exc, TimeoutError)
        or "E_TIMEOUT" in text
        or "超时" in text
        or "timeout" in text.lower()
    )

# 只看文件头，避免为了一张图引入 PIL 依赖（主程序的白名单也是 jpg/jpeg/png/webp/gif）
_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def guess_image_format(image_bytes: bytes) -> str:
    """按文件头猜图片格式，猜不到就按 jpeg 交给上游。"""

    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    for signature, image_format in _SIGNATURES:
        if image_bytes.startswith(signature):
            return image_format
    return "jpeg"


def shrink_image(image_bytes: bytes) -> bytes:
    """把过大的图片等比缩到长边 `_MAX_EDGE`，任何异常都原样返回。

    没有 PIL（或图片本身有问题）时直接返回原图，行为与旧版本一致。
    """

    if not image_bytes or len(image_bytes) <= _SHRINK_THRESHOLD_BYTES:
        return image_bytes
    try:
        from PIL import Image  # 局部导入：环境里没有 PIL 也不影响其它功能
    except Exception:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if getattr(image, "is_animated", False):
                image.seek(0)  # MPO/GIF 取第一帧
            converted = image if image.mode in ("RGB", "L") else image.convert("RGB")
            converted.thumbnail((_MAX_EDGE, _MAX_EDGE))
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            shrunk = buffer.getvalue()
    except Exception:
        return image_bytes
    return shrunk if shrunk and len(shrunk) < len(image_bytes) else image_bytes


class MaizoneImageReader:
    """实现 `qzone_api` 期望的 `get_image_description(image_base64)` 接口。"""

    def __init__(self, plugin: Any, *, cache_size: int = 256) -> None:
        """Args: plugin: MaizonePlugin 实例（用它的 config 与 ctx.llm）。"""

        self._plugin = plugin
        self._cache: dict[str, str] = {}
        self._cache_size = max(16, int(cache_size))
        self._warned_empty = False

    def _read_settings(self) -> tuple[bool, str]:
        """读配置；每次调用都读，WebUI 改完即时生效。"""

        read_config = getattr(self._plugin.config, "read", None)
        enabled = bool(getattr(read_config, "enable_image_read", True))
        task_name = str(getattr(read_config, "image_read_task", "") or "").strip() or "vlm"
        return enabled, task_name

    async def _describe(self, task_name: str, image_base64: str, image_format: str) -> str:
        """发一次多模态请求，返回纯文本描述（空串表示模型没给内容）。"""

        response = await self._plugin.ctx.llm.generate(
            prompt=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
                        },
                    ],
                }
            ],
            task_name=task_name,
            timeout_ms=_CALL_TIMEOUT_MS,
        )
        if isinstance(response, dict):
            return str(response.get("response") or "").strip()
        return ""

    async def get_image_description(self, image_base64: str) -> str:
        """把图片 base64 交给 vlm 任务换成一段文字描述。"""

        raw_base64 = (image_base64 or "").strip()
        if not raw_base64:
            return ""

        enabled, task_name = self._read_settings()
        if not enabled:
            return _PLACEHOLDER

        cache_key = hashlib.sha256(raw_base64.encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        logger = self._plugin.ctx.logger
        try:
            payload = base64.b64decode(raw_base64, validate=False)
        except Exception:
            payload = b""

        shrunk = shrink_image(payload)
        if shrunk is not payload and shrunk:
            raw_base64 = base64.b64encode(shrunk).decode("utf-8")
            logger.debug(f"配图过大已压缩后送识图: {len(payload)} -> {len(shrunk)} 字节")
        image_format = guess_image_format(shrunk or b"")

        description = ""
        last_error: str = ""
        for attempt in range(_RETRY_TIMES + 1):
            try:
                description = await self._describe(task_name, raw_base64, image_format)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - 任何失败都退化为占位符
                last_error = f"{type(exc).__name__}: {exc}"
                # 超时说明模型/网关已经排队到上限，立刻重试多半还是超时，只会把停顿翻倍
                if is_timeout_error(exc) or attempt >= _RETRY_TIMES:
                    break
                logger.warning(f"图片识别失败（task={task_name}），重试一次: {last_error}")
                await asyncio.sleep(_RETRY_DELAY)

        if last_error:
            hint = "（超时，未重试）" if is_timeout_error(Exception(last_error)) else ""
            logger.warning(f"图片识别失败（task={task_name}，上限 {_CALL_TIMEOUT_MS // 1000}s）{hint}: {last_error}")
            return _PLACEHOLDER

        if not description:
            if not self._warned_empty:
                self._warned_empty = True
                logger.warning(
                    f"图片识别未返回描述（task={task_name}），请确认 model_config.toml 的 "
                    f"[model_task_config.{task_name}] 指向能识图的模型"
                )
            return _PLACEHOLDER

        result = f"[图片：{description}]"
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        logger.info(f"图片识别完成：{description[:40]}")
        return result
