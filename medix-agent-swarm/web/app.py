"""
MediX 医疗助手 - Web 服务（FastAPI）

提供聊天 API 与前端页面，包装 medix-agent-swarm 的 Swarm 处理流程。
- POST /api/chat        非流式（兼容）
- POST /api/chat/stream 流式（SSE 推送系统内部状态）
启动：uvicorn web.app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 添加项目根目录到路径（web/ 的上一级 = medix-agent-swarm）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm import process_with_swarm
from loguru import logger

from web.intent_guard import get_guard

guard = get_guard()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="MediX 多智能体医疗助手",
    description="基于 Skills-Agent 两层架构与 Agent Swarm 协作的医疗咨询系统",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    answer: str
    swarm_enabled: bool = False
    agents_involved: List[str] = []
    timeout_occurred: bool = False
    execution_time: float = 0.0
    suggestions: List[str] = []
    disclaimer: str = ""


@app.get("/")
async def index():
    """前端页面"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    处理用户咨询（非流式，兼容旧接口）

    简单问题 -> 单 Agent 快速响应
    复杂问题 -> Swarm 多 Agent 协作
    """
    session_id = req.session_id or f"web-{uuid.uuid4().hex[:8]}"

    # 熔断机制：非医疗类问题直接拦截，不消耗 LLM token
    verdict = guard.check(req.message, session_id=session_id)
    if not verdict["allowed"]:
        return JSONResponse(
            status_code=200,
            content={
                "rejected": True,
                "circuit": verdict["circuit"],
                "message": verdict["message"],
                "session_id": session_id,
            },
        )

    start = time.time()
    try:
        result = await process_with_swarm(req.message, session_id=session_id)
        execution_time = time.time() - start

        logger.info(f"[web] session={session_id} 耗时={execution_time:.1f}s")

        return ChatResponse(**_build_result(result, execution_time, session_id))
    except Exception as e:
        logger.exception(f"[web] 处理失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时出错: {e}",
                "hint": "请稍后重试，或检查 LLM API 配置（config.py）",
            },
        )


# ============================================================
# SSE 流式输出：loguru 日志钩子 -> asyncio 队列 -> 前端事件流
# ============================================================

# 初始化噪音日志黑名单（不推送给前端）
_NOISE_KEYWORDS = (
    "Discovered skill",
    "Discovered ",
    "Registered skill",
    "ConstraintValidator",
    "Total ",
    "Initialized ",
    "ShortTermMemory initialized",
    "SwarmCoordinator initialized",
    "Memory system:",
    "Loading embedding model",
    "Embedding model loaded",
    "Connecting to Qdrant",
    "Collection already exists",
    "Failed to initialize Mem0",
    "Long-term memory disabled",
    "To enable Mem0",
)


def _sse(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_result(result: dict, execution_time: float, session_id: str) -> dict:
    """统一构造响应结构"""
    return {
        "session_id": session_id,
        "answer": result.get("answer", ""),
        "swarm_enabled": result.get("swarm_enabled", False),
        "agents_involved": result.get("agents_involved", []),
        "timeout_occurred": result.get("timeout_occurred", False),
        "execution_time": round(execution_time, 2),
        "suggestions": result.get("suggestions", []),
        "disclaimer": result.get("disclaimer", ""),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式聊天：SSE 推送系统内部状态，完成后一次性返回完整回答

    事件类型：
    - status: 系统内部进度（任务分解 / 路由 / 工具调用 / 知识库检索等）
    - done:   处理完成（含完整回答、建议、免责声明、模式徽章）
    - error:  处理失败
    """
    session_id = req.session_id or f"web-{uuid.uuid4().hex[:8]}"

    # 熔断机制：非医疗类问题在创建处理任务前拦截
    verdict = guard.check(req.message, session_id=session_id)
    if not verdict["allowed"]:
        async def reject_gen():
            yield _sse("rejected", {
                "rejected": True,
                "circuit": verdict["circuit"],
                "message": verdict["message"],
                "session_id": session_id,
            })
        return StreamingResponse(
            reject_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def log_sink(message):
        """loguru sink：把 INFO 日志桥接到 asyncio 队列（线程安全）"""
        record = message.record
        if record["level"].no < 20:  # 只转发 INFO 及以上
            return
        text = record["message"]
        if len(text.strip()) < 3:
            return
        if any(k in text for k in _NOISE_KEYWORDS):
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, text)
        except RuntimeError:
            pass

    sink_id = logger.add(log_sink)

    async def event_gen():
        start = time.time()
        try:
            # 后台执行 Swarm 处理（与事件流并发）
            task = asyncio.create_task(
                process_with_swarm(req.message, session_id=session_id)
            )
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.2)
                # 转发期间产生的日志
                while not q.empty():
                    try:
                        text = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    yield _sse("status", {"message": text})
                if done:
                    break
            result = task.result()
            yield _sse("done", _build_result(result, time.time() - start, session_id))
        except Exception as e:
            logger.exception(f"[web] 流式处理失败: {e}")
            yield _sse("error", {"error": f"处理请求时出错: {e}"})
        finally:
            logger.remove(sink_id)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
