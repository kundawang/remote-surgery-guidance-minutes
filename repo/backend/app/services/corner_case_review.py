"""自动驾驶路测评审：corner case 分析、仿真平台对接与报告生成。

设计要点：
- 仿真 id 使用 SHA-256（标准库 hashlib）派生，跨进程、跨机器稳定可复现，
  不再使用 Python 内建 hash()（其字符串结果随进程随机化）。
- 本地兜底 id 使用 ``LOCAL_`` 前缀，与仿真平台返回的远端 id 命名空间隔离，
  报告中可区分「已提交到仿真平台」与「仅本地创建」。
- 分析失败（模型超时 / 返回格式不对）返回明确的失败状态与错误原因，
  不会把异常伪装成 severity=unknown 的 corner case。
- 仿真平台返回非 2xx 时不静默降级为本地成功，结果中携带状态码与错误信息。
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

SIM_ID_VERSION = "1.5"
SIM_ID_PREFIX = f"SIM_{SIM_ID_VERSION}_"
LOCAL_SIM_ID_PREFIX = f"LOCAL_{SIM_ID_PREFIX}"

# corner case 必备字段，缺失即视为模型返回格式不对
_REQUIRED_CASE_FIELDS = ("description", "timestamp", "category")


def generate_simulation_id(description: str, timestamp: Any, category: str) -> str:
    """根据 corner case 的（描述, 时间点, 分类）生成稳定可复现的仿真 id。

    同一组输入在任何进程、任何机器上都得到同一个 id；任一字段不同则 id 不同。
    仅使用标准库 SHA-256，不引入新依赖。
    """
    payload = "\x1f".join(
        [str(description or ""), str(timestamp or ""), str(category or "")]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{SIM_ID_PREFIX}{int(digest[:8], 16) % 10000:04d}"


def generate_local_simulation_id(description: str, timestamp: Any, category: str) -> str:
    """本地兜底 id：与远端 id 命名空间隔离（LOCAL_ 前缀），不会互相冲突。"""
    return LOCAL_SIM_ID_PREFIX + generate_simulation_id(
        description, timestamp, category
    )[len(SIM_ID_PREFIX):]


def is_local_simulation_id(simulation_id: str) -> bool:
    return bool(simulation_id) and simulation_id.startswith(LOCAL_SIM_ID_PREFIX)


class AnalysisError(Exception):
    """corner case 分析失败（模型超时、返回格式不对等）。"""


class _HeuristicAnalyzer:
    """默认分析器：基于关键词的简单启发式，便于在无模型环境下运行。

    生产环境应注入真实的模型分析器（callable，签名见
    ``CornerCaseReviewService.__init__``）；模型超时或返回格式不对时，
    由服务层统一转换为失败状态。
    """

    KEYWORDS = ("加塞", "急刹", "逆行", "闯红灯", "鬼探头", "cut in", "emergency")

    def __call__(self, transcript: str) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for line in (transcript or "").splitlines():
            text = line.strip()
            if not text:
                continue
            if any(kw in text for kw in self.KEYWORDS):
                cases.append(
                    {
                        "description": text,
                        "timestamp": "",
                        "category": "heuristic",
                        "severity": "medium",
                    }
                )
        return cases


class _UrllibHttpClient:
    """默认 HTTP 客户端（仅标准库）。测试或调用方可注入自定义 client。"""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def post(self, url: str, payload: Dict[str, Any]) -> "HttpResponse":
        return self._request("POST", url, payload)

    def get(self, url: str) -> "HttpResponse":
        return self._request("GET", url, None)

    def _request(
        self, method: str, url: str, payload: Optional[Dict[str, Any]]
    ) -> "HttpResponse":
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return HttpResponse(resp.status, resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # 非 2xx：保留状态码与响应体
            return HttpResponse(exc.code, exc.read().decode("utf-8", "replace"))


class HttpResponse:
    def __init__(self, status_code: int, body: str = ""):
        self.status_code = status_code
        self.body = body

    def json(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.body)
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}


class CornerCaseReviewService:
    """路测评审服务：分析 corner case、提交仿真平台、生成报告。

    参数:
        analyzer: 可调用对象，输入会议文本，返回 corner case dict 列表。
                  抛异常或返回格式不对时，analyze_corner_cases 返回失败状态。
        http_client: 具有 post(url, payload) / get(url) 方法的对象，
                  返回带 status_code / json() 的响应。默认使用标准库实现。
        simulation_platform_url: 仿真平台地址；为空时所有测试仅本地创建。
    """

    def __init__(
        self,
        analyzer: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        http_client: Optional[Any] = None,
        simulation_platform_url: Optional[str] = None,
    ):
        self._analyzer = analyzer or _HeuristicAnalyzer()
        self._http = http_client or _UrllibHttpClient()
        self._platform_url = (simulation_platform_url or "").rstrip("/")

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    def analyze_corner_cases(self, transcript: str) -> Dict[str, Any]:
        """分析会议文本中的 corner case。

        返回字段（签名与字段名保持稳定，前端/报告模板依赖）：
            success: 是否分析成功
            corner_cases: 真实识别到的 corner case 列表（失败时为空）
            count: 真实场景数量（不含任何失败占位）
            error: 失败原因；成功时为 None
        """
        try:
            raw = self._analyzer(transcript)
            cases = self._validate_cases(raw)
        except Exception as exc:  # 模型超时 / 返回格式不对 / 其他异常
            return {
                "success": False,
                "corner_cases": [],
                "count": 0,
                "error": f"分析失败: {exc}",
            }

        for case in cases:
            case["simulation_id"] = generate_simulation_id(
                case["description"], case["timestamp"], case["category"]
            )
        return {
            "success": True,
            "corner_cases": cases,
            "count": len(cases),
            "error": None,
        }

    @staticmethod
    def _validate_cases(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            raise AnalysisError(f"模型返回格式不对: 期望列表, 实际 {type(raw).__name__}")
        cases = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise AnalysisError(f"模型返回格式不对: 第 {i} 项不是对象")
            missing = [f for f in _REQUIRED_CASE_FIELDS if f not in item]
            if missing:
                raise AnalysisError(
                    f"模型返回格式不对: 第 {i} 项缺少字段 {', '.join(missing)}"
                )
            case = dict(item)
            case.setdefault("severity", "medium")
            cases.append(case)
        return cases

    # ------------------------------------------------------------------
    # 仿真平台
    # ------------------------------------------------------------------
    def create_simulation_test(self, corner_case: Dict[str, Any]) -> Dict[str, Any]:
        """把 corner case 提交到仿真平台创建测试。

        返回字段：
            success / simulation_id / submitted / source / status_code / error
        - 远端 2xx：submitted=True, source="remote", id 取平台返回值。
        - 远端非 2xx：success=False，带 status_code 与 error，不静默降级。
        - 平台不可达（网络异常）：本地兜底，id 带 LOCAL_ 前缀，
          submitted=False, source="local"。
        """
        description = corner_case.get("description", "")
        timestamp = corner_case.get("timestamp", "")
        category = corner_case.get("category", "")
        local_id = generate_local_simulation_id(description, timestamp, category)

        result: Dict[str, Any] = {
            "success": False,
            "simulation_id": None,
            "submitted": False,
            "source": None,
            "status_code": None,
            "error": None,
        }

        if not self._platform_url:
            result.update(
                success=True,
                simulation_id=local_id,
                source="local",
                error="未配置仿真平台地址, 仅本地创建",
            )
            return result

        try:
            resp = self._http.post(
                f"{self._platform_url}/tests",
                {
                    "description": description,
                    "timestamp": timestamp,
                    "category": category,
                    "severity": corner_case.get("severity"),
                    "client_reference_id": generate_simulation_id(
                        description, timestamp, category
                    ),
                },
            )
        except Exception as exc:  # 平台不可达：明确的本地兜底，而非假成功
            result.update(
                success=True,
                simulation_id=local_id,
                source="local",
                error=f"仿真平台不可达, 仅本地创建: {exc}",
            )
            return result

        status = getattr(resp, "status_code", None)
        if status is not None and 200 <= status < 300:
            body = resp.json() if hasattr(resp, "json") else {}
            remote_id = (
                body.get("simulation_id")
                or body.get("id")
                or generate_simulation_id(description, timestamp, category)
            )
            result.update(
                success=True,
                simulation_id=remote_id,
                submitted=True,
                source="remote",
                status_code=status,
            )
            return result

        # 非 2xx：不降级为本地成功，带上状态码与错误信息供调用方判断
        detail = ""
        if hasattr(resp, "json"):
            detail = resp.json().get("error", "") or ""
        if not detail:
            detail = getattr(resp, "body", "") or getattr(resp, "text", "") or ""
        result.update(
            status_code=status,
            error=f"仿真平台返回错误 (HTTP {status}): {detail or '无详细信息'}",
        )
        return result

    def get_test_status(self, simulation_id: str) -> Dict[str, Any]:
        """查询仿真测试状态。本地创建的 id 不请求远端，直接标记 local_only。"""
        if is_local_simulation_id(simulation_id):
            return {
                "success": True,
                "simulation_id": simulation_id,
                "status": "local_only",
                "submitted": False,
                "source": "local",
                "status_code": None,
                "error": None,
            }
        if not self._platform_url:
            return {
                "success": False,
                "simulation_id": simulation_id,
                "status": None,
                "submitted": None,
                "source": None,
                "status_code": None,
                "error": "未配置仿真平台地址, 无法查询远端测试状态",
            }
        try:
            resp = self._http.get(f"{self._platform_url}/tests/{simulation_id}")
        except Exception as exc:
            return {
                "success": False,
                "simulation_id": simulation_id,
                "status": None,
                "submitted": None,
                "source": None,
                "status_code": None,
                "error": f"查询仿真平台失败: {exc}",
            }
        status = getattr(resp, "status_code", None)
        if status is not None and 200 <= status < 300:
            body = resp.json() if hasattr(resp, "json") else {}
            return {
                "success": True,
                "simulation_id": simulation_id,
                "status": body.get("status", "unknown"),
                "submitted": True,
                "source": "remote",
                "status_code": status,
                "error": None,
            }
        return {
            "success": False,
            "simulation_id": simulation_id,
            "status": None,
            "submitted": None,
            "source": None,
            "status_code": status,
            "error": f"仿真平台返回错误 (HTTP {status})",
        }

    # ------------------------------------------------------------------
    # 策略 / 摘要 / 报告
    # ------------------------------------------------------------------
    def generate_control_strategies(
        self, corner_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """针对真实 corner case 生成控制策略建议。"""
        strategies = []
        for case in corner_cases or []:
            strategies.append(
                {
                    "simulation_id": case.get("simulation_id"),
                    "description": case.get("description", ""),
                    "category": case.get("category", ""),
                    "strategy": (
                        f"针对「{case.get('category', '未知分类')}」场景回归验证: "
                        f"{case.get('description', '')}"
                    ),
                }
            )
        return {"strategies": strategies, "count": len(strategies)}

    def generate_meeting_summary(
        self,
        analysis_result: Dict[str, Any],
        strategies_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """生成会议纪要摘要。统计只包含真实场景，分析失败单独列出。"""
        cases = analysis_result.get("corner_cases", []) or []
        error = analysis_result.get("error")
        strategies = (strategies_result or {}).get("strategies", []) or []
        summary_lines = [f"本次评审共识别到 {len(cases)} 个极端场景。"]
        if error:
            summary_lines.append(f"分析未完成: {error}")
        return {
            "total_corner_cases": len(cases),
            "analysis_succeeded": bool(analysis_result.get("success", False)),
            "analysis_error": error,
            "strategy_count": len(strategies),
            "summary": "\n".join(summary_lines),
        }

    def generate_markdown_report(
        self,
        analysis_result: Dict[str, Any],
        strategies_result: Optional[Dict[str, Any]] = None,
        simulation_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """生成 Markdown 评审报告。

        「共识别到 N 个极端场景」只统计真实场景；分析失败在
        「分析告警」中单独提示；仿真测试区分「已提交到仿真平台」
        与「仅本地创建」。
        """
        cases = analysis_result.get("corner_cases", []) or []
        error = analysis_result.get("error")
        strategies = (strategies_result or {}).get("strategies", []) or []
        simulations = simulation_results or []

        lines = ["# 自动驾驶路测评审报告", ""]
        lines.append(f"共识别到 {len(cases)} 个极端场景")
        lines.append("")

        if error:
            lines += ["## 分析告警", "", f"- 分析失败: {error}", ""]

        if cases:
            lines += ["## 极端场景列表", ""]
            for i, case in enumerate(cases, 1):
                lines.append(
                    f"{i}. [{case.get('severity', '-')}] "
                    f"{case.get('description', '')} "
                    f"(分类: {case.get('category', '-')}, "
                    f"仿真 id: {case.get('simulation_id', '-')})"
                )
            lines.append("")

        if strategies:
            lines += ["## 控制策略建议", ""]
            for s in strategies:
                lines.append(f"- {s.get('strategy', '')}")
            lines.append("")

        if simulations:
            submitted = [s for s in simulations if s.get("submitted")]
            local_only = [s for s in simulations if not s.get("submitted")]
            lines += ["## 仿真测试", ""]
            lines.append(f"- 已提交到仿真平台: {len(submitted)} 个")
            lines.append(f"- 仅本地创建: {len(local_only)} 个")
            for s in simulations:
                state = "已提交到仿真平台" if s.get("submitted") else "仅本地创建"
                note = f"（{s['error']}）" if s.get("error") else ""
                lines.append(f"- {s.get('simulation_id', '-')}: {state}{note}")
            lines.append("")

        return "\n".join(lines)
