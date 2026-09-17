"""
医疗意图守卫（熔断机制）

防止滥用：非医疗类问题在进入 Swarm 处理链之前被拦截，节省 LLM token。

多级拦截（成本从低到高）：
1. 医疗关键词白名单（零成本）：命中直接放行，防误杀正常医疗咨询
2. 非医疗关键词黑名单（零成本）：命中直接拒绝
3. Embedding 语义判定（零 token，本地模型）：与医疗/非医疗语义中心的余弦相似度
4. （可选）LLM 兜底：灰色地带用 LLM 二分类确认（默认关闭，max_tokens=10 成本≈0）

熔断状态：
- 同一来源（session_id/IP）连续触发 N 次非医疗拦截 → 进入冷却期
- 冷却期内所有请求直接拒绝（连判定都不做），防止恶意刷屏消耗 token
- 冷却期自动恢复；正常医疗提问重置计数

配置（外部文件，避免硬编码）：
- knowledge/data/documents/keywords/guard_config.json  开关与参数
- knowledge/data/documents/keywords/medical_keywords.txt     医疗白名单（每行一个词）
- knowledge/data/documents/keywords/non_medical_keywords.txt 非医疗黑名单（每行一个词）
"""
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ============================================================
# 配置与关键词文件目录
# ============================================================

KEYWORDS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "data" / "documents" / "keywords"
CONFIG_FILE = KEYWORDS_DIR / "guard_config.json"
MEDICAL_KEYWORDS_FILE = KEYWORDS_DIR / "medical_keywords.txt"
NON_MEDICAL_KEYWORDS_FILE = KEYWORDS_DIR / "non_medical_keywords.txt"

# ============================================================
# 兜底关键词（仅当外部关键词文件缺失时使用，避免功能失效）
# 正常维护请编辑 keywords/ 目录下的 txt 文件
# ============================================================

# 命中即放行（避免误杀正常医疗咨询）
MEDICAL_KEYWORDS = (
    # 症状
    "头痛", "头疼", "头晕", "发烧", "发热", "咳嗽", "感冒", "失眠", "疼痛", "呕吐",
    "腹泻", "乏力", "胸闷", "心悸", "心慌", "过敏", "皮疹", "流涕", "鼻塞", "喉咙",
    "呼吸困难", "气喘", "恶心", "便秘", "尿频", "水肿", "麻木", "晕厥", "耳鸣", "酸痛",
    "不适", "不舒服", "难受", "发炎", "肿痛", "出血",
    # 疾病
    "高血压", "糖尿病", "冠心病", "哮喘", "胃病", "胃炎", "肝病", "肝炎", "肾病",
    "肾结石", "癌症", "肿瘤", "炎症", "感染", "贫血", "痛风", "关节炎", "骨质疏松",
    "甲状腺", "中风", "心梗", "脑梗", "肺炎", "支气管炎", "溃疡", "流感", "新冠",
    "慢性病", "体检异常",
    # 药物与治疗
    "药", "用药", "剂量", "副作用", "阿司匹林", "布洛芬", "胰岛素", "抗生素",
    "维生素", "中药", "西药", "处方", "疫苗", "接种", "手术", "化疗", "放疗",
    "降压药", "降糖药", "退烧药", "止痛药", "消炎药", "保健品",
    # 医疗行为
    "体检", "检查", "化验", "检验", "就医", "挂号", "急诊", "康复", "复查",
    "门诊", "住院", "诊断", "治疗", "看医生", "挂科", "科室",
    # 身体指标
    "血压", "血糖", "心率", "体重", "bmi", "胆固醇", "尿酸", "血脂", "血常规",
    "尿常规", "肝功能", "肾功能", "心电图",
    # 健康管理
    "饮食", "运动", "睡眠", "营养", "养生", "减肥", "减重", "备孕", "怀孕",
    "哺乳", "健康", "作息", "锻炼", "食疗", "忌口",
)

# 命中即拒绝（明显非医疗）
NON_MEDICAL_KEYWORDS = (
    # 编程 / 技术
    "代码", "编程", "函数", "算法", "脚本", "程序", "报错", "冒泡", "排序",
    "数据库", "前端", "后端", "接口", "bug", "部署", "服务器", "docker",
    "python", "java", "javascript", "c++", "编译", "调试", "git", "爬虫",
    # 娱乐 / 生活
    "笑话", "电影", "游戏", "音乐", "小说", "明星", "八卦", "追剧", "动漫",
    "天气", "外卖", "美食推荐", "旅游攻略", "穿搭", "美妆", "购物", "股票",
    "基金", "彩票", "运势", "星座", "房价", "工资",
    # 通用闲聊 / 其他领域
    "你是谁", "介绍一下自己", "讲个故事", "写首诗", "翻译", "作文", "作业",
    "数学题", "英语", "历史", "地理", "物理", "化学",
)

# 医疗语义参考句（用于计算"医疗语义中心"）
MEDICAL_REFERENCE_SENTENCES = (
    "我最近身体不舒服，头痛发烧，应该怎么办",
    "感冒了吃什么药比较好",
    "高血压和糖尿病日常需要注意什么",
    "胸闷心悸需要去医院检查吗",
    "体检报告显示血脂偏高，怎么调理",
    "最近总是失眠，有什么改善建议",
    "孩子接种疫苗后发烧正常吗",
    "膝盖疼痛是不是关节炎",
)

# 非医疗语义参考句（用于计算"非医疗语义中心"）
NON_MEDICAL_REFERENCE_SENTENCES = (
    "帮我写一段python代码实现排序",
    "今天天气怎么样，适合出门吗",
    "推荐一部好看的电影",
    "讲个笑话逗我开心",
    "股票和基金应该怎么投资",
    "帮我翻译一段英文",
    "怎么做红烧肉这道菜",
    "最近有什么好玩的游戏",
)


class IntentGuard:
    """
    医疗意图守卫：多级拦截 + 熔断

    Args:
        threshold: 语义判定阈值（医疗/非医疗相似度差值的绝对边界）
        trip_count: 连续非医疗拦截多少次触发熔断
        cooldown: 熔断冷却时间（秒）
        use_embedding: 是否启用 Embedding 语义层（False 则仅关键词）
    """

    def __init__(
        self,
        threshold: float = 0.04,
        trip_count: int = 3,
        cooldown: int = 300,
        use_embedding: bool = True,
        enabled: Optional[bool] = None,
    ):
        """
        Args:
            threshold: 语义判定阈值（医疗/非医疗相似度差值的绝对边界）
            trip_count: 连续非医疗拦截多少次触发熔断
            cooldown: 熔断冷却时间（秒）
            use_embedding: 是否启用 Embedding 语义层（False 则仅关键词）
            enabled: 是否启用拦截；None 时从 guard_config.json 读取（默认）
        """
        # 从外部配置文件读取参数（文件不存在时使用构造默认值）
        config = self._load_config()
        self.enabled = enabled if enabled is not None else config.get("enabled", True)
        self.threshold = config.get("threshold", threshold)
        self.trip_count = int(config.get("trip_count", trip_count))
        self.cooldown = int(config.get("cooldown_seconds", cooldown))
        self.use_embedding = use_embedding

        # 从外部文件加载关键词（文件缺失时回退内置兜底词表）
        self.medical_keywords = self._load_keywords(
            MEDICAL_KEYWORDS_FILE, MEDICAL_KEYWORDS, "medical"
        )
        self.non_medical_keywords = self._load_keywords(
            NON_MEDICAL_KEYWORDS_FILE, NON_MEDICAL_KEYWORDS, "non-medical"
        )

        logger.info(
            f"IntentGuard initialized: enabled={self.enabled}, "
            f"medical_keywords={len(self.medical_keywords)}, "
            f"non_medical_keywords={len(self.non_medical_keywords)}"
        )

        # 熔断状态表: key -> {"count": int, "until": float}
        self._circuit_state: Dict[str, dict] = {}

        # Embedding 模型（复用知识库单例，避免重复加载）
        self._embedder = None
        self._medical_center: Optional[List[float]] = None
        self._non_medical_center: Optional[List[float]] = None

        if use_embedding:
            self._init_embedding()

    # ---------- 配置与关键词加载 ----------

    @staticmethod
    def _load_config() -> dict:
        """从 guard_config.json 读取配置，文件不存在或损坏时返回空 dict"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"IntentGuard config load failed, using defaults: {e}")
        return {}

    @staticmethod
    def _load_keywords(file_path: Path, fallback: tuple, kind: str) -> tuple:
        """
        从外部 txt 文件加载关键词（每行一个，# 开头为注释）
        文件缺失/为空时回退到内置兜底词表
        """
        try:
            if file_path.exists():
                words = []
                for raw in file_path.read_text(encoding="utf-8").splitlines():
                    w = raw.strip()
                    if w and not w.startswith("#"):
                        words.append(w)
                if words:
                    logger.info(f"IntentGuard loaded {len(words)} {kind} keywords from {file_path.name}")
                    return tuple(words)
                logger.warning(f"{file_path.name} is empty, fallback to built-in {kind} keywords")
            else:
                logger.warning(f"{file_path.name} not found, fallback to built-in {kind} keywords")
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}, fallback to built-in keywords")
        return fallback

    # ---------- 初始化 ----------

    def _init_embedding(self):
        """加载 embedding 模型并预计算语义中心（失败则降级为纯关键词）"""
        try:
            from knowledge.milvus_kb import MedicalKnowledgeBase

            kb = MedicalKnowledgeBase()
            model = kb.embedding_model

            med_vecs = model.encode(MEDICAL_REFERENCE_SENTENCES, normalize_embeddings=True)
            non_vecs = model.encode(NON_MEDICAL_REFERENCE_SENTENCES, normalize_embeddings=True)

            self._medical_center = med_vecs.mean(axis=0).tolist()
            self._non_medical_center = non_vecs.mean(axis=0).tolist()
            self._embedder = model
            logger.info("✅ IntentGuard embedding layer ready")
        except Exception as e:
            logger.warning(f"IntentGuard embedding init failed, fallback to keyword-only: {e}")
            self._embedder = None
            self.use_embedding = False

    # ---------- 核心判定 ----------

    def check(self, text: str, session_id: Optional[str] = None, ip: Optional[str] = None) -> dict:
        """
        检查问题是否允许进入医疗问答流程

        Returns:
            {
                "allowed": bool,       # 是否放行
                "reason": str,         # 判定依据（medical_keyword / non_medical_keyword / embedding / circuit / gray_zone）
                "circuit": bool,       # 是否处于熔断期
                "message": str,        # 拒绝时的提示文案
            }
        """
        text = (text or "").strip()
        if not text:
            return self._reject("empty", "请输入您的问题")

        # 守卫未启用：直接放行（配置 guard_config.json 的 enabled=true 启用）
        if not self.enabled:
            return self._allow("disabled")

        key = session_id or ip or "unknown"
        circuit = self._is_circuit_open(key)
        if circuit:
            return self._reject("circuit", "本助手仅支持医疗健康类问题咨询，请稍后再试", circuit=True)

        # 第 1 级：医疗关键词白名单（优先放行，防误杀）
        low = text.lower()
        if any(k in low for k in self.medical_keywords):
            return self._allow("medical_keyword")

        # 第 2 级：非医疗关键词黑名单
        if any(k in low for k in self.non_medical_keywords):
            self._record_rejection(key)
            return self._reject("non_medical_keyword", "本助手仅支持医疗健康类问题咨询，请描述您的症状或健康问题")

        # 第 3 级：Embedding 语义判定
        if self.use_embedding and self._embedder is not None:
            verdict = self._judge_by_embedding(text)
            if verdict == "non_medical":
                self._record_rejection(key)
                return self._reject("embedding", "本助手仅支持医疗健康类问题咨询，请描述您的症状或健康问题")
            if verdict == "gray":
                # 灰色地带：保守放行，避免误杀正常用户
                return self._allow("gray_zone")
            return self._allow("embedding")

        # 无 embedding 时，关键词未命中即放行（避免误杀）
        return self._allow("keyword_fallback")

    # ---------- Embedding 判定 ----------

    def _judge_by_embedding(self, text: str) -> str:
        """返回 medical / non_medical / gray"""
        try:
            vec = self._embedder.encode([text], normalize_embeddings=True)[0]
            sim_med = self._cosine(vec, self._medical_center)
            sim_non = self._cosine(vec, self._non_medical_center)
            diff = sim_med - sim_non

            if diff >= self.threshold:
                return "medical"
            if diff <= -self.threshold:
                return "non_medical"
            return "gray"
        except Exception as e:
            logger.warning(f"Embedding judgement failed: {e}")
            return "gray"

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # ---------- 熔断状态 ----------

    def _is_circuit_open(self, key: str) -> bool:
        state = self._circuit_state.get(key)
        if not state:
            return False
        if state["until"] and time.time() < state["until"]:
            return True
        # 仅当存在熔断时间且已过期时才清理状态（避免误删正常计数）
        if state["until"]:
            self._circuit_state.pop(key, None)
        return False

    def _record_rejection(self, key: str):
        """记录一次非医疗拦截，达到阈值触发熔断"""
        state = self._circuit_state.setdefault(key, {"count": 0, "until": 0})
        state["count"] += 1
        if state["count"] >= self.trip_count:
            state["until"] = time.time() + self.cooldown
            state["count"] = 0
            logger.warning(f"🔴 IntentGuard: {key} 连续 {self.trip_count} 次非医疗问题，熔断 {self.cooldown}s")
        else:
            logger.info(f"IntentGuard: {key} 非医疗拦截 {state['count']}/{self.trip_count}")

    def _record_allowed(self, key: str):
        """正常放行时重置计数"""
        state = self._circuit_state.get(key)
        if state and not state["until"]:
            state["count"] = 0

    # ---------- 结果构造 ----------

    def _allow(self, reason: str) -> dict:
        return {"allowed": True, "reason": reason, "circuit": False, "message": ""}

    def _reject(self, reason: str, message: str, circuit: bool = False) -> dict:
        return {"allowed": False, "reason": reason, "circuit": circuit, "message": message}

    def get_status(self) -> dict:
        """状态概览（用于调试/监控）"""
        now = time.time()
        open_circuits = {k: int(v["until"] - now) for k, v in self._circuit_state.items() if v["until"] and v["until"] > now}
        return {
            "embedding_enabled": self.use_embedding,
            "threshold": self.threshold,
            "trip_count": self.trip_count,
            "cooldown": self.cooldown,
            "open_circuits": open_circuits,
        }


# 全局单例（与知识库共享 embedding 模型）
_guard_instance: Optional[IntentGuard] = None


def get_guard() -> IntentGuard:
    """获取全局守卫单例"""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = IntentGuard()
    return _guard_instance
