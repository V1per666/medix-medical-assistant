/* ============================================================
   MediX 多智能体医疗助手 - 前端逻辑
   ============================================================ */
(() => {
  "use strict";

  const chatEl = document.getElementById("chat");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusPill = document.getElementById("statusPill");
  const welcomeEl = document.getElementById("welcome");

  let sessionId = null;
  let busy = false;

  /* ---------- 会话 ID（保持多轮对话上下文） ---------- */
  function getSessionId() {
    if (sessionId) return sessionId;
    const saved = sessionStorage.getItem("medix_session_id");
    if (saved) { sessionId = saved; return saved; }
    sessionId = "web-" + (crypto.randomUUID ? crypto.randomUUID().slice(0, 8) : Date.now().toString(36));
    sessionStorage.setItem("medix_session_id", sessionId);
    return sessionId;
  }

  /* ---------- 轻量 Markdown 渲染（无外部依赖） ---------- */
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function inline(text) {
    return esc(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }

  function renderMarkdown(text) {
    // 提取代码块
    const blocks = [];
    let body = text.replace(/```([\s\S]*?)```/g, (_, code) => {
      blocks.push(`<pre>${esc(code.trim())}</pre>`);
      return `\u0000${blocks.length - 1}\u0000`;
    });

    const lines = body.split("\n");
    const out = [];
    let i = 0;
    const maxIter = lines.length * 2 + 100;  // 防死循环保护
    let guard = 0;

    while (i < lines.length) {
      if (++guard > maxIter) break;  // 防死循环保护
      const line = lines[i];

      // 表格
      if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|[\s:|-]+\|/.test(lines[i + 1])) {
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith("|")) {
          rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
          i++;
        }
        let html = "<table>";
        rows.forEach((row, ri) => {
          const tag = ri === 0 ? "th" : "td";
          html += "<tr>" + row.map((c) => `<${tag}>${inline(c)}</${tag}>`).join("") + "</tr>";
        });
        html += "</table>";
        out.push(html);
        continue;
      }

      // 标题
      const h = line.match(/^(#{2,4})\s+(.*)/);
      if (h) {
        out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
        i++;
        continue;
      }

      // 分隔线
      if (/^\s*---+\s*$/.test(line)) {
        out.push("<hr>");
        i++;
        continue;
      }

      // 无序列表（收集连续项）
      if (/^\s*[-*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
          i++;
        }
        out.push(`<ul>${items.join("")}</ul>`);
        continue;
      }

      // 有序列表
      if (/^\s*\d+[.、)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.、)]\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*\d+[.、)]\s+/, ""))}</li>`);
          i++;
        }
        out.push(`<ol>${items.join("")}</ol>`);
        continue;
      }

      // 代码块占位
      const ph = line.match(/^\u0000(\d+)\u0000$/);
      if (ph) {
        out.push(blocks[Number(ph[1])]);
        i++;
        continue;
      }

      // 空行
      if (!line.trim()) { i++; continue; }

      // 普通段落（合并连续文本行）
      // 注意：列表/标题判定必须要求符号后跟空白（如 [-*]\s），否则行首 **粗体** 会被误判为列表行导致死循环
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(#|[-*]\s|\d+[.、)]\s|\||---+)/.test(lines[i]) && !/^\u0000\d+\u0000$/.test(lines[i])) {
        para.push(inline(lines[i].trim()));
        i++;
      }
      out.push(`<p>${para.join("<br>")}</p>`);
    }

    return out.join("");
  }

  /* ---------- 消息渲染 ---------- */
  function addUserMsg(text) {
    welcomeEl && welcomeEl.remove();
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `
      <div class="avatar">👤</div>
      <div class="bubble"></div>`;
    el.querySelector(".bubble").textContent = text;
    chatEl.appendChild(el);
    scrollBottom();
    return el;
  }

  function addBotMsg(data) {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `
      <div class="avatar">✚</div>
      <div class="content">
        <div class="bubble"></div>
        <div class="msg-meta"></div>
      </div>`;

    // 模式徽章
    const meta = el.querySelector(".msg-meta");
    const modeBadge = data.swarm_enabled
      ? `<span class="badge swarm">🐝 Swarm 协作 · ${data.agents_involved.length} Agents</span>`
      : `<span class="badge single">🤖 单 Agent</span>`;
    meta.insertAdjacentHTML("beforeend", modeBadge);

    if (data.execution_time) {
      meta.insertAdjacentHTML("beforeend", `<span class="badge time">⏱ ${data.execution_time}s</span>`);
    }
    if (data.timeout_occurred) {
      meta.insertAdjacentHTML("beforeend", `<span class="badge" style="color:var(--rose)">⚠ 部分 Agent 超时</span>`);
    }

    // 回答
    el.querySelector(".bubble").innerHTML = renderMarkdown(data.answer || "（无回答）");

    // 核心建议
    if (data.suggestions && data.suggestions.length) {
      const sug = document.createElement("div");
      sug.className = "sug-box";
      sug.innerHTML = `
        <div class="sug-title">核心建议</div>
        <ul class="sug-list">${data.suggestions.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`;
      el.querySelector(".content").appendChild(sug);
    }

    // 免责声明
    if (data.disclaimer) {
      const note = document.createElement("div");
      note.className = "legal-note";
      note.textContent = data.disclaimer;
      el.querySelector(".content").appendChild(note);
    }

    chatEl.appendChild(el);
    scrollBottom();
    return el;
  }

  function addErrorMsg(err) {
    const el = document.createElement("div");
    el.className = "msg bot error";
    el.innerHTML = `
      <div class="avatar">⚠️</div>
      <div class="bubble"></div>`;
    el.querySelector(".bubble").textContent = err;
    chatEl.appendChild(el);
    scrollBottom();
  }

  /* ---------- 熔断/拦截提示 ---------- */
  function addRejected(data) {
    const el = document.createElement("div");
    el.className = "msg bot rejected";
    el.innerHTML = `
      <div class="avatar">🔒</div>
      <div class="bubble">
        <div class="rejected-title"></div>
        <div class="rejected-body"></div>
      </div>`;
    el.querySelector(".rejected-title").textContent = data.circuit
      ? "⛔ 已触发临时熔断"
      : "⚠️ 超出咨询范围";
    el.querySelector(".rejected-body").textContent =
      data.message || "本助手仅支持医疗健康类问题咨询，请描述您的症状或健康问题";
    chatEl.appendChild(el);
    scrollBottom();
  }

  /* ---------- 思考指示器（实时状态日志面板） ---------- */
  const STATUS_ICONS = [
    [/LeadAgent|分解任务/, "📋"],
    [/Route/, "🧭"],
    [/Created SubTask|subtasks/, "🐝"],
    [/Starting Agent Loop/, "🤖"],
    [/LLM requested/, "🧠"],
    [/Searching|知识库|KB results/, "🔍"],
    [/Analyzing/, "🩺"],
    [/Assessing/, "⚠️"],
    [/Recommending/, "💊"],
    [/deep research|DeepResearch/, "🔬"],
    [/ICD-10/, "📑"],
    [/clinical guidelines/, "📖"],
    [/Web searching/, "🌐"],
    [/Planned|planning/, "🗂️"],
    [/Synthesizing|report generated|Report/, "📊"],
    [/Completed|finished|完成/, "🏁"],
    [/Saved|保存/, "💾"],
    [/constraint|约束/, "🛡️"],
  ];

  function statusIcon(text) {
    for (const [re, icon] of STATUS_ICONS) {
      if (re.test(text)) return icon;
    }
    return "·";
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "msg bot typing";
    el.id = "typingMsg";
    el.innerHTML = `
      <div class="avatar">✚</div>
      <div class="bubble typing-panel">
        <div class="typing-head">
          <svg class="ecg-mini" viewBox="0 0 120 26" preserveAspectRatio="none">
            <path class="ecg-line" d="M0 13 H22 L26 13 L29 5 L33 21 L36 13 H54
                     L58 13 L61 7 L65 19 L68 13 H120"/>
          </svg>
          <span class="typing-label">多智能体协作分析中</span>
        </div>
        <div class="typing-log" id="typingLog"></div>
      </div>`;
    chatEl.appendChild(el);
    scrollBottom();
    return el;
  }
  function hideTyping() {
    const t = document.getElementById("typingMsg");
    t && t.remove();
  }

  function appendStatus(text) {
    const log = document.getElementById("typingLog");
    if (!log) return;
    const row = document.createElement("div");
    row.className = "log-row";
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    row.innerHTML = `<span class="log-ts">${ts}</span><span class="log-ic">${statusIcon(text)}</span><span class="log-tx"></span>`;
    row.querySelector(".log-tx").textContent = text;
    log.appendChild(row);
    while (log.children.length > 40) log.removeChild(log.firstChild);  // 限制行数
    scrollBottom();
  }

  /* ---------- SSE 流式解析 ---------- */
  async function readStream(resp, onEvent, onError) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        let data;
        try {
          data = JSON.parse(dataLine.slice(5).trim());
        } catch (e) {
          continue;  // 坏帧直接忽略
        }
        try {
          onEvent(data);
        } catch (e) {
          // 渲染异常不能静默吞掉，交给上层处理
          onError && onError(e);
          return;
        }
      }
    }
  }

  /* ---------- 发送（流式） ---------- */
  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;

    busy = true;
    setBusy(true);
    addUserMsg(text);
    showTyping();

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: getSessionId() }),
      });

      if (!resp.ok || !resp.body) {
        let errText = "服务器开小差了，请稍后再试";
        try {
          const j = await resp.json();
          errText = j.error || errText;
        } catch (e) { /* 忽略 */ }
        hideTyping();
        addErrorMsg(errText);
        return;
      }

      await readStream(resp, (data) => {
        // 注意：rejected 事件也含 message 字段，必须先于 status 判断
        if (data.rejected !== undefined) {
          hideTyping();
          addRejected(data);        // rejected 事件（熔断/拦截）
        } else if (data.answer !== undefined) {
          hideTyping();
          addBotMsg(data);          // done 事件
        } else if (data.error) {
          hideTyping();
          addErrorMsg(data.error);  // error 事件
        } else if (data.message !== undefined) {
          appendStatus(data.message);  // status 事件
        }
      }, (e) => {
        // 渲染异常：降级为纯文本显示，不再静默丢失
        hideTyping();
        console.error("[MediX] 渲染回答失败:", e);
        addErrorMsg("回答渲染失败，请重试（" + (e && e.message ? e.message : e) + "）");
      });
    } catch (e) {
      hideTyping();
      addErrorMsg("网络连接失败，请检查服务是否运行（uvicorn web.app:app）");
    } finally {
      busy = false;
      setBusy(false);
      inputEl.focus();
    }
  }

  function setBusy(b) {
    statusPill.classList.toggle("busy", b);
    sendBtn.disabled = b;
  }

  function scrollBottom() {
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  /* ---------- 输入自适应高度 ---------- */
  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
    sendBtn.disabled = busy || !inputEl.value.trim();
  }

  /* ---------- 事件绑定 ---------- */
  inputEl.addEventListener("input", autoResize);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = inputEl.value;
      inputEl.value = "";
      autoResize();
      send(v);
    }
  });
  sendBtn.addEventListener("click", () => {
    const v = inputEl.value;
    inputEl.value = "";
    autoResize();
    send(v);
  });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => send(chip.dataset.q));
  });

  autoResize();
})();
