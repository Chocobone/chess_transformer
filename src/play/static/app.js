// 대국 흐름: 설정 창, API 호출, 기보·후보·비교 패널.
"use strict";

const REASONS = {
  checkmate: "체크메이트", stalemate: "스테일메이트", insufficient_material: "기물 부족",
  threefold_repetition: "3회 반복", fifty_moves: "50수 규칙", seventyfive_moves: "75수 규칙",
  fivefold_repetition: "5회 반복", resign: "기권",
};
const LOW_PROB = 0.1; // 모델이 둔 수의 π가 이보다 낮으면 기보에 따로 표시한다
const COMPARE_KEY = "chess-app.compare";
const MODEL_KEY = "chess-app.model";

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;

const app = {
  models: [],
  compareIds: [],
  game: null, // 서버의 대국 상태
  view: null, // 되돌려 보기 중인 ply (0 = 시작 국면). null이면 현재 국면
  thinking: false,
  compareOn: false,
  compareCache: new Map(), // fen → 결과
  compareRequest: 0,
};

const storage = {
  get(key) { try { return window.localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { window.localStorage.setItem(key, value); } catch { /* 저장소를 못 쓰면 기억하지 않는다 */ } },
};

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 4000);
}

const board = new ChessBoard($("board"), { onMove: userMove });

// ---------- 국면 ----------

function plyCount() { return app.game ? app.game.moves.length : 0; }
function viewPly() { return app.view === null ? plyCount() : app.view; }
function isLive() { return app.view === null || app.view === plyCount(); }

function fenAt(ply) {
  return ply === 0 ? app.game.initial_fen : app.game.moves[ply - 1].fen;
}

function userCanMove() {
  return app.game && isLive() && !app.thinking && !app.game.outcome && app.game.legal_moves.length > 0;
}

function modelToMove() {
  return app.game && !app.game.outcome && app.game.turn !== app.game.user_color;
}

// ---------- 그리기 ----------

function render() {
  const game = app.game;
  if (!game) return;
  const ply = viewPly();
  board.set({
    fen: fenAt(ply),
    orientation: game.user_color,
    lastMove: ply > 0 ? game.moves[ply - 1].uci : null,
    check: isLive() ? game.check : null,
    legalMoves: userCanMove() ? game.legal_moves : [],
    interactive: userCanMove(),
  });
  renderPlayers();
  renderInfo();
  renderMoves();
  renderCandidates();
  renderOutcome();
  renderControls();
  renderCompare();
}

function colorName(color) { return color === "white" ? "백" : "흑"; }

function renderPlayers() {
  const game = app.game;
  const modelColor = game.user_color === "white" ? "black" : "white";
  $("opponent-name").textContent = game.model.label;
  $("opponent").className = `player opponent ${modelColor}` + (game.turn === modelColor && !game.outcome ? " to-move" : "");
  $("me").className = `player me ${game.user_color}` + (game.turn === game.user_color && !game.outcome ? " to-move" : "");
  const analysis = game.last_analysis;
  $("opponent-extra").innerHTML = app.thinking
    ? '<span class="thinking">생각 중<span>.</span><span>.</span><span>.</span></span>'
    : analysis ? `<span title="모델이 자기 수를 고를 때 본 예상 승률 (모델 기준)">예상 승률 ${pct(analysis.win_prob)}</span>` : "";
  $("me-extra").textContent = colorName(game.user_color);
}

function renderInfo() {
  const game = app.game;
  const outcome = game.outcome ? `<div class="info-result">${resultText(game.outcome)}</div>` : "";
  $("game-info-body").innerHTML = `
    <div class="info-row"><span class="muted">상대</span><span>${game.model.label}</span></div>
    <div class="info-row"><span class="muted">내 색</span><span>${colorName(game.user_color)}</span></div>
    <div class="info-row"><span class="muted">수 선택</span><span title="학습 정책 그대로">π 샘플링 · T 0.02</span></div>
    ${outcome}`;
}

function resultText(outcome) {
  const who = outcome.winner ? `${colorName(outcome.winner)} 승` : "무승부";
  return `${outcome.result} · ${who} · ${REASONS[outcome.reason] || outcome.reason}`;
}

function renderMoves() {
  const game = app.game;
  const current = viewPly();
  const rows = [];
  for (let i = 0; i < game.moves.length; i += 2) {
    const cell = (ply) => {
      const move = game.moves[ply - 1];
      if (!move) return '<span class="move empty"></span>';
      const classes = ["move", move.by];
      if (ply === current) classes.push("active");
      let extra = "";
      if (move.by === "model" && move.prob !== null) {
        if (move.prob < LOW_PROB) classes.push("low");
        extra += `<span class="prob" title="모델이 이 수에 둔 확률 π">${pct(move.prob)}</span>`;
      }
      if (move.repetition) extra += '<span class="rep" title="이 수로 같은 국면이 세 번째가 되었습니다">⟳</span>';
      return `<span class="${classes.join(" ")}" data-ply="${ply}"><span class="san">${move.san}</span>${extra}</span>`;
    };
    rows.push(`<div class="move-row"><span class="move-no">${i / 2 + 1}</span>${cell(i + 1)}${cell(i + 2)}</div>`);
  }
  if (game.outcome) rows.push(`<div class="move-result">${game.outcome.result}<br><span>${resultText(game.outcome)}</span></div>`);
  const el = $("moves");
  el.innerHTML = rows.join("") || '<div class="empty">아직 둔 수가 없습니다</div>';
  const active = el.querySelector(".move.active");
  if (active && isLive()) el.scrollTop = el.scrollHeight;
  else if (active) active.scrollIntoView({ block: "nearest" });
}

function candidateRows(candidates, played) {
  return candidates.map((c) => `
    <div class="cand${c.uci === played ? " played" : ""}">
      <span class="cand-san">${c.san}</span>
      <span class="bar"><span style="width:${Math.max(c.prob * 100, 0.5)}%"></span></span>
      <span class="cand-prob" title="π(a|s)">${pct(c.prob)}</span>
      <span class="cand-q" title="q(s,a): 이 수를 둔 뒤 모델 쪽 예상 승률">${pct(c.q)}</span>
    </div>`).join("");
}

function renderCandidates() {
  // 보고 있는 국면까지 모델이 마지막으로 둔 수의 후보를 보인다
  const moves = app.game.moves.slice(0, viewPly());
  const index = moves.map((m) => m.by).lastIndexOf("model");
  const body = $("candidates-body");
  if (index < 0 || !moves[index].analysis) {
    $("candidates-caption").textContent = "";
    body.innerHTML = '<div class="empty">모델이 수를 두면 후보가 나타납니다</div>';
    return;
  }
  const move = moves[index], analysis = move.analysis;
  $("candidates-caption").textContent =
    `${Math.floor(index / 2) + 1}수 ${move.san} · 합법수 ${analysis.legal_count}개 · 막대 π, 오른쪽 q`;
  body.innerHTML = candidateRows(analysis.candidates, move.uci);
}

function renderOutcome() {
  const overlay = $("result-overlay");
  const outcome = app.game.outcome;
  if (!outcome || !isLive() || app.dismissedOutcome === app.game.moves.length) { overlay.hidden = true; return; }
  overlay.innerHTML = `<div class="result-card">
      <div class="result-score">${outcome.result}</div>
      <div>${resultText(outcome)}</div>
      <div class="result-actions">
        <button class="button primary" data-action="new">새 대국</button>
        <button class="button" data-action="close">닫기</button>
      </div></div>`;
  overlay.hidden = false;
}

function renderControls() {
  const game = app.game;
  const hasUserMove = game.moves.some((m) => m.by === "user");
  $("undo").disabled = app.thinking || !hasUserMove;
  $("resign").disabled = app.thinking || !!game.outcome;
  $("nav-first").disabled = $("nav-prev").disabled = viewPly() === 0;
  $("nav-next").disabled = $("nav-last").disabled = isLive();
}

// ---------- 비교 패널 ----------

async function renderCompare() {
  const panel = $("compare-panel");
  panel.hidden = !app.compareOn;
  if (!app.compareOn || !app.game) return;
  const ply = viewPly();
  const fen = fenAt(ply);
  const played = ply < plyCount() ? app.game.moves[ply].uci : null;
  $("compare-caption").textContent = ply === 0 ? "시작 국면" : `${Math.ceil(ply / 2)}수 ${ply % 2 ? "백" : "흑"} 다음`;
  const body = $("compare-body");
  const show = (results) => {
    body.innerHTML = results.map((r) => `
      <div class="compare-model">
        <div class="compare-name">${r.label}<span class="muted">${pct(r.win_prob)}</span></div>
        ${candidateRows(r.candidates, played)}
      </div>`).join("");
  };
  if (app.compareCache.has(fen)) { show(app.compareCache.get(fen)); return; }
  const request = ++app.compareRequest;
  body.innerHTML = '<div class="empty">계산 중…</div>';
  try {
    const data = await api("POST", "/api/analyze", { fen, models: app.compareIds });
    app.compareCache.set(fen, data.results);
    if (request === app.compareRequest) show(data.results);
  } catch (error) {
    if (request === app.compareRequest) {
      body.innerHTML = `<div class="empty">${error.status === 400 ? "종료 국면입니다" : error.message}</div>`;
    }
  }
}

// ---------- 대국 진행 ----------

async function userMove(from, to, needsPromotion) {
  let uci = from + to;
  if (needsPromotion) {
    const piece = await board.choosePromotion($("promotion"), to, app.game.user_color === "white" ? "w" : "b");
    if (!piece) { render(); return; }
    uci += piece;
  }
  try {
    app.game = await api("POST", "/api/game/move", { uci });
    app.view = null;
    render();
    await modelMove();
  } catch (error) {
    toast(error.message);
    await refresh();
  }
}

async function modelMove() {
  if (!modelToMove() || app.thinking) return;
  app.thinking = true;
  render();
  try {
    app.game = await api("POST", "/api/game/engine");
  } catch (error) {
    toast(`모델 수를 받지 못했습니다: ${error.message}`);
  } finally {
    app.thinking = false;
    app.view = null;
    render();
  }
}

async function refresh() {
  try {
    app.game = await api("GET", "/api/game");
  } catch (error) {
    if (error.status !== 404) toast(error.message);
    app.game = null;
  }
  app.view = null;
  if (app.game) render();
}

async function startGame(color) {
  const model = document.querySelector(".model-option.selected")?.dataset.model;
  if (!model) { toast("모델을 고르세요"); return; }
  storage.set(MODEL_KEY, model);
  try {
    app.game = await api("POST", "/api/game", { model, color });
  } catch (error) {
    toast(error.message);
    return;
  }
  $("setup").hidden = true;
  app.view = null;
  app.dismissedOutcome = null;
  render();
  await modelMove();
}

// ---------- 설정 창 ----------

function openSetup() {
  const remembered = storage.get(MODEL_KEY);
  const selected = app.models.some((m) => m.id === remembered) ? remembered
    : app.game ? app.game.model.id : (app.models.find((m) => m.id.startsWith("tal")) || app.models[0])?.id;
  const groups = [
    ["기사 어댑터", app.models.filter((m) => !m.id.startsWith("base-"))],
    ["베이스", app.models.filter((m) => m.id.startsWith("base-"))],
  ];
  $("model-list").innerHTML = groups.filter(([, list]) => list.length).map(([title, list]) => `
    <div class="model-group">${title}</div>
    ${list.map((m) => `<button class="model-option${m.id === selected ? " selected" : ""}" data-model="${m.id}">${m.label}</button>`).join("")}
  `).join("");
  $("setup-close").hidden = !app.game;
  $("setup").hidden = false;
}

$("model-list").addEventListener("click", (e) => {
  const option = e.target.closest(".model-option");
  if (!option) return;
  document.querySelectorAll(".model-option").forEach((el) => el.classList.toggle("selected", el === option));
});
document.querySelectorAll(".color-button").forEach((button) => {
  button.addEventListener("click", () => startGame(button.dataset.color));
});
$("setup-close").addEventListener("click", () => { $("setup").hidden = true; });
$("new-game").addEventListener("click", () => { if (!app.thinking) openSetup(); });

// ---------- 조작 ----------

function goTo(ply) {
  if (!app.game) return;
  const clamped = Math.max(0, Math.min(plyCount(), ply));
  app.view = clamped === plyCount() ? null : clamped;
  render();
}

$("moves").addEventListener("click", (e) => {
  const move = e.target.closest(".move[data-ply]");
  if (move) goTo(Number(move.dataset.ply));
});
$("nav-first").addEventListener("click", () => goTo(0));
$("nav-prev").addEventListener("click", () => goTo(viewPly() - 1));
$("nav-next").addEventListener("click", () => goTo(viewPly() + 1));
$("nav-last").addEventListener("click", () => goTo(plyCount()));
document.addEventListener("keydown", (e) => {
  if (!$("setup").hidden || e.target.closest("input, select, textarea")) return;
  const moves = { ArrowLeft: -1, ArrowRight: 1 };
  if (e.key in moves) { e.preventDefault(); goTo(viewPly() + moves[e.key]); }
  if (e.key === "Home") goTo(0);
  if (e.key === "End") goTo(plyCount());
});

$("undo").addEventListener("click", async () => {
  try {
    app.game = await api("POST", "/api/game/undo");
    app.view = null;
    app.dismissedOutcome = null;
    render();
  } catch (error) { toast(error.message); }
});
$("resign").addEventListener("click", async () => {
  if (!window.confirm("기권하시겠습니까?")) return;
  try {
    app.game = await api("POST", "/api/game/resign");
    app.view = null;
    render();
  } catch (error) { toast(error.message); }
});
$("result-overlay").addEventListener("click", (e) => {
  const action = e.target.closest("[data-action]")?.dataset.action;
  if (action === "new") openSetup();
  if (action === "close" || action === "new") {
    app.dismissedOutcome = app.game.moves.length;
    $("result-overlay").hidden = true;
  }
});
$("compare-toggle").addEventListener("change", (e) => {
  app.compareOn = e.target.checked;
  storage.set(COMPARE_KEY, app.compareOn ? "on" : "off");
  if (app.game) renderCompare();
});

// ---------- 시작 ----------

(async function init() {
  app.compareOn = storage.get(COMPARE_KEY) === "on";
  $("compare-toggle").checked = app.compareOn;
  try {
    const data = await api("GET", "/api/models");
    app.models = data.models;
    app.compareIds = data.compare;
  } catch (error) {
    toast(`서버에 연결하지 못했습니다: ${error.message}`);
    return;
  }
  await refresh();
  if (!app.game) openSetup();
  else await modelMove();
})();
