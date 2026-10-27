// 체스판 그리기와 입력. 규칙은 모른다: 서버가 준 합법수(UCI) 목록에 있는 수만 받는다.
"use strict";

const FILES = "abcdefgh";
const PIECE_DIR = "/static/piece/cburnett/";

function parsePlacement(fen) {
  // FEN의 기물 배치 → { "e1": "wK", ... }
  const pieces = {};
  fen.split(" ")[0].split("/").forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += Number(ch); continue; }
      const color = ch === ch.toUpperCase() ? "w" : "b";
      pieces[FILES[f] + (8 - r)] = color + ch.toUpperCase();
      f += 1;
    }
  });
  return pieces;
}

class ChessBoard {
  constructor(element, { onMove }) {
    this.el = element;
    this.onMove = onMove; // (from, to, needsPromotion) => void
    this.orientation = "white";
    this.pieces = {};
    this.fen = null;
    this.lastMove = null;
    this.check = null;
    this.dests = new Map(); // from → Set(to)
    this.promotions = new Set(); // "e7e8" 처럼 승진이 필요한 from+to
    this.interactive = false;
    this.selected = null;
    this.drag = null;
    this.dropped = null; // 끌어 놓아 둔 수 (from+to)

    this.squareLayer = document.createElement("div");
    this.squareLayer.className = "squares";
    this.pieceLayer = document.createElement("div");
    this.pieceLayer.className = "pieces";
    this.coordLayer = document.createElement("div");
    this.coordLayer.className = "coords";
    this.el.append(this.squareLayer, this.pieceLayer, this.coordLayer);

    this.el.addEventListener("pointerdown", (e) => this._pointerDown(e));
    window.addEventListener("pointermove", (e) => this._pointerMove(e));
    window.addEventListener("pointerup", (e) => this._pointerUp(e));
    this.el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  set({ fen, orientation, lastMove = null, check = null, legalMoves = [], interactive = false }) {
    // 새로 둔 수 하나만 움직여 보인다. 끌어 놓은 수는 이미 그 자리에 있으므로 움직이지 않는다
    const animate = this.fen !== null && fen !== this.fen && lastMove && lastMove !== this.lastMove
      && orientation === this.orientation && lastMove.slice(0, 4) !== this.dropped;
    this.dropped = null;
    this.fen = fen;
    this.orientation = orientation;
    this.lastMove = lastMove;
    this.check = check;
    this.interactive = interactive;
    this.dests = new Map();
    this.promotions = new Set();
    for (const uci of legalMoves) {
      const from = uci.slice(0, 2), to = uci.slice(2, 4);
      if (!this.dests.has(from)) this.dests.set(from, new Set());
      this.dests.get(from).add(to);
      if (uci.length === 5) this.promotions.add(from + to);
    }
    if (!this.interactive || !this.dests.has(this.selected)) this.selected = null;
    this.pieces = parsePlacement(fen);
    this._renderSquares();
    this._renderPieces(animate ? lastMove : null);
    this._renderCoords();
  }

  // 화면 좌표 ↔ 칸
  _xy(square) {
    const f = FILES.indexOf(square[0]), r = Number(square[1]) - 1;
    return this.orientation === "white" ? [f, 7 - r] : [7 - f, r];
  }
  _square(x, y) {
    return this.orientation === "white" ? FILES[x] + (8 - y) : FILES[7 - x] + (y + 1);
  }
  _squareAt(clientX, clientY) {
    const rect = this.el.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * 8);
    const y = Math.floor(((clientY - rect.top) / rect.height) * 8);
    return x >= 0 && x < 8 && y >= 0 && y < 8 ? this._square(x, y) : null;
  }

  _renderSquares() {
    const lastFrom = this.lastMove && this.lastMove.slice(0, 2);
    const lastTo = this.lastMove && this.lastMove.slice(2, 4);
    const dests = this.selected ? this.dests.get(this.selected) || new Set() : new Set();
    const html = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const sq = this._square(x, y);
        const classes = ["sq", (x + y) % 2 ? "dark" : "light"];
        if (sq === lastFrom || sq === lastTo) classes.push("last-move");
        if (sq === this.selected) classes.push("selected");
        if (dests.has(sq)) classes.push(this.pieces[sq] ? "dest occupied" : "dest");
        if (sq === this.check) classes.push("check");
        html.push(`<div class="${classes.join(" ")}" data-square="${sq}"></div>`);
      }
    }
    this.squareLayer.innerHTML = html.join("");
  }

  _renderPieces(animateMove) {
    this.pieceLayer.innerHTML = "";
    for (const [sq, piece] of Object.entries(this.pieces)) {
      const img = document.createElement("img");
      img.className = "piece";
      img.src = PIECE_DIR + piece + ".svg";
      img.alt = piece;
      img.draggable = false;
      img.dataset.square = sq;
      const [x, y] = this._xy(sq);
      if (animateMove && sq === animateMove.slice(2, 4)) {
        const [fx, fy] = this._xy(animateMove.slice(0, 2));
        img.style.transform = `translate(${fx * 100}%, ${fy * 100}%)`;
        this.pieceLayer.append(img);
        img.getBoundingClientRect(); // 시작 위치를 확정한 뒤 옮긴다
        img.classList.add("animating");
        img.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
        img.addEventListener("transitionend", () => img.classList.remove("animating"), { once: true });
      } else {
        img.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
        this.pieceLayer.append(img);
      }
    }
  }

  _renderCoords() {
    const html = [];
    for (let i = 0; i < 8; i++) {
      const file = this.orientation === "white" ? FILES[i] : FILES[7 - i];
      const rank = this.orientation === "white" ? 8 - i : i + 1;
      html.push(`<span class="file ${i % 2 ? "on-light" : "on-dark"}" style="left:${i * 12.5}%">${file}</span>`);
      html.push(`<span class="rank ${i % 2 ? "on-dark" : "on-light"}" style="top:${i * 12.5}%">${rank}</span>`);
    }
    this.coordLayer.innerHTML = html.join("");
  }

  _select(square) {
    this.selected = square;
    this._renderSquares();
  }

  _tryMove(from, to, dropped = false) {
    if (!(this.dests.get(from) || new Set()).has(to)) return false;
    this.selected = null;
    this.interactive = false; // 서버 응답으로 다시 그릴 때까지 다음 입력을 막는다
    this.dropped = dropped ? from + to : null;
    this.onMove(from, to, this.promotions.has(from + to));
    return true;
  }

  _pointerDown(e) {
    if (!this.interactive || e.button !== 0) return;
    const sq = this._squareAt(e.clientX, e.clientY);
    if (!sq) return;
    if (this.selected && this.selected !== sq && this._tryMove(this.selected, sq)) return;
    if (this.dests.has(sq)) {
      e.preventDefault();
      const wasSelected = this.selected === sq;
      this._select(sq);
      const img = this.pieceLayer.querySelector(`img[data-square="${sq}"]`);
      if (img) this.drag = { from: sq, img, startX: e.clientX, startY: e.clientY, moved: false, wasSelected };
    } else {
      this._select(null);
    }
  }

  _pointerMove(e) {
    if (!this.drag) return;
    const { img, startX, startY } = this.drag;
    if (!this.drag.moved && Math.hypot(e.clientX - startX, e.clientY - startY) < 4) return;
    this.drag.moved = true;
    const rect = this.el.getBoundingClientRect();
    const size = rect.width / 8;
    img.classList.add("dragging");
    const x = (e.clientX - rect.left - size / 2) / size;
    const y = (e.clientY - rect.top - size / 2) / size;
    img.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
  }

  _pointerUp(e) {
    if (!this.drag) return;
    const { from, img, moved, wasSelected } = this.drag;
    this.drag = null;
    img.classList.remove("dragging");
    if (moved) {
      const to = this._squareAt(e.clientX, e.clientY);
      if (to && to !== from && this._tryMove(from, to, true)) return;
      const [x, y] = this._xy(from);
      img.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
    } else if (wasSelected) {
      this._select(null); // 선택한 기물을 다시 누르면 선택을 푼다
    }
  }

  // 승진 기물 선택. 고르면 "q" | "r" | "b" | "n", 취소하면 null
  choosePromotion(container, to, color) {
    return new Promise((resolve) => {
      const [x] = this._xy(to);
      const fromTop = this._xy(to)[1] === 0;
      container.innerHTML = "";
      ["q", "n", "r", "b"].forEach((p, i) => {
        const button = document.createElement("button");
        button.className = "promotion-piece";
        button.style.left = `${x * 12.5}%`;
        button.style.top = `${(fromTop ? i : 7 - i) * 12.5}%`;
        button.innerHTML = `<img src="${PIECE_DIR}${color}${p.toUpperCase()}.svg" alt="${p}">`;
        button.addEventListener("click", (e) => { e.stopPropagation(); finish(p); });
        container.append(button);
      });
      const finish = (piece) => {
        container.hidden = true;
        container.onclick = null;
        resolve(piece);
      };
      container.onclick = () => finish(null);
      container.hidden = false;
    });
  }
}
