/* remix-editor.js — the drum score as a grid you can edit.
 *
 * One column per 16th step, one row per voice. Click an empty cell to add a
 * hit, click a hit to remove it. Scroll on a kick to change its note
 * (shift + scroll: every kick in that bar). "apply" sends the score to the
 * worker, which re-renders only the drums. MIDI export/import uses the same
 * score, so edits made in a DAW come back here.
 */
(function () {
  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const LANES = [["kick", 34], ["snare", 18], ["hat", 18], ["bell", 18]];
  const HEAD = 16, SW = 9;            // header height, px per 16th step
  const KIND = { drop: "#ff5a1f", build: "#c8781e", verse: "#8e837c", break: "#5a4f49", intro: "#6b2a1c", outro: "#6b2a1c" };
  let score = null, meta = null, canvas = null, wrap = null, onApply = null, dirty = false, playhead = -1;

  const laneAt = y => { let top = HEAD; for (const [name, h] of LANES) { if (y >= top && y < top + h) return name; top += h; } return null; };
  const laneTop = name => { let top = HEAD; for (const [n, h] of LANES) { if (n === name) return top; top += h; } return 0; };
  const height = () => HEAD + LANES.reduce((a, [, h]) => a + h, 0);

  function draw() {
    if (!score) return;
    const bars = meta.N_BARS, W = bars * 16 * SW, H = height(), g = canvas.getContext("2d");
    if (canvas.width !== W) { canvas.width = W; canvas.height = H; canvas.style.width = W + "px"; canvas.style.height = H + "px"; }
    g.fillStyle = "#0e0c0b"; g.fillRect(0, 0, W, H);
    for (const b of meta.blocks) {                                      // sections
      g.fillStyle = KIND[b.kind] || "#444"; g.fillRect(b.out * 16 * SW, 0, b.n * 16 * SW - 1, HEAD - 2);
      g.fillStyle = "#120404"; g.font = "10px Tahoma, sans-serif"; g.fillText(`${b.name}  ${b.out + 1}`, b.out * 16 * SW + 4, 11);
    }
    for (let s = 0; s <= bars * 16; s++) {                              // grid
      g.fillStyle = s % 16 === 0 ? "#4a403b" : s % 4 === 0 ? "#2a2523" : "#1a1716";
      g.fillRect(s * SW, HEAD, 1, H - HEAD);
    }
    let top = HEAD;
    for (const [name, h] of LANES) {
      g.fillStyle = "#000"; g.fillRect(0, top + h - 1, W, 1);
      for (const e of score[name]) {
        const x = e.s * SW + 1, w = SW - 2;
        if (name === "kick") {
          g.fillStyle = `hsl(${(e.pc * 30) % 360} 80% 55%)`; g.fillRect(x, top + 2, w, h - 4);
          g.save(); g.translate(x + 7, top + h - 4); g.rotate(-Math.PI / 2); g.fillStyle = "#000"; g.font = "bold 9px Tahoma, sans-serif"; g.fillText(NAMES[e.pc], 0, 0); g.restore();
        } else {
          const v = Math.min(1, (e.g ?? 0.3) / ({ snare: 0.3, hat: 0.34, bell: 0.5 }[name]));
          g.fillStyle = name === "bell" ? "#e8c35a" : name === "snare" ? "#ff8a5c" : "#cfc6c0"; g.globalAlpha = 0.35 + 0.65 * v;
          g.fillRect(x, top + 3, w, h - 6); g.globalAlpha = 1;
        }
      }
      top += h;
    }
    if (playhead >= 0) { g.fillStyle = "#4ade5a"; g.fillRect(playhead * SW, 0, 2, H); }
  }

  function hitAt(name, s) { return score[name].findIndex(e => e.s === s); }
  function defaultPc(s) {
    const bar = Math.floor(s / 16), inBar = score.kick.filter(e => Math.floor(e.s / 16) === bar);
    if (inBar.length) return inBar[0].pc;
    const prev = score.kick.filter(e => e.s < s).pop(); return prev ? prev.pc : 0;
  }
  function edited() { dirty = true; document.getElementById("e-apply").disabled = false; draw(); }

  function onClick(ev) {
    const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top, name = laneAt(y);
    if (!name) return;
    const s = Math.floor(x / SW), i = hitAt(name, s);
    if (i >= 0) score[name].splice(i, 1);
    else if (name === "kick") score.kick.push({ s, pc: defaultPc(s) });
    else if (name === "hat") score.hat.push({ s, g: s % 4 === 2 ? 0.34 : 0.13 });
    else if (name === "snare") score.snare.push({ s, g: 0.22 });
    else score.bell.push({ s, g: 0.5, semis: (score.bell[score.bell.length - 1] || {}).semis ?? 0 });
    score[name].sort((a, b) => a.s - b.s); edited();
  }
  function onWheel(ev) {
    const r = canvas.getBoundingClientRect(), y = ev.clientY - r.top;
    if (laneAt(y) !== "kick") return;
    const s = Math.floor((ev.clientX - r.left) / SW), i = hitAt("kick", s);
    if (i < 0) return;
    ev.preventDefault();
    const d = ev.deltaY < 0 ? 1 : -1, bar = Math.floor(s / 16);
    const targets = ev.shiftKey ? score.kick.filter(e => Math.floor(e.s / 16) === bar) : [score.kick[i]];
    for (const e of targets) e.pc = (e.pc + d + 12) % 12;
    edited();
  }

  window.EDITOR = {
    init(opts) {
      canvas = document.getElementById("e-grid"); wrap = document.getElementById("e-wrap"); onApply = opts.onApply;
      canvas.addEventListener("click", onClick);
      canvas.addEventListener("wheel", onWheel, { passive: false });
      document.getElementById("e-apply").addEventListener("click", () => { if (score) { dirty = false; document.getElementById("e-apply").disabled = true; onApply(JSON.parse(JSON.stringify(score))); } });
      document.getElementById("e-midi").addEventListener("click", () => {
        if (!score) return;
        const url = URL.createObjectURL(new Blob([self.REMIX.encodeMidi({ ...meta, score })], { type: "audio/midi" }));
        const a = document.createElement("a"); a.href = url; a.download = (opts.name() || "remix") + " (hardtekk drums).mid"; a.click();
      });
      document.getElementById("e-midi-in").addEventListener("change", async ev => {
        const f = ev.target.files[0]; ev.target.value = ""; if (!f || !score) return;
        const s = self.REMIX.decodeMidi(await f.arrayBuffer());
        for (const k of ["kick", "snare", "hat", "bell"]) score[k] = s[k];
        edited(); opts.log(`MIDI loaded: ${s.kick.length} kicks, ${s.snare.length} snares, ${s.hat.length} hats, ${s.bell.length} bells — apply to hear it`);
      });
    },
    load(m) { meta = m; score = JSON.parse(JSON.stringify(m.score)); dirty = false; document.getElementById("e-apply").disabled = true; draw(); },
    playhead(sec) {
      if (!meta) return;
      const s = Math.max(-1, (sec - meta.OB0) / (meta.BAR / 16)), was = playhead; playhead = s;
      if (Math.floor(was) !== Math.floor(s)) {
        draw();
        const x = s * SW; if (x < wrap.scrollLeft || x > wrap.scrollLeft + wrap.clientWidth - 40) wrap.scrollLeft = Math.max(0, x - 80);
      }
    },
  };
})();
