// uPlot wrappers. Colors come from the CSS variables in app.css at render
// time, and every chart rebuilds itself when the OS theme flips.
/* global uPlot */

const lightQuery = matchMedia("(prefers-color-scheme: light)");
const FONT = '11px -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif';

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function theme() {
  return {
    text: cssVar("--text"),
    muted: cssVar("--text-muted"),
    grid: cssVar("--grid"),
    border: cssVar("--border"),
    danger: cssVar("--danger"),
    cpu: cssVar("--series-cpu"),
    mem: cssVar("--series-mem"),
    swap: cssVar("--series-swap"),
    contention: cssVar("--series-contention"),
  };
}

// "#4cc2ff" -> "rgba(76,194,255,a)". Theme colors are all hex.
export function alpha(hex, a) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// A y axis fixed to 0-100 %, and an x axis whose tick labels can be
// hidden (stacked charts share the bottom chart's labels). Pass
// `ySplits: null` and a `yFormat` for an auto-scaled, non-percent y axis.
export function axes(t, { xLabels = true, ySplits = [0, 25, 50, 75, 100], yFormat = (v) => `${v}%` } = {}) {
  return [
    {
      stroke: t.muted,
      font: FONT,
      grid: { stroke: t.grid, width: 1 },
      ticks: { stroke: t.grid, width: 1, size: 4 },
      size: xLabels ? 24 : 6,
      values: xLabels ? clockLabels : (u, splits) => splits.map(() => ""),
    },
    {
      stroke: t.muted,
      font: FONT,
      grid: { stroke: t.grid, width: 1 },
      ticks: { show: false },
      size: ySplits ? 40 : 56,
      ...(ySplits ? { splits: () => ySplits } : {}),
      values: (u, splits) => splits.map(yFormat),
    },
  ];
}

// One-line local clock labels; seconds only when the window is short.
function clockLabels(u, splits) {
  const span = u.scales.x.max - u.scales.x.min;
  const pad = (n) => String(n).padStart(2, "0");
  return splits.map((v) => {
    const d = new Date(v * 1000);
    const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return span < 600 ? `${hm}:${pad(d.getSeconds())}` : hm;
  });
}

// Draws a dashed vertical "now" line at x = getNow().
export function nowMarker(getNow) {
  return (u) => {
    const now = getNow();
    if (now == null) return;
    const x = Math.round(u.valToPos(now, "x", true));
    const { top, height, left, width } = u.bbox;
    if (x < left || x > left + width) return;
    const ctx = u.ctx;
    const t = theme();
    ctx.save();
    ctx.strokeStyle = t.muted;
    ctx.lineWidth = devicePixelRatio;
    ctx.setLineDash([3 * devicePixelRatio, 3 * devicePixelRatio]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = t.muted;
    ctx.font = `${10 * devicePixelRatio}px -apple-system, system-ui, sans-serif`;
    ctx.fillText("now", x + 4 * devicePixelRatio, top + 11 * devicePixelRatio);
    ctx.restore();
  };
}

// A uPlot chart that tracks its container's width and the OS theme.
// `build(theme)` returns uPlot options without width/height.
export class Chart {
  constructor(el, build, height) {
    this.el = el;
    this.build = build;
    this.height = height;
    this.data = null;
    this.plot = null;
    this.onTheme = () => this.#rebuild();
    lightQuery.addEventListener("change", this.onTheme);
    this.resizer = new ResizeObserver(() => {
      const width = this.el.clientWidth;
      if (this.plot && width > 0 && width !== this.plot.width) {
        this.plot.setSize({ width, height: this.height });
      }
    });
    this.resizer.observe(el);
  }

  setData(data) {
    this.data = data;
    if (!this.plot) this.#rebuild();
    else this.plot.setData(data);
  }

  redraw() {
    this.plot?.redraw(false);
  }

  // Drop the plot so the next setData rebuilds it from `build` (for when
  // the options themselves depend on view state, e.g. the chosen metric).
  reset() {
    this.plot?.destroy();
    this.plot = null;
  }

  destroy() {
    lightQuery.removeEventListener("change", this.onTheme);
    this.resizer.disconnect();
    this.plot?.destroy();
    this.plot = null;
  }

  #rebuild() {
    if (!this.data) return;
    this.plot?.destroy();
    const opts = { ...this.build(theme()), width: this.el.clientWidth || 600, height: this.height };
    this.plot = new uPlot(opts, this.data, this.el);
  }
}

// A tiny inline SVG trend line. `values` may contain nulls.
export function sparkline(svg, values, color, { min = 0, max = 100 } = {}) {
  const w = 100;
  const h = 28;
  const pts = [];
  const n = values.length;
  values.forEach((v, i) => {
    if (v == null) return;
    const x = n > 1 ? (i / (n - 1)) * w : w;
    const clamped = Math.min(max, Math.max(min, v));
    const y = h - 1 - ((clamped - min) / (max - min || 1)) * (h - 2);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "none");
  let line = svg.querySelector("polyline");
  if (!line) {
    line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    svg.appendChild(line);
  }
  line.setAttribute("stroke", color);
  line.setAttribute("points", pts.join(" "));
}
