// Small DOM controls shared by the views.

import { escapeHtml } from "./format.js";
import { barWidths } from "./stats.js";

// A segmented control. `options` is [{ value, label }]; calls onChange(value)
// when the selection changes. Returns { set(value), disable(bool) }.
export function segmented(el, options, value, onChange) {
  el.classList.add("segmented");
  el.setAttribute("role", "group");
  el.innerHTML = options
    .map((o) => `<button type="button" data-value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</button>`)
    .join("");
  const buttons = [...el.querySelectorAll("button")];
  const set = (v) => {
    value = v;
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.value === String(v)));
  };
  for (const b of buttons) {
    b.addEventListener("click", () => {
      const v = options.find((o) => String(o.value) === b.dataset.value).value;
      if (v === value) return;
      set(v);
      onChange(v);
    });
  }
  set(value);
  return {
    set,
    disable(disabled) {
      for (const b of buttons) b.disabled = disabled;
    },
  };
}

// A button that needs a second click within `ms` to act. The first click
// swaps in `armedLabel()`; any other outcome reverts it.
export function twoClick(button, armedLabel, onConfirm, ms = 6000) {
  const idleLabel = button.textContent;
  let timer = null;
  const disarm = () => {
    clearTimeout(timer);
    timer = null;
    button.dataset.armed = "false";
    button.textContent = idleLabel;
  };
  button.addEventListener("click", () => {
    if (button.dataset.armed === "true") {
      disarm();
      onConfirm();
      return;
    }
    button.dataset.armed = "true";
    button.textContent = armedLabel();
    timer = setTimeout(disarm, ms);
  });
  return { disarm };
}

// Horizontal bars as HTML. `entries` is [{ label, value, text }]; bars share
// one scale by magnitude, and negative values get the "neg" style.
export function barList(entries) {
  const widths = barWidths(entries.map((e) => e.value));
  return `<ul class="bars">${entries
    .map((e, i) => `
      <li>
        <span class="bar-label" title="${escapeHtml(e.label)}">${escapeHtml(e.label)}</span>
        <span class="bar-track"><span class="bar${e.value < 0 ? " neg" : ""}" style="width:${widths[i].toFixed(1)}%"></span></span>
        <span class="bar-value num">${escapeHtml(e.text ?? String(e.value))}</span>
      </li>`)
    .join("")}</ul>`;
}
