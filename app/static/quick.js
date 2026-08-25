/* Quick Convert: paste -> classify -> XML, on one screen.

   Reuses the same engine as the document flow; only the input path differs.
   Edits re-generate locally without another model call. */

const SUPPORTED_ELEMENTS = [
  "radio", "checkbox", "radio_grid", "checkbox_grid",
  "textarea", "text", "number", "select", "html",
];
const GRID_ELEMENTS = new Set(["radio_grid", "checkbox_grid"]);
const OPTION_ELEMENTS = new Set(["radio", "checkbox", "select"]);

const $ = (id) => document.getElementById(id);

let questions = [];
let resourceCatalog = [];
let subjectTypes = [];
let converting = false;

/* -- convert ------------------------------------------------------------ */

const input = $("paste-input");

input.addEventListener("input", () => {
  const lines = input.value ? input.value.split("\n").length : 0;
  $("paste-count").textContent = input.value ? `${lines} line(s)` : "";
});

input.addEventListener("keydown", (event) => {
  // Ctrl+Enter matches the Sublime muscle memory this replaces.
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    convert();
  }
});
$("convert-btn").addEventListener("click", convert);

async function convert() {
  const text = input.value.trim();
  if (!text || converting) return;

  converting = true;
  $("convert-btn").disabled = true;
  setStatus("Converting…");

  try {
    const body = await jsonOrThrow(
      await fetch("/api/quick-convert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
    );
    questions = body.questions;
    renderCards();
    showXml(body);
    renderWarnings(body.warnings, body);
    setStatus(`Converted ${questions.length} question(s).`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    converting = false;
    $("convert-btn").disabled = false;
  }
}

/** Re-generate from the edited questions — no second model call. */
async function regenerate() {
  if (!questions.length) return;
  try {
    const body = await jsonOrThrow(
      await fetch("/api/quick-generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ questions }),
      })
    );
    showXml(body);
    setStatus("XML updated.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

/* -- xml pane ----------------------------------------------------------- */

let lastFragments = "";

function showXml(body) {
  lastFragments = body.xml || "";
  paintXml();
  $("xml-meta").textContent = body.well_formed === false
    ? "not well-formed"
    : lastFragments
      ? `${(lastFragments.match(/<suspend\/>/g) || []).length} question(s)`
      : "";
}

function wrapped() {
  if (!$("quick-wrap").checked || !lastFragments) return lastFragments;
  return `<survey xmlns:atm1d="http://decipherinc.com/atm1d" xmlns:ss="http://decipherinc.com/ss">\n\n${lastFragments}\n\n</survey>\n`;
}

function paintXml() {
  const pane = $("quick-xml");
  pane.replaceChildren();
  const text = wrapped();
  if (!text) return;

  // Tag / attribute / value colouring, built from text nodes so nothing in the
  // questionnaire can inject markup into the page.
  const pattern = /(<\/?)([\w:.-]+)|([\w:.-]+)=("[^"]*")|(\$\{res\.[^}]*\})/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) pane.append(text.slice(cursor, match.index));
    if (match[2]) {
      pane.append(match[1], span("t", match[2]));
    } else if (match[3]) {
      pane.append(span("a", match[3]), "=", span("v", match[4]));
    } else {
      pane.append(span("r", match[5]));
    }
    cursor = pattern.lastIndex;
  }
  pane.append(text.slice(cursor));
}

function span(className, text) {
  const node = document.createElement("span");
  node.className = className;
  node.textContent = text;
  return node;
}

$("quick-wrap").addEventListener("change", paintXml);

$("copy-btn").addEventListener("click", async () => {
  const text = wrapped();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    flashCopy("Copied");
  } catch {
    // Clipboard access can be refused; selecting the text still works.
    const range = document.createRange();
    range.selectNodeContents($("quick-xml"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    flashCopy("Selected — press Ctrl+C");
  }
});

function flashCopy(message) {
  const state = $("copy-state");
  state.textContent = message;
  setTimeout(() => { state.textContent = ""; }, 2500);
}

/* -- cards -------------------------------------------------------------- */

function renderCards() {
  $("quick-cards-section").hidden = questions.length === 0;
  $("quick-cards").replaceChildren(...questions.map(card));
}

function card(question, position) {
  const node = $("quick-card-template").content.cloneNode(true);
  const root = node.querySelector(".quick-card");
  root.classList.add(question.needs_review ? "flagged" : "confident");
  root.dataset.position = String(position);

  root.querySelector(".label-pill").textContent = question.label;
  root.querySelector(".qc-title").textContent = runsText(question.title) || "(no title)";
  root.querySelector(".ai-notes").textContent = question.ai_notes || "";

  const badge = root.querySelector(".confidence");
  badge.textContent = `${question.needs_review ? "🔴" : "🟢"} ${Math.round((question.confidence || 0) * 100)}%`;
  badge.classList.add(question.needs_review ? "flag" : "good");

  renderRouting(root, question);

  const select = root.querySelector(".element-select");
  SUPPORTED_ELEMENTS.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === question.element;
    select.append(option);
  });

  buildResourceSelect(root, question);
  buildSubjectSelect(root, question);

  field(root, "title").value = runsText(question.title);
  field(root, "comment").value = runsText(question.comment);
  field(root, "options").value = optionsText(question.options);
  field(root, "rows").value = optionsText(question.rows);
  field(root, "cols").value = optionsText(question.cols);
  field(root, "routing_notes").value = (question.routing_notes || []).join("\n");
  applyVisibility(root, question.element);

  const body = root.querySelector(".qc-body");
  const toggle = root.querySelector(".toggle");
  if (question.needs_review) open(body, toggle, true);
  toggle.addEventListener("click", () => open(body, toggle, body.hidden));

  // Any edit updates the model and repaints the XML pane.
  root.querySelectorAll("textarea, select").forEach((control) => {
    control.addEventListener("change", () => commit(root));
  });
  return node;
}

function open(body, toggle, isOpen) {
  body.hidden = !isOpen;
  toggle.setAttribute("aria-expanded", String(isOpen));
  toggle.textContent = isOpen ? "Collapse" : "Edit";
}

function applyVisibility(root, element) {
  root.querySelector(".options-field").hidden = !OPTION_ELEMENTS.has(element);
  root.querySelector(".grid-fields").hidden = !GRID_ELEMENTS.has(element);
  root.querySelector(".subject-field").hidden = !GRID_ELEMENTS.has(element);
}

function commit(root) {
  const position = Number(root.dataset.position);
  const question = questions[position];
  if (!question) return;

  const element = root.querySelector(".element-select").value;
  const resource = root.querySelector(".resource-select").value;

  question.element = element;
  question.subject_type = root.querySelector(".subject-select").value;
  question.comment_resource = resource || null;
  question.title = toRuns(field(root, "title").value);
  question.comment = toRuns(field(root, "comment").value);
  question.options = toOptions(field(root, "options").value);
  question.rows = toOptions(field(root, "rows").value);
  question.cols = toOptions(field(root, "cols").value);
  question.routing_notes = field(root, "routing_notes").value
    .split("\n").map((line) => line.trim()).filter(Boolean);

  applyVisibility(root, element);
  root.querySelector(".custom-comment").hidden = resource !== "";
  root.querySelector(".qc-title").textContent = runsText(question.title) || "(no title)";
  renderRouting(root, question);
  regenerate();
}

function renderRouting(root, question) {
  const notes = question.routing_notes || [];
  const container = root.querySelector(".routing-notes");
  container.hidden = notes.length === 0;
  container.querySelector(".routing-list").replaceChildren(
    ...notes.map((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    })
  );
}

function buildResourceSelect(root, question) {
  const select = root.querySelector(".resource-select");
  select.replaceChildren(
    ...resourceCatalog.map((entry) => {
      const option = document.createElement("option");
      option.value = entry.label;
      option.textContent = `${entry.label} — ${entry.text}`;
      return option;
    })
  );
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "Custom text…";
  select.append(custom);
  select.value = question.comment_resource || "";
  root.querySelector(".custom-comment").hidden = select.value !== "";
}

function buildSubjectSelect(root, question) {
  const select = root.querySelector(".subject-select");
  select.replaceChildren(
    ...subjectTypes.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    })
  );
  select.value = question.subject_type || "none";
}

/* -- helpers ------------------------------------------------------------ */

const field = (root, name) => root.querySelector(`[data-field="${name}"]`);
const runsText = (runs) => (runs || []).map((run) => run.text).join("").trim();
const optionsText = (options) => (options || [])
  .map((option) => (option.code ? `${option.raw_text} | ${option.code}` : option.raw_text))
  .join("\n");

function toRuns(text) {
  return text.trim() ? [{ text, bold: false, italic: false, underline: false, color: null }] : [];
}

/** Split "Other | 97" back into text and code, matching the server. */
function toOptions(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const match = line.match(/^(.*?)(?:\s*[|(\[{]\s*(\d{1,3})\s*[)\]}]?)$/);
    return match
      ? { raw_text: match[1].trim(), code: match[2], bold: false, italic: false }
      : { raw_text: line, code: null, bold: false, italic: false };
  });
}

async function jsonOrThrow(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch { /* fall through */ }
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(
      typeof detail === "string" ? detail
        : Array.isArray(detail) ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : `Request failed (${response.status}).`
    );
  }
  return payload;
}

function setStatus(message, isError = false) {
  const node = $("quick-status");
  node.textContent = message;
  node.style.color = isError ? "var(--flag)" : "";
}

function renderWarnings(warnings, body) {
  const all = [...(warnings || [])];
  if (body && body.well_formed === false) all.unshift(`Generated XML is not well-formed: ${body.error}`);
  $("quick-warnings").replaceChildren(
    ...all.map((text) => {
      const node = document.createElement("div");
      node.className = "warning";
      node.textContent = `⚠ ${text}`;
      return node;
    })
  );
}

/* -- boot --------------------------------------------------------------- */

(async function boot() {
  try {
    const body = await (await fetch("/api/resources")).json();
    resourceCatalog = body.resources || [];
    subjectTypes = body.subject_types || [];
  } catch {
    subjectTypes = ["brand", "category", "product", "statement", "none"];
  }

  const badge = $("ai-status");
  try {
    const status = await (await fetch("/api/ai-status")).json();
    badge.textContent = status.available
      ? `AI ready · ${status.model}`
      : status.reachable ? `AI model missing · ${status.model}` : "AI offline · fallback mode";
    badge.classList.add(status.available ? "up" : "down");
    badge.title = status.detail || "";
    if (!status.available && status.detail) renderWarnings([status.detail]);
  } catch {
    badge.textContent = "AI status unknown";
    badge.classList.add("down");
  }
})();
