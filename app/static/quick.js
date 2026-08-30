/* Quick Convert: paste -> classify -> XML, on one screen.

   Reuses the same engine as the document flow; only the input path differs.
   Edits re-generate locally without another model call. */

const SUPPORTED_ELEMENTS = [
  "radio", "checkbox", "radio_grid", "checkbox_grid",
  "textarea", "text", "number", "select", "select_slider", "html",
  // Programmer content: generates nothing. Must be listed, or a card
  // classified this way would silently fall back to the first option.
  "not_a_question",
];
const GRID_ELEMENTS = new Set(["radio_grid", "checkbox_grid"]);
const OPTION_ELEMENTS = new Set(["radio", "checkbox", "select", "select_slider"]);
const NON_QUESTION_ELEMENTS = new Set(["not_a_question"]);

const $ = (id) => document.getElementById(id);

let questions = [];
let resourceCatalog = [];
let subjectTypes = [];
let converting = false;
let history = [];

const HISTORY_KEY = "quick-convert-history";
const HISTORY_LIMIT = 50;

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
  setStatus("");
  startConverting();

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
    showTimings(body.timings);
    // Purely additive: a snapshot of this conversion, leaving the workspace as
    // it is. Later converts add entries, they never rewrite existing ones.
    addHistory(text, body);
    setStatus(`Converted ${questions.length} question(s).`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    stopConverting();
    converting = false;
    $("convert-btn").disabled = false;
  }
}

/* A conversion can run for a minute on CPU inference, so the wait needs a
   visible heartbeat. No ETA is promised — the point is showing it is alive. */
let convertTimer = null;

function startConverting() {
  const started = Date.now();
  const panel = $("converting");
  const text = $("converting-text");
  panel.hidden = false;
  text.textContent = "Converting… 0s";

  clearInterval(convertTimer);
  convertTimer = setInterval(() => {
    text.textContent = `Converting… ${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);
}

function stopConverting() {
  clearInterval(convertTimer);
  convertTimer = null;
  $("converting").hidden = true;
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
      : "nothing to generate";
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
  if (NON_QUESTION_ELEMENTS.has(question.element)) {
    root.classList.add("non-question");
    root.querySelector(".qc-title").textContent =
      "Programmer instruction — not a respondent question. No XML generated.";
  }

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

  // Any edit updates the model and repaints the XML pane. The command input
  // is excluded: it is interpreted on demand, not on every keystroke.
  root.querySelectorAll("textarea, select").forEach((control) => {
    control.addEventListener("change", () => commit(root));
  });
  wireCommandBox(root);
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


/* -- session history ----------------------------------------------------

   Kept in sessionStorage rather than a plain array: it survives an accidental
   reload and still disappears when the tab closes, which is the intended
   lifetime. It is not durable storage — see the README's known limitations. */

function loadHistory() {
  try {
    const stored = sessionStorage.getItem(HISTORY_KEY);
    history = stored ? JSON.parse(stored) : [];
  } catch {
    history = [];
  }
  if (!Array.isArray(history)) history = [];
  renderHistory();
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // A full or unavailable store must not break converting; the in-memory
    // list still shows this session's entries.
  }
}

function addHistory(text, body) {
  const labels = (body.questions || []).map((question) => question.label);
  history.unshift({
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: new Date().toISOString(),
    text,
    xml: body.xml || "",
    labels,
    count: labels.length,
  });
  history = history.slice(0, HISTORY_LIMIT);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const section = $("history-section");
  section.hidden = history.length === 0;
  $("history-count").textContent = history.length
    ? `${history.length} conversion${history.length === 1 ? "" : "s"}`
    : "";
  $("history-list").replaceChildren(...history.map(historyEntry));
}

function historyEntry(entry) {
  const node = $("history-template").content.cloneNode(true);
  const root = node.querySelector(".history-entry");

  const summary = entry.labels.length
    ? entry.labels.join(", ")
    : firstLine(entry.text);
  root.querySelector(".history-title").textContent = summary;
  root.querySelector(".history-meta").textContent =
    `${entry.count} question${entry.count === 1 ? "" : "s"} · ${clockTime(entry.at)}`;
  root.querySelector(".history-input").textContent = entry.text;
  root.querySelector(".history-xml").textContent = entry.xml;

  const body = root.querySelector(".history-body");
  const toggle = root.querySelector(".history-toggle");
  toggle.addEventListener("click", () => {
    body.hidden = !body.hidden;
    toggle.setAttribute("aria-expanded", String(!body.hidden));
  });

  // Each entry copies its own XML, independent of the active pane.
  root.querySelector(".history-copy").addEventListener("click", async (event) => {
    event.stopPropagation();
    const state = root.querySelector(".history-state");
    try {
      await navigator.clipboard.writeText(entry.xml);
      flashHistory(root, "Copied");
    } catch {
      const range = document.createRange();
      range.selectNodeContents(root.querySelector(".history-xml"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      body.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
      flashHistory(root, "Selected — press Ctrl+C");
    }
    if (state) state.scrollIntoView?.({ block: "nearest" });
  });

  root.querySelector(".history-restore").addEventListener("click", () => {
    input.value = entry.text;
    input.dispatchEvent(new Event("input"));
    input.focus();
    setStatus("Loaded that text back into the paste box. Convert to run it again.");
  });

  return node;
}

function flashHistory(root, message) {
  const state = root.querySelector(".history-state");
  state.textContent = message;
  setTimeout(() => { state.textContent = ""; }, 2500);
}

const firstLine = (text) => {
  const line = (text || "").split("\n").find((candidate) => candidate.trim());
  return line ? line.trim().slice(0, 80) : "(empty)";
};

function clockTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

$("clear-history").addEventListener("click", () => {
  history = [];
  saveHistory();
  renderHistory();
});

/* -- performance note ---------------------------------------------------- */

function showTimings(timings) {
  const note = $("perf-note");
  const usable = (timings || []).filter((entry) => entry.total_seconds);
  if (!usable.length) {
    note.hidden = true;
    return;
  }

  const total = usable.reduce((sum, entry) => sum + entry.total_seconds, 0);
  const rates = usable.map((entry) => entry.output_tokens_per_second).filter(Boolean);
  const slowest = rates.length ? Math.min(...rates) : null;

  let text = `Model time: ${total.toFixed(1)}s across ${usable.length} call(s)`;
  if (slowest) {
    text += ` · ${slowest} output tokens/sec`;
    // Sustained single-digit tokens/sec on an 8B model means CPU inference.
    if (slowest < 10) {
      text += " — that rate indicates CPU inference. A smaller model will be much faster;"
        + " see DECIPHER_OLLAMA_MODEL.";
    }
  }
  note.textContent = text;
  note.hidden = false;
}

loadHistory();


/* -- natural-language corrections ---------------------------------------

   Additive by design: the dropdown and text fields above stay the primary,
   zero-latency way to fix anything. This costs a model round trip, so it is
   opt-in and always shows what it would change before anything is saved. */

function wireCommandBox(root) {
  const input = root.querySelector(".command-input");
  const proposal = root.querySelector(".command-proposal");
  const state = root.querySelector(".command-state");
  let pending = null;

  const setState = (message, kind = "") => {
    state.className = `command-state muted ${kind}`.trim();
    state.textContent = message;
  };

  async function run() {
    const instruction = input.value.trim();
    if (!instruction) return;

    const position = Number(root.dataset.position);
    const question = questions[position];
    if (!question) return;

    proposal.hidden = true;
    pending = null;
    setState("Interpreting…");

    try {
      const body = await jsonOrThrow(
        await fetch("/api/quick-command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, instruction }),
        })
      );

      if (!body.understood) {
        // Never guess: an unclear instruction says so and points at the fields.
        setState(body.reason || "Not sure what you meant, please use the fields below.", "err");
        return;
      }

      pending = body;
      setState("");
      renderProposal(root, body);
      proposal.hidden = false;
    } catch (error) {
      setState(error.message, "err");
    }
  }

  root.querySelector(".command-run").addEventListener("click", run);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      run();
    }
  });

  root.querySelector(".command-reject").addEventListener("click", () => {
    pending = null;
    proposal.hidden = true;
    setState("Discarded.");
  });

  root.querySelector(".command-accept").addEventListener("click", async () => {
    if (!pending) return;
    const position = Number(root.dataset.position);
    const before = questions[position];

    questions[position] = pending.question;
    proposal.hidden = true;
    input.value = "";
    setState("Applied.", "ok");

    try {
      // Recorded so both correction routes feed the same library.
      await fetch("/api/quick-command/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ before, after: pending.question, instruction: pending.instruction || "" }),
      });
    } catch {
      // Recording is best effort; the edit itself already stands.
    }

    pending = null;
    renderCards();
    regenerate();
  });
}

function renderProposal(root, body) {
  root.querySelector(".command-reason").textContent = body.reason || "";

  const rows = (body.changes || []).map((change) => {
    const tr = document.createElement("tr");
    const field = document.createElement("th");
    field.className = "field";
    field.textContent = change.field;

    const before = document.createElement("td");
    before.className = "before";
    before.textContent = change.before || "(empty)";

    const after = document.createElement("td");
    after.className = "after";
    after.textContent = change.after || "(empty)";

    tr.append(field, before, after);
    return tr;
  });
  root.querySelector(".diff tbody").replaceChildren(...rows);
}
