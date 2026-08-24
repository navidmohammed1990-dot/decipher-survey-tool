/* Review surface for the parse -> classify -> review -> export pipeline.
   Dependency-free on purpose: it ships with the server, no build step. */

const SUPPORTED_ELEMENTS = [
  "radio", "checkbox", "radio_grid", "checkbox_grid",
  "textarea", "text", "number", "select", "html",
];
const GRID_ELEMENTS = new Set(["radio_grid", "checkbox_grid"]);
const OPTION_ELEMENTS = new Set(["radio", "checkbox", "select"]);

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

let draft = { questions: [], summary: {}, review_threshold: 0.75 };

/* -- step navigation ---------------------------------------------------- */

function showStep(name) {
  document.querySelectorAll(".step").forEach((button) => {
    button.classList.toggle("active", button.dataset.step === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `panel-${name}`;
  });
  if (name === "export") renderExport();
}

function enableStep(name, enabled = true) {
  document.querySelector(`.step[data-step="${name}"]`).disabled = !enabled;
}

document.querySelectorAll(".step").forEach((button) => {
  button.addEventListener("click", () => {
    if (!button.disabled) showStep(button.dataset.step);
  });
});
$("to-export").addEventListener("click", () => showStep("export"));

/* -- upload and classify ------------------------------------------------ */

const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) run(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) run(file);
});

async function run(file) {
  try {
    setStatus(`Parsing ${file.name}…`);
    const body = new FormData();
    body.append("file", file);
    const parsed = await jsonOrThrow(await fetch("/api/parse", { method: "POST", body }));
    renderParseStats(parsed);

    setStatus(`Classifying ${parsed.questions.length} question(s)… this can take a moment.`);
    const classified = await jsonOrThrow(
      await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      })
    );

    draft = classified;
    renderWarnings("review-warnings", classified.warnings);
    renderReview();
    enableStep("review");
    enableStep("export");
    showStep("review");
    setStatus(`Parsed and classified ${file.name}.`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function jsonOrThrow(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    /* fall through to the status-based message below */
  }
  if (!response.ok) {
    throw new Error(payload?.detail ? detailText(payload.detail) : `Request failed (${response.status}).`);
  }
  return payload;
}

function detailText(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  return JSON.stringify(detail);
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

/* -- rendering ---------------------------------------------------------- */

function statTile(label, value, className) {
  const tile = el("div", `stat${className ? " " + className : ""}`);
  tile.append(el("div", "value", String(value)), el("div", "label", label));
  return tile;
}

function renderParseStats(parsed) {
  const s = parsed.stats;
  $("parse-stats").replaceChildren(
    statTile("Questions", s.questions),
    statTile("Paragraphs", s.non_empty_paragraphs),
    statTile("Tables", s.tables),
    statTile("Bold runs", s.bold_runs),
    statTile("Italic runs", s.italic_runs)
  );
  renderWarnings("parse-warnings", parsed.warnings);
}

function renderWarnings(containerId, warnings) {
  $(containerId).replaceChildren(
    ...(warnings || []).map((text) => el("div", "warning", `⚠ ${text}`))
  );
}

function summaryTiles(target) {
  const s = draft.summary || {};
  $(target).replaceChildren(
    statTile("Questions", s.total ?? draft.questions.length),
    statTile("Needs review", s.flagged ?? 0, "flagged"),
    statTile("Confident", s.confident ?? 0, "confident")
  );
}

function recomputeSummary() {
  const flagged = draft.questions.filter((q) => q.needs_review).length;
  draft.summary = {
    total: draft.questions.length,
    flagged,
    confident: draft.questions.length - flagged,
  };
}

function renderReview() {
  recomputeSummary();
  summaryTiles("review-summary");

  const filter = $("filter").value;
  const visible = draft.questions.filter((question) => {
    if (filter === "flagged") return question.needs_review;
    if (filter === "confident") return !question.needs_review;
    return true;
  });

  $("question-list").replaceChildren(...visible.map(questionCard));
}

$("filter").addEventListener("change", renderReview);

function questionCard(question) {
  const node = $("question-template").content.cloneNode(true);
  const card = node.querySelector(".question-card");
  card.classList.add(question.needs_review ? "flagged" : "confident");
  card.dataset.label = question.label;

  card.querySelector(".label-pill").textContent = question.label;
  card.querySelector(".qc-title").textContent = runsText(question.title) || "(no title)";
  card.querySelector(".ai-notes").textContent = question.ai_notes || "";

  const badge = card.querySelector(".confidence");
  const percent = Math.round((question.confidence || 0) * 100);
  badge.textContent = question.needs_review ? `🔴 ${percent}%` : `🟢 ${percent}%`;
  badge.classList.add(question.needs_review ? "flag" : "good");

  const select = card.querySelector(".element-select");
  SUPPORTED_ELEMENTS.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === question.element;
    select.append(option);
  });
  select.addEventListener("change", () => applyElementVisibility(card, select.value));

  field(card, "title").value = runsText(question.title);
  field(card, "comment").value = runsText(question.comment);
  field(card, "options").value = optionsText(question.options);
  field(card, "rows").value = optionsText(question.rows);
  field(card, "cols").value = optionsText(question.cols);
  field(card, "dev_notes").value = question.dev_notes || "";
  applyElementVisibility(card, question.element);

  const body = card.querySelector(".qc-body");
  const toggle = card.querySelector(".toggle");
  // Anything the AI was unsure about opens already expanded.
  if (question.needs_review) openCard(body, toggle, true);
  toggle.addEventListener("click", () => openCard(body, toggle, body.hidden));

  card.querySelector(".save").addEventListener("click", () => saveCard(card));
  card.querySelector(".preview").addEventListener("click", () => previewCard(card));
  return node;
}

function openCard(body, toggle, open) {
  body.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  toggle.textContent = open ? "Collapse" : "Edit";
}

function applyElementVisibility(card, element) {
  card.querySelector(".options-field").hidden = !OPTION_ELEMENTS.has(element);
  card.querySelector(".grid-fields").hidden = !GRID_ELEMENTS.has(element);
}

const field = (card, name) => card.querySelector(`[data-field="${name}"]`);
const runsText = (runs) => (runs || []).map((run) => run.text).join("").trim();
const optionsText = (options) => (options || []).map((option) => option.raw_text).join("\n");

/* -- editing ------------------------------------------------------------ */

async function saveCard(card) {
  const label = card.dataset.label;
  const state = card.querySelector(".save-state");
  state.className = "save-state muted";
  state.textContent = "Saving…";

  const patch = {
    element: card.querySelector(".element-select").value,
    title: field(card, "title").value,
    comment: field(card, "comment").value,
    options: field(card, "options").value,
    rows: field(card, "rows").value,
    cols: field(card, "cols").value,
    dev_notes: field(card, "dev_notes").value,
  };

  try {
    const updated = await jsonOrThrow(
      await fetch(`/api/questions/${encodeURIComponent(label)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
    );
    const index = draft.questions.findIndex((question) => question.label === label);
    if (index !== -1) draft.questions[index] = updated;

    state.className = "save-state ok";
    state.textContent = "Saved";
    recomputeSummary();
    summaryTiles("review-summary");
    card.classList.remove("flagged", "confident");
    card.classList.add(updated.needs_review ? "flagged" : "confident");

    const badge = card.querySelector(".confidence");
    badge.className = `confidence ${updated.needs_review ? "flag" : "good"}`;
    badge.textContent = `${updated.needs_review ? "🔴" : "🟢"} ${Math.round((updated.confidence || 0) * 100)}%`;
    card.querySelector(".qc-title").textContent = runsText(updated.title) || "(no title)";
  } catch (error) {
    state.className = "save-state err";
    state.textContent = error.message;
  }
}

async function previewCard(card) {
  const label = card.dataset.label;
  const output = card.querySelector(".xml-preview");
  output.hidden = false;
  output.textContent = "Generating…";

  try {
    const body = await jsonOrThrow(
      await fetch(`/api/generate/${encodeURIComponent(label)}`, { method: "POST" })
    );
    output.textContent = body.xml;
  } catch (error) {
    output.textContent = error.message;
  }
}

/* -- export ------------------------------------------------------------- */

async function renderExport() {
  recomputeSummary();
  summaryTiles("export-summary");

  const output = $("export-output");
  output.textContent = "Generating…";
  try {
    const body = await jsonOrThrow(
      await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wrap: $("wrap-toggle").checked }),
      })
    );
    output.textContent = body.xml;
    const warnings = [...(body.warnings || [])];
    if (!body.well_formed) warnings.unshift(`Generated XML is not well-formed: ${body.error}`);
    renderWarnings("export-warnings", warnings);
  } catch (error) {
    output.textContent = "";
    renderWarnings("export-warnings", [error.message]);
  }
}

$("wrap-toggle").addEventListener("change", renderExport);
$("download-xml").addEventListener("click", () => {
  window.location.href = `/api/export.xml?wrap=${$("wrap-toggle").checked}`;
});

/* -- misc --------------------------------------------------------------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

(async function checkAi() {
  const badge = $("ai-status");
  try {
    const body = await (await fetch("/api/ai-status")).json();
    badge.textContent = body.available ? `AI ready · ${body.model}` : "AI offline · fallback mode";
    badge.classList.add(body.available ? "up" : "down");
  } catch {
    badge.textContent = "AI status unknown";
    badge.classList.add("down");
  }
})();
