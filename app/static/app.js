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
let parsedDocument = null;
let activeJobId = null;
let pollTimer = null;
let resourceCatalog = [];
let subjectTypes = [];

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
    $("picker").hidden = true;
    $("progress-panel").hidden = true;

    const body = new FormData();
    body.append("file", file);
    parsedDocument = await jsonOrThrow(await fetch("/api/parse", { method: "POST", body }));
    renderParseStats(parsedDocument);
    renderPicker();
    setStatus(`Found ${classifiableQuestions().length} question(s). Choose which to classify.`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

/* -- question picker: parse covers everything, the SP picks the batch ---- */

function classifiableQuestions() {
  return (parsedDocument?.questions || []).filter((q) => !q.is_preamble && q.label);
}

function renderPicker() {
  const questions = classifiableQuestions();
  const done = new Set(draft.questions.map((q) => q.label));

  $("picker-list").replaceChildren(
    ...questions.map((question) => {
      const row = el("label", "picker-row");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = question.label;
      box.checked = !done.has(question.label);
      box.addEventListener("change", updatePickerCount);

      row.append(box, el("span", "pk-label", question.label),
                 el("span", "pk-title", question.title_text || "(no title)"));
      if (done.has(question.label)) row.append(el("span", "pk-done", "✓ classified"));
      return row;
    })
  );

  $("picker-hint").textContent = done.size
    ? `${done.size} already classified. A new batch merges into the same review set.`
    : "Large questionnaires are quicker to review in batches.";
  $("picker").hidden = false;
  updatePickerCount();
}

const pickerBoxes = () => [...document.querySelectorAll("#picker-list input[type=checkbox]")];
const selectedLabels = () => pickerBoxes().filter((b) => b.checked).map((b) => b.value);

function updatePickerCount() {
  const count = selectedLabels().length;
  $("picker-count").textContent = `${count} selected`;
  $("classify-btn").disabled = count === 0;
}

$("select-all").addEventListener("click", () => {
  pickerBoxes().forEach((box) => { box.checked = true; });
  updatePickerCount();
});
$("select-none").addEventListener("click", () => {
  pickerBoxes().forEach((box) => { box.checked = false; });
  updatePickerCount();
});
$("apply-range").addEventListener("click", applyRange);
$("range-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); applyRange(); }
});

/** Accepts "Q1-Q15", "1-15", or a comma-separated list of labels. */
function applyRange() {
  const raw = $("range-input").value.trim();
  if (!raw) return;

  const labels = classifiableQuestions().map((q) => q.label);
  const match = raw.match(/^\s*([A-Za-z]*)(\d+)\s*[-–—]\s*([A-Za-z]*)(\d+)\s*$/);
  let wanted;

  if (match) {
    const prefix = (match[1] || match[3] || "Q").toUpperCase();
    const from = Number(match[2]);
    const to = Number(match[4]);
    const [lo, hi] = from <= to ? [from, to] : [to, from];
    wanted = new Set();
    for (let n = lo; n <= hi; n += 1) wanted.add(`${prefix}${n}`);
  } else {
    wanted = new Set(raw.split(/[,\s]+/).filter(Boolean).map((s) => s.toUpperCase()));
  }

  const matched = labels.filter((label) => wanted.has(label));
  pickerBoxes().forEach((box) => { box.checked = wanted.has(box.value); });
  updatePickerCount();

  setStatus(
    matched.length
      ? `Selected ${matched.length} question(s) matching "${raw}".`
      : `No questions matched "${raw}".`,
    matched.length === 0
  );
}

/* -- classification job with real progress ------------------------------ */

$("classify-btn").addEventListener("click", startClassification);
$("cancel-btn").addEventListener("click", cancelClassification);

async function startClassification() {
  const labels = selectedLabels();
  if (!labels.length) return;

  try {
    $("classify-btn").disabled = true;
    const job = await jsonOrThrow(
      await fetch("/api/classify/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document: parsedDocument, labels }),
      })
    );
    activeJobId = job.job_id;
    showProgress(0, job.total, null);
    $("progress-panel").hidden = false;
    setStatus("");
    pollJob();
  } catch (error) {
    setStatus(error.message, true);
    $("classify-btn").disabled = false;
  }
}

function pollJob() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!activeJobId) return;
    try {
      const status = await jsonOrThrow(await fetch(`/api/classify/${activeJobId}/status`));
      showProgress(status.completed, status.total, status.estimated_remaining_seconds);
      renderWarnings(
        "progress-warning",
        status.slow
          ? ["This is taking longer than expected — you can keep waiting, or cancel and " +
             "process this in smaller batches. Questions already done are kept either way."]
          : []
      );

      if (status.state === "running") {
        pollJob();
      } else {
        await finishJob(status);
      }
    } catch (error) {
      setStatus(error.message, true);
      activeJobId = null;
      $("classify-btn").disabled = false;
    }
  }, 1000);
}

async function cancelClassification() {
  if (!activeJobId) return;
  $("cancel-btn").disabled = true;
  try {
    await fetch(`/api/classify/${activeJobId}/cancel`, { method: "POST" });
  } finally {
    $("cancel-btn").disabled = false;
  }
}

async function finishJob(status) {
  clearTimeout(pollTimer);
  activeJobId = null;
  $("classify-btn").disabled = false;
  $("progress-panel").hidden = true;

  draft = await jsonOrThrow(await fetch("/api/questions"));
  renderReview();
  renderPicker();
  enableStep("review");
  enableStep("export");
  showStep("review");

  const done = status.state === "cancelled"
    ? `Cancelled after ${status.completed} of ${status.total}. Completed questions were kept.`
    : `Classified ${status.completed} question(s).`;
  const fell = status.fallback_count
    ? ` ${status.fallback_count} used the offline fallback and need review.`
    : "";
  setStatus(done + fell);
}

function showProgress(completed, total, remainingSeconds) {
  const percent = total ? Math.round((completed / total) * 100) : 0;
  $("progress-bar").style.width = `${percent}%`;
  $("progress-label").textContent = `Classifying ${completed} of ${total}…`;
  $("progress-detail").textContent =
    remainingSeconds === null || remainingSeconds === undefined
      ? "Estimating time remaining…"
      : `About ${formatDuration(remainingSeconds)} remaining`;
}

function formatDuration(seconds) {
  if (seconds < 45) return `${Math.max(1, Math.round(seconds))}s`;
  const minutes = Math.round(seconds / 60);
  return minutes <= 1 ? "1 minute" : `${minutes} minutes`;
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
  renderRoutingNotes(card, question);

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
  buildResourceSelect(card, question);
  buildSubjectSelect(card, question);
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

/** Routing text is shown for context and never reaches the generated XML. */
function renderRoutingNotes(card, question) {
  const notes = question.routing_notes || [];
  const container = card.querySelector(".routing-notes");
  container.hidden = notes.length === 0;
  container.querySelector(".routing-list").replaceChildren(
    ...notes.map((text) => el("li", null, text))
  );
}

function openCard(body, toggle, open) {
  body.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  toggle.textContent = open ? "Collapse" : "Edit";
}

function applyElementVisibility(card, element) {
  card.querySelector(".options-field").hidden = !OPTION_ELEMENTS.has(element);
  card.querySelector(".grid-fields").hidden = !GRID_ELEMENTS.has(element);
  card.querySelector(".subject-field").hidden = !GRID_ELEMENTS.has(element);
}

/** Comment source: a resource tag, or custom text.
 *
 *  Preview text sits next to each tag so the programmer can see what they are
 *  picking; the generator still emits only the ${res.X} reference.
 */
function buildResourceSelect(card, question) {
  const select = card.querySelector(".resource-select");
  select.replaceChildren();

  for (const entry of resourceCatalog) {
    const option = document.createElement("option");
    option.value = entry.label;
    option.textContent = `${entry.label} — ${entry.text}`;
    select.append(option);
  }
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "Custom text…";
  select.append(custom);

  select.value = question.comment_resource || "";
  applyCommentMode(card, select.value);
  select.addEventListener("change", () => applyCommentMode(card, select.value));
}

function applyCommentMode(card, value) {
  card.querySelector(".custom-comment").hidden = value !== "";
}

function buildSubjectSelect(card, question) {
  const select = card.querySelector(".subject-select");
  select.replaceChildren(
    ...subjectTypes.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    })
  );
  select.value = question.subject_type || "none";

  // A grid's subject picks between the SRBrand / SRStatement variants, so the
  // comment tag has to follow it.
  select.addEventListener("change", () => {
    const element = card.querySelector(".element-select").value;
    if (!GRID_ELEMENTS.has(element)) return;
    const prefix = element === "radio_grid" ? "SR" : "MR";
    const suffix = select.value === "none" ? "Statement"
      : select.value.charAt(0).toUpperCase() + select.value.slice(1);
    const resource = card.querySelector(".resource-select");
    if (resource.value !== "" && [...resource.options].some((o) => o.value === prefix + suffix)) {
      resource.value = prefix + suffix;
      applyCommentMode(card, resource.value);
    }
  });
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
    subject_type: card.querySelector(".subject-select").value,
    // "" means custom text; the server reads it as "clear the tag".
    comment_resource: card.querySelector(".resource-select").value,
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
    renderRoutingNotes(card, updated);
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

(async function loadResources() {
  try {
    const body = await (await fetch("/api/resources")).json();
    resourceCatalog = body.resources || [];
    subjectTypes = body.subject_types || [];
  } catch {
    resourceCatalog = [];
    subjectTypes = ["brand", "category", "product", "statement", "none"];
  }
})();

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
