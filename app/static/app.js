/* Phase 1 review surface: upload a questionnaire and inspect what the parser
   extracted. Deliberately dependency-free — the review UI proper arrives in a
   later phase. */

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let parsed = null;
/** Blocks by index, including those nested in table cells. */
let blockIndex = new Map();

/* -- upload ------------------------------------------------------------- */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) upload(fileInput.files[0]);
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
  if (file) upload(file);
});

async function upload(file) {
  setStatus(`Parsing ${file.name}…`);
  resultsEl.hidden = true;

  const body = new FormData();
  body.append("file", file);

  try {
    const response = await fetch("/api/parse", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.detail || `Upload failed (${response.status}).`, true);
      return;
    }
    parsed = payload;
    blockIndex = indexBlocks(payload.blocks);
    render();
    setStatus(`Parsed ${file.name}.`);
  } catch (error) {
    setStatus(`Could not reach the server: ${error.message}`, true);
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

/** Walk blocks and nested cell blocks into a lookup by index. */
function indexBlocks(blocks, into = new Map()) {
  for (const block of blocks) {
    into.set(block.index, block);
    if (block.kind === "table") {
      for (const row of block.rows) {
        for (const cell of row.cells) indexBlocks(cell.blocks, into);
      }
    }
  }
  return into;
}

/* -- rendering ---------------------------------------------------------- */

function render() {
  renderStats();
  renderWarnings();
  renderQuestions();
  renderBlocks();
  document.getElementById("json-output").textContent = JSON.stringify(parsed, null, 2);
  resultsEl.hidden = false;
}

function renderStats() {
  const s = parsed.stats;
  const tiles = [
    ["Questions", s.questions],
    ["Paragraphs", s.non_empty_paragraphs],
    ["Tables", s.tables],
    ["Runs", s.runs],
    ["Bold runs", s.bold_runs],
    ["Italic runs", s.italic_runs],
  ];
  document.getElementById("stats").replaceChildren(
    ...tiles.map(([label, value]) => {
      const tile = el("div", "stat");
      tile.append(el("div", "value", String(value)), el("div", "label", label));
      return tile;
    })
  );
}

function renderWarnings() {
  document.getElementById("warnings").replaceChildren(
    ...parsed.warnings.map((text) => el("div", "warning", `⚠ ${text}`))
  );
}

function renderQuestions() {
  const container = document.getElementById("view-questions");
  container.replaceChildren(
    ...parsed.questions.map((question) => {
      const card = el("div", "question");

      const heading = el("h3");
      const pill = el("span", "label-pill", question.label || "preamble");
      if (question.is_preamble) pill.classList.add("preamble");
      heading.append(pill);

      const title = el("span", "title");
      title.append(...runNodes(question.title_runs));
      heading.append(title);
      card.append(heading);

      card.append(
        el(
          "div",
          "meta",
          `blocks ${question.start_index}–${question.end_index} · ` +
            `${question.block_indices.length} block(s)` +
            (question.pattern ? ` · matched: ${question.pattern}` : "")
        )
      );

      for (const index of question.block_indices) {
        if (index === question.title_block_index) continue;
        const block = blockIndex.get(index);
        if (block) card.append(blockNode(block));
      }
      return card;
    })
  );
}

function renderBlocks() {
  const container = document.getElementById("view-blocks");
  const card = el("div", "question");
  card.append(...parsed.blocks.map(blockNode));
  container.replaceChildren(card);
}

function blockNode(block) {
  if (block.kind === "table") return tableNode(block);

  const line = el("div", "block-line");
  line.append(el("span", "idx", String(block.index)));

  const marker = block.list_info?.marker || block.literal_marker;
  line.append(el("span", "marker", marker ? String(marker) : ""));

  const text = el("span", "text");
  if (block.runs.length) {
    text.append(...runNodes(block.runs));
  } else {
    text.append(document.createTextNode(" "));
  }
  line.append(text);

  if (block.style && block.style !== "Normal") {
    line.append(el("span", "style-tag", block.style));
  }
  return line;
}

function tableNode(block) {
  const wrapper = el("div");
  wrapper.append(
    el("div", "meta", `Table (block ${block.index}) — ${block.n_rows} × ${block.n_cols}`)
  );

  const table = el("table", "grid");
  for (const row of block.rows) {
    const tr = el("tr");
    for (const cell of row.cells) {
      const td = el(row.is_header ? "th" : "td");
      if (cell.grid_span > 1) td.colSpan = cell.grid_span;
      for (const inner of cell.blocks) {
        const p = el("div");
        p.append(...runNodes(inner.runs || []));
        td.append(p);
      }
      tr.append(td);
    }
    table.append(tr);
  }
  wrapper.append(table);
  return wrapper;
}

/** Render formatting runs as real <strong>/<em>/<u> nodes. */
function runNodes(runs) {
  return (runs || []).map((run) => {
    let node = document.createTextNode(run.text);
    for (const [flag, tag] of [["bold", "strong"], ["italic", "em"], ["underline", "u"]]) {
      if (run[flag]) {
        const wrapper = document.createElement(tag);
        wrapper.append(node);
        node = wrapper;
      }
    }
    return node;
  });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* -- tabs --------------------------------------------------------------- */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".view").forEach((view) => {
      view.hidden = view.id !== `view-${tab.dataset.view}`;
    });
  });
});
