/*
  MAGI console client.

  One WebSocket, one state object, one render pass. No framework: the whole
  surface is a dozen elements whose text and data-state change, and a Pi should
  not need a CDN to draw its own screen.

  The load-bearing rule here is that the display never asserts more than the
  daemon told it. Anything unknown reads "—", not a plausible default: a
  console whose panels look confident when the socket is dead is worse than one
  that shows nothing.
*/

const SLOTS = ["top", "left", "right"];

// The anime numbers them MELCHIOR-1, BALTHASAR-2, CASPER-3, and lays them out
// with 2 at the top. Keyed by name so a roster from the YAML that does not
// match the canon still renders, just without the numbering.
const CANON = {
  MELCHIOR:  { index: 1, slot: "right" },
  BALTHASAR: { index: 2, slot: "top" },
  CASPAR:    { index: 3, slot: "left" },
};

// Node states, with the label pair each one shows. The kanji is chrome; the
// English beside it is what the operator actually reads.
const STATE_LABELS = {
  idle:     { jp: "待&#x6A5F;", en: "STANDBY" },
  thinking: { jp: "&#x5BE9;&#x8B70;&#x4E2D;", en: "DELIBERATING" },
  spoke:    { jp: "&#x767A;&#x8A00;", en: "POSITION SET" },
  majority: { jp: "&#x627F;&#x8A8D;", en: "IN CONSENSUS" },
  dissent:  { jp: "&#x5426;&#x5B9A;", en: "DISSENTING" },
  absent:   { jp: "&#x6B20;&#x5E2D;", en: "NO RESPONSE" },
};

const OUTCOME = {
  UNANIMOUS: { jp: "&#x5168;&#x4F1A;&#x4E00;&#x81F4;", core: "unanimous" },
  MAJORITY:  { jp: "&#x591A;&#x6570;&#x6C7A;",         core: "majority" },
  DEADLOCK:  { jp: "&#x8A0E;&#x8B70;&#x6C7A;&#x88C2;", core: "deadlock" },
};

const $ = (id) => document.getElementById(id);

const state = {
  advisors: [],          // [{name, archetype, model}]
  nodes: new Map(),      // name -> {el, nameEl, stateEl, summaryEl}
  debating: false,
  calls: 0,
  socket: null,        // PTT audio rides this same connection
};

// ── Boot ──────────────────────────────────────────────────────────

async function boot() {
  const config = await fetch("/api/config").then((r) => r.json());

  state.advisors = config.advisors;
  $("m-node").textContent = config.node_id;
  $("m-engine").textContent = config.engine.replace("autogen_", "").toUpperCase();
  $("m-priority").textContent = `A${"-".repeat(Math.max(0, config.max_rounds - 1))}`;

  buildTriad(config.advisors);
  wireInput(config.speech_available);
  connect();
}

function buildTriad(advisors) {
  const triad = $("triad");
  advisors.forEach((advisor, i) => {
    const canon = CANON[advisor.name] || {};
    const slot = canon.slot && !triad.querySelector(`[data-slot="${canon.slot}"]`)
      ? canon.slot
      : SLOTS[i % SLOTS.length];

    const el = document.createElement("article");
    el.className = "node";
    el.dataset.slot = slot;
    el.dataset.state = "idle";
    el.innerHTML = `
      <div class="node-name">${advisor.name}${canon.index ? ` <span class="idx">&bull; ${canon.index}</span>` : ""}</div>
      <div class="node-role"></div>
      <div class="node-model"></div>
      <div class="node-state"></div>
      <div class="node-summary"></div>`;
    // textContent for both: an archetype and a model tag both come from
    // config/magi.yaml, which is a file someone edits by hand.
    el.querySelector(".node-role").textContent = advisor.archetype || "";
    // Which model is behind each seat is the one thing about this system that
    // changes most often — rotating them is the whole point — so the screen
    // says which one is answering rather than leaving it to whoever remembers
    // what the YAML said.
    el.querySelector(".node-model").textContent = advisor.model || "";
    triad.appendChild(el);

    state.nodes.set(advisor.name, {
      el,
      stateEl: el.querySelector(".node-state"),
      summaryEl: el.querySelector(".node-summary"),
    });
    // A node that has never been asked anything still owes the operator a
    // state. Left empty it reads as a panel that failed to load rather than
    // one waiting for a question.
    setNode(advisor.name, "idle", "");
  });
}

// ── Transport ─────────────────────────────────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ui/stream`);

  state.socket = ws;
  ws.onopen = () => setFlag("flag-link", "LINK ACTIVE", "ok");

  ws.onmessage = (event) => {
    const { topic, data } = JSON.parse(event.data);
    if (topic === "question") onQuestion(data);
    else if (topic === "turn") onTurn(data);
    else if (topic === "verdict") onVerdict(data);
    else if (topic === "status") onStatus(data);
    else if (topic === "draft") onDraft(data);
    else if (topic === "stt") onStt(data);
  };

  const dropped = () => {
    setFlag("flag-link", "LINK LOST", "alarm");
    // Fixed 2 s, not exponential backoff: this is a kiosk on a LAN beside the
    // machine it talks to. The failure it recovers from is a daemon restart,
    // and an operator watching a blank screen should not wait 30 s for it.
    setTimeout(connect, 2000);
  };
  ws.onclose = dropped;
  ws.onerror = () => ws.close();
}

// ── Events ────────────────────────────────────────────────────────

function onQuestion(data) {
  state.debating = true;
  state.calls = 0;
  $("m-code").textContent = (data.debate_id || "").slice(0, 4).toUpperCase() || "----";
  $("verdict").hidden = true;
  setFlag("flag-phase", "DELIBERATION", "busy");
  setFlag("flag-round", `RUNDE 1/${data.max_rounds || "-"}`, null);
  // Stale from the previous debate otherwise, and a call count that belongs to
  // a question already answered is worse than no call count.
  setFlag("flag-calls", "0 CALLS", null);
  document.body.dataset.phase = "deliberating";
  $("core").dataset.state = "deliberating";
  $("core-state").textContent = "IN SESSION";

  // Everyone starts thinking at once: phase A runs the advisors in parallel.
  for (const [name, node] of state.nodes) {
    setNode(name, "thinking", "");
    node.summaryEl.textContent = "";
  }
}

function onTurn(data) {
  setNode(data.advisor, "spoke", data.turn.summary);
  setFlag("flag-round", `RUNDE ${data.round_index}`, null);
}

function onVerdict(record) {
  state.debating = false;
  document.body.dataset.phase = "resolved";

  const spec = OUTCOME[record.outcome] || { jp: "?", core: "deadlock" };
  const panel = $("verdict");
  panel.hidden = false;
  panel.dataset.outcome = record.outcome;
  $("verdict-kanji").innerHTML = spec.jp;
  $("verdict-flag").textContent = record.outcome;
  $("verdict-meta").textContent =
    `${record.rounds_used} ROUNDS · ${record.duration_s.toFixed(0)}s · ` +
    `${record.llm_calls} CALLS · ${record.terminated_by.toUpperCase()}`;
  $("verdict-answer").textContent = record.verdict.answer;

  showNote("verdict-dissent", record.verdict.dissent && `DISSENT: ${record.verdict.dissent}`);
  showNote("verdict-split", record.verdict.disagreement_point &&
    `SPLIT ON: ${record.verdict.disagreement_point}`);

  // Colour each node by whether it ended inside the agreeing bloc. This is the
  // one honest analogue of the anime's 承認/否定 vote: the advisors do not cast
  // a binary vote on the question, but they do declare who they stand with, and
  // that is what is being shown.
  const present = new Set(record.advisors_present || []);
  const bloc = blocOf(record);
  // Final summaries come from the record, not from the live turns: a browser
  // that connected mid-debate — or after it — has seen no `turn` frames, and
  // replay_last hands it only the verdict. Reading them from here is what makes
  // a late-joining kiosk show the same screen as one that watched throughout.
  const finalTurn = new Map(record.turns.map((t) => [t.advisor, t.turn]));
  for (const [name] of state.nodes) {
    if (!present.has(name)) {
      setNode(name, "absent", "no response from this node");
      continue;
    }
    setNode(name, bloc.has(name) ? "majority" : "dissent",
            finalTurn.get(name)?.summary || "");
  }

  $("core").dataset.state = spec.core;
  $("core-state").textContent = record.outcome;
  setFlag("flag-phase", "RESOLVED", "ok");
  setFlag("flag-calls", `${record.llm_calls} CALLS`, null);
}

/*
  Who ended up in the agreeing bloc, recomputed from the turns.

  The daemon's tally is authoritative and is what produced `outcome`; this only
  decides which nodes to paint green. It mirrors consensus.py's rule — mutual
  agreement, self-references dropped — so the two never disagree on screen. A
  UNANIMOUS verdict short-circuits: everyone present is in.
*/
function blocOf(record) {
  const present = new Set(record.advisors_present || []);
  if (record.outcome === "UNANIMOUS") return present;

  const claims = new Map();
  for (const t of record.turns) {
    claims.set(t.advisor, new Set(
      (t.turn.agrees_with || [])
        .map((n) => n.trim().toUpperCase())
        .filter((n) => present.has(n) && n !== t.advisor)
    ));
  }

  let best = new Set();
  for (const name of present) {
    const bloc = new Set([name]);
    for (const other of present) {
      if (other === name) continue;
      if (claims.get(name)?.has(other) && claims.get(other)?.has(name)) bloc.add(other);
    }
    if (bloc.size > best.size) best = bloc;
  }
  return best.size > 1 ? best : new Set();
}

function onStatus(status) {
  const temp = status.temp_c;
  $("flag-temp").textContent = temp == null ? "TEMP —" : `TEMP ${temp.toFixed(1)}°C`;

  const condition = status.condition || "unknown";
  const level = { GREEN: "ok", CAUTION: "caution", WARNING: "warning", EMERGENCY: "alarm" };
  setFlag("flag-temp", null, level[condition] || "unknown");
  setFlag(
    "flag-condition",
    status.throttling?.length ? status.throttling.join(" ") : `CONDITION ${condition}`,
    level[condition] || "unknown"
  );
}

// ── Draft ─────────────────────────────────────────────────────────

/*
  One line per PTT press, each with its own delete.

  Deleting by id, never by position: a press still being transcribed can land
  between the operator reading a line and tapping its button, and a positional
  delete would then remove whichever line arrived in that slot instead.
*/
function onDraft(data) {
  const list = $("draft-lines");
  const panel = $("draft");
  panel.hidden = data.lines.length === 0;
  $("draft-count").textContent = data.lines.length
    ? `${data.lines.length} LINE${data.lines.length > 1 ? "S" : ""}` +
      (data.full ? " · FULL" : "")
    : "";

  list.replaceChildren(...data.lines.map((line, i) => {
    const li = document.createElement("li");
    li.className = "draft-line";

    const n = document.createElement("span");
    n.className = "n";
    n.textContent = String(i + 1).padStart(2, "0");

    const t = document.createElement("span");
    t.className = "t";
    // textContent, not innerHTML: this string came out of a speech model and
    // is echoed straight back to the screen.
    t.textContent = line.text;

    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn-tiny";
    del.textContent = "DEL";
    del.title = "drop this line";
    del.addEventListener("click", () =>
      fetch(`/api/draft/${line.id}`, { method: "DELETE" }));

    li.append(n, t, del);
    return li;
  }));
}

// What the node heard, or why it heard nothing. Each state needs its own words
// because the operator's next move differs: press again, press longer, or stop
// pressing because the node is broken.
const STT_HINT = {
  transcribing: "listening…",
  empty: "nothing was heard — hold the button while you speak",
  discarded: "no clear speech in that recording",
  full: "the draft is full — send it or clear it",
  unavailable: "speech recognition is not installed on this node",
};

function onStt(data) {
  const hint = $("stt-hint");
  if (data.state === "ok") {
    hint.hidden = true;
    return;
  }
  hint.hidden = false;
  hint.dataset.state = data.state;
  hint.textContent = STT_HINT[data.state] || data.detail || data.state;
}

// ── DOM helpers ───────────────────────────────────────────────────

function setNode(name, nodeState, summary) {
  const node = state.nodes.get(name);
  if (!node) return;
  node.el.dataset.state = nodeState;
  const label = STATE_LABELS[nodeState];
  node.stateEl.innerHTML = `${label.jp}<span class="en">${label.en}</span>`;
  if (summary !== undefined) node.summaryEl.textContent = summary || "";
}

function setFlag(id, text, level) {
  const el = $(id);
  if (!el) return;
  if (text !== null && text !== undefined) el.textContent = text;
  if (level !== null && level !== undefined) el.dataset.state = level;
}

function showNote(id, text) {
  const el = $(id);
  el.hidden = !text;
  if (text) el.textContent = text;
}

// ── Input ─────────────────────────────────────────────────────────

function wireInput(speechAvailable) {
  const form = $("ask");
  const input = $("question");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const typed = input.value.trim();

    // Typing and dictating are the same act, so SEND commits whichever is
    // there. Typed text goes straight through; an empty box means the operator
    // dictated, and SEND commits the draft they have just read back.
    if (typed) {
      input.value = "";
      await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: typed }),
      });
      return;
    }
    await fetch("/api/send", { method: "POST" });
  });

  $("draft-clear").addEventListener("click", () =>
    fetch("/api/draft", { method: "DELETE" }));

  wirePtt(speechAvailable);
  wireFullscreen();
}

// ── Fullscreen ────────────────────────────────────────────────────

/*
  The kiosk is already fullscreen via `chromium --kiosk`, so this is for every
  other way the console gets opened: a laptop on the LAN, a tablet, the Pi's
  own browser started by hand. Without it those all show the console inside
  browser chrome, which on an 800x480 panel costs a third of the triad.

  Hidden entirely where the API does not exist rather than left as a button
  that does nothing — the display's rule is that it never shows a control it
  cannot honour.
*/
function wireFullscreen() {
  const btn = $("fullscreen");
  const root = document.documentElement;
  const request = root.requestFullscreen || root.webkitRequestFullscreen;
  if (!request) return;

  btn.hidden = false;

  btn.addEventListener("click", async () => {
    try {
      if (fullscreenElement()) {
        await (document.exitFullscreen?.() ?? document.webkitExitFullscreen?.());
      } else {
        // Must be called on the element, not on the captured reference.
        await (root.requestFullscreen?.() ?? root.webkitRequestFullscreen?.());
      }
    } catch (err) {
      // Denied by policy, or an iframe without `allow="fullscreen"`. Say so on
      // the hint line instead of leaving a button that silently does nothing.
      onStt({ state: "error", detail: `fullscreen refused: ${err.name}` });
    }
  });

  // Esc leaves fullscreen without going through the button, so the label is
  // driven by the event rather than by what was last clicked.
  for (const event of ["fullscreenchange", "webkitfullscreenchange"]) {
    document.addEventListener(event, syncFullscreenLabel);
  }
  syncFullscreenLabel();
}

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function syncFullscreenLabel() {
  const active = Boolean(fullscreenElement());
  $("fs-label").textContent = active ? "EXIT" : "FULLSCREEN";
  $("fullscreen").dataset.state = active ? "busy" : "unknown";
}

// ── Push to talk ──────────────────────────────────────────────────

let recorder = null;
let chunks = [];

/*
  Hold to record, release to transcribe.

  The blob goes out as a binary frame on the WebSocket that is already open,
  rather than a POST per press: no handshake per sentence, and a failure shows
  up on the same LINK indicator the operator is already watching.

  The microphone is requested on the first press, not at load. A kiosk that
  demands permission before anyone has asked it for anything gets the
  permission denied, and then the button is dead for the session.
*/
function wirePtt(speechAvailable) {
  const ptt = $("ptt");

  if (!speechAvailable) {
    // Present but honestly disabled, rather than hidden. The control is part of
    // the machine's identity; pretending it does not exist would misrepresent
    // what this node is for, and pretending it works would be worse.
    ptt.disabled = true;
    ptt.title = "speech recognition is not installed — ./setup_env.sh --voice";
    ptt.querySelector(".ptt-label").textContent = "NO MIC";
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    // getUserMedia needs a secure context: over plain http only localhost
    // qualifies, so a browser opening the node by LAN address has no
    // microphone at all. Saying so beats a button that silently does nothing.
    ptt.disabled = true;
    ptt.querySelector(".ptt-label").textContent = "NO MIC";
    ptt.title = window.isSecureContext
      ? "this browser has no MediaRecorder"
      : "no microphone over plain http — open the node at localhost, or use https";
    return;
  }

  ptt.disabled = false;
  ptt.addEventListener("pointerdown", (e) => { e.preventDefault(); startRecording(ptt); });
  for (const event of ["pointerup", "pointerleave", "pointercancel"]) {
    ptt.addEventListener(event, () => stopRecording(ptt));
  }
}

async function startRecording(ptt) {
  if (recorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      // The tracks are stopped explicitly: leaving them live keeps the
      // browser's recording indicator on between presses, which on a kiosk
      // reads as "this thing is always listening".
      for (const track of stream.getTracks()) track.stop();
      const blob = new Blob(chunks, { type: recorder.mimeType });
      recorder = null;
      if (blob.size > 0 && state.socket?.readyState === WebSocket.OPEN) {
        state.socket.send(await blob.arrayBuffer());
      }
    };
    recorder.start();
    ptt.classList.add("recording");
  } catch (err) {
    ptt.classList.remove("recording");
    onStt({ state: "error", detail: `microphone unavailable: ${err.name}` });
  }
}

function stopRecording(ptt) {
  ptt.classList.remove("recording");
  if (recorder?.state === "recording") recorder.stop();
}

boot();
