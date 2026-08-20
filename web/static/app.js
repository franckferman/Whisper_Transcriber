/* ============================================================
   whispr - app.js
   Vanilla JS, no frameworks.
   ============================================================ */

(function () {
  "use strict";

  // ---- State ---------------------------------------------------------------

  let selectedBackend = null;
  let currentJobId = null;
  let selectedFile = null;
  let backendsData = {};

  // ---- Auth (only used when the server sets WHISPR_AUTH_TOKEN) --------------

  let authToken = sessionStorage.getItem("whispr_token") || "";

  function authHeaders() {
    return authToken ? { "X-Whispr-Token": authToken } : {};
  }

  // fetch wrapper: attaches the token, and on 401 prompts once and retries
  async function apiFetch(url, opts = {}, allowPrompt = true) {
    opts.headers = Object.assign({}, opts.headers || {}, authHeaders());
    const res = await fetch(url, opts);
    if (res.status === 401 && allowPrompt) {
      const entered = prompt("This whispr instance requires an access token:");
      if (entered) {
        authToken = entered.trim();
        sessionStorage.setItem("whispr_token", authToken);
        return apiFetch(url, opts, false);
      }
    }
    return res;
  }

  function withToken(url) {
    return authToken ? url + (url.includes("?") ? "&" : "?") +
      "token=" + encodeURIComponent(authToken) : url;
  }

  // ---- DOM references -------------------------------------------------------

  const tabFile       = document.getElementById("tab-file");
  const tabUrl        = document.getElementById("tab-url");
  const panelFile     = document.getElementById("panel-file");
  const panelUrl      = document.getElementById("panel-url");
  const fileDrop      = document.getElementById("file-drop");
  const fileInput     = document.getElementById("file-input");
  const browseBtn     = document.getElementById("browse-btn");
  const fileNameHint  = document.getElementById("file-name-hint");
  const urlInput      = document.getElementById("url-input");
  const backendList   = document.getElementById("backend-list");
  const openaiKeyRow  = document.getElementById("openai-key-row");
  const openaiKeyInput = document.getElementById("openai-key-input");
  const fasterWhisperRow   = document.getElementById("faster-whisper-row");
  const fwModelSelect      = document.getElementById("fw-model-select");
  const whisperCppRow      = document.getElementById("whisper-cpp-row");
  const languageSel   = document.getElementById("language-select");
  const workersInput  = document.getElementById("workers-input");
  const transcribeBtn = document.getElementById("transcribe-btn");
  const formCard      = document.getElementById("form-card");
  const progressCard  = document.getElementById("progress-card");
  const progressStatus = document.getElementById("progress-status");
  const spinner       = document.getElementById("spinner");
  const logBox        = document.getElementById("log-box");
  const resultsCard   = document.getElementById("results-card");
  const downloadRow   = document.getElementById("download-row");
  const resetBtn      = document.getElementById("reset-btn");

  // ---- Tabs ----------------------------------------------------------------

  function activateTab(tab) {
    [tabFile, tabUrl].forEach(t => {
      t.classList.remove("tab--active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("tab--active");
    tab.setAttribute("aria-selected", "true");

    panelFile.classList.toggle("tab-panel--hidden", tab !== tabFile);
    panelUrl.classList.toggle("tab-panel--hidden", tab !== tabUrl);
  }

  tabFile.addEventListener("click", () => activateTab(tabFile));
  tabUrl.addEventListener("click", () => activateTab(tabUrl));

  // ---- File drop -----------------------------------------------------------

  browseBtn.addEventListener("click", () => fileInput.click());
  fileDrop.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
  });
  fileDrop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      selectedFile = fileInput.files[0];
      fileNameHint.textContent = selectedFile.name;
      fileDrop.style.borderColor = "var(--accent)";
    }
  });

  fileDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileDrop.classList.add("file-drop--over");
  });
  fileDrop.addEventListener("dragleave", () => {
    fileDrop.classList.remove("file-drop--over");
  });
  fileDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    fileDrop.classList.remove("file-drop--over");
    if (e.dataTransfer.files.length) {
      selectedFile = e.dataTransfer.files[0];
      fileNameHint.textContent = selectedFile.name;
      fileDrop.style.borderColor = "var(--accent)";
    }
  });

  // ---- Backend detection ---------------------------------------------------

  async function loadBackends() {
    try {
      const res = await apiFetch("/api/backends");
      backendsData = await res.json();
      renderBackends(backendsData);
    } catch (err) {
      backendList.innerHTML = `<p class="loading-text" style="color:#f97583">
        Failed to detect backends: ${escHtml(String(err))}
      </p>`;
    }
  }

  function renderBackends(data) {
    backendList.innerHTML = "";

    // Priority order for auto-selection
    const order = ["faster_whisper", "whisper_cpp", "openai"];
    const labels = {
      faster_whisper: "faster-whisper",
      whisper_cpp:    "whisper.cpp",
      openai:         "OpenAI API",
    };

    let autoSelected = false;

    order.forEach((key) => {
      const info = data[key];
      if (!info) return;

      const option = document.createElement("label");
      option.className = "backend-option" +
        (info.available ? "" : " backend-option--disabled");

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "backend";
      radio.value = key;
      radio.disabled = !info.available;

      // Auto-select first available (faster_whisper > whisper_cpp > openai)
      if (info.available && !autoSelected) {
        radio.checked = true;
        option.classList.add("backend-option--selected");
        selectedBackend = key;
        autoSelected = true;
      }

      const dot = document.createElement("span");
      dot.className = "backend-dot " +
        (info.available ? "backend-dot--green" : "backend-dot--grey");

      const nameSpan = document.createElement("span");
      nameSpan.className = "backend-name";
      nameSpan.textContent = labels[key] || key;

      option.appendChild(radio);
      option.appendChild(dot);
      option.appendChild(nameSpan);

      if (!info.available) {
        const col = document.createElement("div");
        col.style.marginLeft = "auto";
        col.style.textAlign = "right";

        const reasonEl = document.createElement("span");
        reasonEl.className = "backend-reason";
        reasonEl.textContent = info.reason || "Unavailable";
        col.appendChild(reasonEl);

        if (info.install_hint) {
          const hintEl = document.createElement("div");
          hintEl.className = "backend-hint";
          hintEl.textContent = info.install_hint;
          col.appendChild(hintEl);
        }
        option.appendChild(col);
      }

      option.addEventListener("click", () => {
        if (!info.available) return;
        document.querySelectorAll(".backend-option").forEach(el => {
          el.classList.remove("backend-option--selected");
        });
        option.classList.add("backend-option--selected");
        radio.checked = true;
        selectedBackend = key;
        updateExtraOptions();
      });

      backendList.appendChild(option);
    });

    updateExtraOptions();
  }

  function updateExtraOptions() {
    fasterWhisperRow.hidden = selectedBackend !== "faster_whisper";
    openaiKeyRow.hidden     = selectedBackend !== "openai";
    whisperCppRow.hidden    = selectedBackend !== "whisper_cpp";
  }

  // ---- Transcribe ----------------------------------------------------------

  transcribeBtn.addEventListener("click", startTranscription);

  async function startTranscription() {
    // Validate
    const isFileMode = tabFile.getAttribute("aria-selected") === "true";

    if (isFileMode && !selectedFile) {
      alert("Please select a file first.");
      return;
    }
    if (!isFileMode && !urlInput.value.trim()) {
      alert("Please enter a URL.");
      return;
    }
    if (!selectedBackend) {
      alert("No backend selected or available.");
      return;
    }

    const fmtChecks = document.querySelectorAll('input[name="fmt"]:checked');
    const formats   = Array.from(fmtChecks).map(c => c.value);
    if (!formats.length) {
      alert("Select at least one output format.");
      return;
    }

    // Build form data
    const fd = new FormData();
    fd.append("backend",  selectedBackend);
    fd.append("language", languageSel.value);
    fd.append("formats",  formats.join(","));
    fd.append("workers",  workersInput.value);

    if (isFileMode) {
      fd.append("source_type", "file");
      fd.append("file", selectedFile);
    } else {
      fd.append("source_type", "url");
      fd.append("url", urlInput.value.trim());
    }

    if (selectedBackend === "faster_whisper") {
      fd.append("fw_model", fwModelSelect.value);
    }
    if (selectedBackend === "openai" && openaiKeyInput.value.trim()) {
      fd.append("openai_key", openaiKeyInput.value.trim());
    }
    // whisper.cpp binary/model are pinned server-side -- nothing to send

    // Disable button, show progress
    transcribeBtn.disabled = true;
    progressCard.hidden = false;
    resultsCard.hidden  = true;
    logBox.innerHTML    = "";
    progressStatus.textContent = "Starting...";
    spinner.className   = "spinner";

    // Submit job
    let jobId;
    try {
      const res  = await apiFetch("/api/transcribe", { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      jobId = data.job_id;
      currentJobId = jobId;
    } catch (err) {
      showError("Failed to start job: " + err.message);
      return;
    }

    progressStatus.textContent = "Running...";

    // Open WebSocket (token goes in the query string; browsers can't set ws headers)
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(withToken(`${proto}://${location.host}/ws/${jobId}`));

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }

      if (msg.type === "log") {
        appendLog(msg.level, msg.msg);
        progressStatus.textContent = msg.msg.slice(0, 80);
      } else if (msg.type === "done") {
        onJobDone(msg);
      } else if (msg.type === "error") {
        showError(msg.msg);
      }
    };

    ws.onerror = () => {
      showError("WebSocket connection error.");
    };

    ws.onclose = () => {
      // If status is still "Running", poll once
      if (progressStatus.textContent.startsWith("Running")) {
        pollStatus(jobId);
      }
    };
  }

  async function pollStatus(jobId) {
    try {
      const res  = await apiFetch(`/api/status/${jobId}`, {}, false);
      const data = await res.json();
      if (data.status === "done") {
        onJobDone({ job_id: jobId, formats: [] });
      } else if (data.status === "error") {
        showError("Job failed. Check log for details.");
      }
    } catch {}
  }

  function onJobDone(msg) {
    progressStatus.textContent = "Done!";
    spinner.className = "spinner spinner--done";
    appendLog("INFO", "All done.");

    // Build download buttons
    downloadRow.innerHTML = "";
    const fmts = msg.formats && msg.formats.length
      ? msg.formats
      : ["txt"];

    fmts.forEach((fmt) => {
      const a = document.createElement("a");
      a.className = "btn btn--download";
      a.href = withToken(`/api/download/${msg.job_id}/${fmt}`);
      a.download = "";
      a.innerHTML = `<span class="dl-icon">&#8615;</span> ${fmt.toUpperCase()}`;
      downloadRow.appendChild(a);
    });

    resultsCard.hidden = false;
    transcribeBtn.disabled = false;
  }

  function showError(msg) {
    progressStatus.textContent = "Error";
    spinner.style.borderColor = "#f97583";
    spinner.style.borderTopColor = "#f97583";
    spinner.style.animation = "none";
    appendLog("ERROR", msg);
    transcribeBtn.disabled = false;
  }

  function appendLog(level, msg) {
    const span = document.createElement("span");
    span.className = "log-line log-line--" + (level || "INFO");
    span.textContent = msg;
    logBox.appendChild(span);
    logBox.appendChild(document.createTextNode("\n"));
    logBox.scrollTop = logBox.scrollHeight;
  }

  // ---- Reset ---------------------------------------------------------------

  resetBtn.addEventListener("click", () => {
    resultsCard.hidden  = true;
    progressCard.hidden = true;
    logBox.innerHTML    = "";
    selectedFile        = null;
    fileNameHint.textContent = "No file selected";
    fileDrop.style.borderColor = "";
    fileInput.value     = "";
    urlInput.value      = "";
    transcribeBtn.disabled = false;
    currentJobId        = null;
  });

  // ---- Helpers -------------------------------------------------------------

  function escHtml(str) {
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
  }

  // ---- Init ----------------------------------------------------------------

  loadBackends();

})();
