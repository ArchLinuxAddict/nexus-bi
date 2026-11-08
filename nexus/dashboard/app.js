const state = {
  apiKey: localStorage.getItem("nexus_api_key") || "",
  analysis: null,
  functions: [],
  selectedFunction: null,
  reconstruction: null,
  binaries: [],
  modelAnalysis: null,
  checkpoint: "",
  javaBinary: null,
  javaAnalysis: null,
  javaReconstruction: null,
};

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  if (options.body && typeof options.body === "string") headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = text; }
  if (!response.ok) {
    const message = payload && payload.error ? payload.error.message : (payload && payload.detail ? payload.detail : text);
    throw new Error(`${response.status}: ${message}`);
  }
  return payload;
}

function setStatus(selector, message, kind = "muted") {
  const element = $(selector);
  if (!element) return;
  element.textContent = message;
  element.className = kind;
}

function switchView(name) {
  document.querySelectorAll(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  if (name === "overview") loadOverview();
  if (name === "binaries") loadBinaries();
  if (name === "datasets") loadDatasets();
  if (name === "model") loadModel();
  if (name === "foundation") loadFoundation();
  if (name === "tools") loadTools();
  if (name === "residual") loadResidual();
  if (name === "reconstruction") renderReconstructionState();
}

function renderReconstructionState() {
  const empty = $("#reconstruction-empty");
  const content = $("#reconstruction-content");
  if (!empty || !content) return;
  const hasAnalysis = Boolean(state.analysis);
  empty.classList.toggle("hidden", hasAnalysis);
  content.classList.toggle("hidden", !hasAnalysis);
}

async function restoreLastAnalysis() {
  const analysisId = localStorage.getItem("nexus_last_analysis");
  if (!analysisId) return;
  try {
    const result = await api(`/v1/analyses/${analysisId}`);
    state.analysis = { ...(result.summary || {}), analysis_id: result.analysis_id, binary_id: result.binary_id };
    renderAnalysis();
  } catch (error) {
    localStorage.removeItem("nexus_last_analysis");
  }
}

async function loadHealth() {
  try {
    const health = await api("/healthz");
    setStatus("#health", health.status, "badge good");
    const version = await api("/v1/version");
    $("#version").textContent = `${version.name} ${version.version}`;
  } catch (error) {
    setStatus("#health", "unavailable", "badge bad");
  }
}

async function loadOverview() {
  try {
    const capabilities = await api("/v1/capabilities");
    const rows = Object.entries(capabilities).map(([name, item]) => {
      const kind = item.status === "available" ? "good" : item.status === "experimental" ? "warn" : "bad";
      return `<tr><td><code>${esc(name)}</code></td><td><span class="badge ${kind}">${esc(item.status)}</span></td><td>${esc(item.reason)}</td></tr>`;
    }).join("");
    $("#capabilities").innerHTML = `<table><thead><tr><th>Capability</th><th>Status</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table>`;
    const hardware = await api("/v1/hardware");
    $("#hardware").innerHTML = `<table><tbody>
      <tr><th>CPU</th><td>${esc(hardware.host.cpu_model)} (${esc(hardware.host.logical_cpus)} logical)</td></tr>
      <tr><th>Memory</th><td>${(hardware.host.memory_total_bytes / 1073741824).toFixed(1)} GiB</td></tr>
      <tr><th>Disk free</th><td>${(hardware.host.disk_free_bytes / 1073741824).toFixed(1)} GiB</td></tr>
      <tr><th>GPU compute</th><td>${hardware.gpu.compute_available ? "available" : "unavailable"}</td></tr>
      <tr><th>KVM</th><td>${hardware.isolation.kvm ? "present" : "absent"}</td></tr>
      <tr><th>QEMU</th><td>${Object.values(hardware.isolation.qemu).some(Boolean) ? "present" : "absent"}</td></tr>
      <tr><th>Large training</th><td>${hardware.large_training_ready ? "possible" : "disabled without supported GPU"}</td></tr>
    </tbody></table>`;
    const binaries = await api("/v1/binaries?limit=8");
    $("#overview-binaries").innerHTML = binaries.length ? `<table><thead><tr><th>ID</th><th>SHA-256</th><th>Size</th><th>Machine</th></tr></thead><tbody>${binaries.map((item) => `<tr><td><code>${esc(item.id)}</code></td><td><code>${esc(item.sha256.slice(0, 16))}…</code></td><td>${item.size}</td><td>${esc(item.machine || "unknown")}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No binaries uploaded.</div>`;
  } catch (error) {
    $("#capabilities").innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}



async function loadTools() {
  try {
    const matrix = await api("/v1/tools");
    const tools = Object.values(matrix.tools || {});
    $("#tool-matrix").innerHTML = `<p>Status: <strong>${esc(matrix.status)}</strong> · missing: ${esc((matrix.missing_required || []).join(", ") || "none")}</p><table><thead><tr><th>Tool</th><th>Status</th><th>Version</th><th>Capabilities</th></tr></thead><tbody>${tools.map((tool) => `<tr><td>${esc(tool.name)}</td><td>${esc(tool.status)}</td><td>${esc(tool.version)}</td><td>${esc((tool.capabilities || []).join(", "))}</td></tr>`).join("")}</tbody></table>`;
  } catch (error) {
    $("#tool-matrix").textContent = error.message;
  }
}

async function uploadJavaArtifact() {
  const file = $("#java-file")?.files?.[0];
  if (!file) return;
  try {
    setStatus("#java-upload-status", "uploading…", "muted");
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/v1/binaries", { method: "POST", body: form, headers: state.apiKey ? { "X-API-Key": state.apiKey } : {} });
    if (!response.ok) throw new Error(await response.text());
    const uploaded = await response.json();
    state.javaBinary = uploaded;
    setStatus("#java-upload-status", `uploaded ${uploaded.id} · ${uploaded.format}`, "muted");
    await analyzeJavaArtifact();
  } catch (error) {
    setStatus("#java-upload-status", error.message, "muted");
  }
}

async function analyzeJavaArtifact() {
  if (!state.javaBinary) return;
  try {
    setStatus("#java-upload-status", "analyzing…", "muted");
    const analysis = await api("/v1/java/analyze-binary", { method: "POST", body: JSON.stringify({ binary_id: state.javaBinary.id }) });
    state.javaAnalysis = analysis;
    const summary = analysis.summary || {};
    const nativeMethods = summary.native_methods || [];
    $("#java-summary").innerHTML = `<table><tbody>
      <tr><th>Format</th><td>${esc(analysis.format)}</td></tr>
      <tr><th>Classes</th><td>${esc(summary.classes ?? "")}</td></tr>
      <tr><th>Methods</th><td>${esc(summary.methods ?? "")}</td></tr>
      <tr><th>Fields</th><td>${esc(summary.fields ?? "")}</td></tr>
      <tr><th>Native methods</th><td>${esc(nativeMethods.length)}</td></tr>
      </tbody></table>`;
    setStatus("#java-upload-status", "analysis ready", "muted");
  } catch (error) {
    setStatus("#java-upload-status", error.message, "muted");
  }
}

async function reconstructJavaArtifact() {
  if (!state.javaBinary) { setStatus("#java-reconstruction-status", "upload an artifact first", "muted"); return; }
  try {
    setStatus("#java-reconstruction-status", "decompiling with JADX/CFR…", "muted");
    const result = await api("/v1/java/reconstruct-binary", {
      method: "POST",
      body: JSON.stringify({
        binary_id: state.javaBinary.id,
        project_name: $("#java-project-name").value || "nexus_java_reconstruction",
        verify: $("#java-verify").checked,
      }),
    });
    state.javaReconstruction = result;
    const files = Object.keys(result.files || {}).filter((file) => file.endsWith(".java"));
    $("#java-file-select").classList.toggle("hidden", files.length === 0);
    $("#java-file-select").innerHTML = files.map((file) => `<option value="${esc(file)}">${esc(file)}</option>`).join("");
    $("#java-evidence").textContent = JSON.stringify(result.evidence_map || {}, null, 2);
    if (files.length) await showJavaSource(files[0]);
    const verification = result.verification?.status || "not requested";
    const reason = result.verification?.reason ? ` (${result.verification.reason})` : "";
    const selection = result.evidence_map?.source_selection;
    const sourceCount = selection ? ` · ${selection.source_files_returned}/${selection.source_files_total} sources` : "";
    setStatus("#java-reconstruction-status", `${result.classification}${sourceCount} · verify=${verification}${reason}`, "muted");
  } catch (error) {
    setStatus("#java-reconstruction-status", error.message, "muted");
  }
}

async function showJavaSource(file) {
  if (!state.javaReconstruction || !file) return;
  $("#java-source").textContent = state.javaReconstruction.files?.[file] || "";
}

async function loadCheckpoints() {
  try {
    const checkpoints = await api("/v1/foundation/checkpoints");
    const select = $("#model-checkpoint");
    if (!select) return;
    const usable = checkpoints.filter((item) => item.path && /(?:^|\/)final\.pt$|best\.pt$|last\.pt$/.test(item.path));
    select.innerHTML = usable.map((item) => `<option value="${esc(item.path)}">${esc(item.relative_path)} · step ${esc(item.step ?? "?")} · ${esc(item.status ?? "unknown")}</option>`).join("");
    const canonical = usable.find((item) => item.path.includes("nexus-bi-unified-first-run") && item.path.endsWith("final.pt"));
    const preferred = canonical || usable[0];
    if (preferred) {
      select.value = preferred.path;
      state.checkpoint = preferred.path;
    }
  } catch (error) {
    const select = $("#model-checkpoint");
    if (select) select.innerHTML = `<option value="">no checkpoints available</option>`;
  }
}

async function runModelAnalysis() {
  if (!state.analysis) return;
  const checkpoint = $("#model-checkpoint").value;
  if (!checkpoint) { setStatus("#model-status", "no checkpoint selected", "muted"); return; }
  try {
    setStatus("#model-status", "running model…", "muted");
    const result = await api("/v1/foundation/analyze", { method: "POST", body: JSON.stringify({ binary_id: state.analysis.binary_id, checkpoint, max_functions: 20 }) });
    state.modelAnalysis = result;
    const functions = result.functions || [];
    $("#model-functions").innerHTML = functions.length ? `<table><thead><tr><th>Function</th><th>Purpose</th><th>Confidence</th><th>IR classes</th><th>CFG edge rate</th></tr></thead><tbody>${functions.map((fn) => `<tr><td><code>${esc(fn.function_id)}</code></td><td>${esc(fn.predicted_purpose)} (${Number(fn.purpose_probability || 0).toFixed(3)})</td><td>${Number(fn.confidence || 0).toFixed(3)}</td><td>${esc((fn.predicted_ir_classes || []).join(", "))}</td><td>${Number(fn.cfg_predicted_edge_rate || 0).toFixed(3)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No functions encoded.</div>`;
    $("#model-raw").classList.remove("hidden");
    $("#model-raw").textContent = JSON.stringify(result, null, 2);
    setStatus("#model-status", result.status === "no_functions" ? "no encodable functions" : `${result.encoded_function_count ?? 0}/${result.deterministic_function_count ?? 0} functions encoded`, "muted");
  } catch (error) {
    setStatus("#model-status", error.message, "muted");
  }
}

async function runModelReconstruction() {
  if (!state.analysis) return;
  const checkpoint = $("#model-checkpoint").value;
  const functionId = state.selectedFunction?.function_id || state.functions?.[0]?.function_id;
  if (!checkpoint || !functionId) { setStatus("#reconstruction-status", "select a function and checkpoint first", "muted"); return; }
  try {
    setStatus("#reconstruction-status", "running model reconstruction…", "muted");
    const result = await api("/v1/foundation/reconstruct", { method: "POST", body: JSON.stringify({ binary_id: state.analysis.binary_id, function_id: functionId, checkpoint, language: "c" }) });
    $("#model-reconstruction").textContent = JSON.stringify(result, null, 2);
    setStatus("#reconstruction-status", result.status || "completed", "muted");
  } catch (error) {
    setStatus("#reconstruction-status", error.message, "muted");
  }
}

async function loadBinaries() {
  try {
    const binaries = await api("/v1/binaries?limit=100");
    state.binaries = binaries;
    $("#binary-list").innerHTML = binaries.length ? `<table><thead><tr><th>ID</th><th>SHA-256</th><th>Machine</th><th>Size</th><th></th></tr></thead><tbody>${binaries.map((item) => `<tr><td><code>${esc(item.id)}</code></td><td><code>${esc(item.sha256.slice(0, 20))}…</code></td><td>${esc(item.machine || "unknown")}</td><td>${item.size}</td><td><button data-analyze="${esc(item.id)}">Analyze</button></td></tr>`).join("")}</tbody></table>` : `<div class="empty">No binaries uploaded.</div>`;
    document.querySelectorAll("[data-analyze]").forEach((button) => button.addEventListener("click", () => analyzeBinary(button.dataset.analyze)));
  } catch (error) {
    $("#binary-list").innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

async function analyzeBinary(binaryId) {
  try {
    setStatus("#upload-status", "analyzing…", "muted");
    const analysis = await api("/v1/analyses", { method: "POST", body: JSON.stringify({ binary_id: binaryId }) });
    state.analysis = analysis;
    if (analysis.analysis_id) localStorage.setItem("nexus_last_analysis", analysis.analysis_id);
    setStatus("#upload-status", `analysis ${analysis.analysis_id || "created"}`, "muted");
    switchView("analysis");
    renderAnalysis();
    if (!state.checkpoint) await loadCheckpoints();
    if (state.checkpoint) await runModelAnalysis();
  } catch (error) {
    setStatus("#upload-status", error.message, "muted");
  }
}

function renderAnalysis() {
  const analysis = state.analysis;
  if (!analysis) return;
  $("#analysis-empty").classList.add("hidden");
  $("#analysis-content").classList.remove("hidden");
  $("#analysis-summary").innerHTML = `<table><tbody>
    <tr><th>Binary</th><td><code>${esc(analysis.sha256)}</code></td></tr>
    <tr><th>Machine</th><td>${esc(analysis.machine)}</td></tr>
    <tr><th>Entry</th><td><code>${esc(analysis.entry)}</code></td></tr>
    <tr><th>Functions</th><td>${analysis.function_count ?? analysis.functions}</td></tr>
    <tr><th>Instructions</th><td>${analysis.instruction_count ?? analysis.instructions}</td></tr>
    <tr><th>Security findings</th><td>${analysis.security_findings_count ?? analysis.security_findings}</td></tr>
    <tr><th>JNI</th><td>${analysis.jni ? "detected" : "not detected"}</td></tr>
    <tr><th>Validation</th><td>${esc(analysis.validation?.status || "unknown")}</td></tr>
    <tr><th>Warnings</th><td>${esc((analysis.warnings || []).join("; ") || "none")}</td></tr>
  </tbody></table><div class="muted">${esc((analysis.limitations || []).join(" · "))}</div>`;
  const analysisId = analysis.analysis_id || analysis.id;
  renderReconstructionState();
  api(`/v1/analyses/${analysisId}/functions`).then((functions) => {
    state.functions = functions;
    $("#function-list").innerHTML = functions.length ? `<table><thead><tr><th>Name</th><th>Entry</th><th>Confidence</th><th>Instructions</th><th></th></tr></thead><tbody>${functions.map((fn) => `<tr><td><code>${esc(fn.names?.[0]?.value || fn.name || fn.function_id)}</code></td><td><code>${esc(fn.entry_va)}</code></td><td>${Number(fn.confidence || 0).toFixed(2)}</td><td>${fn.blocks?.reduce((sum, block) => sum + (block.instruction_ids?.length || 0), 0) || ""}</td><td><button data-function="${esc(fn.function_id)}">Inspect</button></td></tr>`).join("")}</tbody></table>` : `<div class="empty">No functions recovered.</div>`;
    document.querySelectorAll("[data-function]").forEach((button) => button.addEventListener("click", () => selectFunction(button.dataset.function)));
  }).catch((error) => { $("#function-list").innerHTML = `<div class="empty">${esc(error.message)}</div>`; });
  api(`/v1/analyses/${analysisId}/security`).then(renderSecurity).catch(() => {});
  api(`/v1/analyses/${analysisId}/callgraph`).then(renderCallgraph).catch(() => {});
}

async function selectFunction(functionId) {
  const analysis = state.analysis;
  const analysisId = analysis.analysis_id || analysis.id;
  try {
    const fn = await api(`/v1/analyses/${analysisId}/functions/${functionId}`);
    state.selectedFunction = fn;
    const names = (fn.names || []).map((item) => item.value).join(", ");
    $("#function-detail").classList.remove("empty");
    $("#function-detail").innerHTML = `<table><tbody>
      <tr><th>ID</th><td><code>${esc(fn.function_id)}</code></td></tr>
      <tr><th>Names</th><td>${esc(names || "unnamed")}</td></tr>
      <tr><th>Entry</th><td><code>${esc(fn.entry_va)}</code></td></tr>
      <tr><th>Ranges</th><td>${(fn.ranges || []).map((range) => `<code>${esc(range.start_va)}-${esc(range.end_va)}</code>`).join("<br>")}</td></tr>
      <tr><th>Confidence</th><td>${Number(fn.confidence || 0).toFixed(2)}</td></tr>
      <tr><th>Purpose</th><td>${esc(fn.semantic?.purpose || "unknown")} (${Number(fn.semantic?.confidence || 0).toFixed(2)})</td></tr>
      <tr><th>Unknowns</th><td>${(fn.unknowns || []).map((item) => esc(item.question)).join("<br>") || "none"}</td></tr>
    </tbody></table>`;
    const cfg = await api(`/v1/analyses/${analysisId}/cfg?function_id=${encodeURIComponent(functionId)}`);
    renderCfg(cfg);
    const dataflow = await api(`/v1/analyses/${analysisId}/dataflow?function_id=${encodeURIComponent(functionId)}`);
    $("#dataflow").innerHTML = `<table><thead><tr><th>Kind</th><th>Count</th></tr></thead><tbody>
      <tr><td>Def-use chains</td><td>${(dataflow.def_use_chains || []).length}</td></tr>
      <tr><td>Memory accesses</td><td>${(dataflow.memory_accesses || []).length}</td></tr>
      <tr><td>Stack slots</td><td>${(dataflow.stack_slots || []).length}</td></tr>
      <tr><td>Globals</td><td>${(dataflow.globals || []).length}</td></tr>
      <tr><td>Argument candidates</td><td>${(dataflow.argument_candidates || []).length}</td></tr>
    </tbody></table><div class="muted">${esc((dataflow.limitations || []).join(" · "))}</div>`;
    const instructions = await api(`/v1/analyses/${analysisId}/disassembly?limit=10000`);
    const ranges = fn.ranges || [];
    const inside = instructions.filter((instruction) => ranges.some((range) => Number.parseInt(instruction.va, 16) >= Number.parseInt(range.start_va, 16) && Number.parseInt(instruction.va, 16) < Number.parseInt(range.end_va, 16)));
    $("#disassembly").innerHTML = inside.map((instruction) => `<div><span class="muted">${esc(instruction.va)}</span> <code>${esc(instruction.bytes)}</code> ${esc(instruction.mnemonic)} ${esc((instruction.operands || []).map((operand) => operand.value).join(", "))}</div>`).join("") || `<div class="empty">No instructions available.</div>`;
  } catch (error) {
    $("#function-detail").textContent = error.message;
  }
}

function graphPositions(nodes, width, height) {
  const positions = new Map();
  const count = Math.max(1, nodes.length);
  const columns = Math.ceil(Math.sqrt(count));
  nodes.forEach((node, index) => {
    positions.set(node.node_id, {
      x: 40 + (index % columns) * ((width - 80) / Math.max(1, columns - 1 || 1)),
      y: 40 + Math.floor(index / columns) * ((height - 80) / Math.max(1, Math.ceil(count / columns) - 1 || 1)),
    });
  });
  return positions;
}

function renderCfg(cfg) {
  const nodes = (cfg.nodes || []).slice(0, 40);
  const edges = (cfg.edges || []).filter((edge) => nodes.some((node) => node.node_id === edge.source) && nodes.some((node) => node.node_id === edge.target)).slice(0, 80);
  const width = 900, height = 300;
  const positions = graphPositions(nodes, width, height);
  const svg = [`<svg class="cfg-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">`];
  for (const edge of edges) {
    const source = positions.get(edge.source), target = positions.get(edge.target);
    if (!source || !target) continue;
    svg.push(`<line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="#58a6ff" stroke-opacity="0.55" marker-end="url(#arrow)" />`);
  }
  svg.push(`<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#58a6ff"/></marker></defs>`);
  for (const node of nodes) {
    const position = positions.get(node.node_id);
    svg.push(`<circle cx="${position.x}" cy="${position.y}" r="16" fill="#1c2330" stroke="#30363d" />`);
    svg.push(`<text x="${position.x}" y="${position.y + 4}" text-anchor="middle" class="node-label">${esc((node.node_id || "").slice(-8))}</text>`);
  }
  svg.push("</svg>");
  $("#cfg-view").innerHTML = svg.join("") + `<div class="muted">${esc((cfg.limitations || []).join(" · "))}</div>`;
}

function renderCallgraph(callgraph) {
  const nodes = (callgraph.nodes || []).slice(0, 60);
  const edges = (callgraph.edges || []).slice(0, 120);
  $("#callgraph").innerHTML = `<div class="muted">${nodes.length} nodes · ${edges.length} edges</div><table><thead><tr><th>Caller</th><th>Target</th><th>Kind</th><th>Confidence</th></tr></thead><tbody>${edges.map((edge) => `<tr><td><code>${esc((nodes.find((node) => node.node_id === edge.source) || {}).name || edge.source)}</code></td><td><code>${esc((nodes.find((node) => node.node_id === edge.target) || {}).name || edge.target)}</code></td><td>${esc(edge.kind)}</td><td>${Number(edge.confidence || 0).toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
}

function renderSecurity(findings) {
  if (!findings || !findings.length) {
    $("#security-empty").classList.remove("hidden");
    $("#security-list").innerHTML = "";
    return;
  }
  $("#security-empty").classList.add("hidden");
  $("#security-list").innerHTML = findings.map((finding) => `<div class="finding ${esc(finding.severity)}"><strong>${esc(finding.title)}</strong> <span class="badge">${esc(finding.severity)}</span><div>${esc(finding.description)}</div><div class="muted">${esc(finding.function_name || "")} ${esc(finding.address || "")} · confidence ${Number(finding.confidence || 0).toFixed(2)} · ${esc(finding.status)}</div><div class="muted">${esc(finding.remediation || "")}</div></div>`).join("");
}

async function loadDatasets() {
  try {
    const datasets = await api("/v1/datasets");
    $("#dataset-list").className = datasets.length ? "card" : "card empty";
    $("#dataset-list").innerHTML = datasets.length ? `<table><thead><tr><th>ID</th><th>Path</th><th>Samples</th><th>Created</th></tr></thead><tbody>${datasets.map((item) => `<tr><td><code>${esc(item.id)}</code></td><td><code>${esc(item.path)}</code></td><td>${esc(item.summary?.samples ?? "")}</td><td>${esc(item.created_at)}</td></tr>`).join("")}</tbody></table>` : "No datasets registered.";
  } catch (error) {
    $("#dataset-list").textContent = error.message;
  }
}

async function loadModel() {
  try {
    const info = await api("/v1/model");
    $("#model-status").innerHTML = `<p>${esc(info.status)}</p><p class="muted">${esc(info.policy)}</p>${info.models?.length ? `<table><thead><tr><th>ID</th><th>SHA-256</th><th>Created</th></tr></thead><tbody>${info.models.map((item) => `<tr><td><code>${esc(item.id)}</code></td><td><code>${esc(item.sha256.slice(0, 20))}…</code></td><td>${esc(item.created_at)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No model checkpoint registered.</div>`}`;
    const experiments = await api("/v1/experiments");
    $("#experiment-list").innerHTML = experiments.length ? `<table><thead><tr><th>ID</th><th>Created</th><th>Status</th></tr></thead><tbody>${experiments.map((item) => `<tr><td><code>${esc(item.id)}</code></td><td>${esc(item.created_at)}</td><td>${esc(item.summary?.status || "")}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No experiments registered.</div>`;
  } catch (error) {
    $("#model-status").textContent = error.message;
  }
}



async function loadResidual() {
  try {
    const status = await api("/v1/residual/status");
    $("#residual-status").innerHTML = `<table><tbody>
      <tr><th>Checkpoint</th><td><code>${esc(status.checkpoint)}</code></td></tr>
      <tr><th>Architecture</th><td>${esc(status.architecture_version)} / ${esc(status.profile)}</td></tr>
      <tr><th>Parameters</th><td>${esc(status.parameters?.total)}</td></tr>
      <tr><th>Steps</th><td>${esc(status.metadata?.steps)}</td></tr>
      <tr><th>Train records</th><td>${esc(status.metadata?.train_records)}</td></tr>
      </tbody></table>`;
    if (status.evaluation?.metrics) renderResidualMetrics(status.evaluation);
  } catch (error) {
    $("#residual-status").textContent = error.message;
  }
}

function renderResidualMetrics(report) {
  const metrics = report.metrics || {};
  $("#residual-metrics").innerHTML = `<p>${esc(report.samples)} held-out samples</p><table><tbody>${Object.entries(metrics).map(([key, value]) => `<tr><th>${esc(key)}</th><td>${value == null ? "n/a" : Number(value).toFixed(6)}</td></tr>`).join("")}</tbody></table>`;
}

async function evaluateResidual() {
  try {
    setStatus("#residual-evaluate-status", "evaluating…", "muted");
    const report = await api("/v1/residual/evaluate", { method: "POST", body: "{}" });
    renderResidualMetrics(report);
    setStatus("#residual-evaluate-status", "done", "muted");
  } catch (error) {
    setStatus("#residual-evaluate-status", error.message, "muted");
  }
}

async function runResidual() {
  const toolText = $("#residual-tool-text").value;
  if (!toolText.trim()) { setStatus("#residual-run-status", "enter tool output first", "muted"); return; }
  try {
    setStatus("#residual-run-status", "running…", "muted");
    const result = await api("/v1/residual/infer", {
      method: "POST",
      body: JSON.stringify({
        tool_text: toolText,
        target_text: $("#residual-target-text").value,
        language: $("#residual-language").value,
        tool: $("#residual-tool").value,
        jni_method: $("#residual-jni-method").value,
        jni_descriptor: $("#residual-jni-descriptor").value,
        tool_status: "completed",
        compile_status: "unknown",
        warning_count: 0,
        cached: false,
        max_new_tokens: 128,
      }),
    });
    const score = (value) => value == null ? "n/a" : Number(value).toFixed(6);
    $("#residual-scores").innerHTML = `<table><tbody>
      <tr><th>Candidate exact</th><td>${result.candidate_exact == null ? "n/a" : String(result.candidate_exact)}</td></tr>
      <tr><th>Candidate score (experimental)</th><td>${score(result.candidate_score)}</td></tr>
      <tr><th>Failure probability (experimental)</th><td>${score(result.failure_probability)}</td></tr>
      <tr><th>Confidence (experimental)</th><td>${score(result.confidence)}</td></tr>
      <tr><th>Teacher-forced token accuracy</th><td>${score(result.teacher_forced_token_accuracy)}</td></tr>
      <tr><th>Teacher-forced exact</th><td>${result.teacher_forced_exact == null ? "n/a" : String(result.teacher_forced_exact)}</td></tr>
      </tbody></table>`;
    $("#residual-generated").textContent = result.canonical_text || result.generated_text || "";
    setStatus("#residual-run-status", "done", "muted");
  } catch (error) {
    setStatus("#residual-run-status", error.message, "muted");
  }
}

async function loadFoundation() {
  try {
    const info = await api("/v1/foundation/model");
    const latest = info.latest_run || {};
    $("#foundation-status").innerHTML = `<table><tbody>
      <tr><th>Status</th><td>${esc(info.status)}</td></tr>
      <tr><th>Architecture</th><td><code>${esc(info.architecture_version)}</code></td></tr>
      <tr><th>Latest run</th><td>${latest.run_name ? `<code>${esc(latest.run_name)}</code> · ${esc(latest.status)}` : "none recorded"}</td></tr>
      <tr><th>Parameter count</th><td>${latest.parameter_count ? Number(latest.parameter_count).toLocaleString() : "not available"}</td></tr>
      <tr><th>Foundation model trained at scale</th><td>${info.honesty.foundation_model_trained_at_scale ? "claimed" : "not claimed"}</td></tr>
      <tr><th>Pretrained weights</th><td>${info.honesty.pretrained_weights_loaded ? "loaded" : "none (random initialization)"}</td></tr>
    </tbody></table>`;
    const profile = info.profiles.profiles[info.profiles.profiles.compact ? "compact" : Object.keys(info.profiles.profiles)[0]];
    $("#foundation-architecture").innerHTML = `<table><tbody>
      <tr><th>Model family</th><td>one unified checkpoint; modality encoders + graph trunk + hierarchical aggregation + task heads + reconstruction decoder</td></tr>
      <tr><th>Profiles</th><td>${Object.keys(info.profiles.profiles).map((name) => `<code>${esc(name)}</code>`).join(" · ")}</td></tr>
      <tr><th>CPU profile params</th><td>${Number(profile.estimated_parameters).toLocaleString()} (estimate)</td></tr>
      <tr><th>d_model / heads</th><td>${profile.config.d_model} / ${profile.config.n_heads}</td></tr>
      <tr><th>Instruction / decoder layers</th><td>${profile.config.n_instruction_layers} / ${profile.config.n_decoder_layers}</td></tr>
      <tr><th>Context limits</th><td>${profile.config.max_instructions} instructions · ${profile.config.max_blocks} blocks · ${profile.config.max_bytes} bytes</td></tr>
    </tbody></table>`;
    $("#foundation-modalities").innerHTML = `<table><tbody>${Object.entries(info.modalities).map(([name, value]) => `<tr><th><code>${esc(name)}</code></th><td>${esc(value)}</td></tr>`).join("")}</tbody></table>`;
    const runs = info.training_runs || [];
    $("#foundation-runs").innerHTML = runs.length ? `<table><thead><tr><th>Run</th><th>Status</th><th>Step</th><th>Parameters</th><th>Validation</th><th>Resources</th></tr></thead><tbody>${runs.map((run) => `<tr><td><code>${esc(run.run_name)}</code></td><td>${esc(run.status)}</td><td>${esc((run.progress || {}).step ?? "")}/${esc((run.progress || {}).max_steps ?? "")}</td><td>${run.parameter_count ? Number(run.parameter_count).toLocaleString() : ""}</td><td>${esc((run.validation || {}).weighted_validation_score ?? "")}</td><td>${esc((run.resources || {}).peak_rss_mb ?? "")} MB peak · ${esc((run.resources || {}).wall_seconds ?? "")} s</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No foundation training runs recorded.</div>`;
    const checkpoints = info.checkpoints || [];
    $("#foundation-checkpoints").innerHTML = checkpoints.length ? `<table><thead><tr><th>Path</th><th>SHA-256</th><th>Step</th><th>Status</th><th>Initialization</th></tr></thead><tbody>${checkpoints.map((item) => `<tr><td><code>${esc(item.relative_path)}</code></td><td><code>${esc((item.sha256 || "").slice(0, 16))}…</code></td><td>${esc(item.step ?? "")}</td><td>${esc(item.status ?? "")}</td><td>${esc(item.initialization ?? "")}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No foundation checkpoints recorded.</div>`;
    const evaluation = await api("/v1/foundation/evaluation");
    if (evaluation.report) {
      const report = evaluation.report;
      const categories = report.capability_categories || {};
      $("#foundation-evaluation").innerHTML = `<table><thead><tr><th>Category</th><th>Status</th><th>Metrics</th></tr></thead><tbody>${Object.entries(categories).map(([name, value]) => `<tr><td><code>${esc(name)}</code></td><td>${value && value.status ? esc(value.status) : "unavailable"}</td><td>${esc(JSON.stringify((value && (value.metrics || value.note || value.reason)) || {}).slice(0, 240))}</td></tr>`).join("")}</tbody></table><div class="muted">${esc((report.limitations || []).join(" · "))}</div>`;
      const candidates = (report.reconstruction || {}).candidates || [];
      $("#foundation-unresolved").innerHTML = candidates.length ? `<table><thead><tr><th>Method</th><th>Confidence</th><th>Unresolved findings</th><th>Compilation</th></tr></thead><tbody>${candidates.map((candidate) => `<tr><td><code>${esc(candidate.method)}</code></td><td>${candidate.confidence === null || candidate.confidence === undefined ? "not available" : Number(candidate.confidence).toFixed(3)}</td><td>${esc((candidate.unresolved_operations || []).join("; ")) || "none reported"}</td><td>${esc((candidate.statuses || {}).compilation || "unknown")}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No reconstruction candidates in the latest evaluation.</div>`;
    } else {
      $("#foundation-evaluation").innerHTML = `<div class="empty">${esc(evaluation.status || "No evaluation report recorded.")}</div>`;
      $("#foundation-unresolved").innerHTML = `<div class="empty">No evaluation report recorded.</div>`;
    }
  } catch (error) {
    $("#foundation-status").innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

async function runReconstruction(language) {
  if (!state.analysis) return;
  const analysisId = state.analysis.analysis_id || state.analysis.id;
  try {
    setStatus("#reconstruction-status", "generating…", "muted");
    const functionIds = state.selectedFunction ? [state.selectedFunction.function_id] : undefined;
    const result = await api("/v1/reconstructions", { method: "POST", body: JSON.stringify({ analysis_id: analysisId, language, function_ids: functionIds }) });
    state.reconstruction = result;
    $("#reconstruction-empty").classList.add("hidden");
    $("#reconstruction-content").classList.remove("hidden");
    const files = Object.keys(result.files || {});
    $("#source-file-select").innerHTML = files.map((file) => `<option value="${esc(file)}">${esc(file)}</option>`).join("");
    $("#evidence-map").innerHTML = `<h3>Evidence map</h3><pre class="code">${esc(JSON.stringify(result.evidence_map || {}, null, 2))}</pre>`;
    const protection = result.evidence_map?.protection;
    const protectionNote = protection?.runtime_decryption ? " · runtime decryption detected" : "";
    const jniNote = result.classification === "jni_semantic_reconstruction"
      ? " · semantic Java recovered from JNI calls; use Java/APK for JADX source"
      : result.classification === "jni_bridge_reconstruction"
        ? " · recovered JNI facade from RegisterNatives; native bodies are C, not Java"
        : "";
    setStatus("#reconstruction-status", `${result.classification}${result.evidence_map?.backend ? " · " + result.evidence_map.backend : ""}${protectionNote}${jniNote} · ${result.reconstruction_id || "not stored"}`, "muted");
    const selected = (result.functions || []).find((item) => item.function_id === state.selectedFunction?.function_id);
    const preferred = selected?.generated_name ? `src/${selected.generated_name}.c` : files[0];
    await showSource(preferred || files[0]);
  } catch (error) {
    setStatus("#reconstruction-status", error.message, "muted");
  }
}

async function showSource(file) {
  if (!state.reconstruction || !file) return;
  const reconstructionId = state.reconstruction.reconstruction_id;
  if (reconstructionId) {
    try {
      const text = await fetch(`/v1/reconstructions/${reconstructionId}/files/${file}`, { headers: state.apiKey ? { "X-API-Key": state.apiKey } : {} }).then(async (response) => { if (!response.ok) throw new Error(await response.text()); return response.text(); });
      $("#source-view").textContent = text;
      return;
    } catch (error) {
      $("#source-view").textContent = error.message;
      return;
    }
  }
  $("#source-view").textContent = state.reconstruction.files[file] || "";
}

async function verifyReconstruction() {
  if (!state.reconstruction || !state.reconstruction.reconstruction_id) return;
  try {
    $("#verification-result").textContent = "compiling…";
    const result = await api(`/v1/reconstructions/${state.reconstruction.reconstruction_id}/verify`, { method: "POST" });
    $("#verification-result").innerHTML = `<pre class="code">${esc(JSON.stringify(result, null, 2))}</pre>`;
  } catch (error) {
    $("#verification-result").textContent = error.message;
  }
}

async function runSmoke() {
  try {
    $("#smoke-result").textContent = "running CPU smoke test…";
    const result = await api("/v1/model/smoke-test?steps=2", { method: "POST" });
    $("#smoke-result").textContent = JSON.stringify(result, null, 2);
    loadModel();
  } catch (error) {
    $("#smoke-result").textContent = error.message;
  }
}

document.querySelectorAll(".tabs button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("#binary-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    setStatus("#upload-status", "uploading…", "muted");
    const result = await api("/v1/binaries", { method: "POST", body: form });
    setStatus("#upload-status", `uploaded ${result.id}`, "muted");
    await loadBinaries();
    await analyzeBinary(result.id);
  } catch (error) {
    setStatus("#upload-status", error.message, "muted");
  }
});
$("#api-key").value = state.apiKey;
$("#api-key").addEventListener("change", (event) => { state.apiKey = event.target.value.trim(); localStorage.setItem("nexus_api_key", state.apiKey); });
$("#reconstruct-c").addEventListener("click", () => runReconstruction("c"));
$("#reconstruct-java").addEventListener("click", () => runReconstruction("java"));
$("#reconstruct-model-c").addEventListener("click", runModelReconstruction);
$("#java-upload-form").addEventListener("submit", (event) => { event.preventDefault(); uploadJavaArtifact(); });
$("#java-reconstruct").addEventListener("click", reconstructJavaArtifact);
$("#java-file-select").addEventListener("change", (event) => showJavaSource(event.target.value));
$("#residual-evaluate").addEventListener("click", evaluateResidual);
$("#residual-run").addEventListener("click", runResidual);
$("#run-model-analysis").addEventListener("click", runModelAnalysis);
$("#model-checkpoint").addEventListener("change", (event) => { state.checkpoint = event.target.value; });
$("#source-file-select").addEventListener("change", (event) => showSource(event.target.value));
$("#verify-reconstruction").addEventListener("click", verifyReconstruction);
$("#run-smoke").addEventListener("click", runSmoke);

loadHealth();
loadOverview();
loadBinaries();
loadCheckpoints();
restoreLastAnalysis();
