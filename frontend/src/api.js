// Every call to the FastAPI backend lives here, so no component has to know
// about URLs, status codes or blob downloads.

const BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, options);
  if (response.ok) return response;

  // FastAPI puts the useful message in `detail`. Surfacing it verbatim
  // matters here: the messages name the actual fix ("run ollama pull ...",
  // "ask for fewer test cases"), which a generic "request failed" would hide.
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (body?.detail) detail = body.detail;
  } catch {
    // Non-JSON error body; the status line is all we have.
  }
  throw new Error(detail);
}

async function json(path, body) {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

export async function fetchHealth() {
  const response = await request("/health");
  return response.json();
}

// No test-case count is sent: how many cases a requirement needs is derived
// from the requirement itself, server-side. `refresh` asks the server to run
// the model again instead of replaying an identical earlier result.
export function generateTestCases({ requirementText, topK, refresh = false }) {
  return json("/test-cases", {
    requirement_text: requirementText,
    top_k: topK,
    refresh,
  });
}

export function generateTestScript({ requirementText, testCases }) {
  return json("/test-script", {
    requirement_text: requirementText,
    test_cases: testCases,
  });
}

export async function uploadRequirement(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await request("/requirements/upload", {
    method: "POST",
    body: form,
  });
  return response.json();
}

export function downloadTestCases({ testCases, requirementId }) {
  return download("/test-cases/export", {
    test_cases: testCases,
    requirement_id: requirementId,
  });
}

export function downloadTestScript({ script, requirementId }) {
  return download("/test-script/export", {
    script,
    requirement_id: requirementId,
  });
}

async function download(path, body) {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const blob = await response.blob();
  // The server names the file (requirement ID + timestamp); read it back off
  // the header rather than rebuilding it here, so the two can't drift.
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : "download";

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoked on the next tick, not immediately: some browsers read the object
  // URL asynchronously after the click, and tearing it down in the same task
  // makes the download silently produce nothing.
  setTimeout(() => URL.revokeObjectURL(url), 0);
  return filename;
}
