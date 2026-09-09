import { useEffect, useRef, useState } from "react";
import {
  downloadTestCases,
  downloadTestScript,
  fetchHealth,
  generateTestCases,
  generateTestScript,
  uploadRequirement,
} from "./api.js";
import DatasheetTable from "./components/DatasheetTable.jsx";
import RetrievedContext from "./components/RetrievedContext.jsx";
import ScriptPane from "./components/ScriptPane.jsx";

// No progress bar for generation: a 7B model on CPU takes long enough that a
// fake progress animation would be a lie. An elapsed-second counter is
// honest and tells the user the request is still alive.
function useElapsed(running) {
  const [seconds, setSeconds] = useState(0);
  const startRef = useRef(0);

  useEffect(() => {
    if (!running) return undefined;
    startRef.current = Date.now();
    setSeconds(0);
    const timer = setInterval(
      () => setSeconds(Math.round((Date.now() - startRef.current) / 1000)),
      1000
    );
    return () => clearInterval(timer);
  }, [running]);

  return seconds;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [requirementText, setRequirementText] = useState("");
  const [uploadedName, setUploadedName] = useState("");

  const [result, setResult] = useState(null);
  const [script, setScript] = useState(null);

  const [busy, setBusy] = useState(null); // "cases" | "script" | "export" | null
  const [error, setError] = useState("");

  const elapsed = useElapsed(busy === "cases" || busy === "script");

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function run(kind, action) {
    setBusy(kind);
    setError("");
    try {
      return await action();
    } catch (exception) {
      setError(exception.message);
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const uploaded = await run("upload", () => uploadRequirement(file));
    if (uploaded) {
      setRequirementText(uploaded.requirement_text);
      setUploadedName(uploaded.filename);
    }
    // Clear the input so re-selecting the same file fires change again.
    event.target.value = "";
  }

  async function onGenerateTestCases({ refresh = false } = {}) {
    // A new set of test cases invalidates any script written from the old
    // ones, so it's cleared rather than left to be downloaded by mistake.
    setScript(null);
    const generated = await run("cases", () =>
      generateTestCases({ requirementText, refresh })
    );
    if (generated) setResult(generated);
  }

  async function onGenerateScript() {
    const generated = await run("script", () =>
      generateTestScript({ requirementText, testCases: result.test_cases })
    );
    if (generated) setScript(generated);
  }

  const hasTestCases = Boolean(result?.test_cases?.length);
  const generating = busy === "cases" || busy === "script";
  const canGenerate = requirementText.trim().length > 0 && !busy;

  return (
    <div className="app">
      <header className="masthead">
        <h1>Test Case Generator</h1>
        <span className="subtitle">I-ETMS Protect · onboard segment</span>
        <div className="status">
          <span>
            knowledge base <b>{health ? health.knowledge_base_chunks.toLocaleString() : "—"}</b>{" "}
            chunks
          </span>
          <span>
            <span
              className={`status-dot ${health?.llm_available ? "ready" : health ? "down" : ""}`}
            />
            <b>{health?.llm_model ?? "model unknown"}</b>
          </span>
        </div>
      </header>

      <div className="workbench">
        <aside className="rail">
          <div className="field">
            <label htmlFor="requirement">Requirement</label>
            <p className="help">
              Paste the requirement text, including its ID, or load it from a text file.
            </p>
            <textarea
              id="requirement"
              value={requirementText}
              onChange={(event) => setRequirementText(event.target.value)}
              placeholder={"15 Speed Enforcement\n\nL2R9479\n\nThe onboard shall …"}
              spellCheck="false"
            />
          </div>

          <label className="file-drop">
            <input type="file" accept=".txt,.md" onChange={onUpload} />
            {uploadedName ? `Loaded ${uploadedName} — replace` : "Load from .txt file"}
          </label>

          <button
            className="btn btn-primary btn-block"
            onClick={() => onGenerateTestCases()}
            disabled={!canGenerate}
          >
            {busy === "cases" ? `Generating… ${elapsed}s` : "Generate test cases"}
          </button>

          <p className="help">
            How many test cases you get is derived from the requirement — one per
            verifiable behaviour it states — not from a number you pick.
          </p>

          {error && (
            <div className="notice error">
              <p>{error}</p>
            </div>
          )}

          {generating && !error && (
            <div className="notice working">
              <p>
                {busy === "cases"
                  ? "Retrieving context and generating."
                  : "Writing the automation script."}{" "}
                <span className="elapsed">{elapsed}s elapsed</span>
              </p>
            </div>
          )}

          {health && health.knowledge_base_chunks === 0 && (
            <div className="notice">
              <p>
                The knowledge base is empty. Run{" "}
                <code>python scripts/run_ingestion.py</code> before generating.
              </p>
            </div>
          )}
        </aside>

        <main className="results">
          {hasTestCases ? (
            <>
              <div className="results-head">
                <h2>{result.requirement_id || "Requirement"} — Datasheet</h2>
                <span className="meta">
                  {result.test_cases.length} cases · mean confidence{" "}
                  {Math.round(result.mean_confidence * 100)}% ·{" "}
                  {result.cached
                    ? `reused an earlier identical run (${result.elapsed_seconds}s)`
                    : `${result.elapsed_seconds}s`}
                </span>
                {result.cached && (
                  <button
                    className="btn btn-small"
                    disabled={!canGenerate}
                    onClick={() => onGenerateTestCases({ refresh: true })}
                  >
                    Regenerate
                  </button>
                )}
              </div>
              <DatasheetTable testCases={result.test_cases} />
              <RetrievedContext
                chunks={result.retrieved}
                subdivisions={result.track_subdivisions}
              />
              <ScriptPane script={script?.script} elapsedSeconds={script?.elapsed_seconds} />
            </>
          ) : (
            <div className="empty">
              <h3>No test cases yet</h3>
              <p>
                Paste a requirement on the left and generate. Retrieved context and
                per-case confidence appear here alongside the datasheet.
              </p>
            </div>
          )}
        </main>
      </div>

      <footer className="actions">
        <button
          className="btn"
          disabled={!hasTestCases || busy === "export"}
          onClick={() =>
            run("export", () =>
              downloadTestCases({
                testCases: result.test_cases,
                requirementId: result.requirement_id,
              })
            )
          }
        >
          Download test cases (.xlsx)
        </button>

        <button
          className="btn btn-primary"
          disabled={!hasTestCases || generating}
          onClick={onGenerateScript}
        >
          {busy === "script" ? `Generating script… ${elapsed}s` : "Generate test script"}
        </button>

        <button
          className="btn"
          disabled={!script?.script || busy === "export"}
          onClick={() =>
            run("export", () =>
              downloadTestScript({
                script: script.script,
                requirementId: script.requirement_id,
              })
            )
          }
        >
          Download test script (.txt)
        </button>

        <span className="hint spacer">
          {hasTestCases
            ? "Rows flagged for review scored below the confidence threshold."
            : "Downloads unlock once test cases are generated."}
        </span>
      </footer>
    </div>
  );
}
