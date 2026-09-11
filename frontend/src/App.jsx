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
      1000,
    );
    return () => clearInterval(timer);
  }, [running]);

  return seconds;
}

// Strips a leading "<ID> " from uploaded text once the ID has its own field,
// so it isn't shown (and edited) twice. Mirrors the backend's
// requirement_parser._ID_AT_START_RE closely enough for the common case;
// anything it doesn't match is left in the body untouched.
const ID_AT_START_RE = /^([A-Za-z]{1,6}\d[\w.-]*)\b\s*/;

// The subdivisions with track data on disk (data/track_data/<id>/), labelled
// with their display name the way the ingested reports name them. Picking
// one here overrides config/track_mapping.yaml's requirement -> subdivision
// lookup for this generation, rather than replacing it — leaving it unset
// falls back to the mapping file exactly as before.
const SUBDIVISION_OPTIONS = [
  { id: "08101", label: "8101 - Ginger" },
  { id: "08102", label: "8102 - Nutmeg" },
  { id: "08214", label: "8214 - South Morrill" },
  { id: "08236", label: "8236 - Test Track Powder Ri" },
  { id: "08880", label: "8880 - Spokane" },
];

function splitLeadingId(text) {
  const match = ID_AT_START_RE.exec(text.trim());
  if (!match) return { id: "", body: text };
  return { id: match[1], body: text.trim().slice(match[0].length) };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [requirementId, setRequirementId] = useState("");
  const [requirementText, setRequirementText] = useState("");
  const [subdivisionId, setSubdivisionId] = useState("");
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
      if (uploaded.requirement_id) {
        // Backend already parsed the ID; drop the matching lead-in from the
        // body so it isn't duplicated between the two fields.
        setRequirementId(uploaded.requirement_id);
        setRequirementText(splitLeadingId(uploaded.requirement_text).body);
      } else {
        setRequirementText(uploaded.requirement_text);
      }
      setUploadedName(uploaded.filename);
    }
    // Clear the input so re-selecting the same file fires change again.
    event.target.value = "";
  }

  // The ID has its own required field now, but the backend still identifies
  // a requirement by finding an ID at the start of the pasted text (see
  // requirement_parser.py) — so the two fields are stitched back together
  // here rather than adding a parallel requirement_id parameter server-side.
  const combinedRequirementText = `${requirementId.trim()} ${requirementText.trim()}`;

  async function onGenerateTestCases({ refresh = false } = {}) {
    // A new set of test cases invalidates any script written from the old
    // ones, so it's cleared rather than left to be downloaded by mistake.
    setScript(null);
    const generated = await run("cases", () =>
      generateTestCases({
        requirementText: combinedRequirementText,
        subdivisionId,
        refresh,
      }),
    );
    if (generated) setResult(generated);
  }

  async function onGenerateScript() {
    const generated = await run("script", () =>
      generateTestScript({
        requirementText: combinedRequirementText,
        testCases: result.test_cases,
        subdivisionId,
      }),
    );
    if (generated) setScript(generated);
  }

  const hasTestCases = Boolean(result?.test_cases?.length);
  const generating = busy === "cases" || busy === "script";
  const canGenerate =
    requirementId.trim().length > 0 &&
    requirementText.trim().length > 0 &&
    !busy;

  return (
    <div className="app">
      <header className="masthead">
        <h1>Test Case Generator</h1>
        <span className="subtitle">I-ETMS Protect · onboard segment</span>
        <div className="status">
          <span>
            knowledge base{" "}
            <b>
              {health ? health.knowledge_base_chunks.toLocaleString() : "—"}
            </b>{" "}
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
            <label htmlFor="requirement-id">
              Requirement ID <span className="required">*</span>
            </label>
            <p className="help">
              Required — identifies this requirement everywhere downstream.
            </p>
            <input
              id="requirement-id"
              type="text"
              value={requirementId}
              onChange={(event) => setRequirementId(event.target.value)}
              placeholder="L2R9479"
              required
              spellCheck="false"
            />
          </div>

          <div className="field">
            <label htmlFor="subdivision-id">Subdivision</label>
            <p className="help">select the subdivision</p>
            <select
              id="subdivision-id"
              value={subdivisionId}
              onChange={(event) => setSubdivisionId(event.target.value)}
            >
              <option value="">Select Subdivision</option>
              {SUBDIVISION_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="requirement">Requirement</label>
            <p className="help">
              Paste the requirement text, or load it from a text file.
            </p>
            <textarea
              id="requirement"
              value={requirementText}
              onChange={(event) => setRequirementText(event.target.value)}
              placeholder={"15 Speed Enforcement\n\nThe onboard shall …"}
              spellCheck="false"
            />
          </div>

          <label className="file-drop">
            <input type="file" accept=".txt,.md" onChange={onUpload} />
            {uploadedName
              ? `Loaded ${uploadedName} — replace`
              : "Load from .txt file"}
          </label>

          <button
            className="btn btn-primary btn-block"
            onClick={() => onGenerateTestCases()}
            disabled={!canGenerate}
          >
            {busy === "cases"
              ? `Generating… ${elapsed}s`
              : "Generate test cases"}
          </button>

          <p className="help">
            How many test cases you get is derived from the requirement — one
            per verifiable behaviour it states — not from a number you pick.
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
              <ScriptPane
                script={script?.script}
                elapsedSeconds={script?.elapsed_seconds}
              />
            </>
          ) : (
            <div className="empty">
              <h3>No test cases yet</h3>
              <p>
                Paste a requirement on the left and generate. Retrieved context
                and per-case confidence appear here alongside the datasheet.
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
              }),
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
          {busy === "script"
            ? `Generating script… ${elapsed}s`
            : "Generate test script"}
        </button>

        <button
          className="btn"
          disabled={!script?.script || busy === "export"}
          onClick={() =>
            run("export", () =>
              downloadTestScript({
                script: script.script,
                requirementId: script.requirement_id,
              }),
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
