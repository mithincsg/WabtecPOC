import { useEffect, useRef, useState } from "react";
import {
  downloadTestCases,
  downloadTestScript,
  fetchHealth,
  fetchSubdivisions,
  generateTestCases,
  generateTestScript,
  lookupFolder,
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

// The feature a requirement belongs to comes from the Change Approval Form
// workbook, so it can be shown as soon as the Requirement No is typed — long
// before a generation would reveal it, and it is the same value the
// generated rows and the exported spreadsheet carry in their Folder column.
// Keyed on the dedicated Requirement No field, not the pasted requirement
// text, so the folder reflects exactly the ID the user entered rather than
// whatever parsing the free-text box happens to find. Debounced because the
// lookup runs on every keystroke, and aborted on the next one so a slow
// answer can't overwrite a newer requirement's folder.
function useFolderLookup(requirementNumber) {
  const [folder, setFolder] = useState(null);

  useEffect(() => {
    if (!requirementNumber.trim()) {
      setFolder(null);
      return undefined;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      lookupFolder({ requirementText: requirementNumber, signal: controller.signal })
        .then(setFolder)
        // A failed lookup is not worth an error banner: the folder is a
        // preview of something generation resolves again anyway.
        .catch(() => {});
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [requirementNumber]);

  return folder;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [requirementNumber, setRequirementNumber] = useState("");
  const [requirementText, setRequirementText] = useState("");
  const [uploadedName, setUploadedName] = useState("");
  // Which track_data subdivision to draw block/milepost/signal values from.
  // This picker is the only thing that selects track data — there is no
  // requirement -> subdivision map behind it any more — so a choice is
  // required before generating.
  const [subdivision, setSubdivision] = useState("");
  const [subdivisions, setSubdivisions] = useState([]);
  // Only start showing the Requirement No validation once the user has
  // tried to generate with it empty — flagging it red before they've typed
  // anything would just look broken.
  const [requirementNumberTouched, setRequirementNumberTouched] = useState(false);
  // Same treatment for the subdivision: only flag it once they've tried to
  // generate without one.
  const [subdivisionTouched, setSubdivisionTouched] = useState(false);

  const [result, setResult] = useState(null);
  const [script, setScript] = useState(null);

  const [busy, setBusy] = useState(null); // "cases" | "script" | "export" | null
  const [error, setError] = useState("");

  const elapsed = useElapsed(busy === "cases" || busy === "script");
  const folderLookup = useFolderLookup(requirementNumber);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    // An empty list is a valid answer (no track data ingested yet), so a
    // failure here leaves the picker showing only "Use requirement mapping"
    // rather than blocking generation.
    fetchSubdivisions()
      .then(setSubdivisions)
      .catch(() => setSubdivisions([]));
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
      // Pre-fill Requirement No from whatever the document itself carries,
      // but only if the field is still empty — an upload shouldn't clobber
      // an ID the user already typed in by hand.
      if (uploaded.requirement_id && !requirementNumber.trim()) {
        setRequirementNumber(uploaded.requirement_id);
      }
    }
    // Clear the input so re-selecting the same file fires change again.
    event.target.value = "";
  }

  // Requirement No is a separate, mandatory field so the folder and the
  // track-data mapping are keyed on exactly what the user entered there,
  // not on whatever an ID-detecting regex finds inside pasted free text.
  // The backend still only takes one requirement_text, so it's composed
  // here with the number leading — parse_requirement reads an ID at the
  // very start of the text, and skipping the prepend when it's already
  // there avoids a duplicated line for the common "pasted the ID in
  // already" case.
  function composedRequirementText() {
    const number = requirementNumber.trim();
    const text = requirementText.trim();
    if (!number) return text;
    if (text.toLowerCase().startsWith(number.toLowerCase())) return text;
    return `${number}\n\n${text}`;
  }

  async function onGenerateTestCases({ refresh = false } = {}) {
    if (!requirementNumber.trim() || !subdivision) {
      setRequirementNumberTouched(true);
      setSubdivisionTouched(true);
      return;
    }
    // A new set of test cases invalidates any script written from the old
    // ones, so it's cleared rather than left to be downloaded by mistake.
    setScript(null);
    const generated = await run("cases", () =>
      generateTestCases({
        requirementText: composedRequirementText(),
        subdivision,
        refresh,
      })
    );
    if (generated) setResult(generated);
  }

  async function onGenerateScript() {
    const generated = await run("script", () =>
      generateTestScript({
        requirementText: composedRequirementText(),
        testCases: result.test_cases,
        subdivision,
      })
    );
    if (generated) setScript(generated);
  }

  const hasTestCases = Boolean(result?.test_cases?.length);
  const generating = busy === "cases" || busy === "script";
  const requirementNumberMissing = requirementNumberTouched && !requirementNumber.trim();
  const subdivisionMissing = subdivisionTouched && !subdivision;
  const canGenerate =
    requirementNumber.trim().length > 0 &&
    requirementText.trim().length > 0 &&
    subdivision.length > 0 &&
    !busy;

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
            <label htmlFor="requirement-no">
              Requirement No <span className="required">*</span>
            </label>
            <p className="help">
              The requirement being tested, e.g. L2R9479. Required — it's what the
              folder lookup, track-data mapping and datasheet are keyed on.
            </p>
            <input
              id="requirement-no"
              className={`requirement-no-input ${requirementNumberMissing ? "invalid" : ""}`}
              type="text"
              value={requirementNumber}
              onChange={(event) => {
                setRequirementNumber(event.target.value);
                setRequirementNumberTouched(true);
              }}
              onBlur={() => setRequirementNumberTouched(true)}
              placeholder="L2R9479"
              spellCheck="false"
              required
              aria-required="true"
              aria-invalid={requirementNumberMissing}
            />
            {requirementNumberMissing && (
              <p className="field-error">Requirement No is required.</p>
            )}
          </div>

          {requirementNumber.trim() && (
            <div className={`folder-card ${folderLookup?.folder ? "" : "unmapped"}`}>
              <span className="folder-req">
                {folderLookup?.requirement_id || requirementNumber.trim()}
              </span>
              {folderLookup?.folder ? (
                <span className="folder-note">
                  This requirement's test cases will go into the{" "}
                  <span className="folder-name">{folderLookup.folder}</span> folder.
                </span>
              ) : (
                <span className="folder-note">
                  {folderLookup
                    ? "No feature mapped for this requirement — its test cases will be left without a folder. Add a row to data/CAF.xlsx and re-run scripts/convert_caf_mapping.py to fix it."
                    : "Looking up the folder…"}
                </span>
              )}
            </div>
          )}

          <div className="field">
            <label htmlFor="requirement">Requirement text</label>
            <p className="help">Paste the requirement text, or load it from a text file.</p>
            <textarea
              id="requirement"
              value={requirementText}
              onChange={(event) => setRequirementText(event.target.value)}
              placeholder={"15 Speed Enforcement\n\nThe onboard shall …"}
              spellCheck="false"
            />
          </div>

          <div className="field">
            <label htmlFor="subdivision">Subdivision</label>
            <p className="help">
              Which track under <code>data/track_data/</code> the blocks, mileposts,
              signals and switches should come from. Required — without it the
              model gets no track data and writes TODO placeholders instead of
              values.
            </p>
            <select
              id="subdivision"
              className={subdivisionMissing ? "invalid" : ""}
              value={subdivision}
              onChange={(event) => {
                setSubdivision(event.target.value);
                if (event.target.value) setSubdivisionTouched(false);
              }}
              aria-invalid={subdivisionMissing}
            >
              <option value="">Select a subdivision…</option>
              {subdivisions.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.label || entry.id}
                </option>
              ))}
            </select>
            {subdivision ? (
              <p className="help">
                Track values will come only from{" "}
                <b>{subdivisions.find((e) => e.id === subdivision)?.label || subdivision}</b>.
              </p>
            ) : null}
            {subdivisionMissing ? (
              <p className="field-error">Subdivision is required.</p>
            ) : null}
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
