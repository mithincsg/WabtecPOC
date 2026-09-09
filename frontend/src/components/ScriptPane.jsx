// The generated automation script, shown before it's downloaded so the
// reviewer can see the branch-per-test-case structure lines up with the
// datasheet above it.

export default function ScriptPane({ script, elapsedSeconds }) {
  if (!script) return null;

  const lines = script.split("\n").length;
  return (
    <section className="script-pane">
      <header>
        <span>Test script draft</span>
        <span className="meta">
          {lines} lines
          {elapsedSeconds ? ` · ${elapsedSeconds}s` : ""}
        </span>
      </header>
      <pre>{script}</pre>
    </section>
  );
}
