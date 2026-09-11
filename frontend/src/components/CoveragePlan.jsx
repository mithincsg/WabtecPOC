// The behaviour list the plan call agreed before any row was written.
//
// Coverage is judged against behaviours, not against row count — a run that
// produced fifteen rows covering seven behaviours is worse than nine rows
// covering nine — so the list the rows were written from is worth being able
// to open. Collapsed by default, like the retrieved context.

export default function CoveragePlan({ behaviours, rowCount }) {
  if (!behaviours || behaviours.length === 0) return null;

  return (
    <details className="context">
      <summary>
        Behaviour plan — {behaviours.length} behaviour
        {behaviours.length === 1 ? "" : "s"} · {rowCount} row
        {rowCount === 1 ? "" : "s"} written
      </summary>
      <ol className="context-list">
        {behaviours.map((behaviour, index) => (
          <li key={index}>
            <p className="excerpt">{behaviour}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}
