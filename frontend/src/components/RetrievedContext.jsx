// Which knowledge-base chunks the model actually saw. Collapsed by default:
// it's the first thing to open when a test case looks wrong, and noise the
// rest of the time.

export default function RetrievedContext({ chunks, subdivisions }) {
  if (!chunks.length) return null;

  return (
    <details className="context">
      <summary>
        Retrieved context — {chunks.length} chunk{chunks.length === 1 ? "" : "s"}
        {subdivisions.length > 0 && ` · track ${subdivisions.join(", ")}`}
      </summary>
      <ul className="context-list">
        {chunks.map((chunk) => (
          <li key={chunk.chunk_id}>
            <div className="context-source">
              <span className="tag">{chunk.document_type}</span>
              <span className="path">{chunk.source}</span>
              <span className="scores">
                {chunk.similarity !== null && `cos ${chunk.similarity.toFixed(2)}`}
                {chunk.similarity !== null && chunk.bm25_score !== null && " · "}
                {chunk.bm25_score !== null && `bm25 ${chunk.bm25_score.toFixed(1)}`}
                {` · ${chunk.matched_by.join(" + ")}`}
              </span>
            </div>
            <p className="excerpt">{chunk.excerpt}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}
