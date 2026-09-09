// The confidence column. Shows the overall score, the three components that
// produced it, and which existing test case the row was compared against —
// a bare number would tell a reviewer nothing about whether to trust it.

const ASPECTS = [
  { min: 0.75, key: "clear", label: "clear" },
  { min: 0.5, key: "approach", label: "approach" },
  { min: 0, key: "stop", label: "stop" },
];

export function aspectFor(score) {
  return ASPECTS.find((a) => score >= a.min) ?? ASPECTS[ASPECTS.length - 1];
}

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export default function ConfidenceCell({ confidence }) {
  const aspect = aspectFor(confidence.overall);
  const compared = confidence.closest_existing_id || confidence.closest_existing_source;

  return (
    <td className="confidence">
      {/* The dot is redundant with the number for sighted users and invisible
          to assistive tech, so the confidence level is stated in the cell's
          accessible name rather than carried by colour alone. */}
      <div className="score" aria-label={`confidence ${percent(confidence.overall)}, ${aspect.label}`}>
        <span className={`aspect aspect-${aspect.key}`} aria-hidden="true" />
        <span>{percent(confidence.overall)}</span>
      </div>

      <ul className="score-parts">
        <li>
          <span>context</span>
          <span>{percent(confidence.retrieval)}</span>
        </li>
        <li>
          <span>grounding</span>
          <span>{percent(confidence.grounding)}</span>
        </li>
        <li>
          <span>vs existing</span>
          <span>{percent(confidence.similarity_to_existing)}</span>
        </li>
      </ul>

      {compared ? (
        <p className="matched-existing">
          closest existing: <code>{compared}</code>
        </p>
      ) : (
        <p className="matched-existing">no existing case to compare</p>
      )}

      {confidence.needs_review && <span className="review-flag">Review</span>}
    </td>
  );
}
