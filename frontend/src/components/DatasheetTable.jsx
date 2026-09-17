// Column labels and order match the delivered workbook.
const COLUMNS = [
  { key: "s_no", label: "S_no", className: "num" },
  { key: "requirement", label: "Requirement", className: "tight" },
  { key: "description", label: "Description", className: "description" },
  { key: "folder", label: "Folder" },
  { key: "optimization_technique", label: "Optimization_Technique" },
  { key: "test_type", label: "Test_Type", className: "tight" },
  { key: "test_technique", label: "Test_Technique" },
  { key: "retired", label: "Retired?", className: "tight" },
  { key: "scorable", label: "Scorable", className: "tight" },
  { key: "comments", label: "Comments" },
];

// The Description column has a shape the datasheets follow: a leading
// "Test Scenario:" line, then conditions, then a "Verify," line. Emphasising
// those two lines makes a 15-line cell scannable without altering the text,
// which has to stay byte-identical to what gets exported.
function Description({ text }) {
  return (
    <>
      {text.split("\n").map((line, index) => {
        const trimmed = line.trim();
        if (/^test\s*scenario\s*:/i.test(trimmed)) {
          return (
            <span className="scenario" key={index}>
              {line}
            </span>
          );
        }
        if (/^verify,?$/i.test(trimmed)) {
          return (
            <span className="verify" key={index}>
              {line}
            </span>
          );
        }
        return `${line}\n`;
      })}
    </>
  );
}

export default function DatasheetTable({ testCases }) {
  return (
    <div className="sheet-scroll">
      <table className="sheet">
        <thead>
          <tr>
            {COLUMNS.map((column) => (
              <th
                key={column.key}
                className={column.key === "comments" ? "comments-head" : undefined}
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {testCases.map((testCase) => (
            <tr key={testCase.s_no}>
              {COLUMNS.map((column) => (
                <td key={column.key} className={column.className}>
                  {column.key === "description" ? (
                    <Description text={testCase.description} />
                  ) : (
                    testCase[column.key]
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
