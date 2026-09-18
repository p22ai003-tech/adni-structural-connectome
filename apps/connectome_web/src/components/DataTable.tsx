import { useMemo, useState } from "react";
import { formatNumber } from "../api";

interface DataTableHeaderGroup {
  label: string;
  columns: string[];
  tone: "strength" | "burden";
}

interface DataTableProps {
  rows: Record<string, unknown>[];
  columns?: string[];
  maxHeight?: number | null;
  compact?: boolean;
  initialRows?: number;
  caption?: string;
  listColumns?: string[];
  rowClassName?: (
    row: Record<string, unknown>,
    rowIndex: number,
  ) => string | undefined;
  cellClassName?: (
    row: Record<string, unknown>,
    column: string,
    value: unknown,
  ) => string | undefined;
  headerGroups?: DataTableHeaderGroup[];
}

function displayValue(value: unknown): string {
  if (typeof value === "number") return formatNumber(value, 4);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return value === null || value === undefined || value === ""
    ? "—"
    : String(value);
}

export function DataTable({
  rows,
  columns,
  maxHeight = 430,
  compact = false,
  initialRows = 100,
  caption,
  listColumns = [],
  rowClassName,
  cellClassName,
  headerGroups = [],
}: DataTableProps) {
  const [showAll, setShowAll] = useState(false);
  const resolvedColumns = useMemo(
    () => columns ?? (rows[0] ? Object.keys(rows[0]) : []),
    [columns, rows],
  );
  const visibleRows = showAll ? rows : rows.slice(0, initialRows);
  const headerGroupByColumn = useMemo(() => {
    const mapping = new Map<string, DataTableHeaderGroup>();
    headerGroups.forEach((group) => {
      group.columns.forEach((column) => mapping.set(column, group));
    });
    return mapping;
  }, [headerGroups]);
  const hasHeaderGroups = headerGroupByColumn.size > 0;

  if (!rows.length) {
    return <div className="table-empty">No rows available.</div>;
  }

  return (
    <div className={`data-table-shell ${compact ? "compact" : ""}`}>
      {caption ? <div className="table-caption">{caption}</div> : null}
      <div
        className={`data-table-scroll ${maxHeight === null ? "show-all-rows" : ""}`}
        style={maxHeight === null ? undefined : { maxHeight }}
        role="region"
        aria-label={caption ?? "Scrollable data table"}
        tabIndex={0}
      >
        <table
          className={`data-table ${hasHeaderGroups ? "has-header-groups" : ""}`}
        >
          <thead>
            <tr>
              {resolvedColumns.map((column, columnIndex) => {
                const group = headerGroupByColumn.get(column);
                if (!hasHeaderGroups) {
                  return (
                    <th key={column} scope="col">
                      {column}
                    </th>
                  );
                }
                if (!group) {
                  return (
                    <th key={column} scope="col" rowSpan={2}>
                      {column}
                    </th>
                  );
                }
                const previousColumn = resolvedColumns[columnIndex - 1];
                if (
                  previousColumn &&
                  headerGroupByColumn.get(previousColumn) === group
                ) {
                  return null;
                }
                const contiguousColumns = resolvedColumns
                  .slice(columnIndex)
                  .findIndex(
                    (candidate) =>
                      headerGroupByColumn.get(candidate) !== group,
                  );
                const colSpan =
                  contiguousColumns === -1
                    ? resolvedColumns.length - columnIndex
                    : contiguousColumns;
                return (
                  <th
                    key={group.label}
                    className={`table-header-group table-header-group-${group.tone}`}
                    colSpan={colSpan}
                    scope="colgroup"
                  >
                    {group.label}
                  </th>
                );
              })}
            </tr>
            {hasHeaderGroups ? (
              <tr>
                {resolvedColumns.map((column) => {
                  const group = headerGroupByColumn.get(column);
                  return group ? (
                    <th
                      key={column}
                      className={`table-subheader table-subheader-${group.tone}`}
                      scope="col"
                    >
                      {column}
                    </th>
                  ) : null;
                })}
              </tr>
            ) : null}
          </thead>
          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr
                key={`${rowIndex}-${String(row[resolvedColumns[0]])}`}
                className={rowClassName?.(row, rowIndex)}
              >
                {resolvedColumns.map((column) => (
                  <td
                    key={column}
                    title={displayValue(row[column])}
                    className={cellClassName?.(row, column, row[column])}
                  >
                    {listColumns.includes(column) ? (
                      <ul className="table-cell-list">
                        {String(row[column] ?? "")
                          .split(";")
                          .map((item) => item.trim())
                          .filter(Boolean)
                          .map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                      </ul>
                    ) : (
                      displayValue(row[column])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > initialRows ? (
        <button
          className="table-more"
          type="button"
          onClick={() => setShowAll((value) => !value)}
        >
          {showAll
            ? `Show first ${initialRows}`
            : `Show all ${rows.length.toLocaleString()} rows`}
        </button>
      ) : null}
    </div>
  );
}
