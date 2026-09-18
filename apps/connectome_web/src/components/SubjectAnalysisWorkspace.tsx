import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  downloadRows,
  formatNumber,
  getJson,
} from "../api";
import type {
  AnalysisMetric,
  SubjectAnalysis,
} from "../types";
import { ErrorState, Loading } from "./AsyncState";
import { DataTable } from "./DataTable";
import { GroupDistributionPlot } from "./ScientificPlots";
import { SectionHeading } from "./SectionHeading";

const GROUPS = ["CN", "MCI", "AD"] as const;

function pColumn(rows: Record<string, unknown>[]): string | undefined {
  const candidates = ["bm_q", "mwu_q", "q", "welch_q", "kw_q"];
  return candidates.find((column) =>
    rows.some((row) => row[column] !== undefined && row[column] !== null),
  );
}

function groupMedian(
  rows: Record<string, unknown>[],
  group: string,
): unknown {
  return rows.find((row) => String(row.group) === group)?.median;
}

export function SubjectAnalysisWorkspace({
  sectionId,
  metrics,
}: {
  sectionId: string;
  metrics: AnalysisMetric[];
}) {
  const [metric, setMetric] = useState(metrics[0]?.id ?? "");
  const [hideOutliers, setHideOutliers] = useState(true);
  const [showSubjectRows, setShowSubjectRows] = useState(false);
  useEffect(() => {
    if (!metrics.some((item) => item.id === metric)) {
      setMetric(metrics[0]?.id ?? "");
    }
  }, [metric, metrics]);
  const analysis = useQuery({
    queryKey: ["subject-analysis", sectionId, metric],
    queryFn: ({ signal }) =>
      getJson<SubjectAnalysis>(
        `/sections/${sectionId}/analysis/subject/${encodeURIComponent(metric)}`,
        signal,
      ),
    enabled: Boolean(metric),
  });
  const selectedMetric = metrics.find((item) => item.id === metric);
  const qColumn = useMemo(
    () => pColumn(analysis.data?.data.pairwise ?? []),
    [analysis.data],
  );
  const significantCount = useMemo(() => {
    if (!qColumn || !analysis.data) return 0;
    return analysis.data.data.pairwise.filter(
      (row) => Number(row[qColumn]) < 0.05,
    ).length;
  }, [analysis.data, qColumn]);

  return (
    <section className="panel">
      <SectionHeading index="01" title="Subject-level analysis" />
      <div className="control-row wrap">
        <label className="control grow">
          <span>Metric</span>
          <select
            value={metric}
            onChange={(event) => setMetric(event.target.value)}
          >
            {metrics.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={hideOutliers}
            onChange={(event) => setHideOutliers(event.target.checked)}
          />
          <span>Visual-only 1.5-IQR trim</span>
        </label>
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={showSubjectRows}
            onChange={(event) =>
              setShowSubjectRows(event.target.checked)
            }
          />
          <span>Show subject-level rows</span>
        </label>
      </div>
      {selectedMetric?.description ? (
        <div className="definition-callout">
          <strong>MEASURE</strong>
          <span>{selectedMetric.description}</span>
        </div>
      ) : null}

      {analysis.isLoading ? (
        <Loading label="Loading authoritative subject analysis" />
      ) : analysis.isError ? (
        <ErrorState error={analysis.error} />
      ) : analysis.data ? (
        <>
          <div className="metric-strip">
            {GROUPS.map((group) => (
              <div className="metric-tile" key={group}>
                <span>{group} median</span>
                <strong>
                  {formatNumber(
                    groupMedian(
                      analysis.data!.data.descriptives,
                      group,
                    ),
                    5,
                  )}
                </strong>
              </div>
            ))}
            <div className="metric-tile">
              <span>Adjusted pairwise p&lt;0.05</span>
              <strong>{significantCount}</strong>
            </div>
          </div>
          <GroupDistributionPlot
            rows={
              hideOutliers
                ? analysis.data.data.display_rows
                : analysis.data.data.rows
            }
            valueKey={metric}
            title={selectedMetric?.label ?? metric}
            yTitle={selectedMetric?.label ?? metric}
          />
          <p className="method-note">
            {analysis.data.data.display_filter}
            {hideOutliers
              ? ` Hidden for display: ${GROUPS.map(
                  (group) =>
                    `${group} ${analysis.data!.data.display_removed[group] ?? 0}`,
                ).join(" · ")}.`
              : " No observations are hidden in this view."}
          </p>
          <div className="table-grid two">
            <DataTable
              rows={analysis.data.data.descriptives}
              compact
              caption="Group descriptives"
            />
            <DataTable
              rows={analysis.data.data.pairwise}
              compact
              caption="Stored pairwise inference"
            />
          </div>
          <div className="analysis-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() =>
                downloadRows(
                  `${sectionId}_${metric}_subject_rows.csv`,
                  analysis.data!.data.rows,
                )
              }
            >
              Download subject rows
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() =>
                downloadRows(
                  `${sectionId}_${metric}_pairwise.csv`,
                  analysis.data!.data.pairwise,
                )
              }
              disabled={!analysis.data.data.pairwise.length}
            >
              Download inference
            </button>
          </div>
          {showSubjectRows ? (
            <DataTable
              rows={analysis.data.data.rows}
              compact
              maxHeight={560}
              initialRows={100}
              caption="Authenticated subject-level source rows"
            />
          ) : null}
          <div className="provenance-line">
            Source: <code>{analysis.data.data.source}</code>. Statistics:{" "}
            {analysis.data.data.statistic_source}. The React client does not
            recompute p-values.
          </div>
        </>
      ) : null}
    </section>
  );
}
