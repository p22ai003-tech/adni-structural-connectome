import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { downloadRows, getJson } from "../api";
import type {
  AnalysisMetric,
  NodeCatalog,
  NodeRanking,
  NodeValues,
} from "../types";
import { ErrorState, Loading } from "./AsyncState";
import { DataTable } from "./DataTable";
import {
  GroupDistributionPlot,
  NodeRankingPlot,
} from "./ScientificPlots";
import { SectionHeading } from "./SectionHeading";

const PAIRS = [
  ["CN", "MCI"],
  ["CN", "AD"],
  ["MCI", "AD"],
] as const;

interface RepeatedNodes {
  metric: string;
  top_n: number;
  method: string;
  rows: Record<string, unknown>[];
}

export function NodeAnalysisWorkspace({
  sectionId,
  metrics,
}: {
  sectionId: string;
  metrics: AnalysisMetric[];
}) {
  const [metric, setMetric] = useState(metrics[0]?.id ?? "");
  const [pairIndex, setPairIndex] = useState(0);
  const [topN, setTopN] = useState(20);
  const [onlyFdr, setOnlyFdr] = useState(false);
  const [node, setNode] = useState<number | null>(null);
  const [hideOutliers, setHideOutliers] = useState(true);
  const [showSubjectRows, setShowSubjectRows] = useState(false);
  const [groupA, groupB] = PAIRS[pairIndex];

  useEffect(() => {
    if (!metrics.some((item) => item.id === metric)) {
      setMetric(metrics[0]?.id ?? "");
    }
    setNode(null);
  }, [metric, metrics]);

  const catalog = useQuery({
    queryKey: ["node-catalog", sectionId, metric],
    queryFn: ({ signal }) =>
      getJson<NodeCatalog>(
        `/sections/${sectionId}/analysis/nodes/${encodeURIComponent(metric)}`,
        signal,
      ),
    enabled: Boolean(metric),
  });
  const ranking = useQuery({
    queryKey: ["node-ranking", sectionId, metric, groupA, groupB],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams({ group_a: groupA, group_b: groupB });
      return getJson<NodeRanking>(
        `/sections/${sectionId}/analysis/nodes/${encodeURIComponent(metric)}/ranking?${params.toString()}`,
        signal,
      );
    },
    enabled: Boolean(metric),
  });
  const repeated = useQuery({
    queryKey: ["node-repeated", sectionId, metric, topN],
    queryFn: ({ signal }) =>
      getJson<RepeatedNodes>(
        `/sections/${sectionId}/analysis/nodes/${encodeURIComponent(metric)}/repeated?top_n=${topN}`,
        signal,
      ),
    enabled: Boolean(metric),
  });

  useEffect(() => {
    if (node !== null || !catalog.data?.data.nodes.length) return;
    const firstRanked = Number(ranking.data?.data.rows[0]?.node);
    const firstCatalog = Number(catalog.data.data.nodes[0]?.node);
    setNode(Number.isFinite(firstRanked) ? firstRanked : firstCatalog);
  }, [catalog.data, node, ranking.data]);

  const nodeData = useQuery({
    queryKey: ["node-values", sectionId, metric, node],
    queryFn: ({ signal }) =>
      getJson<NodeValues>(
        `/sections/${sectionId}/analysis/nodes/${encodeURIComponent(metric)}/${node}`,
        signal,
      ),
    enabled: node !== null && Boolean(metric),
  });
  const filteredRanking = useMemo(() => {
    const rows = ranking.data?.data.rows ?? [];
    return (onlyFdr
      ? rows.filter((row) => Number(row.q_value) < 0.05)
      : rows
    ).slice(0, topN);
  }, [onlyFdr, ranking.data, topN]);
  const selectedMetric = metrics.find((item) => item.id === metric);

  return (
    <>
      <section className="panel">
        <SectionHeading index="01" title="Pairwise AAL3 node ranking" />
        <div className="control-row wrap">
          <label className="control grow">
            <span>Node metric</span>
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
          <label className="control">
            <span>Diagnostic contrast</span>
            <select
              value={pairIndex}
              onChange={(event) => setPairIndex(Number(event.target.value))}
            >
              {PAIRS.map(([first, second], index) => (
                <option key={`${first}-${second}`} value={index}>
                  {first} vs {second}
                </option>
              ))}
            </select>
          </label>
          <label className="control compact-number-control">
            <span>Top nodes</span>
            <input
              type="number"
              min={5}
              max={80}
              step={5}
              value={topN}
              onChange={(event) =>
                setTopN(
                  Math.min(80, Math.max(5, Number(event.target.value))),
                )
              }
            />
          </label>
          <label className="toggle-control">
            <input
              type="checkbox"
              checked={onlyFdr}
              onChange={(event) => setOnlyFdr(event.target.checked)}
            />
            <span>Only BH q&lt;0.05</span>
          </label>
        </div>
        {selectedMetric?.description ? (
          <div className="definition-callout">
            <strong>MEASURE</strong>
            <span>{selectedMetric.description}</span>
          </div>
        ) : null}
        {ranking.isLoading || repeated.isLoading ? (
          <Loading label="Computing node-level contrast" />
        ) : ranking.isError ? (
          <ErrorState error={ranking.error} />
        ) : repeated.isError ? (
          <ErrorState error={repeated.error} />
        ) : ranking.data && repeated.data ? (
          <>
            {filteredRanking.length ? (
              <div className="analysis-split">
                <NodeRankingPlot
                  rows={filteredRanking}
                  groupA={groupA}
                  groupB={groupB}
                  topN={topN}
                />
                <DataTable
                  rows={filteredRanking}
                  columns={[
                    "rank",
                    "node_name",
                    "atlas_label",
                    `median_${groupA}`,
                    `median_${groupB}`,
                    "median_diff_a_minus_b",
                    "higher_group",
                    "p_value",
                    "q_value",
                    "cliffs_delta",
                  ]}
                  maxHeight={600}
                  initialRows={80}
                  compact
                  caption={`${groupA} vs ${groupB} · ${metric}`}
                />
              </div>
            ) : (
              <div className="table-empty">
                No nodes pass BH q&lt;0.05 for this contrast. Disable the FDR
                filter to inspect the complete p-ranked table.
              </div>
            )}
            <p className="method-note">{ranking.data.data.method}</p>
            <DataTable
              rows={repeated.data.data.rows.slice(0, topN)}
              columns={[
                "rank",
                "node_name",
                "atlas_label",
                "top_n_occurrences",
                "fdr_significant_comparisons",
                "best_p",
                "best_q",
                "comparisons",
                "directions",
              ]}
              maxHeight={430}
              initialRows={80}
              compact
              caption="Regions repeated across all three pairwise rankings"
            />
            <div className="analysis-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={() =>
                  downloadRows(
                    `${sectionId}_${metric}_${groupA}_${groupB}_nodes.csv`,
                    ranking.data!.data.rows,
                  )
                }
              >
                Download complete ranking
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() =>
                  downloadRows(
                    `${sectionId}_${metric}_repeated_top${topN}.csv`,
                    repeated.data!.data.rows,
                  )
                }
              >
                Download repeated regions
              </button>
            </div>
          </>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading index="02" title="AAL3 region viewer" />
        {catalog.isLoading ? (
          <Loading label="Loading AAL3 node catalog" />
        ) : catalog.isError ? (
          <ErrorState error={catalog.error} />
        ) : catalog.data ? (
          <>
            <div className="control-row wrap">
              <label className="control grow">
                <span>AAL3 region</span>
                <select
                  value={node ?? ""}
                  onChange={(event) => setNode(Number(event.target.value))}
                >
                  {catalog.data.data.nodes.map((row) => (
                    <option key={String(row.node)} value={Number(row.node)}>
                      {String(row.node_name)} · {String(row.atlas_label)}
                      {Number(row.kw_q) < 0.05
                        ? ` · omnibus q=${Number(row.kw_q).toPrecision(3)}`
                        : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={hideOutliers}
                  onChange={(event) =>
                    setHideOutliers(event.target.checked)
                  }
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
            {nodeData.isLoading ? (
              <Loading label="Loading selected AAL3 values" />
            ) : nodeData.isError ? (
              <ErrorState error={nodeData.error} />
            ) : nodeData.data ? (
              <>
                <GroupDistributionPlot
                  rows={
                    hideOutliers
                      ? nodeData.data.data.display_rows
                      : nodeData.data.data.rows
                  }
                  valueKey="value"
                  title={`${selectedMetric?.label ?? metric} · ${nodeData.data.data.node_name}`}
                  yTitle={selectedMetric?.label ?? metric}
                />
                <DataTable
                  rows={nodeData.data.data.descriptives}
                  compact
                  caption="Selected-region group descriptives"
                />
                {showSubjectRows ? (
                  <DataTable
                    rows={nodeData.data.data.rows}
                    compact
                    maxHeight={520}
                    initialRows={100}
                    caption="Authenticated selected-region subject rows"
                  />
                ) : null}
                <p className="method-note">
                  {nodeData.data.data.display_filter} Source:{" "}
                  <code>{nodeData.data.data.source}</code>.
                </p>
              </>
            ) : null}
          </>
        ) : null}
      </section>
    </>
  );
}
