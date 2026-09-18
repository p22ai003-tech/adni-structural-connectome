import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data } from "plotly.js";
import { getJson } from "../api";
import type { MetricCatalog, MetricPayload } from "../types";
import { ErrorState, Loading } from "./AsyncState";
import { Plot } from "./Plot";

const GROUP_COLORS: Record<string, string> = {
  CN: "#38bdf8",
  MCI: "#a3e635",
  AD: "#fb7185",
};

export function MetricPanel({
  sectionId,
  catalog,
}: {
  sectionId: string;
  catalog: MetricCatalog;
}) {
  const [metric, setMetric] = useState(catalog.metrics[0] ?? "");
  const metricData = useQuery({
    queryKey: ["section-metric", sectionId, metric],
    queryFn: ({ signal }) =>
      getJson<MetricPayload>(
        `/sections/${sectionId}/metrics/${encodeURIComponent(metric)}`,
        signal,
      ),
    enabled: Boolean(metric),
  });
  const traces = useMemo<Data[]>(() => {
    if (!metricData.data) return [];
    return ["CN", "MCI", "AD"].map((group) => {
      const values = metricData.data!.data.rows
        .filter((row) => String(row.group) === group)
        .map((row) => Number(row[metric]))
        .filter(Number.isFinite);
      return {
        type: "box",
        name: group,
        y: values,
        marker: { color: GROUP_COLORS[group] },
        line: { color: GROUP_COLORS[group] },
        fillcolor: `${GROUP_COLORS[group]}55`,
        boxpoints: "outliers",
        jitter: 0.25,
        pointpos: 0,
        hovertemplate: `${group}<br>${metric}=%{y:.5g}<extra></extra>`,
      } as Data;
    });
  }, [metric, metricData.data]);

  if (!catalog.available || !catalog.metrics.length) return null;
  return (
    <div className="panel metric-panel">
      <div className="panel-header">
        <div>
          <span className="eyebrow">Subject-level source table</span>
          <h3>Explore a recorded metric</h3>
        </div>
        <label className="control compact-control">
          <span>Metric</span>
          <select
            value={metric}
            onChange={(event) => setMetric(event.target.value)}
          >
            {catalog.metrics.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      {metricData.isLoading ? (
        <Loading label="Loading subject values" />
      ) : metricData.isError ? (
        <ErrorState error={metricData.error} />
      ) : metricData.data ? (
        <>
          <Plot
            data={traces}
            layout={{
              height: 430,
              title: {
                text: metric,
                x: 0.02,
                xanchor: "left",
                font: { size: 15 },
              },
              yaxis: { title: { text: metric } },
              showlegend: false,
            }}
          />
          <div className="provenance-line">
            Source: {metricData.data.data.source}. Values are unchanged
            subject-level observations; this panel is descriptive.
          </div>
        </>
      ) : null}
    </div>
  );
}
