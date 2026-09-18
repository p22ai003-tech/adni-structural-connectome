import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data } from "plotly.js";
import { formatNumber, getJson } from "../api";
import type {
  ConnectomeSubject,
  MatrixPayload,
  MatrixSummary,
} from "../types";
import { DataTable } from "../components/DataTable";
import { ErrorState, Loading } from "../components/AsyncState";
import { Plot } from "../components/Plot";
import { SectionHeading } from "../components/SectionHeading";

interface EdgePayload {
  metadata: Record<string, unknown>;
  positive_only: boolean;
  offset: number;
  limit: number;
  total_rows: number;
  rows: Record<string, unknown>[];
}

export function MatrixViewerPage() {
  const [subjectId, setSubjectId] = useState("");
  const [weight, setWeight] = useState("fd_sum");
  const subjects = useQuery({
    queryKey: ["connectome-subjects"],
    queryFn: ({ signal }) =>
      getJson<ConnectomeSubject[]>("/connectomes/subjects", signal),
  });
  useEffect(() => {
    if (!subjectId && subjects.data?.data.length) {
      setSubjectId(subjects.data.data[0].subject_id);
    }
  }, [subjectId, subjects.data]);
  const selected = subjects.data?.data.find(
    (subject) => subject.subject_id === subjectId,
  );
  useEffect(() => {
    if (
      selected &&
      !selected.available_weights.includes(weight) &&
      selected.available_weights.length
    ) {
      setWeight(selected.available_weights[0]);
    }
  }, [selected, weight]);

  const summary = useQuery({
    queryKey: ["matrix-summary", subjectId, weight],
    queryFn: ({ signal }) =>
      getJson<MatrixSummary>(
        `/connectomes/${subjectId}/${weight}/summary`,
        signal,
      ),
    enabled: Boolean(subjectId && weight),
  });
  const matrix = useQuery({
    queryKey: ["matrix", subjectId, weight],
    queryFn: ({ signal }) =>
      getJson<MatrixPayload>(
        `/connectomes/${subjectId}/${weight}/matrix`,
        signal,
      ),
    enabled: Boolean(subjectId && weight),
  });
  const edges = useQuery({
    queryKey: ["matrix-edges", subjectId, weight],
    queryFn: ({ signal }) =>
      getJson<EdgePayload>(
        `/connectomes/${subjectId}/${weight}/edges?limit=100`,
        signal,
      ),
    enabled: Boolean(subjectId && weight),
  });
  const heatmap = useMemo<Data[]>(() => {
    if (!matrix.data) return [];
    const payload = matrix.data.data;
    const labels = payload.labels.map((label) => label.label);
    return [
      {
        type: "heatmap",
        z: payload.matrix,
        x: labels,
        y: labels,
        colorscale:
          weight.includes("md") ||
          weight.includes("ad") ||
          weight.includes("rd")
            ? "Viridis"
            : "YlGnBu",
        colorbar: {
          title: { text: weight },
          thickness: 12,
        },
        hovertemplate:
          "%{y} ↔ %{x}<br>Value=%{z:.5g}<extra></extra>",
      } as Data,
    ];
  }, [matrix.data, weight]);

  if (subjects.isLoading) {
    return <Loading label="Indexing connectome matrices" />;
  }
  if (subjects.isError) return <ErrorState error={subjects.error} />;
  if (!subjects.data) return null;

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Tractography · Matrix QC</span>
          <h1>Structural Connectome Matrix Viewer</h1>
          <p>
            Inspect one subject and one allow-listed weight at a time without
            exposing filesystem paths to the browser.
          </p>
        </div>
        <div className="status-pill">
          {subjects.data.data.length.toLocaleString()} subjects indexed
        </div>
      </div>
      <section className="panel">
        <div className="control-row matrix-controls">
          <label className="control grow">
            <span>Subject</span>
            <select
              value={subjectId}
              onChange={(event) => setSubjectId(event.target.value)}
            >
              {subjects.data.data.map((subject) => (
                <option key={subject.subject_id} value={subject.subject_id}>
                  {subject.subject_id} · {subject.group} · density{" "}
                  {formatNumber(subject.density, 3)}
                </option>
              ))}
            </select>
          </label>
          <label className="control">
            <span>Matrix weight</span>
            <select
              value={weight}
              onChange={(event) => setWeight(event.target.value)}
            >
              {(selected?.available_weights ?? []).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        {summary.isLoading || matrix.isLoading || edges.isLoading ? (
          <Loading label="Loading matrix and edge summaries" />
        ) : summary.isError ? (
          <ErrorState error={summary.error} />
        ) : matrix.isError ? (
          <ErrorState error={matrix.error} />
        ) : edges.isError ? (
          <ErrorState error={edges.error} />
        ) : summary.data && matrix.data && edges.data ? (
          <>
            <div className="metric-strip matrix-metrics">
              <div>
                <span>Positive edges</span>
                <strong>
                  {formatNumber(summary.data.data.summary.positive_edges, 0)}
                </strong>
              </div>
              <div>
                <span>Density</span>
                <strong>
                  {formatNumber(summary.data.data.summary.density, 4)}
                </strong>
              </div>
              <div>
                <span>Median positive weight</span>
                <strong>
                  {formatNumber(summary.data.data.summary.median, 5)}
                </strong>
              </div>
              <div>
                <span>Symmetric</span>
                <strong>
                  {summary.data.data.summary.symmetric ? "Yes" : "No"}
                </strong>
              </div>
            </div>
            <SectionHeading
              index={1}
              title={`${subjectId} · ${weight.toUpperCase()} MATRIX`}
            />
            <Plot
              data={heatmap}
              layout={{
                height: 760,
                margin: { l: 115, r: 70, t: 30, b: 115 },
                xaxis: {
                  tickfont: { size: 7 },
                  showticklabels: false,
                },
                yaxis: {
                  tickfont: { size: 7 },
                  showticklabels: false,
                  autorange: "reversed",
                },
              }}
            />
            <p className="method-note">
              Full 166 × 166 AAL3 matrix. Hover reveals the mapped parcel labels;
              axes are visually suppressed to prevent 166 overlapping labels.
            </p>
            <SectionHeading index={2} title="TOP POSITIVE EDGES" />
            <DataTable
              rows={edges.data.data.rows}
              columns={[
                "source_index",
                "source_label",
                "target_index",
                "target_label",
                "value",
              ]}
              initialRows={100}
              maxHeight={520}
              compact
              caption={`Top 100 by weight from ${edges.data.data.total_rows.toLocaleString()} positive edges`}
            />
          </>
        ) : null}
      </section>
    </main>
  );
}
