import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data, Shape } from "plotly.js";
import { getJson } from "../api";
import type {
  Artifact,
  ModelTaskCatalog,
  ModelTaskData,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { DataTable } from "../components/DataTable";
import { Plot } from "../components/Plot";
import { SectionHeading } from "../components/SectionHeading";

const GROUP_COLORS: Record<string, string> = {
  CN: "#38bdf8",
  MCI: "#a3e635",
  AD: "#fb7185",
};

function displayModel(value: unknown): string {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function primaryRows(rows: Record<string, unknown>[]) {
  const hasRole = rows.some((row) => row.analysis_role !== undefined);
  return hasRole
    ? rows.filter((row) => String(row.analysis_role) === "primary")
    : rows;
}

function PerformancePlot({
  data,
}: {
  data: ModelTaskData;
}) {
  const rows = primaryRows(data.performance);
  if (data.kind === "classification") {
    const metrics = [
      ["balanced_accuracy", "Balanced accuracy", "#38bdf8"],
      ["macro_auroc_ovr", "Macro AUROC", "#a78bfa"],
      ["macro_f1", "Macro F1", "#a3e635"],
    ] as const;
    return (
      <Plot
        data={metrics.map(
          ([metric, label, color]) =>
            ({
              type: "bar",
              name: label,
              x: rows.map((row) => displayModel(row.model)),
              y: rows.map((row) => Number(row[metric])),
              marker: { color },
              hovertemplate:
                `${label}<br>%{x}<br>%{y:.3f}<extra></extra>`,
            }) as Data,
        )}
        layout={{
          height: 460,
          title: {
            text: "Cross-validated model discrimination",
            font: { size: 16 },
          },
          barmode: "group",
          yaxis: {
            title: { text: "Cross-validated score" },
            range: [0, 1],
          },
          legend: { orientation: "h" },
          margin: { b: 130 },
        }}
      />
    );
  }
  const traces: Data[] = [
    {
      type: "bar",
      name: "MAE",
      x: rows.map((row) => displayModel(row.model)),
      y: rows.map((row) => Number(row.mae)),
      marker: { color: "#38bdf8" },
      hovertemplate: "%{x}<br>MAE=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "R²",
      x: rows.map((row) => displayModel(row.model)),
      y: rows.map((row) => Number(row.r2)),
      yaxis: "y2",
      line: { color: "#fb7185", width: 3 },
      marker: { color: "#fb7185", size: 8 },
      hovertemplate: "%{x}<br>R²=%{y:.3f}<extra></extra>",
    },
  ];
  return (
    <Plot
      data={traces}
      layout={{
        height: 460,
        title: {
          text: "Cross-validated score prediction",
          font: { size: 16 },
        },
        yaxis: { title: { text: "Mean absolute error" }, rangemode: "tozero" },
        yaxis2: {
          title: { text: "R²" },
          overlaying: "y",
          side: "right",
          zeroline: true,
        },
        legend: { orientation: "h" },
        margin: { b: 130, r: 70 },
      }}
    />
  );
}

function ConfusionPlot({
  rows,
  model,
}: {
  rows: Record<string, unknown>[];
  model: string;
}) {
  const selected = rows.filter((row) => String(row.model) === model);
  const trueKey = selected.some((row) => row.true_group !== undefined)
    ? "true_group"
    : "true_class";
  const predictedKey = selected.some(
    (row) => row.predicted_group !== undefined,
  )
    ? "predicted_group"
    : "predicted_class";
  const labels = Array.from(
    new Set(
      selected.flatMap((row) => [
        String(row[trueKey]),
        String(row[predictedKey]),
      ]),
    ),
  ).sort();
  const matrix = labels.map((truth) =>
    labels.map((prediction) => {
      const match = selected.find(
        (row) =>
          String(row[trueKey]) === truth &&
          String(row[predictedKey]) === prediction,
      );
      return Number(match?.n ?? 0);
    }),
  );
  return (
    <Plot
      data={[
        {
          type: "heatmap",
          x: labels,
          y: labels,
          z: matrix,
          colorscale: "Blues",
          text: matrix.map((row) => row.map(String)),
          texttemplate: "%{text}",
          hovertemplate:
            "True=%{y}<br>Predicted=%{x}<br>n=%{z}<extra></extra>",
          colorbar: { title: { text: "n" } },
        } as unknown as Data,
      ]}
      layout={{
        height: 430,
        title: {
          text: `Pooled outer-fold confusion · ${displayModel(model)}`,
          font: { size: 15 },
        },
        xaxis: { title: { text: "Predicted class" } },
        yaxis: {
          title: { text: "True class" },
          autorange: "reversed",
        },
        showlegend: false,
      }}
    />
  );
}

function RegressionPredictionPlot({
  rows,
  model,
}: {
  rows: Record<string, unknown>[];
  model: string;
}) {
  const selected = rows.filter(
    (row) =>
      String(row.model) === model &&
      (!row.analysis_role || String(row.analysis_role) === "primary"),
  );
  const values = selected.flatMap((row) => [
    Number(row.true_score),
    Number(row.predicted_score),
  ]).filter(Number.isFinite);
  const lower = Math.min(...values);
  const upper = Math.max(...values);
  const identity: Partial<Shape> = {
    type: "line",
    x0: lower,
    x1: upper,
    y0: lower,
    y1: upper,
    line: { color: "#94a3b8", dash: "dash", width: 2 },
  };
  return (
    <Plot
      data={["CN", "MCI", "AD"].map((group) => {
        const groupRows = selected.filter(
          (row) => String(row.group) === group,
        );
        return {
          type: "scatter",
          mode: "markers",
          name: group,
          x: groupRows.map((row) => Number(row.true_score)),
          y: groupRows.map((row) => Number(row.predicted_score)),
          marker: {
            color: GROUP_COLORS[group],
            size: 8,
            opacity: 0.72,
          },
          hovertemplate:
            `${group}<br>Observed=%{x:.2f}<br>Predicted=%{y:.2f}<extra></extra>`,
        } as Data;
      })}
      layout={{
        height: 430,
        title: {
          text: `Observed versus predicted · ${displayModel(model)}`,
          font: { size: 15 },
        },
        xaxis: { title: { text: "Observed clinical score" } },
        yaxis: { title: { text: "Cross-validated prediction" } },
        shapes: [identity],
        legend: { orientation: "h" },
      }}
    />
  );
}

function RocPlot({
  rows,
  model,
}: {
  rows: Record<string, unknown>[];
  model: string;
}) {
  const selected = rows.filter(
    (row) => String(row.model) === model,
  );
  const classes = Array.from(
    new Set(selected.map((row) => String(row.class))),
  ).sort();
  return (
    <Plot
      data={classes.map((label) => {
        const values = selected
          .filter((row) => String(row.class) === label)
          .sort((first, second) => Number(first.point) - Number(second.point));
        return {
          type: "scatter",
          mode: "lines",
          name: label,
          x: values.map((row) => Number(row.fpr)),
          y: values.map((row) => Number(row.tpr)),
          line: {
            color: GROUP_COLORS[label] ?? "#a78bfa",
            width: 3,
          },
          hovertemplate:
            `${label}<br>FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>`,
        } as Data;
      })}
      layout={{
        height: 430,
        title: {
          text: `One-vs-rest ROC · ${displayModel(model)}`,
          font: { size: 15 },
        },
        xaxis: { title: { text: "False-positive rate" }, range: [0, 1] },
        yaxis: { title: { text: "True-positive rate" }, range: [0, 1] },
        shapes: [
          {
            type: "line",
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 1,
            line: { color: "#64748b", dash: "dash", width: 1.5 },
          },
        ],
        legend: { orientation: "h" },
      }}
    />
  );
}

function ImportancePlot({
  rows,
  model,
  topN,
  title = "Top recorded features",
  valueLabel = "Recorded model importance",
}: {
  rows: Record<string, unknown>[];
  model: string;
  topN: number;
  title?: string;
  valueLabel?: string;
}) {
  const selected = rows
    .filter(
      (row) =>
        String(row.model) === model &&
        (!row.analysis_role || String(row.analysis_role) === "primary"),
    )
    .sort((first, second) => Number(first.rank) - Number(second.rank))
    .slice(0, topN)
    .reverse();
  return (
    <Plot
      data={[
        {
          type: "bar",
          orientation: "h",
          x: selected.map((row) =>
            Number(row.importance ?? row.mean_abs_contribution),
          ),
          y: selected.map((row) =>
            String(row.feature).replaceAll("__", " · "),
          ),
          marker: {
            color: selected.map((row) => {
              const group = String(row.feature_group);
              if (group.includes("DTI")) return "#38bdf8";
              if (group.includes("EDR")) return "#fb7185";
              if (group.includes("Graph")) return "#a3e635";
              return "#a78bfa";
            }),
          },
          customdata: selected.map((row) => [
            row.feature_group,
            row.selected_fold_count,
            row.importance_method,
          ]) as Array<Array<string | number>>,
          hovertemplate:
            "%{y}<br>importance=%{x:.4g}<br>family=%{customdata[0]}" +
            "<br>selected folds=%{customdata[1]}<extra></extra>",
        } as Data,
      ]}
      layout={{
        height: 700,
        title: {
          text: `${title} · ${displayModel(model)}`,
          font: { size: 15 },
        },
        xaxis: { title: { text: valueLabel } },
        yaxis: { automargin: true },
        showlegend: false,
        margin: { l: 300 },
      }}
    />
  );
}

export function MlDiagnosticsPage({ section }: { section: Section }) {
  const [taskId, setTaskId] = useState("diagnosis");
  const [model, setModel] = useState("");
  const [topFeatures, setTopFeatures] = useState(20);
  const [showPredictions, setShowPredictions] = useState(false);
  const catalog = useQuery({
    queryKey: ["model-task-catalog"],
    queryFn: ({ signal }) =>
      getJson<ModelTaskCatalog>("/models/tasks", signal),
  });
  const task = useQuery({
    queryKey: ["model-task", taskId],
    queryFn: ({ signal }) =>
      getJson<ModelTaskData>(`/models/tasks/${taskId}`, signal),
  });
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(`/sections/${section.id}/artifacts`, signal),
  });
  const modelOptions = useMemo(
    () =>
      Array.from(
        new Set(
          primaryRows(task.data?.data.performance ?? []).map((row) =>
            String(row.model),
          ),
        ),
      ),
    [task.data],
  );
  useEffect(() => {
    if (!modelOptions.includes(model)) {
      setModel(modelOptions[0] ?? "");
    }
  }, [model, modelOptions]);
  const selectedPerformance = primaryRows(
    task.data?.data.performance ?? [],
  ).find((row) => String(row.model) === model);

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>
            Recorded cross-validation, error analysis, feature attribution,
            and task-specific cohort scope. No model is refitted in the
            browser or API.
          </p>
        </div>
        <div className="page-stat">
          <strong>{catalog.data?.data.tasks.length ?? "—"}</strong>
          <span>prediction tasks</span>
        </div>
      </div>

      <section className="panel">
        <SectionHeading index="01" title="Cross-validated performance" />
        {catalog.isLoading || task.isLoading ? (
          <Loading label="Loading recorded ML diagnostics" />
        ) : catalog.isError ? (
          <ErrorState error={catalog.error} />
        ) : task.isError ? (
          <ErrorState error={task.error} />
        ) : catalog.data && task.data ? (
          <>
            <div className="control-row wrap">
              <label className="control grow">
                <span>Prediction task</span>
                <select
                  value={taskId}
                  onChange={(event) => setTaskId(event.target.value)}
                >
                  {catalog.data.data.tasks.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Model</span>
                <select
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                >
                  {modelOptions.map((item) => (
                    <option key={item} value={item}>
                      {displayModel(item)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <PerformancePlot data={task.data.data} />
            {selectedPerformance ? (
              <DataTable
                rows={[selectedPerformance]}
                compact
                caption="Selected model · primary recorded analysis"
              />
            ) : null}
            <div className="mapping-warning">
              <strong>Interpretation boundary:</strong>{" "}
              {catalog.data.data.contract.validation}{" "}
              {catalog.data.data.contract.clinical_scope}
            </div>
          </>
        ) : null}
      </section>

      {task.data ? (
        <>
          <section className="panel">
            <SectionHeading index="02" title="Error structure" />
            {task.data.data.kind === "classification" ? (
              <div className="plot-grid two">
                <ConfusionPlot
                  rows={task.data.data.confusion}
                  model={model}
                />
                {task.data.data.roc.length ? (
                  <RocPlot rows={task.data.data.roc} model={model} />
                ) : (
                  <div className="table-empty">
                    A recorded ROC curve is not available for this task.
                  </div>
                )}
              </div>
            ) : (
              <div className="plot-grid two">
                <RegressionPredictionPlot
                  rows={task.data.data.predictions}
                  model={model}
                />
                <DataTable
                  rows={task.data.data.detail.filter(
                    (row) =>
                      String(row.model) === model &&
                      (!row.analysis_role ||
                        String(row.analysis_role) === "primary"),
                  )}
                  compact
                  maxHeight={430}
                  caption="Performance stratified by diagnostic group"
                />
              </div>
            )}
            {task.data.data.kind === "classification" ? (
              <DataTable
                rows={task.data.data.detail.filter(
                  (row) =>
                    String(row.model) === model &&
                    (!row.analysis_role ||
                      String(row.analysis_role) === "primary"),
                )}
                compact
                maxHeight={430}
                caption="Per-class sensitivity, specificity and F1"
              />
            ) : null}
            <div className="analysis-actions">
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={showPredictions}
                  onChange={(event) =>
                    setShowPredictions(event.target.checked)
                  }
                />
                Show subject-level cross-validated predictions
              </label>
            </div>
            {showPredictions ? (
              <DataTable
                rows={task.data.data.predictions.filter(
                  (row) =>
                    String(row.model) === model &&
                    (!row.analysis_role ||
                      String(row.analysis_role) === "primary"),
                )}
                compact
                maxHeight={560}
                initialRows={120}
                caption="Authenticated cross-validated prediction rows"
              />
            ) : null}
          </section>

          <section className="panel">
            <SectionHeading index="03" title="Feature attribution" />
            <div className="control-row">
              <label className="control compact-number-control">
                <span>Top features</span>
                <input
                  type="number"
                  min={10}
                  max={60}
                  step={5}
                  value={topFeatures}
                  onChange={(event) =>
                    setTopFeatures(
                      Math.max(
                        10,
                        Math.min(60, Number(event.target.value) || 20),
                      ),
                    )
                  }
                />
              </label>
            </div>
            <div
              className={
                task.data.data.shap.length ? "plot-grid two" : "plot-grid"
              }
            >
              <ImportancePlot
                rows={task.data.data.importance}
                model={model}
                topN={topFeatures}
              />
              {task.data.data.shap.length ? (
                <ImportancePlot
                  rows={task.data.data.shap}
                  model={model}
                  topN={topFeatures}
                  title="Top mean absolute TreeSHAP contributions"
                  valueLabel="Mean absolute contribution"
                />
              ) : null}
            </div>
            <p className="method-note">
              {catalog.data?.data.contract.interpretation}
            </p>
          </section>

          <section className="panel">
            <SectionHeading index="04" title="Execution status and audit" />
            <div className="table-grid two">
              <DataTable
                rows={task.data.data.status}
                compact
                caption="Declared model execution status"
              />
              <DataTable
                rows={task.data.data.stage}
                compact
                caption="Task-specific stage summary"
              />
            </div>
          </section>
        </>
      ) : null}

      <section className="panel">
        <SectionHeading index="OUTPUTS" title="Complete recorded outputs" />
        {artifacts.isLoading ? (
          <Loading label="Indexing ML outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
