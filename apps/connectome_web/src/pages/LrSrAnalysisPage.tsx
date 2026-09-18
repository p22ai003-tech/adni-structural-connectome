import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data, Shape } from "plotly.js";
import { formatNumber, getJson } from "../api";
import type {
  ExceptionConfig,
  ExceptionInference,
  ExceptionSummary,
  FeatureEda,
  FeatureFamilies,
  NetworkMapping,
  NetworkMeasureRanked,
  NetworkMeasureRankedGroup,
  ExceptionSpecificity,
  TractDistribution,
  TractRanges,
  TractSummary,
} from "../types";
import { DataTable } from "../components/DataTable";
import { ErrorState, Loading } from "../components/AsyncState";
import { Plot } from "../components/Plot";
import { SectionHeading } from "../components/SectionHeading";

const GROUP_COLORS: Record<string, string> = {
  Overall: "#94a3b8",
  CN: "#38bdf8",
  MCI: "#a3e635",
  AD: "#fb7185",
};
const PANELS = ["Overall", "CN", "MCI", "AD"] as const;

function median(values: number[]): number | null {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return null;
  const middle = Math.floor(finite.length / 2);
  return finite.length % 2
    ? finite[middle]
    : (finite[middle - 1] + finite[middle]) / 2;
}

function visuallyTrimmed(
  rows: ExceptionInference["plot_rows"],
): ExceptionInference["plot_rows"] {
  const strata = Array.from(
    new Set(
      rows.map((row) => `${row.Group}\u0000${row["Edge class"]}`),
    ),
  );
  return strata.flatMap((stratum) => {
    const [group, edgeClass] = stratum.split("\u0000");
    const groupRows = rows.filter(
      (row) =>
        row.Group === group && row["Edge class"] === edgeClass,
    );
    const values = groupRows
      .map((row) => Number(row["Subject median"]))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    if (values.length < 4) return groupRows;
    const quantile = (position: number) => {
      const index = (values.length - 1) * position;
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      return (
        values[lower] +
        (values[upper] - values[lower]) * (index - lower)
      );
    };
    const q1 = quantile(0.25);
    const q3 = quantile(0.75);
    const iqr = q3 - q1;
    return groupRows.filter((row) => {
      const value = Number(row["Subject median"]);
      return value >= q1 - 1.5 * iqr && value <= q3 + 1.5 * iqr;
    });
  });
}

function explicitInference(row: Record<string, unknown>): string {
  const metric = String(row.Metric ?? "measure")
    .replace(
      "Exception − non-exception strength",
      "within-subject exception-minus-non-exception strength change",
    )
    .toLowerCase();
  const direction = String(row["Observed direction"] ?? "");
  const [first, second] = String(row.Contrast ?? "").split(" vs ");
  const higher = direction.replace(" higher", "");
  const lower = higher === first ? second : first;
  return higher && lower
    ? `${higher} has higher ${metric} than ${lower}.`
    : `No directional median interpretation is available for ${metric}.`;
}

function TractLengthPanel() {
  const summary = useQuery({
    queryKey: ["tract-summary"],
    queryFn: ({ signal }) =>
      getJson<TractSummary[]>("/lr-sr/tract-length/summary", signal),
  });
  const distributions = useQuery({
    queryKey: ["tract-distributions"],
    queryFn: async ({ signal }) => {
      const result = [];
      for (const panel of PANELS) {
        result.push(
          await getJson<TractDistribution>(
            `/lr-sr/tract-length/distribution?panel=${panel}`,
            signal,
          ),
        );
      }
      return result;
    },
  });
  const ranges = useQuery({
    queryKey: ["tract-ranges"],
    queryFn: ({ signal }) =>
      getJson<TractRanges>("/lr-sr/tract-length/ranges", signal),
  });
  if (summary.isLoading || distributions.isLoading || ranges.isLoading) {
    return <Loading label="Loading complete tract-length distributions" />;
  }
  if (summary.isError) return <ErrorState error={summary.error} />;
  if (distributions.isError) {
    return <ErrorState error={distributions.error} />;
  }
  if (ranges.isError) return <ErrorState error={ranges.error} />;
  if (!summary.data || !distributions.data || !ranges.data) return null;
  const summaryRows = summary.data.data.map((row) => ({
    Cohort: row.Panel,
    Cases: row.n_subjects,
    "Positive edges": row.n_edges,
    "Median length (mm)": row.median,
    "Q1 (mm)": row.q1,
    "Q3 (mm)": row.q3,
    "IQR (mm)": row.iqr,
    "Lower fence (mm)": row.lower_fence,
    "Upper fence (mm)": row.upper_fence,
  }));
  const groupDistributions = distributions.data.filter(
    (item) => item.data.panel !== "Overall",
  );

  return (
    <>
      <div className="tract-range-strip">
        <div>
          <strong>POPULATION MEDIAN</strong>
          <span>
            {formatNumber(
              ranges.data.data.population_median_mm,
              2,
            )}{" "}
            mm
          </span>
        </div>
        {ranges.data.data.ranges.map((range) => (
          <div key={range.code}>
            <strong>{range.code}</strong>
            <span>{range.label}</span>
          </div>
        ))}
      </div>
      <div className="plot-grid tract-grid">
        {distributions.data.map((item) => {
          const data = item.data;
          const stats = data.full_summary;
          const shapes: Partial<Shape>[] = [
            { x: stats.q1, dash: "dot" as const, width: 1.5 },
            { x: stats.median, dash: "solid" as const, width: 2.5 },
            { x: stats.q3, dash: "dot" as const, width: 1.5 },
          ].map(({ x, dash, width }) => ({
            type: "line",
            x0: x,
            x1: x,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#e2e8f0", dash, width },
          }));
          return (
            <Plot
              key={data.panel}
              className="compact-plot"
              data={[
                {
                  type: "histogram",
                  x: data.values,
                  nbinsx: 36,
                  marker: { color: GROUP_COLORS[data.panel] },
                  opacity: 0.84,
                  hovertemplate:
                    "Length=%{x:.2f} mm<br>Sampled edges=%{y}<extra></extra>",
                } as Data,
              ]}
              layout={{
                height: 315,
                title: {
                  text: `${data.panel.toUpperCase()} · median ${formatNumber(stats.median, 2)} mm`,
                  x: 0.03,
                  xanchor: "left",
                  font: { size: 14 },
                },
                shapes,
                bargap: 0.03,
                xaxis: { title: { text: "Length (mm)" } },
                yaxis: { title: { text: "Sampled edges" } },
                showlegend: false,
              }}
            />
          );
        })}
        <Plot
          className="compact-plot"
          data={groupDistributions.map(
            (item) =>
              ({
                type: "box",
                name: item.data.panel,
                y: item.data.values,
                marker: { color: GROUP_COLORS[item.data.panel] },
                line: { color: GROUP_COLORS[item.data.panel] },
                fillcolor: `${GROUP_COLORS[item.data.panel]}66`,
                boxpoints: false,
                hovertemplate:
                  `${item.data.panel}<br>Length=%{y:.2f} mm<extra></extra>`,
              }) as Data,
          )}
          layout={{
            height: 315,
            title: {
              text: "CN / MCI / AD PERCENTILES",
              x: 0.03,
              xanchor: "left",
              font: { size: 14 },
            },
            yaxis: { title: { text: "Length (mm)" } },
            showlegend: false,
          }}
        />
      </div>
      <DataTable
        rows={summaryRows}
        maxHeight={300}
        initialRows={10}
        compact
      />
      <p className="method-note">
        Histograms use deterministic display samples. Every displayed median,
        quartile and fence uses all 5,408,318 loaded positive edge
        observations.
      </p>
    </>
  );
}

function ExceptionPlot({
  inference,
  shortLabel,
}: {
  inference: ExceptionInference;
  shortLabel: string;
}) {
  const ordinary = inference.plot_rows.filter(
    (row) => row["Edge class"] === "Non-exception",
  );
  const exceptional = inference.plot_rows.filter(
    (row) => row["Edge class"] === "Exception",
  );
  const trimmedEdgeRows = visuallyTrimmed([
    ...ordinary,
    ...exceptional,
  ]);
  const edgeTraces = ["CN", "MCI", "AD"].map((group) => {
    const nonException = trimmedEdgeRows
      .filter(
        (row) =>
          row.Group === group &&
          row["Edge class"] === "Non-exception",
      )
      .map((row) => Number(row["Subject median"]));
    const exception = trimmedEdgeRows
      .filter(
        (row) =>
          row.Group === group &&
          row["Edge class"] === "Exception",
      )
      .map((row) => Number(row["Subject median"]));
    return {
      type: "box",
      name: group,
      x: [
        ...nonException.map(() => `${group}<br>Non-exception`),
        ...exception.map(() => `${group}<br>Exception`),
      ],
      y: [...nonException, ...exception],
      marker: { color: GROUP_COLORS[group] },
      line: { color: GROUP_COLORS[group], width: 2 },
      fillcolor: `${GROUP_COLORS[group]}55`,
      boxpoints: false,
      hovertemplate:
        `${group}<br>%{x}<br>Subject median=%{y:.5g}<extra></extra>`,
    } as Data;
  });
  const edgeAnnotations = ["CN", "MCI", "AD"].flatMap((group) =>
    [
      ["Non-exception", ordinary],
      ["Exception", exceptional],
    ].flatMap(([edgeClass, rows]) => {
      const value = median(
        (rows as ExceptionInference["plot_rows"])
          .filter((row) => row.Group === group)
          .map((row) => Number(row["Subject median"])),
      );
      return value === null
        ? []
        : [
            {
              x: `${group}<br>${edgeClass}`,
              // Plotly annotations on a logarithmic axis use log10
              // coordinates rather than the raw data value.
              y: Math.log10(value),
              text: `median ${formatNumber(value, 4)}`,
              showarrow: false,
              yshift: 14,
              font: { color: "#f8fafc", size: 10 },
              bgcolor: "rgba(7,10,16,0.78)",
              bordercolor: GROUP_COLORS[group],
              borderpad: 3,
            },
          ];
    }),
  );
  return (
    <div className="plot-grid">
      <Plot
        data={edgeTraces}
        layout={{
          height: 500,
          title: {
            text: "LR STRENGTH BY DIAGNOSIS AND EDGE CLASS",
            font: { size: 14 },
          },
          annotations: edgeAnnotations,
          yaxis: {
            title: { text: shortLabel },
            type: "log",
          },
          legend: { orientation: "h" },
          margin: { b: 82 },
        }}
      />
      <p className="method-note plot-grid-note">
        The panel compares exception and non-exception strengths directly; it
        does not subtract one from the other or form a ratio. The boxes apply a
        per-group and edge-class 1.5-IQR visual trim so extreme points do not
        compress the distributions. Every displayed median and inferential
        test still uses the complete subject cohort.
      </p>
    </div>
  );
}

function InteractionPanel({
  inference,
}: {
  inference: ExceptionInference;
}) {
  const paired = new Map<
    string,
    {
      group: string;
      nonException?: number;
      exception?: number;
    }
  >();
  inference.plot_rows.forEach((row) => {
    if (
      row["Edge class"] !== "Non-exception" &&
      row["Edge class"] !== "Exception"
    ) {
      return;
    }
    const current = paired.get(row.Subject) ?? { group: row.Group };
    if (row["Edge class"] === "Exception") {
      current.exception = Number(row["Subject median"]);
    } else {
      current.nonException = Number(row["Subject median"]);
    }
    paired.set(row.Subject, current);
  });
  const interactionSubjects = Array.from(paired.entries()).flatMap(
    ([subject, row]) => {
      const exception = Number(row.exception);
      const nonException = Number(row.nonException);
      if (
        !Number.isFinite(exception) ||
        !Number.isFinite(nonException) ||
        exception <= 0 ||
        nonException <= 0
      ) {
        return [];
      }
      return [
        {
          subject,
          group: row.group,
          logContrast: Math.log(exception) - Math.log(nonException),
          multiplier: exception / nonException,
        },
      ];
    },
  );
  const displayedInteractionSubjects = ["CN", "MCI", "AD"].flatMap(
    (group) => {
      const groupRows = interactionSubjects.filter(
        (row) => row.group === group,
      );
      const values = groupRows
        .map((row) => row.multiplier)
        .sort((a, b) => a - b);
      if (values.length < 4) return groupRows;
      const quantile = (position: number) => {
        const index = (values.length - 1) * position;
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        return (
          values[lower] +
          (values[upper] - values[lower]) * (index - lower)
        );
      };
      const q1 = quantile(0.25);
      const q3 = quantile(0.75);
      const iqr = q3 - q1;
      return groupRows.filter(
        (row) =>
          row.multiplier >= q1 - 1.5 * iqr &&
          row.multiplier <= q3 + 1.5 * iqr,
      );
    },
  );
  const pairwise = inference.interaction_rows.filter(
    (row) => row.Test === "Mann-Whitney U",
  );
  const interactionAnnotations = ["CN", "MCI", "AD"].flatMap((group) => {
    const value = median(
      interactionSubjects
        .filter((row) => row.group === group)
        .map((row) => row.multiplier),
    );
    return value === null
      ? []
      : [
          {
            x: group,
            // The displayed multiplier is on a logarithmic axis.
            y: Math.log10(value),
            text: `median ${formatNumber(value, 2)}×`,
            showarrow: false,
            yshift: 16,
            font: { color: "#f8fafc", size: 11 },
            bgcolor: "rgba(7,10,16,0.82)",
            bordercolor: GROUP_COLORS[group],
            borderpad: 3,
          },
        ];
  });
  const pairwiseRows = pairwise.map((row) => {
    const first = String(row["First group"] ?? "");
    const second = String(row["Second group"] ?? "");
    const direction = String(row["Observed direction"] ?? "");
    const higher = direction.replace(" higher", "");
    const lower = higher === first ? second : first;
    return {
      Contrast: row.Contrast,
      N: row.N,
      "Median separation, first":
        Number.isFinite(Number(row["Median A"]))
          ? `${formatNumber(Math.exp(Number(row["Median A"])), 3)}×`
          : "—",
      "Median separation, second":
        Number.isFinite(Number(row["Median B"]))
          ? `${formatNumber(Math.exp(Number(row["Median B"])), 3)}×`
          : "—",
      Interpretation:
        higher && lower
          ? `${higher} has a larger exception-to-non-exception strength separation than ${lower}.`
          : "No directional interpretation available.",
      "Cliff delta": row.Effect,
      "Pairwise Holm p": row["Adjusted p"],
    };
  });
  return (
    <>
      <div className="interaction-results-grid">
        <Plot
          data={["CN", "MCI", "AD"].map(
            (group) =>
              ({
                type: "box",
                name: group,
                y: displayedInteractionSubjects
                  .filter((row) => row.group === group)
                  .map((row) => row.multiplier),
                customdata: displayedInteractionSubjects
                  .filter((row) => row.group === group)
                  .map((row) => row.subject),
                marker: { color: GROUP_COLORS[group] },
                line: { color: GROUP_COLORS[group], width: 2 },
                fillcolor: `${GROUP_COLORS[group]}55`,
                boxpoints: false,
                hovertemplate:
                  `${group}<br>Subject=%{customdata}<br>Exception/non-exception strength separation=%{y:.3f}×<extra></extra>`,
              }) as Data,
          )}
          layout={{
            height: 450,
            title: {
              text: "SUBJECT-LEVEL EDGE-CLASS SEPARATION BY DIAGNOSIS",
              font: { size: 14 },
            },
            annotations: interactionAnnotations,
            yaxis: {
              title: {
                text: "Exception / non-exception strength (×)",
              },
              type: "log",
            },
            showlegend: false,
          }}
        />
        <div className="interaction-table-column">
          <DataTable
            rows={pairwiseRows}
            columns={[
              "Contrast",
              "N",
              "Median separation, first",
              "Median separation, second",
              "Interpretation",
              "Cliff delta",
              "Pairwise Holm p",
            ]}
            rowClassName={(row) =>
              row.Contrast === "CN vs AD"
                ? "table-row-significant"
                : undefined
            }
            initialRows={10}
            maxHeight={null}
            compact
          />
          <p className="significance-key">
            Yellow row: Holm-adjusted p &lt; 0.05.
          </p>
        </div>
      </div>
      <p className="method-note">
        The interaction box plot uses a per-diagnosis 1.5-IQR visual trim so
        extreme multipliers do not compress the display. Median labels and all
        tests use the complete 530-subject cohort.
      </p>
    </>
  );
}

function ExceptionSection() {
  const [measure, setMeasure] = useState("fd_sum");
  const config = useQuery({
    queryKey: ["edr-config"],
    queryFn: ({ signal }) =>
      getJson<ExceptionConfig>("/lr-sr/exceptions/config", signal),
  });
  const summary = useQuery({
    queryKey: ["edr-summary", measure],
    queryFn: ({ signal }) =>
      getJson<ExceptionSummary>(
        `/lr-sr/exceptions/summary?measure=${measure}`,
        signal,
      ),
  });
  const inference = useQuery({
    queryKey: ["edr-inference", measure],
    queryFn: ({ signal }) =>
      getJson<ExceptionInference>(
        `/lr-sr/exceptions/inference?measure=${measure}&table=full`,
        signal,
      ),
  });
  if (config.isLoading) return <Loading label="Loading EDR definition" />;
  if (config.isError) return <ErrorState error={config.error} />;
  if (!config.data) return null;
  const definition = config.data.data.measures[measure];
  const pairwiseInference =
    inference.data?.data.rows
      .filter(
        (row) =>
          row.Question === "Across groups" &&
          row.Test === "Mann-Whitney U" &&
          ["Exception strength", "Non-exception strength"].includes(
            String(row.Metric),
          ),
      )
      .map((row) => ({
        Metric: row.Metric,
        Contrast: row.Contrast,
        N: row.N,
        "Median first group": row["Median A"],
        "Median second group": row["Median B"],
        "Median difference": row["Median difference"],
        Interpretation: explicitInference(row),
        "Cliff delta": row.Effect,
        "Pairwise Holm p": row["Adjusted p"],
      })) ?? [];
  return (
    <>
      <div className="control-row">
        <label className="control grow">
          <span>Exception basis</span>
          <select
            value={measure}
            onChange={(event) => setMeasure(event.target.value)}
          >
            {Object.entries(config.data.data.measures).map(([key, info]) => (
              <option key={key} value={key}>
                {info.label}
              </option>
            ))}
          </select>
        </label>
        <div className="definition-callout">
          <strong>{definition.status}</strong>
          <span>{definition.interpretation}</span>
        </div>
      </div>
      <p className="section-intro">
        {config.data.data.rule} The exception calculation is performed
        separately for every subject; group labels are used only after
        subject-level summaries have been created.
      </p>

      {summary.isLoading || inference.isLoading ? (
        <Loading label="Computing cached EDR summaries" />
      ) : summary.isError ? (
        <ErrorState error={summary.error} />
      ) : inference.isError ? (
        <ErrorState error={inference.error} />
      ) : summary.data && inference.data ? (
        <>
          <div className="plot-grid two">
            <Plot
              data={["CN", "MCI", "AD"].map((group) => {
                const rows = summary.data!.data.rows.filter(
                  (row) =>
                    String(row.Cohort) === group &&
                    ["SR", "MR", "LR"].includes(String(row.Range)),
                );
                return {
                  type: "bar",
                  name: group,
                  x: rows.map((row) => String(row.Range)),
                  y: rows.map((row) => Number(row["Pooled exception (%)"])),
                  marker: { color: GROUP_COLORS[group] },
                  text: rows.map(
                    (row) =>
                      `${formatNumber(row["Pooled exception (%)"], 2)}%`,
                  ),
                  textposition: "outside",
                  hovertemplate:
                    `${group}<br>%{x}<br>Pooled rate=%{y:.3f}%<extra></extra>`,
                } as Data;
              })}
              layout={{
                height: 390,
                barmode: "group",
                title: {
                  text: "EXCEPTION-RATE PROFILE",
                  font: { size: 14 },
                },
                yaxis: { title: { text: "Pooled exception rate (%)" } },
                legend: { orientation: "h" },
              }}
            />
            <Plot
              data={["CN", "MCI", "AD"].map((group) => {
                const rows = summary.data!.data.rows.filter(
                  (row) =>
                    String(row.Cohort) === group &&
                    ["SR", "MR", "LR"].includes(String(row.Range)),
                );
                return {
                  type: "scatter",
                  mode: "lines+markers",
                  name: group,
                  x: rows.map((row) => String(row.Range)),
                  y: rows.map((row) =>
                    Number(row["Median subject rate (%)"]),
                  ),
                  line: { color: GROUP_COLORS[group], width: 3 },
                  marker: { color: GROUP_COLORS[group], size: 8 },
                  hovertemplate:
                    `${group}<br>%{x}<br>Median subject rate=%{y:.3f}%<extra></extra>`,
                } as Data;
              })}
              layout={{
                height: 390,
                title: {
                  text: "SUBJECT-WEIGHTED EXCEPTION PROFILE",
                  font: { size: 14 },
                },
                yaxis: { title: { text: "Median subject rate (%)" } },
                legend: { orientation: "h" },
              }}
            />
          </div>
          <DataTable
            rows={summary.data.data.rows}
            columns={[
              "Cohort",
              "Range",
              "Cases",
              "Candidate edges",
              "Median non-exception measure",
              "Median exception measure",
              "Exception edges",
              "Pooled exception (%)",
              "Median exceptions / case",
              "Median subject rate (%)",
            ]}
            headerGroups={[
              {
                label: "STRENGTH MEASURES",
                columns: [
                  "Median non-exception measure",
                  "Median exception measure",
                ],
                tone: "strength",
              },
              {
                label: "EXCEPTION BURDEN / RATE",
                columns: [
                  "Exception edges",
                  "Pooled exception (%)",
                  "Median exceptions / case",
                  "Median subject rate (%)",
                ],
                tone: "burden",
              },
            ]}
            initialRows={20}
            maxHeight={540}
            compact
          />
          <p className="method-note">
            The three exception-burden fields are different: pooled rate
            weights every candidate edge equally; median count/case is a count;
            median subject rate weights every subject equally.
          </p>

          <SectionHeading
            index="2A"
            title="LR exception versus non-exception strength across CN, MCI and AD"
          />
          <ExceptionPlot
            inference={inference.data.data}
            shortLabel={definition.short_label}
          />
          <DataTable
            rows={pairwiseInference}
            columns={[
              "Metric",
              "Contrast",
              "N",
              "Median first group",
              "Median second group",
              "Median difference",
              "Interpretation",
              "Cliff delta",
              "Pairwise Holm p",
            ]}
            maxHeight={520}
            initialRows={20}
            compact
          />
          <p className="method-note">
            All three diagnostic contrasts are shown separately for exception
            strength and non-exception strength. The diagnosis-specific
            exception-versus-non-exception question is tested in the
            interaction section below rather than by displaying a raw
            subtraction. The visible pairwise p-values are Holm-corrected
            within each three-pair metric family. Analysis remains exploratory
            and unadjusted for age, sex, acquisition, or site.
          </p>

          <details className="definitions-box">
            <summary>Definitions and statistical interpretation</summary>
            <ul>
              <li>
                <strong>Candidate edges:</strong> positive finite edges in the
                stated subject-specific EDR length range.
              </li>
              <li>
                <strong>Exception edges:</strong> candidates above their
                adaptive-bin mean plus three sample standard deviations.
              </li>
              <li>
                <strong>Non-exception / exception measure:</strong> median
                selected edge measure within each subject, then the group
                median of those subject medians.
              </li>
              <li>
                <strong>Requested contrast:</strong> Mann–Whitney U with Cliff
                delta, Holm-corrected across the four prespecified strength
                questions.
              </li>
              <li>
                <strong>Why per subject:</strong> this avoids treating millions
                of edges from the same person as independent observations.
              </li>
            </ul>
          </details>

          <SectionHeading
            index="2B"
            title="Does exception strength change disproportionately with diagnosis?"
          />
          <InteractionPanel inference={inference.data.data} />

        </>
      ) : null}
    </>
  );
}

function ModellingFeatureSection() {
  const [family, setFamily] = useState("");
  const [status, setStatus] = useState("");
  const [search, setSearch] = useState("");
  const families = useQuery({
    queryKey: ["feature-families"],
    queryFn: ({ signal }) =>
      getJson<FeatureFamilies>("/models/features/families", signal),
  });
  const eda = useQuery({
    queryKey: ["feature-eda", family, status, search],
    queryFn: ({ signal }) => {
      const params = new URLSearchParams({ limit: "500" });
      if (family) params.set("family", family);
      if (status) params.set("status", status);
      if (search) params.set("search", search);
      return getJson<FeatureEda>(
        `/models/features/eda?${params.toString()}`,
        signal,
      );
    },
  });
  const mapping = useQuery({
    queryKey: ["network-mapping"],
    queryFn: ({ signal }) =>
      getJson<NetworkMapping>("/networks/mapping", signal),
  });
  if (families.isLoading || eda.isLoading || mapping.isLoading) {
    return <Loading label="Loading model-feature audit" />;
  }
  if (families.isError) return <ErrorState error={families.error} />;
  if (eda.isError) return <ErrorState error={eda.error} />;
  if (mapping.isError) return <ErrorState error={mapping.error} />;
  if (!families.data || !eda.data || !mapping.data) return null;
  const familyOptions = families.data.data.families.map((row) =>
    String(row["Requested feature family"]),
  );
  return (
    <>
      <div className="metric-strip">
        <div>
          <span>Candidate inputs</span>
          <strong>
            {families.data.data.candidate_features.toLocaleString()}
          </strong>
        </div>
        <div>
          <span>Subjects</span>
          <strong>{families.data.data.subjects.toLocaleString()}</strong>
        </div>
        <div>
          <span>Displayed after filter</span>
          <strong>{eda.data.data.total_rows.toLocaleString()}</strong>
        </div>
        <div>
          <span>Network binding</span>
          <strong className="status-warning">Mapped, not direct</strong>
        </div>
      </div>
      <SectionHeading
        index="3A"
        title="High-level metrics captured by each feature family"
      />
      <div className="feature-family-inventory">
        <DataTable
          rows={families.data.data.metric_inventory}
          maxHeight={null}
          initialRows={20}
          compact
          listColumns={[
            "High-level metrics captured",
            "Representation",
            "Actual source blocks",
          ]}
        />
      </div>
      <SectionHeading
        index="3B"
        title="Candidate feature EDA"
      />
      <div className="feature-filter-grid">
        <label className="control">
          <span>Feature family</span>
          <select
            value={family}
            onChange={(event) => setFamily(event.target.value)}
          >
            <option value="">All families</option>
            {familyOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label className="control">
          <span>Model status</span>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value)}
          >
            <option value="">All statuses</option>
            <option value="selected_in_cv">Selected in CV</option>
            <option value="retained_not_selected">
              Retained, not selected
            </option>
            <option value="dropped_correlated">Dropped: correlated</option>
            <option value="dropped_low_variance">
              Dropped: low variance
            </option>
            <option value="dropped_missingness">
              Dropped: missingness
            </option>
            <option value="dropped_sparse">Dropped: sparse</option>
          </select>
        </label>
        <label className="control">
          <span>Feature name</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="e.g. lr_exception"
          />
        </label>
      </div>
      <DataTable
        rows={eda.data.data.rows}
        columns={[
          "Feature",
          "Dashboard family",
          "Feature group",
          "Model status",
          "Selected CV folds",
          "n",
          "Variance",
          "SD",
          "Nulls",
          "Nulls (%)",
          "Zeros",
          "Zeros (%)",
        ]}
        maxHeight={560}
        initialRows={100}
        compact
        caption={`Showing ${eda.data.data.rows.length.toLocaleString()} of ${eda.data.data.total_rows.toLocaleString()} matched features`}
      />
      <p className="method-note">
        Candidate inputs are columns presented to preprocessing and selection.
        “Selected in CV” is a fold-selection record, not a causal or biological
        importance claim. Variance and SD use sample ddof = 1.
      </p>

      <SectionHeading
        index="3C"
        title="AAL3 anatomical and functional-network mapping"
      />
      <div className="mapping-warning">
        <strong>Mapping-sensitive:</strong> {mapping.data.data.status}. The
        functional-network crosswalk is not a native voxelwise AAL3-to-Yeo
        overlap and structural tractography does not establish functional
        connectivity.
      </div>
      <div className="plot-grid two">
        <DataTable
          rows={mapping.data.data.functional}
          maxHeight={360}
          initialRows={20}
          compact
          caption="Approximate functional crosswalk"
        />
        <DataTable
          rows={mapping.data.data.anatomical}
          maxHeight={360}
          initialRows={20}
          compact
          caption="Deterministic anatomical grouping"
        />
      </div>
    </>
  );
}

const TOP_COUNT_COLUMN = "Top-3 count";
const FAMILY_COLORS: Record<string, string> = {
  "F1 whole-brain": "#2a78d6",
  "F2 SR/LR": "#eb6834",
  "F3 exceptions": "#1baf7a",
  "F4 architecture": "#eda100",
  covariate: "#94a3b8",
};
const BUCKET_COLORS = ["#eb6834", "#eda100", "#2a78d6"];
const PRETTY_RANGE: Record<string, string> = {
  "(8.999, 27.0]": "MMSE 9–27",
  "(27.0, 29.0]": "MMSE 28–29",
  "(29.0, 30.0]": "MMSE 30",
};
const EXPLAIN_TARGETS = [
  { key: "mmse", cg: "mmse_band", title: "MMSE — three severity bands", short: "MMSE", metric: "pooled out-of-fold R²" },
  { key: "cdr_bin", cg: "cdr_3level", title: "CDR — three levels (0 / 0.5 / ≥ 1)", short: "CDR", metric: "pooled out-of-fold AUC" },
];
const ICE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"];

/** The API sends JSON null for a missing q; guard it explicitly. Without this,
 * `null < 0.001` coerces to 0 and a missing value renders as "***". */
function isRealNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Compact q-value for in-cell display: scientific below 0.001, else 3 decimals. */
function formatPValue(q: number | null | undefined): string {
  if (!isRealNumber(q)) return "—";
  if (q < 0.001) return q.toExponential(1);
  return q.toFixed(3);
}

function significanceStars(q: number | null | undefined): string {
  if (!isRealNumber(q)) return "ns";
  if (q < 0.001) return "***";
  if (q < 0.01) return "**";
  if (q < 0.05) return "*";
  return "ns";
}

function RankedGroupBlock({
  group,
  networks,
  nSubjects,
  contrastLabel,
  secondGroup,
}: {
  group: NetworkMeasureRankedGroup;
  networks: string[];
  nSubjects: number;
  contrastLabel: string;
  secondGroup: string;
}) {
  const cellByKey = new Map(
    group.cells.map((cell) => [`${cell.network}|${cell.measure}`, cell]),
  );
  const TOP_N = 3;
  const matrixRows = networks.map((network) => {
    const row: Record<string, unknown> = { Network: network };
    let topCount = 0;
    group.measures.forEach((label) => {
      const cell = cellByKey.get(`${network}|${label}`);
      if (!cell || !cell.available) {
        row[label] = "—";
        return;
      }
      const arrow = cell.direction === "down" ? "↓" : "↑";
      row[label] =
        `#${cell.rank} ${arrow} ${significanceStars(cell.bm_q)} ` +
        `q=${formatPValue(cell.bm_q)}`;
      if (cell.rank !== undefined && cell.rank <= TOP_N) topCount += 1;
    });
    row[TOP_COUNT_COLUMN] = topCount;
    return row;
  });
  const maxTopCount = Math.max(
    ...matrixRows.map((row) => Number(row[TOP_COUNT_COLUMN]) || 0),
  );
  const headlineRows = group.cells
    .filter((cell) => cell.available && cell.significant)
    .sort(
      (first, second) =>
        Math.abs(second.cliffs_delta ?? 0) - Math.abs(first.cliffs_delta ?? 0),
    )
    .map((cell, index) => ({
      Rank: index + 1,
      Network: cell.network,
      Measure: cell.measure,
      [`Direction (${contrastLabel})`]:
        cell.direction === "down"
          ? `↓ lower in ${secondGroup}`
          : `↑ higher in ${secondGroup}`,
      [`Cliff's δ (${contrastLabel})`]: cell.cliffs_delta,
      "q (BH-FDR)": cell.bm_q,
    }));
  return (
    <div className="ranked-group">
      <h3>{group.label}</h3>
      <p className="section-intro">{group.description}</p>
      <DataTable
        rows={matrixRows}
        columns={["Network", ...group.measures, TOP_COUNT_COLUMN]}
        maxHeight={null}
        initialRows={networks.length}
        compact
        caption={`Ranked matrix — ${group.label} (n = ${nSubjects}; ${contrastLabel}). Cell = rank · direction (↓ lower in ${secondGroup}, ↑ higher in ${secondGroup}) · significance · BH-FDR q. Yellow = top ${TOP_N} rank for that measure.`}
        cellClassName={(row, column, value) => {
          if (column === TOP_COUNT_COLUMN) {
            return Number(value) === maxTopCount && maxTopCount > 0
              ? "cell-top-rank cell-top-count-max"
              : "cell-top-count";
          }
          if (column === "Network") return undefined;
          const match = /^#(\d+)/.exec(String(value ?? ""));
          return match && Number(match[1]) <= TOP_N
            ? "cell-top-rank"
            : undefined;
        }}
      />
      {headlineRows.length > 0 ? (
        <DataTable
          rows={headlineRows}
          maxHeight={null}
          initialRows={headlineRows.length}
          compact
          caption={`Significant signals — ${group.label}, ranked by ${contrastLabel} effect size`}
        />
      ) : (
        <p className="section-intro">
          <strong>No network survives correction in this group.</strong> Every
          network × measure cell is non-significant after BH-FDR, so this
          effect is not localizable to any single functional network.
        </p>
      )}
    </div>
  );
}

function NetworkRankedSection() {
  const [contrast, setContrast] = useState("cn_ad");
  const ranked = useQuery({
    queryKey: ["network-measure-ranked", contrast],
    queryFn: ({ signal }) =>
      getJson<NetworkMeasureRanked>(
        `/networks/measure-ranked?contrast=${contrast}`,
        signal,
      ),
  });
  if (ranked.isLoading) {
    return <Loading label="Loading network x measure ranked matrix" />;
  }
  if (ranked.isError) return <ErrorState error={ranked.error} />;
  if (!ranked.data) return null;
  const payload = ranked.data.data;
  const [firstGroup, secondGroup] = payload.contrast_label.split("-");
  return (
    <>
      <div className="control-row">
        <label className="control grow">
          <span>Group contrast</span>
          <select
            value={contrast}
            onChange={(event) => setContrast(event.target.value)}
          >
            {payload.contrasts.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="definition-callout">
          <strong>{payload.contrast_label}</strong>
          <span>
            Rank, direction and q below are all for this contrast. ↓ = lower in{" "}
            {secondGroup}, ↑ = higher in {secondGroup}.
          </span>
        </div>
      </div>
      <p className="section-intro">
        Nested views of the same question — the whole connectome, then only
        SR/LR tracts, then only the SR/LR EDR exceptions. Each cell is{" "}
        <strong>rank · direction · significance · q</strong>: the
        within-measure rank by {payload.contrast_label} |Cliff&apos;s δ| (#1 =
        largest effect); the direction from {firstGroup} to {secondGroup},
        where <strong>↓ means lower in {secondGroup}</strong> and{" "}
        <strong>↑ means higher in {secondGroup}</strong> (so FA reads ↓ and
        MD/RD/AxD read ↑); and the Brunner–Munzel {payload.contrast_label}{" "}
        significance with its BH-FDR
        q-value, corrected within each measure across the 10 functional
        networks (*** &lt;0.001, ** &lt;0.01, * &lt;0.05). Rank, direction and
        q are all rank-based, so they always agree. Ranges are the
        CN-referenced Q1/Q3 cutoffs and exceptions follow the Chakraborty et
        al. mean+3SD-at-distance rule, so every group uses the same
        aggregation and the same tests.
      </p>
      {payload.groups.map((group) => (
        <RankedGroupBlock
          key={group.key}
          group={group}
          networks={payload.networks}
          nSubjects={payload.n_subjects}
          contrastLabel={payload.contrast_label}
          secondGroup={secondGroup}
        />
      ))}
    </>
  );
}


function ExceptionSpecificitySection() {
  // all hooks must run before any early return (React rules of hooks)
  const [contrast, setContrast] = useState("cn_ad");
  const spec = useQuery({
    queryKey: ["exception-specificity"],
    queryFn: ({ signal }) =>
      getJson<ExceptionSpecificity>("/networks/exception-specificity", signal),
  });
  if (spec.isLoading) return <Loading label="Loading exception-specificity artifacts" />;
  if (spec.isError) return <ErrorState error={spec.error} />;
  if (!spec.data) return null;
  const payload = spec.data.data;
  const summary = payload.tier_delta_summary ?? [];
  const delta = (payload.tier_delta ?? []).filter((r) => r.contrast === contrast);
  const nets = Array.from(new Set(delta.map((r) => String(r.network))));
  const cols = Array.from(new Set(delta.map((r) => `${r.range}-${r.measure}`)));
  const z = nets.map((n) =>
    cols.map((c) => {
      const row = delta.find((r) => String(r.network) === n && `${r.range}-${r.measure}` === c);
      return row ? Number(row.gain_vs_range) : null;
    }),
  );
  const arch = (payload.architecture ?? []).map((r) => {
    const q = Number(r[`${contrast}_bm_q`]);
    const a3 = Number(r[`adni3_${contrast}_p`]);
    return {
      Feature: r.feature,
      "δ (CN-MCI)": r.cn_mci_cliffs_delta,
      "q (CN-MCI)": r.cn_mci_bm_q,
      "δ (CN-AD)": r.cn_ad_cliffs_delta,
      "q (CN-AD)": r.cn_ad_bm_q,
      "δ (MCI-AD)": r.mci_ad_cliffs_delta,
      "q (MCI-AD)": r.mci_ad_bm_q,
      "ADNI-3 check": Number.isFinite(a3)
        ? a3 < 0.05
          ? `robust (p=${a3.toFixed(3)})`
          : `phase-sensitive (p=${a3.toFixed(3)})`
        : "—",
      _sig: Number.isFinite(q) && q < 0.05,
    };
  });
  return (
    <>
      <p className="section-intro">
        Does the exception tier carry signal the conventional spectrum misses?
        Left: the like-for-like gain in |Cliff&apos;s δ| when the same
        network × measure is computed on exception edges instead of all edges
        of the same range (blue = exceptions lose signal, red = gain). Right:
        subject-level exception-architecture features that have no
        conventional counterpart, each with an ADNI-3-only sensitivity check
        (the acquisition-phase confound). Below: simple, leakage-guarded
        MMSE/CDR models (repeated 5×5-fold CV; ElasticNet-Huber,
        HistGradientBoosting, ExtraTrees) on the incremental feature ladder
        F0 covariates → +F1 whole-brain → +F2 SR/LR → +F3 exceptions →
        +F4 exception architecture.
      </p>
      <div className="control-row">
        <label className="control grow">
          <span>Contrast</span>
          <select value={contrast} onChange={(e) => setContrast(e.target.value)}>
            <option value="cn_mci">CN-MCI</option>
            <option value="cn_ad">CN-AD</option>
            <option value="mci_ad">MCI-AD</option>
          </select>
        </label>
        {summary.map((s) => (
          <div key={s.contrast} className="definition-callout">
            <strong>{s.contrast.replace("_", "-").toUpperCase()}</strong>
            <span>
              mean Δ|δ| {s.mean_gain_vs_range.toFixed(3)} · {s.pop_ups} pop-ups ·{" "}
              {s.drop_outs} drop-outs / {s.cells} cells
            </span>
          </div>
        ))}
      </div>
      {nets.length > 0 ? (
        <Plot
          data={[
            {
              type: "heatmap",
              z,
              x: cols,
              y: nets,
              zmid: 0,
              colorscale: [
                [0, "#2a78d6"],
                [0.5, "#f5f7fa"],
                [1, "#c0392b"],
              ],
              colorbar: { title: { text: "Δ|δ| exc − range" } },
              hovertemplate: "%{y} · %{x}<br>gain=%{z:.3f}<extra></extra>",
            } as Data,
          ]}
          layout={{
            height: 420,
            title: {
              text: `Exception-vs-range gain in |Cliff's δ| (${contrast.replace("_", "-").toUpperCase()})`,
              x: 0.02,
              xanchor: "left",
              font: { size: 14 },
            },
            margin: { l: 120, b: 110 },
            xaxis: { tickangle: -40 },
          }}
        />
      ) : null}
      <DataTable
        rows={arch.map(({ _sig, ...rest }) => rest)}
        maxHeight={null}
        initialRows={arch.length}
        compact
        caption="Exception-architecture features (no conventional counterpart). ADNI-3 check flags rows that do not survive the phase-restricted cohort."
        cellClassName={(row, column) =>
          column === "ADNI-3 check" && String(row[column] ?? "").startsWith("robust")
            ? "cell-top-rank"
            : undefined
        }
      />
      <p className="section-intro">
        The cognition-model workspace — target distributions, feature
        engineering, the MMSE/CDR models, SHAP explanations and diagnostics —
        lives on its own page:{" "}
        <a href="#/section/lr-sr-models">LR-SR Models</a>.
      </p>
    </>
  );
}


function CognitionModelsBody() {
  const spec = useQuery({
    queryKey: ["exception-specificity"],
    queryFn: ({ signal }) =>
      getJson<ExceptionSpecificity>("/networks/exception-specificity", signal),
  });
  if (spec.isLoading) return <Loading label="Loading model artifacts" />;
  if (spec.isError) return <ErrorState error={spec.error} />;
  if (!spec.data) return null;
  const payload = spec.data.data;
  const perm = payload.ml_permutation ?? {};
  return (
    <>
      {payload.pipeline_steps ? (
        <>
          <h3>Feature engineering &amp; preprocessing</h3>
          <p className="section-intro">
            Every step below is applied in this order, and anything fitted from
            data (imputation, scaling, selection) is fitted inside each
            training fold only.
            {payload.selection_experiment ? (
              <>
                {" "}In-fold univariate selection (k = 40) was tested and not
                adopted:{" "}
                {payload.selection_experiment
                  .map(
                    (r) =>
                      `${r.target === "cdr_bin" ? "CDR" : "MMSE"} ${r.metric} ${r.selection === "none" ? "without" : "with"} selection = ${r.value.toFixed(3)}`,
                  )
                  .join("; ")}
                .
              </>
            ) : null}
          </p>
          <DataTable
            rows={(payload.pipeline_steps ?? []).map((r) => ({
              "#": r.order, Step: r.step, Performed: r.performed, Detail: r.detail,
            }))}
            maxHeight={null}
            initialRows={12}
            compact
            caption="The modelling pipeline, in order"
          />
          <div className="plot-grid two">
            <DataTable
              rows={(payload.feature_dictionary ?? []).map((r) => ({
                Feature: r.feature.replace(/_/g, " "), Family: r.family, Construction: r.construction,
              }))}
              maxHeight={420}
              initialRows={12}
              compact
              caption={`Feature dictionary — all ${payload.feature_dictionary?.length ?? 0} model inputs (Appendix A of the thesis)`}
            />
            <DataTable
              rows={(payload.preprocessing_audit ?? [])
                .slice()
                .sort((a, b) => Math.abs(Number(b.skew)) - Math.abs(Number(a.skew)))
                .slice(0, 12)
                .map((r) => ({
                  Feature: String(r.feature).replace(/_/g, " "),
                  "missing %": r.missing_pct, Skew: r.skew,
                  "outliers |z|>4": r.outliers_gt4rz, "max |ρ|": r.max_abs_r,
                }))}
              maxHeight={420}
              initialRows={12}
              compact
              caption="Distribution audit — the 12 most skewed features (rank metrics and tree models are used precisely because of this skew)"
            />
          </div>
        </>
      ) : null}
          <h3>Target distributions</h3>
          <p className="section-intro">
            The Clinical Dementia Rating is a discrete scale — it can only take
            the values 0, 0.5, 1, 2 or 3, so there is no score between 0 and
            0.5 and the binary split “0 vs ≥ 0.5” partitions every subject:
            “CDR 0” <em>is</em> the “below 0.5” class. Levels 1 and above hold
            only {(payload.cdr_level_counts ?? []).filter((r) => r.level >= 1).reduce((a, r) => a + r.total, 0)}{" "}
            subjects — CDR 2 has six and CDR 3 has one, so they cannot be
            stratified into five folds, let alone estimated per-class. A
            three-level ordinal alternative (0 / 0.5 / ≥ 1) is viable and
            informative: the frank-dementia class separates best of all
            (AUC 0.80), the questionable-dementia class (0.5) sits near
            chance (0.55) — the same extremes-clear, boundary-fuzzy pattern
            as the MMSE bands. The binary 0 vs ≥ 0.5 split remains the
            primary target for balance (106/96); the three-level results appear in the
            model-improvement section below, side by side with MMSE. MMSE severity bands are equal-sized by construction
            (quantile cuts).
          </p>
          <div className="plot-grid two">
            {payload.cdr_level_counts ? (
              <Plot
                data={["CN", "MCI", "AD"].map(
                  (g) =>
                    ({
                      type: "bar",
                      name: g,
                      x: (payload.cdr_level_counts ?? []).map((r) => `CDR ${r.level}`),
                      y: (payload.cdr_level_counts ?? []).map((r) => r[g as "CN" | "MCI" | "AD"]),
                      marker: { color: GROUP_COLORS[g] },
                      text: (payload.cdr_level_counts ?? []).map((r) => r[g as "CN" | "MCI" | "AD"] || ""),
                      textposition: "inside",
                    }) as Data,
                )}
                layout={{
                  height: 400,
                  barmode: "stack",
                  title: {
                    text: "CDR levels (nearest visit) — split is 0 vs ≥ 0.5",
                    x: 0.02, xanchor: "left", font: { size: 13 },
                  },
                  margin: { b: 110 },
                  yaxis: { title: { text: "subjects" } },
                  legend: { orientation: "h", y: -0.3 },
                  shapes: [{
                    type: "line", x0: 0.5, x1: 0.5, xref: "x", y0: 0, y1: 1, yref: "paper",
                    line: { color: "#52514e", width: 2, dash: "dash" },
                  }],
                }}
              />
            ) : null}
            {payload.mmse_bucket_definition ? (
              <Plot
                data={["CN", "MCI", "AD"].map(
                  (g) =>
                    ({
                      type: "bar",
                      name: g,
                      x: (payload.mmse_bucket_definition ?? []).map((r) => `${r.label} · ${PRETTY_RANGE[r.range] ?? r.range}`),
                      y: (payload.mmse_bucket_definition ?? []).map((r) => r[g as "CN" | "MCI" | "AD"]),
                      marker: { color: GROUP_COLORS[g] },
                      text: (payload.mmse_bucket_definition ?? []).map((r) => r[g as "CN" | "MCI" | "AD"] || ""),
                      textposition: "inside",
                    }) as Data,
                )}
                layout={{
                  height: 400,
                  barmode: "stack",
                  title: {
                    text: "MMSE severity bands (equal-sized quantile cuts)",
                    x: 0.02, xanchor: "left", font: { size: 13 },
                  },
                  margin: { b: 110 },
                  yaxis: { title: { text: "subjects" } },
                  legend: { orientation: "h", y: -0.3 },
                }}
              />
            ) : null}
          </div>


      {payload.improvement_check ? (
        <>
          <h3>Model-improvement sweep (adversarially verified)</h3>
          <p className="section-intro">
            Roughly thirty configurations across five independent levers
            (boosting tuning, new model classes, ensembling, representation
            changes, target reformulation) were tested under the identical
            leakage-safe protocol, each verified by an independent re-run. One
            change survived: in-fold age-and-sex residualisation of the
            features. Its effect on each target is shown below, in the same
            left/right layout as the rest of the page.
          </p>
          <div className="plot-grid two">
            <DataTable
              rows={(payload.improvement_mmse ?? []).map((r) => ({
                Metric: r.metric,
                Baseline: r.baseline.toFixed(3),
                "Age/sex residualised": r.residualised.toFixed(3),
                Δ: (r.residualised - r.baseline >= 0 ? "+" : "") + (r.residualised - r.baseline).toFixed(3),
              }))}
              maxHeight={null}
              initialRows={4}
              compact
              caption="MMSE continuous score (regression) — confirmed improvement (better in 5 of 5 seeds; permutation p = 0.005)"
              cellClassName={(row, column) =>
                column === "Age/sex residualised" ? "cell-top-rank" : undefined
              }
            />
            <DataTable
              rows={payload.improvement_check.map((r) => ({
                Configuration: r.config.replace(/_/g, " "),
                "macro AUC": r.macro_auc.toFixed(3),
                "bal-acc": r.bal_acc.toFixed(3),
                "AUC 0": r.auc_0.toFixed(3),
                "AUC 0.5": r.auc_05.toFixed(3),
                "AUC ≥1": r.auc_ge1.toFixed(3),
              }))}
              maxHeight={null}
              initialRows={4}
              compact
              caption="CDR (three-level 0 / 0.5 / ≥1) — binary AUC 0.602 unmoved by any lever; residualisation lifts the ≥1 class"
              cellClassName={(row, column) =>
                column === "AUC ≥1" && Number(row[column]) >= 0.8 ? "cell-top-rank" : undefined
              }
            />
          </div>
        </>
      ) : null}
      {payload.ml_ladder ? (
        <>
          <p className="section-intro">
            <strong>Cohort:</strong> n = {payload.ml_cohort?.n} with
            nearest-visit scores (CDR balance{" "}
            {payload.ml_cohort?.cdr_balance?.join(" / ")}; median score-to-scan
            lag {Math.round(payload.ml_cohort?.median_lag_days ?? 0)} days —
            reported as a limitation). Permutation test on the full ladder:{" "}
            {Object.entries(perm)
              .map(
                ([t, v]) =>
                  `${t.toUpperCase()} ${v.metric.replace("_mean", "")}=${v.observed.toFixed(3)} (null ${v.null_mean.toFixed(3)}, p=${v.perm_p.toFixed(4)}, ${v.model})`,
              )
              .join(" · ")}
          </p>
          <h3>Final-model discrimination — MMSE and CDR side by side</h3>
          <p>
            One accepted model per target rather than a configuration ladder:
            in-fold median imputation, in-fold residualisation of every
            connectome feature on age and sex, then ExtraTrees (400 trees,
            balanced classes), scored out of fold over repeated 5×5
            cross-validation with the out-of-fold probabilities averaged across
            the five seeds. Both targets are three-class, so the two columns
            read as a mirror — a structural mirror, not an evidential one.
            Each AUC carries its bootstrap 95% interval (2000 resamples) on the
            row directly beneath it.
          </p>
          <p>
            Every class is scored one against the rest: first over all 202
            subjects, then <em>within</em> each diagnostic group — the same
            cohort-wide model’s probabilities subset by group, not refitted.
            <strong>
              {" "}That within-group block is a negative result, and it is the
              honest headline of this panel.
            </strong>{" "}
            Every within-group AUC that can be computed falls between 0.41 and
            0.63, and all but one of their intervals contain 0.5 (the exception
            is the impaired MMSE band within MCI, 0.633 [0.52–0.75]), so the pooled
            numbers above them are carried by separation <em>between</em>
            diagnostic groups rather than by severity grading inside one. The
            three diagnostic rows underneath say the same thing from the other
            side: the diagnosis label alone, used as nothing more than
            CN &lt; MCI &lt; AD, predicts CDR ≥ 1 at 0.897 against this
            model’s 0.804 and the impaired MMSE band at 0.779 against 0.697;
            restricted to non-AD subjects the model’s CDR ≥ 1 AUC falls to
            0.567; and each class’s own probability identifies AD better than
            it identifies the class it was trained on. The connectome model
            does not beat the label it is standing on.
          </p>
          <p>
            Two blanks appear for different reasons and are printed
            differently. “no cases” means the class does not occur in that
            group at all — no CN subject is CDR ≥ 1, no AD subject is CDR 0.
            “— (n)” means the class occurs but with only n subjects on the
            smaller side, below the threshold of ten at which the bootstrap
            interval stops spanning most of the range; the AUC computes but
            carries no information. Within AD both targets are close to
            degenerate: 23 of 24 AD subjects sit in the lowest MMSE band and 19
            of 24 are CDR ≥ 1. Note also that within CN only CDR 0 and 0.5
            occur and within AD only CDR 0.5 and ≥ 1, so in those rows the two
            populated cells are one binary comparison read from both sides, not
            two independent measurements. The last two rows are alternative
            estimators shown for robustness, always on all 202 subjects: they
            matter in both directions — logistic regression reproduces the
            CDR ≥ 1 result exactly (0.804), so that number is not an ExtraTrees
            artefact, and it edges the accepted model out on MMSE overall
            (0.602 against 0.599).
          </p>
          <div className="plot-grid two">
            {[
              {
                tkey: "mmse_band",
                caption:
                  "MMSE severity bands (classification) — final model, pooled out-of-fold. Balanced-accuracy chance = 0.333. Last five rows: three confound checks, then two alternative estimators, all on 202 subjects.",
              },
              {
                tkey: "cdr_3level",
                caption:
                  "CDR three levels (classification) — final model, pooled out-of-fold. Balanced-accuracy chance = 0.333. Last five rows: three confound checks, then two alternative estimators, all on 202 subjects.",
              },
            ].map(({ tkey, caption }) => {
              const rows = (payload.final_model_table ?? []).filter(
                (r) => r.target === tkey,
              );
              if (!rows.length) return null;
              const cols = Array.from(
                new Map<string, number>(
                  rows.map((r) => [r.col_label, r.col_order] as [string, number]),
                ),
              )
                .sort((a, b) => a[1] - b[1])
                .map(([label]) => label);
              const byRow = new Map<number, Record<string, string>>();
              rows.forEach((r) => {
                const rec = byRow.get(r.row_order) ?? { Metric: r.row_label };
                rec[r.col_label] = r.value;
                byRow.set(r.row_order, rec);
              });
              const ordered = Array.from(byRow.entries())
                .sort((a, b) => a[0] - b[0])
                .map(([, rec]) => rec);
              return (
                <DataTable
                  key={tkey}
                  rows={ordered}
                  columns={["Metric", ...cols]}
                  headerGroups={[
                    {
                      label: "PER CLASS (ONE VS REST)",
                      columns: cols.filter((c) => c !== "Macro"),
                      tone: "strength",
                    },
                  ]}
                  maxHeight={null}
                  initialRows={24}
                  compact
                  caption={caption}
                />
              );
            })}
          </div>
          {payload.final_model_curves ? (
            <>
              {["roc", "pr"].map((kind) => (
                <div className="plot-grid two" key={kind}>
                  {[
                    { tkey: "mmse_band", name: "MMSE bands", pal: BUCKET_COLORS },
                    {
                      tkey: "cdr_3level",
                      name: "CDR 0 / 0.5 / ≥ 1",
                      pal: ["#2a78d6", "#eda100", "#eb6834"],
                    },
                  ].map(({ tkey, name, pal }) => {
                    const curves = (payload.final_model_curves ?? []).filter(
                      (r) => r.target === tkey,
                    );
                    const cw = (payload.final_model_classwise ?? []).filter(
                      (r) => r.target === tkey,
                    );
                    if (!curves.length) return null;
                    const clsIds = Array.from(
                      new Set(curves.map((r) => r.cls)),
                    ).sort((a, b) => a - b);
                    const bases = clsIds.map(
                      (c) => cw.find((r) => r.cls === c)?.baseline_pr ?? 0,
                    );
                    const spread = Math.max(...bases) - Math.min(...bases);
                    const meanBase =
                      bases.reduce((a, b) => a + b, 0) / (bases.length || 1);
                    const data = clsIds.map((c) => {
                      const rows = curves
                        .filter((r) => r.kind === kind && r.cls === c)
                        .sort((a, b) => a.x - b.x);
                      const info = cw.find((r) => r.cls === c);
                      const tag =
                        kind === "roc"
                          ? `AUC ${info?.auc != null ? info.auc.toFixed(2) : "?"}`
                          : `AP ${info?.pr_auc != null ? info.pr_auc.toFixed(2) : "?"}, chance ${info ? info.baseline_pr.toFixed(2) : "?"}`;
                      return {
                        type: "scatter",
                        mode: "lines",
                        x: rows.map((r) => r.x),
                        y: rows.map((r) => r.y),
                        name: `${rows[0]?.label ?? c} (${tag})`,
                        line: { color: pal[c], width: 2 },
                      } as Data;
                    });
                    const shapes =
                      kind === "roc"
                        ? [
                            {
                              type: "line" as const,
                              x0: 0,
                              y0: 0,
                              x1: 1,
                              y1: 1,
                              line: {
                                color: "#94a3b8",
                                width: 1.5,
                                dash: "dot" as const,
                              },
                            },
                          ]
                        : spread > 0.05
                          ? clsIds.map((c, i) => ({
                              type: "line" as const,
                              x0: 0,
                              x1: 1,
                              y0: bases[i],
                              y1: bases[i],
                              line: {
                                color: pal[c],
                                width: 1,
                                dash: "dot" as const,
                              },
                            }))
                          : [
                              {
                                type: "line" as const,
                                x0: 0,
                                x1: 1,
                                y0: meanBase,
                                y1: meanBase,
                                line: {
                                  color: "#94a3b8",
                                  width: 1.5,
                                  dash: "dot" as const,
                                },
                              },
                            ];
                    const title =
                      kind === "roc"
                        ? `${name} — ROC (one vs rest); dotted diagonal = chance`
                        : spread > 0.05
                          ? `${name} — precision–recall; dotted = each class's chance level`
                          : `${name} — precision–recall; dotted = chance (all three prevalences ≈ ${meanBase.toFixed(2)})`;
                    return (
                      <Plot
                        key={tkey}
                        data={data}
                        layout={{
                          height: 340,
                          title: {
                            text: title,
                            x: 0.02,
                            xanchor: "left",
                            font: { size: 13 },
                          },
                          xaxis: {
                            title: {
                              text: kind === "roc" ? "false positive rate" : "recall",
                            },
                            range: [0, 1],
                          },
                          yaxis: {
                            title: {
                              text: kind === "roc" ? "true positive rate" : "precision",
                            },
                            range: [0, 1],
                          },
                          shapes,
                          legend: { orientation: "h", y: -0.28 },
                          margin: { b: 90 },
                        }}
                      />
                    );
                  })}
                </div>
              ))}
            </>
          ) : null}
          <div className="definition-callout">
            <strong>Reading the two metric families</strong>
            <span>
              AUC asks how well the model ranks one class above the rest and is
              0.5 at chance whatever the class sizes. PR-AUC asks how precise
              the model is where it commits, and its chance level is the class
              prevalence — the row directly beneath it — so a PR-AUC of 0.46 for
              CDR ≥ 1 against a chance level of 0.144 is a real gain, while
              0.386 for CDR 0.5 against a chance level of 0.332 is very nearly
              none. R² and Spearman ρ
              appear only where MMSE is modelled as a continuous score, since
              neither AUC nor precision–recall is defined for a continuous
              target.
            </span>
          </div>
      {payload.ml_r2 ? (
        <>
          <h3>Model explanation — MMSE and CDR side by side</h3>
          <p>
            Both columns are now explained on the same three-class targets that
            were scored above, so a class label means the same thing here as it
            does in the tables. Attribution comes from a single in-sample fit of
            a 600-tree explanation model, while the discrimination figures above
            are out-of-fold from a 400-tree model; the two answer different
            questions and their magnitudes are not comparable.
          </p>
          <h4>Overall — what pushes a subject toward impairment</h4>
          <p>
            Both panels are scored on a single impairment contrast, the lowest
            MMSE tertile against the rest and CDR ≥ 0.5 against the rest, so a
            positive SHAP value means the same thing on both sides: this
            feature pushed this subject toward the impaired class. Each row is
            labelled with the share of the model’s total attribution that the
            feature carries, measured against all 105 features rather than
            against the handful shown, and with the split of that attribution
            into the part pushing toward impairment (↑) and the part pushing
            away from it (↓). Note how flat the ranking is: no single feature
            carries more than about six per cent of the total, and for CDR the
            largest single contributor is sex. Attribution comes from one
            in-sample fit of the explanation model, whereas the discrimination
            figures above are out-of-fold.
          </p>
          <div className="plot-grid two">
            {["mmse_low", "cdr_impaired"].map((ok) => {
              const ov = (payload.shap_overall ?? []).filter((r) => r.target === ok);
              if (!ov.length) return null;
              const feats = Array.from(new Set(ov.map((r) => r.feature))).sort(
                (a, b) =>
                  (ov.find((z) => z.feature === a)?.share_pct ?? 0) -
                  (ov.find((z) => z.feature === b)?.share_pct ?? 0),
              );
              const head = ov[0];
              return (
                <Plot
                  key={ok}
                  data={feats.flatMap((f, fi) =>
                    ["CN", "MCI", "AD"].map(
                      (g) =>
                        ({
                          type: "scattergl",
                          mode: "markers",
                          name: g,
                          legendgroup: g,
                          showlegend: fi === 0,
                          x: ov
                            .filter((r) => r.feature === f && r.group === g)
                            .map((r) => r.shap),
                          y: ov
                            .filter((r) => r.feature === f && r.group === g)
                            .map((_, k) => fi + (((k * 7) % 20) / 20) * 0.5 - 0.25),
                          marker: { color: GROUP_COLORS[g], size: 5, opacity: 0.55 },
                          hovertemplate: `${g}<br>SHAP=%{x:.3f}<extra></extra>`,
                        }) as Data,
                    ),
                  )}
                  layout={{
                    height: 400,
                    title: {
                      text: `${head.short} — ${head.target_label}: ${head.n_pos} vs ${head.n_neg}`,
                      x: 0.02,
                      xanchor: "left",
                      font: { size: 13 },
                    },
                    xaxis: {
                      title: { text: "SHAP value  (positive → impaired)" },
                      zeroline: true,
                      zerolinecolor: "#94a3b8",
                    },
                    yaxis: {
                      tickvals: feats.map((_, i) => i),
                      ticktext: feats.map((f) => {
                        const r = ov.find((z) => z.feature === f);
                        return `${f.replace(/_/g, " ")} (${r ? r.share_pct.toFixed(1) : "?"}%)`;
                      }),
                      range: [-0.6, feats.length - 0.4],
                    },
                    annotations: feats.map((f, i) => {
                      const r = ov.find((z) => z.feature === f);
                      const up = r ? r.pos_pct : 50;
                      return {
                        xref: "paper" as const,
                        x: 1.01,
                        xanchor: "left" as const,
                        y: i,
                        yanchor: "middle" as const,
                        showarrow: false,
                        font: { size: 10, color: "#475569" },
                        text: `↑${up.toFixed(0)}% ↓${(100 - up).toFixed(0)}%`,
                      };
                    }),
                    margin: { l: 190, r: 96, b: 70 },
                    legend: { orientation: "h", y: -0.22 },
                  }}
                />
              );
            })}
          </div>
          <h4>Breakdown — by class and by diagnostic group</h4>
          <div className="plot-grid two">
            {EXPLAIN_TARGETS.map((t) => {
              const cg = (payload.shap_class_group ?? []).filter((r) => r.target === t.cg);
              const classes = Array.from(new Set(cg.map((r) => r.cls))).sort();
              return (
                <div key={t.key}>
                  <h3>{t.title}</h3>


                  {classes.map((cls) => {
                    const bee = (payload.shap_beeswarm_class ?? []).filter(
                      (r) => r.target === t.cg && r.cls === cls,
                    );
                    if (!bee.length) return null;
                    const meanAbs = (f: string) => {
                      const v = bee.filter((r) => r.feature === f);
                      return v.length
                        ? v.reduce((a2, r) => a2 + Math.abs(r.shap), 0) / v.length
                        : 0;
                    };
                    const feats6 = Array.from(new Set(bee.map((r) => r.feature))).sort(
                      (a, b) => meanAbs(a) - meanAbs(b),
                    );
                    const label = bee[0]?.cls_label ?? String(cls);
                    const split = (f: string) => {
                      const v = bee.filter((r) => r.feature === f).map((r) => r.shap);
                      const pos = v.reduce((a2, x) => a2 + Math.max(x, 0), 0);
                      const neg = v.reduce((a2, x) => a2 - Math.min(x, 0), 0);
                      return pos + neg > 0 ? (100 * pos) / (pos + neg) : 50;
                    };
                    const shareOf = (f: string) =>
                      cg.find((r) => r.cls === cls && r.group === "All" && r.feature === f)
                        ?.share_pct;
                    return (
                      <Plot
                        key={`bee-${t.key}-${cls}`}
                        data={feats6.flatMap((f, fi) =>
                          ["CN", "MCI", "AD"].map(
                            (g) =>
                              ({
                                type: "scattergl",
                                mode: "markers",
                                name: g,
                                legendgroup: g,
                                showlegend: fi === 0,
                                x: bee
                                  .filter((r) => r.feature === f && r.group === g)
                                  .map((r) => r.shap),
                                y: bee
                                  .filter((r) => r.feature === f && r.group === g)
                                  .map((r, k) => fi + ((k * 7) % 20) / 20 * 0.5 - 0.25),
                                marker: { color: GROUP_COLORS[g], size: 5, opacity: 0.55 },
                                hovertemplate: `${g}<br>SHAP=%{x:.3f}<extra></extra>`,
                              }) as Data,
                          ),
                        )}
                        layout={{
                          height: 330,
                          title: {
                            text: `Per-subject SHAP — “${label}” (dots = subjects, colour = group)`,
                            x: 0.02, xanchor: "left", font: { size: 13 },
                          },
                          xaxis: { title: { text: "SHAP value" }, zeroline: true, zerolinecolor: "#94a3b8" },
                          yaxis: {
                            tickvals: feats6.map((_, i) => i),
                            ticktext: feats6.map((f) => {
                              const sh = shareOf(f);
                              return `${f.replace(/_/g, " ")}${sh != null ? ` (${sh.toFixed(1)}%)` : ""}`;
                            }),
                            range: [-0.6, feats6.length - 0.4],
                          },
                          annotations: feats6.map((f, i) => ({
                            xref: "paper" as const,
                            x: 1.01,
                            xanchor: "left" as const,
                            y: i,
                            yanchor: "middle" as const,
                            showarrow: false,
                            font: { size: 10, color: "#475569" },
                            text: `↑${split(f).toFixed(0)}% ↓${(100 - split(f)).toFixed(0)}%`,
                          })),
                          margin: { l: 170, r: 96, b: 70 },
                          legend: { orientation: "h", y: -0.22 },
                        }}
                      />
                    );
                  })}
                </div>
              );
            })}
          </div>
          {payload.inference_summary ? (
            <>
              <h3>What the models confirm</h3>
              {payload.inference_summary.agreement.map((line) => (
                <p key={line.slice(0, 40)} className="section-intro">
                  {line}
                </p>
              ))}
              {[
                { tgt: "mmse_band", label: "MMSE bands", pal: BUCKET_COLORS },
                {
                  tgt: "cdr_3level",
                  label: "CDR levels",
                  pal: ["#2a78d6", "#eda100", "#eb6834"],
                },
              ].flatMap(({ tgt, label, pal }) => {
                const cg = (payload.shap_class_group ?? []).filter((r) => r.target === tgt);
                const pdpAll = (payload.pdp_class_group ?? []).filter((r) => r.target === tgt);
                const pdpFeats = Array.from(new Set(pdpAll.map((r) => r.feature)));
                const clsIds = Array.from(new Set(cg.map((r) => r.cls))).sort((a, b) => a - b);
                return clsIds.map((cls) => {
                  const clsLabel =
                    cg.find((r) => r.cls === cls)?.cls_label ?? String(cls);
                  const rows = ["All", "CN", "MCI", "AD"].flatMap((g) => {
                    const top = cg
                      .filter((r) => r.cls === cls && r.group === g)
                      .sort((a, b) => b.share_pct - a.share_pct)
                      .slice(0, 3);
                    if (!top.length) return [];
                    const fam = (payload.shap_family_group ?? [])
                      .filter((r) => r.target === tgt && r.cls === cls && r.group === g)
                      .sort((x, y2) => y2.share_pct - x.share_pct);
                    const SHORT: Record<string, string> = {
                      "F1 whole-brain": "F1",
                      "F2 SR/LR": "F2",
                      "F3 exceptions": "F3",
                      "F4 architecture": "F4",
                      covariate: "cov",
                    };
                    return [
                      {
                        Group: g,
                        n: top[0].n,
                        "Top drivers (% of total attribution ± seed sd)": top
                          .map(
                            (r) =>
                              `${r.feature.replace(/_/g, " ")} (${r.share_pct.toFixed(1)} ± ${r.share_sd.toFixed(1)}%)`,
                          )
                          .join(", "),
                        "Family shares (%)": fam
                          .map((r) => `${SHORT[r.family] ?? r.family} ${r.share_pct.toFixed(0)}`)
                          .join(" · "),
                      },
                    ];
                  });
                  if (!rows.length) return null;
                  const topFeat = cg
                    .filter((r) => r.cls === cls && r.group === "All")
                    .sort((a, b) => b.share_pct - a.share_pct)
                    .map((r) => r.feature)
                    .find((f) => pdpFeats.includes(f));
                  const feat = topFeat ?? pdpFeats[0];
                  return (
                    <div key={`${tgt}-${cls}`} className="class-section">
                      <h4>
                        {label} — {clsLabel}
                      </h4>
                      <DataTable
                        rows={rows}
                        columns={[
                          "Group",
                          "n",
                          "Top drivers (% of total attribution ± seed sd)",
                          "Family shares (%)",
                        ]}
                        maxHeight={null}
                        initialRows={6}
                        compact
                        cellClassName={(row, column) =>
                          column === "Group" && row[column] === "All"
                            ? "cell-top-rank"
                            : undefined
                        }
                        caption={`${label} — ${clsLabel}: strongest drivers overall and within each diagnostic group`}
                      />
                      {feat ? (
                        <div className="plot-grid two">
                          {["All", "CN", "MCI", "AD"].map((g) => {
                            const rowsG = pdpAll.filter(
                              (r) => r.feature === feat && r.group === g,
                            );
                            if (!rowsG.length) return null;
                            const cIds = Array.from(new Set(rowsG.map((r) => r.cls))).sort(
                              (a, b) => a - b,
                            );
                            return (
                              <Plot
                                key={`${tgt}-${cls}-${g}`}
                                data={cIds.map((c) => {
                                  const line = rowsG
                                    .filter((r) => r.cls === c)
                                    .sort((a, b) => a.grid - b.grid);
                                  return {
                                    type: "scatter",
                                    mode: "lines",
                                    x: line.map((r) => r.grid),
                                    y: line.map((r) => r.prob),
                                    name: line[0]?.cls_label ?? String(c),
                                    line: {
                                      color: pal[c],
                                      width: c === cls ? 3 : 1.5,
                                      dash: c === cls ? undefined : ("dot" as const),
                                    },
                                  } as Data;
                                })}
                                layout={{
                                  height: 300,
                                  title: {
                                    text: `${g === "All" ? "All 202 subjects" : `within ${g}`} — P(class) vs ${feat.replace(/_/g, " ")}`,
                                    x: 0.02,
                                    xanchor: "left",
                                    font: { size: 12 },
                                  },
                                  xaxis: { title: { text: feat.replace(/_/g, " ") } },
                                  yaxis: { title: { text: "predicted probability" } },
                                  legend: { orientation: "h", y: -0.3 },
                                  margin: { b: 96 },
                                }}
                              />
                            );
                          })}
                        </div>
                      ) : null}
                    </div>
                  );
                });
              })}
              {payload.pdp_range_family ? (
                <div className="class-section">
                  <h4>Where the signal lives — edge range and exception status</h4>
                  <p>
                    The same network and the same measure — default-mode
                    diffusivity — held fixed while only the edge range changes,
                    pooled over all 202 subjects rather than split by diagnosis.
                    Each row is one range; the two columns are the two targets.
                    Because the vertical scale is what matters here, the swing
                    printed on each panel is the change in predicted probability
                    across the feature’s 5th to 95th percentile.
                  </p>
                  {Array.from(
                    new Map(
                      (payload.pdp_range_family ?? []).map((r) => [
                        r.range_label,
                        r.range_order,
                      ]),
                    ),
                  )
                    .sort((a, b) => a[1] - b[1])
                    .map(([rangeLabel]) => (
                      <div className="plot-grid two" key={rangeLabel}>
                        {[
                          { tgt: "mmse_band", short: "MMSE", pal: BUCKET_COLORS },
                          {
                            tgt: "cdr_3level",
                            short: "CDR",
                            pal: ["#2a78d6", "#eda100", "#eb6834"],
                          },
                        ].map(({ tgt, short, pal }) => {
                          const rows = (payload.pdp_range_family ?? []).filter(
                            (r) => r.target === tgt && r.range_label === rangeLabel,
                          );
                          if (!rows.length) return null;
                          const cIds = Array.from(new Set(rows.map((r) => r.cls))).sort(
                            (a, b) => a - b,
                          );
                          const swing = Math.max(
                            ...cIds.map((c) => {
                              const v = rows.filter((r) => r.cls === c).map((r) => r.prob);
                              return Math.max(...v) - Math.min(...v);
                            }),
                          );
                          const miss = rows[0].missing_pct;
                          return (
                            <Plot
                              key={`${rangeLabel}-${tgt}`}
                              data={cIds.map((c) => {
                                const line = rows
                                  .filter((r) => r.cls === c)
                                  .sort((a, b) => a.grid - b.grid);
                                return {
                                  type: "scatter",
                                  mode: "lines",
                                  x: line.map((r) => r.grid),
                                  y: line.map((r) => r.prob),
                                  name: line[0]?.cls_label ?? String(c),
                                  line: { color: pal[c], width: 2.5 },
                                } as Data;
                              })}
                              layout={{
                                height: 320,
                                title: {
                                  text: `${short} · ${rangeLabel} — swing ${(swing * 100).toFixed(1)} pp · ${miss.toFixed(1)}% imputed`,
                                  x: 0.02,
                                  xanchor: "left",
                                  font: { size: 12 },
                                },
                                xaxis: {
                                  title: { text: `${rows[0].feature.replace(/_/g, " ")}` },
                                },
                                yaxis: { title: { text: "predicted probability" } },
                                legend: { orientation: "h", y: -0.3 },
                                margin: { b: 96 },
                              }}
                            />
                          );
                        })}
                      </div>
                    ))}
                  <p>
                    <strong>What this shows.</strong> The relationship is a
                    short-range phenomenon. Short-range default-mode diffusivity
                    moves the predicted probability by 3.4 points on the impaired
                    MMSE band and 4.4 on CDR ≥ 1, while the long-range version of
                    the identical measure moves no class by more than 0.7 points
                    on either instrument — barely above the 0.4-point swing a
                    label-shuffled null produces, which is to say nothing at all. The exception tier sits in between and
                    behaves differently on the two instruments: short-range
                    exception edges carry 3.5 points on CDR ≥ 1, close to the
                    full short-range effect, but only 1.5 on the impaired MMSE
                    band.
                  </p>
                  <p>
                    <strong>Read the long-range panels with care.</strong> Their
                    flatness is not purely biological. Long-range edges are
                    measured in fewer subjects, so 11.4 per cent of the
                    long-range values and 24.8 per cent of the long-range
                    exception values are median-imputed rather than observed, and
                    imputing a quarter of a column toward its centre flattens any
                    curve mechanically. The honest statement is that long-range
                    diffusivity contributes no detectable predictive signal here,
                    with the caveat that it is also the most sparsely measured
                    quantity in the feature set.
                  </p>
                </div>
              ) : null}
              <h4>How to read these panels</h4>
              <p>
                <strong>What the measures are.</strong> A feature ending in
                DIFF is a diffusivity composite: the mean of the robust
                z-scores of mean, radial and axial diffusivity over that
                network’s edges. It does <em>not</em> contain fractional
                anisotropy, which is carried separately as FA. Higher DIFF
                therefore means faster, less restricted diffusion — the
                direction that indicates degraded white-matter microstructure.
                The SR prefix restricts the average to short-range edges, the
                shortest quartile of connection lengths referenced to the
                control distribution; WB takes the same network’s whole-brain
                edges.
              </p>
              <p>
                <strong>Which end is which.</strong> The two instruments run in
                opposite directions. On the MMSE the High band (29–30) is the
                cognitively intact end and the Low band (9–27) the impaired
                one; on the CDR, 0 is unimpaired and ≥ 1 is frank dementia. A
                feature that raises the probability of the Low MMSE band should
                therefore lower the probability of CDR 0, and that is what the
                partial-dependence panels show.
              </p>
              <p>
                <strong>The one relationship that holds.</strong> The same
                diffusivity axis — default-mode network diffusivity, with global
                radial diffusivity close behind — is among the strongest
                contributors in every class of both targets, and it enters the
                impaired and intact classes with opposite signs: per-subject
                attributions for the Low and High MMSE bands correlate at
                −0.98, and CDR 0 against CDR ≥ 1 at −0.97. This is one signal
                read from both ends, not a different signature per class.
                Sweeping DMN diffusivity across its observed range raises the
                probability of the impaired MMSE band by 3.3 percentage points
                and lowers the intact band by 2.7; on the CDR it lowers CDR 0 by
                2.8 points and raises CDR ≥ 1 by 4.2. The curves are monotone
                and clear a label-shuffled null of about 0.4 points, so the
                direction is real.
              </p>
              <p>
                <strong>How much it is worth.</strong> Not much, and the panels
                should not be read as showing predictive strength. The same
                full-range sweep changes the model’s predicted class for six of
                202 subjects on the MMSE and for none on the CDR. Between 84 and
                88 per cent of each curve’s movement happens above the feature’s
                median, which is where the AD subjects sit — median SR DMN
                diffusivity is 1.08 in AD against −0.03 in controls — so the
                relationship is largely the same between-group separation the
                discrimination tables identified, seen from the attribution
                side. Attribution describes what the fitted model used; whether
                that use generalises is the AUC question, answered above.
              </p>
              <p>
                <strong>What not to read into the ranking.</strong> The short-
                and whole-brain versions of a network measure are near-duplicate
                quantities — SR and WB DMN diffusivity correlate at 0.98 — and
                they are interchangeable to the model: remove either and the
                other takes first place in every class. Their relative order
                carries no meaning. Ordering a single fit produced, in which SR
                led the impaired band and WB the intact one, reproduced in only
                28 per cent of 200 refits and became an exact coin flip when
                subjects were resampled. Attribution here is therefore averaged
                over five seeds, but even so only the leading feature is firmly
                placed; positions below it move by roughly half a percentage
                point between fits and should be read as a set, not a ranking.
              </p>
            </>
          ) : null}
        </>
      ) : null}


        </>
      ) : (
        <p className="section-intro">Model artifacts are still being computed.</p>
      )}
    </>
  );
}

export function LrSrModelsPage() {
  return (
    <main className="content-page lr-sr-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Tractography · Cognition models</span>
          <h1>LR–SR Models</h1>
          <p>
            Target distributions, feature engineering, and the MMSE/CDR models
            built on the tiered connectome features — with per-class,
            per-group explanations.
          </p>
        </div>
        <div className="status-pill reference">
          <span className="pulse-dot" />
          Precomputed, leakage-guarded
        </div>
      </div>
      <section className="panel">
        <SectionHeading index={1} title="FEATURES → TARGETS → MODELS" />
        <CognitionModelsBody />
      </section>
    </main>
  );
}

export function LrSrAnalysisPage() {
  return (
    <main className="content-page lr-sr-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Tractography · Work-along analysis</span>
          <h1>LR–SR Structural Pathway Analysis</h1>
          <p>
            One coherent view of tract geometry, subject-specific EDR
            exceptions, strength contrasts, and candidate modelling features.
          </p>
        </div>
        <div className="status-pill reference">
          <span className="pulse-dot" />
          Reference-equivalent calculations
        </div>
      </div>

      <section className="panel">
        <SectionHeading index={1} title="TRACT-LENGTH DISTRIBUTIONS" />
        <p className="section-intro">
          Positive upper-triangle AAL3 <code>len_mean</code> edges, summarized
          over the complete 530-subject analysis cohort.
        </p>
        <TractLengthPanel />
      </section>

      <section className="panel">
        <SectionHeading index={2} title="EDR EXCEPTIONS" />
        <ExceptionSection />
      </section>

      <section className="panel">
        <SectionHeading index={3} title="MODELLING FEATURES" />
        <ModellingFeatureSection />
      </section>

      <section className="panel">
        <SectionHeading index={4} title="NETWORK × MEASURE RANKED MATRIX" />
        <NetworkRankedSection />
      </section>

      <section className="panel">
        <SectionHeading index={5} title="EXCEPTION-SPECIFIC SIGNAL & SIMPLE MODELS" />
        <ExceptionSpecificitySection />
      </section>
    </main>
  );
}
