import { useMemo } from "react";
import type { Data } from "plotly.js";
import { Plot } from "./Plot";

const GROUPS = ["CN", "MCI", "AD"] as const;
const COLORS: Record<string, string> = {
  CN: "#38bdf8",
  MCI: "#a3e635",
  AD: "#fb7185",
};

function finiteValues(
  rows: Record<string, unknown>[],
  group: string,
  valueKey: string,
): number[] {
  return rows
    .filter((row) => String(row.group) === group)
    .map((row) => Number(row[valueKey]))
    .filter(Number.isFinite);
}

export function GroupDistributionPlot({
  rows,
  valueKey,
  title,
  yTitle,
  height = 460,
}: {
  rows: Record<string, unknown>[];
  valueKey: string;
  title: string;
  yTitle?: string;
  height?: number;
}) {
  const traces = useMemo<Data[]>(
    () =>
      GROUPS.map((group) => {
        const values = finiteValues(rows, group, valueKey);
        return {
          type: "violin",
          name: group,
          y: values,
          box: { visible: true },
          meanline: { visible: true },
          points: false,
          spanmode: "hard",
          scalemode: "width",
          fillcolor: `${COLORS[group]}66`,
          line: { color: COLORS[group], width: 2 },
          hovertemplate:
            `${group}<br>${valueKey}=%{y:.5g}` +
            `<br>n=${values.length}<extra></extra>`,
        } as Data;
      }),
    [rows, valueKey],
  );
  const annotations = GROUPS.flatMap((group) => {
    const values = finiteValues(rows, group, valueKey).sort(
      (first, second) => first - second,
    );
    if (!values.length) return [];
    const middle = Math.floor(values.length / 2);
    const value =
      values.length % 2
        ? values[middle]
        : (values[middle - 1] + values[middle]) / 2;
    return [
      {
        x: group,
        y: value,
        text: `median ${value.toLocaleString(undefined, {
          maximumSignificantDigits: 5,
        })}`,
        showarrow: false,
        yshift: 15,
        font: { color: "#f8fafc", size: 10 },
        bgcolor: "rgba(7,10,16,0.78)",
        bordercolor: COLORS[group],
        borderpad: 3,
      },
    ];
  });
  return (
    <Plot
      data={traces}
      layout={{
        height,
        title: {
          text: title,
          x: 0.02,
          xanchor: "left",
          font: { size: 16 },
        },
        xaxis: {
          title: { text: "Diagnostic group" },
          categoryorder: "array",
          categoryarray: [...GROUPS],
        },
        yaxis: { title: { text: yTitle ?? valueKey } },
        annotations,
        showlegend: false,
      }}
    />
  );
}

export function CohortCountPlot({
  rows,
}: {
  rows: Record<string, unknown>[];
}) {
  const ordered = GROUPS.map((group) => {
    const row = rows.find((item) => String(item.group) === group);
    return {
      group,
      count: Number(row?.n_subjects ?? 0),
      final: Number(row?.n_with_final_connectomes ?? 0),
    };
  });
  const traces: Data[] = [
    {
      type: "bar",
      name: "Analysis cohort",
      x: ordered.map((row) => row.group),
      y: ordered.map((row) => row.count),
      marker: {
        color: ordered.map((row) => COLORS[row.group]),
      },
      text: ordered.map((row) => String(row.count)),
      textposition: "outside",
      hovertemplate: "%{x}: %{y} subjects<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Final connectomes",
      x: ordered.map((row) => row.group),
      y: ordered.map((row) => row.final),
      marker: {
        symbol: "diamond",
        size: 12,
        color: "#f5c842",
        line: { color: "#111827", width: 1 },
      },
      hovertemplate: "%{x}: %{y} final connectomes<extra></extra>",
    },
  ];
  return (
    <Plot
      data={traces}
      layout={{
        height: 390,
        title: {
          text: "Cohort and production completeness",
          x: 0.02,
          xanchor: "left",
          font: { size: 16 },
        },
        yaxis: { title: { text: "Subjects" }, rangemode: "tozero" },
        barmode: "group",
        legend: {
          orientation: "h",
        },
      }}
    />
  );
}

export function NodeRankingPlot({
  rows,
  groupA,
  groupB,
  topN,
}: {
  rows: Record<string, unknown>[];
  groupA: string;
  groupB: string;
  topN: number;
}) {
  const display: Array<
    Record<string, unknown> & { score: number; label: string }
  > = rows.slice(0, topN).map((row) => {
    const p = Math.max(Number(row.p_value), 1e-300);
    return {
      ...row,
      score: -Math.log10(p),
      label: String(row.atlas_label ?? row.node_name ?? row.node),
    };
  });
  const customdata: Array<Array<string | number>> = display
    .map((row) => [
      String(row["node_name"] ?? ""),
      Number(row["p_value"]),
      Number(row["q_value"]),
      Number(row["cliffs_delta"]),
      String(row["higher_group"] ?? ""),
    ])
    .reverse();
  const trace: Data = {
    type: "bar",
    orientation: "h",
    x: display.map((row) => row.score).reverse(),
    y: display.map((row) => row.label).reverse(),
    marker: {
      color: display
        .map((row) =>
          String(row["higher_group"]) === groupA
            ? COLORS[groupA]
            : COLORS[groupB],
        )
        .reverse(),
    },
    customdata,
    hovertemplate:
      "AAL3=%{y}<br>node=%{customdata[0]}" +
      "<br>p=%{customdata[1]:.3g}<br>BH q=%{customdata[2]:.3g}" +
      "<br>Cliff delta=%{customdata[3]:.3f}" +
      "<br>higher median=%{customdata[4]}<extra></extra>",
  };
  return (
    <Plot
      data={[trace]}
      layout={{
        height: Math.max(420, Math.min(940, 34 * display.length + 160)),
        title: {
          text: `${groupA} vs ${groupB}: node ranking`,
          x: 0.02,
          xanchor: "left",
          font: { size: 16 },
        },
        xaxis: { title: { text: "−log10 Mann–Whitney p" } },
        yaxis: { automargin: true },
        showlegend: false,
        margin: { l: 210, r: 28, t: 60, b: 56 },
      }}
    />
  );
}
