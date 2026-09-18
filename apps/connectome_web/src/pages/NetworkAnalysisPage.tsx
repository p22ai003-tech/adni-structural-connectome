import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data, Layout } from "plotly.js";
import { getJson } from "../api";
import type {
  Artifact,
  NetworkAffectedness,
  NetworkAnalysisCatalog,
  NetworkBlocks,
  NetworkDistribution,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { DataTable } from "../components/DataTable";
import { Plot } from "../components/Plot";
import { SectionHeading } from "../components/SectionHeading";

const GROUPS = ["CN", "MCI", "AD"] as const;
const GROUP_COLORS: Record<string, string> = {
  CN: "#38bdf8",
  MCI: "#a3e635",
  AD: "#fb7185",
};

function titleCase(value: string): string {
  return value
    .replaceAll("__", " × ")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function median(values: number[]): number {
  const ordered = [...values].filter(Number.isFinite).sort(
    (first, second) => first - second,
  );
  if (!ordered.length) return Number.NaN;
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function visualIqrTrim(
  rows: Record<string, unknown>[],
): Record<string, unknown>[] {
  const strata = Array.from(
    new Set(
      rows.map(
        (row) => `${String(row.group)}\u0000${String(row.network)}`,
      ),
    ),
  );
  return strata.flatMap((stratum) => {
    const [group, network] = stratum.split("\u0000");
    const subset = rows.filter(
      (row) =>
        String(row.group) === group &&
        String(row.network) === network,
    );
    const values = subset
      .map((row) => Number(row.value))
      .filter(Number.isFinite)
      .sort((first, second) => first - second);
    if (values.length < 4) return subset;
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
    return subset.filter((row) => {
      const value = Number(row.value);
      return value >= q1 - 1.5 * iqr && value <= q3 + 1.5 * iqr;
    });
  });
}

function NetworkDistributionPlot({
  rows,
  medianRows,
  metric,
}: {
  rows: Record<string, unknown>[];
  medianRows: Record<string, unknown>[];
  metric: string;
}) {
  const traces: Data[] = GROUPS.map((group) => {
    const groupRows = rows.filter((row) => String(row.group) === group);
    return {
      type: "violin",
      name: group,
      legendgroup: group,
      scalegroup: group,
      offsetgroup: group,
      x: groupRows.map((row) => String(row.network)),
      y: groupRows.map((row) => Number(row.value)),
      box: { visible: true },
      meanline: { visible: true },
      points: false,
      spanmode: "hard",
      scalemode: "width",
      fillcolor: `${GROUP_COLORS[group]}55`,
      line: { color: GROUP_COLORS[group], width: 2 },
      hovertemplate:
        `${group}<br>%{x}<br>${titleCase(metric)}=%{y:.5g}` +
        "<extra></extra>",
    } as Data;
  });
  const medianTraces: Data[] = GROUPS.map((group) => {
    const networks = Array.from(
      new Set(medianRows.map((row) => String(row.network))),
    );
    return {
      type: "scatter",
      mode: "markers",
      name: `${group} median`,
      legendgroup: group,
      showlegend: false,
      x: networks,
      y: networks.map((network) =>
        median(
          medianRows
            .filter(
              (row) =>
                String(row.group) === group &&
                String(row.network) === network,
            )
            .map((row) => Number(row.value)),
        ),
      ),
      marker: {
        color: "#f8fafc",
        line: { color: GROUP_COLORS[group], width: 2 },
        size: 7,
        symbol: "diamond",
      },
      hovertemplate:
        `${group} median<br>%{x}<br>%{y:.5g}<extra></extra>`,
    } as Data;
  });
  return (
    <Plot
      data={[...traces, ...medianTraces]}
      layout={{
        height: 570,
        title: {
          text: `${titleCase(metric)} across mapped systems`,
          font: { size: 16 },
        },
        ...({ violinmode: "group" } as unknown as Partial<Layout>),
        xaxis: {
          title: { text: "Mapped network / anatomical system" },
          tickangle: -28,
        },
        yaxis: { title: { text: titleCase(metric) } },
        legend: { orientation: "h" },
        margin: { b: 145 },
      }}
    />
  );
}

function AffectednessPlot({
  rows,
}: {
  rows: Record<string, unknown>[];
}) {
  const display = [...rows]
    .filter((row) => Number.isFinite(Number(row.cn_ad_cliffs_delta)))
    .sort(
      (first, second) =>
        Math.abs(Number(second.cn_ad_cliffs_delta)) -
        Math.abs(Number(first.cn_ad_cliffs_delta)),
    )
    .slice(0, 30)
    .reverse();
  return (
    <Plot
      data={[
        {
          type: "bar",
          orientation: "h",
          x: display.map((row) => Number(row.cn_ad_cliffs_delta)),
          y: display.map(
            (row) =>
              `${String(row.network)} · ${titleCase(
                String(row.metric),
              )}`,
          ),
          marker: {
            color: display.map((row) =>
              Number(row.cn_ad_cliffs_delta) >= 0
                ? GROUP_COLORS.CN
                : GROUP_COLORS.AD,
            ),
          },
          customdata: display.map((row) => [
            row.feature_family,
            row.cn_ad_bm_p,
            row.cn_ad_bm_q,
            row.mean_CN,
            row.mean_AD,
          ]),
          hovertemplate:
            "%{y}<br>Cliff delta=%{x:.3f}" +
            "<br>family=%{customdata[0]}<br>p=%{customdata[1]:.3g}" +
            "<br>BH q=%{customdata[2]:.3g}" +
            "<br>CN mean=%{customdata[3]:.5g}" +
            "<br>AD mean=%{customdata[4]:.5g}<extra></extra>",
        } as Data,
      ]}
      layout={{
        height: 850,
        title: {
          text: "Largest recorded CN–AD network effects",
          font: { size: 16 },
        },
        xaxis: {
          title: {
            text: "Cliff delta (positive = CN tends higher; negative = AD tends higher)",
          },
          zeroline: true,
        },
        yaxis: { automargin: true },
        showlegend: false,
        margin: { l: 250 },
      }}
    />
  );
}

function blockMatrix(
  rows: Record<string, unknown>[],
  networks: string[],
  group: string,
  metric: string,
): number[][] {
  const lookup = new Map<string, number>();
  rows
    .filter((row) => String(row.group) === group)
    .forEach((row) => {
      const first = String(row.net_a);
      const second = String(row.net_b);
      const value = Number(row[metric]);
      lookup.set(`${first}\u0000${second}`, value);
      lookup.set(`${second}\u0000${first}`, value);
    });
  return networks.map((first) =>
    networks.map(
      (second) => lookup.get(`${first}\u0000${second}`) ?? Number.NaN,
    ),
  );
}

function BlockHeatmap({
  rows,
  networks,
  group,
  metric,
}: {
  rows: Record<string, unknown>[];
  networks: string[];
  group: string;
  metric: string;
}) {
  const cn = blockMatrix(rows, networks, "CN", metric);
  const matrix =
    group === "AD − CN"
      ? blockMatrix(rows, networks, "AD", metric).map((line, lineIndex) =>
          line.map((value, columnIndex) => value - cn[lineIndex][columnIndex]),
        )
      : blockMatrix(rows, networks, group, metric);
  const difference = group === "AD − CN";
  return (
    <Plot
      data={[
        {
          type: "heatmap",
          x: networks,
          y: networks,
          z: matrix,
          colorscale: difference ? "RdBu" : "Viridis",
          reversescale: difference,
          zmid: difference ? 0 : undefined,
          colorbar: {
            title: { text: difference ? "Δ" : titleCase(metric) },
          },
          hovertemplate:
            `${group}<br>%{y} ↔ %{x}<br>%{z:.5g}<extra></extra>`,
        } as Data,
      ]}
      layout={{
        height: 520,
        title: {
          text: `${group} · ${titleCase(metric)}`,
          font: { size: 15 },
        },
        xaxis: { tickangle: -45 },
        yaxis: { autorange: "reversed" },
        showlegend: false,
        margin: { l: 125, b: 125 },
      }}
    />
  );
}

export function NetworkAnalysisPage({ section }: { section: Section }) {
  const [scheme, setScheme] = useState<"functional" | "anatomical">(
    "functional",
  );
  const [family, setFamily] = useState<
    "microstructure" | "graph" | "coupling"
  >("microstructure");
  const [metric, setMetric] = useState("");
  const [visualTrim, setVisualTrim] = useState(true);
  const [showSubjectRows, setShowSubjectRows] = useState(false);
  const [blockMetric, setBlockMetric] = useState<
    "density" | "mean_weight"
  >("mean_weight");
  const catalog = useQuery({
    queryKey: ["network-analysis-catalog"],
    queryFn: ({ signal }) =>
      getJson<NetworkAnalysisCatalog>(
        "/networks/analysis/catalog",
        signal,
      ),
  });
  const familyOptions = useMemo(
    () =>
      catalog.data?.data.schemes
        .find((item) => item.id === scheme)
        ?.families.filter((item) => item.available) ?? [],
    [catalog.data, scheme],
  );
  const metricOptions = useMemo(
    () =>
      familyOptions.find((item) => item.id === family)?.metrics ?? [],
    [family, familyOptions],
  );
  useEffect(() => {
    if (!familyOptions.some((item) => item.id === family)) {
      setFamily(familyOptions[0]?.id ?? "microstructure");
    }
  }, [family, familyOptions]);
  useEffect(() => {
    if (!metricOptions.includes(metric)) {
      setMetric(metricOptions[0] ?? "");
    }
  }, [metric, metricOptions]);
  const distribution = useQuery({
    queryKey: ["network-distribution", scheme, family, metric],
    queryFn: ({ signal }) =>
      getJson<NetworkDistribution>(
        `/networks/analysis/distribution?scheme=${scheme}` +
          `&family=${family}&metric=${encodeURIComponent(metric)}`,
        signal,
      ),
    enabled: Boolean(metric),
  });
  const affectedness = useQuery({
    queryKey: ["network-affectedness", scheme],
    queryFn: ({ signal }) =>
      getJson<NetworkAffectedness>(
        `/networks/analysis/affectedness?scheme=${scheme}`,
        signal,
      ),
  });
  const blocks = useQuery({
    queryKey: ["network-blocks", scheme, blockMetric],
    queryFn: ({ signal }) =>
      getJson<NetworkBlocks>(
        `/networks/analysis/blocks?scheme=${scheme}&metric=${blockMetric}`,
        signal,
      ),
  });
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(`/sections/${section.id}/artifacts`, signal),
  });

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>
            Subject-level structural-connectome measures aggregated over
            documented functional-network approximations and deterministic
            anatomical systems.
          </p>
        </div>
        <div className="page-stat">
          <strong>{affectedness.data?.data.rows.length ?? "—"}</strong>
          <span>network–metric summaries</span>
        </div>
      </div>

      <section className="panel">
        <SectionHeading index="01" title="Network-resolved distributions" />
        {catalog.isLoading ? (
          <Loading label="Loading network analysis catalog" />
        ) : catalog.isError ? (
          <ErrorState error={catalog.error} />
        ) : catalog.data ? (
          <>
            <div className="control-row wrap">
              <label className="control grow">
                <span>Mapping scheme</span>
                <select
                  value={scheme}
                  onChange={(event) =>
                    setScheme(
                      event.target.value as "functional" | "anatomical",
                    )
                  }
                >
                  {catalog.data.data.schemes.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Feature family</span>
                <select
                  value={family}
                  onChange={(event) =>
                    setFamily(
                      event.target.value as
                        | "microstructure"
                        | "graph"
                        | "coupling",
                    )
                  }
                >
                  {familyOptions.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Metric</span>
                <select
                  value={metric}
                  onChange={(event) => setMetric(event.target.value)}
                >
                  {metricOptions.map((item) => (
                    <option key={item} value={item}>
                      {titleCase(item)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={visualTrim}
                  onChange={(event) =>
                    setVisualTrim(event.target.checked)
                  }
                />
                Per-group/network visual 1.5-IQR trim
              </label>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={showSubjectRows}
                  onChange={(event) =>
                    setShowSubjectRows(event.target.checked)
                  }
                />
                Show subject-by-network rows
              </label>
            </div>
            <div className="mapping-warning">
              <strong>Mapping boundary:</strong>{" "}
              {catalog.data.data.mapping_contract[scheme]}{" "}
              {catalog.data.data.mapping_contract.modality}
            </div>
          </>
        ) : null}
        {distribution.isLoading ? (
          <Loading label="Loading subject-by-network values" />
        ) : distribution.isError ? (
          <ErrorState error={distribution.error} />
        ) : distribution.data ? (
          <>
            <NetworkDistributionPlot
              rows={
                visualTrim
                  ? visualIqrTrim(distribution.data.data.rows)
                  : distribution.data.data.rows
              }
              medianRows={distribution.data.data.rows}
              metric={metric}
            />
            <p className="method-note">
              The visual trim affects only the plotted distributions.
              Recorded statistics and the table below always use the complete
              precomputed subject-by-network values.
            </p>
            <DataTable
              rows={distribution.data.data.statistics}
              compact
              maxHeight={470}
              caption="Recorded network-wise inference"
            />
            <p className="provenance-line">
              {distribution.data.provenance.calculation}
            </p>
            {showSubjectRows ? (
              <DataTable
                rows={distribution.data.data.rows}
                compact
                maxHeight={560}
                initialRows={120}
                caption="Authenticated subject-by-network rows"
              />
            ) : null}
          </>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading
          index="02"
          title="Cross-family network affectedness"
        />
        {affectedness.isLoading ? (
          <Loading label="Loading cross-family effects" />
        ) : affectedness.isError ? (
          <ErrorState error={affectedness.error} />
        ) : affectedness.data ? (
          <div className="analysis-split">
            <AffectednessPlot rows={affectedness.data.data.rows} />
            <DataTable
              rows={affectedness.data.data.rows}
              compact
              maxHeight={850}
              initialRows={40}
              caption="Recorded effects and BH-FDR"
            />
          </div>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading
          index="03"
          title="Within- and between-system structural blocks"
        />
        <div className="control-row">
          <label className="control compact-control">
            <span>Block measure</span>
            <select
              value={blockMetric}
              onChange={(event) =>
                setBlockMetric(
                  event.target.value as "density" | "mean_weight",
                )
              }
            >
              <option value="mean_weight">Mean structural weight</option>
              <option value="density">Positive-edge density</option>
            </select>
          </label>
        </div>
        {blocks.isLoading ? (
          <Loading label="Loading structural block matrices" />
        ) : blocks.isError ? (
          <ErrorState error={blocks.error} />
        ) : blocks.data ? (
          <>
            <div className="plot-grid two">
              {["CN", "MCI", "AD", "AD − CN"].map((group) => (
                <BlockHeatmap
                  key={group}
                  rows={blocks.data.data.rows}
                  networks={blocks.data.data.networks}
                  group={group}
                  metric={blockMetric}
                />
              ))}
            </div>
            <DataTable
              rows={blocks.data.data.statistics}
              compact
              maxHeight={480}
              initialRows={50}
              caption="Within/between block inference"
            />
          </>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading index="OUTPUTS" title="Complete recorded outputs" />
        {artifacts.isLoading ? (
          <Loading label="Indexing network outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
