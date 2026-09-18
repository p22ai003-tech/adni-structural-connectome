import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJson, downloadRows } from "../api";
import type {
  Artifact,
  CouplingAalCatalog,
  CouplingAalRanking,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { DataTable } from "../components/DataTable";
import { NodeRankingPlot } from "../components/ScientificPlots";
import { SectionHeading } from "../components/SectionHeading";

const CONTRASTS = [
  ["CN", "MCI"],
  ["CN", "AD"],
  ["MCI", "AD"],
] as const;

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function CouplingAalPage({ section }: { section: Section }) {
  const [topology, setTopology] = useState("strength");
  const [microstructure, setMicrostructure] = useState("fa_mean");
  const [valueMetric, setValueMetric] = useState("coupling_index");
  const [contrast, setContrast] = useState("CN vs MCI");
  const [topN, setTopN] = useState(20);
  const [fdrOnly, setFdrOnly] = useState(false);
  const [nominalOnly, setNominalOnly] = useState(false);
  const [groupA, groupB] = contrast.split(" vs ");
  const catalog = useQuery({
    queryKey: ["coupling-aal-catalog"],
    queryFn: ({ signal }) =>
      getJson<CouplingAalCatalog>("/coupling-aal/catalog", signal),
  });
  const ranking = useQuery({
    queryKey: [
      "coupling-aal-ranking",
      topology,
      microstructure,
      valueMetric,
      groupA,
      groupB,
    ],
    queryFn: ({ signal }) =>
      getJson<CouplingAalRanking>(
        `/coupling-aal/ranking?topology_metric=${topology}` +
          `&microstructure_metric=${microstructure}` +
          `&value_metric=${valueMetric}` +
          `&group_a=${groupA}&group_b=${groupB}`,
        signal,
      ),
  });
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(`/sections/${section.id}/artifacts`, signal),
  });
  const filteredRows = useMemo(() => {
    const rows = ranking.data?.data.rows ?? [];
    if (fdrOnly) {
      return rows.filter((row) => Number(row.q_value) < 0.05);
    }
    if (nominalOnly) {
      return rows.filter((row) => Number(row.p_value) < 0.05);
    }
    return rows;
  }, [fdrOnly, nominalOnly, ranking.data]);
  const fdrCount = (ranking.data?.data.rows ?? []).filter(
    (row) => Number(row.q_value) < 0.05,
  ).length;
  const nominalCount = (ranking.data?.data.rows ?? []).filter(
    (row) => Number(row.p_value) < 0.05,
  ).length;

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>
            Regional AAL3 topology–microstructure coupling computed within
            each subject, then tested between diagnostic groups at the
            subject level.
          </p>
        </div>
        <div className="page-stat">
          <strong>{ranking.data?.data.rows.length ?? "—"}</strong>
          <span>tested AAL3 regions</span>
        </div>
      </div>

      <section className="panel">
        <SectionHeading
          index="01"
          title="Regional coupling specification"
        />
        {catalog.isLoading ? (
          <Loading label="Loading regional coupling contract" />
        ) : catalog.isError ? (
          <ErrorState error={catalog.error} />
        ) : catalog.data ? (
          <>
            <div className="control-row wrap">
              <label className="control grow">
                <span>Topology metric</span>
                <select
                  value={topology}
                  onChange={(event) => setTopology(event.target.value)}
                >
                  {catalog.data.data.topology_metrics.map((item) => (
                    <option key={item} value={item}>
                      {titleCase(item)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Microstructure metric</span>
                <select
                  value={microstructure}
                  onChange={(event) =>
                    setMicrostructure(event.target.value)
                  }
                >
                  {catalog.data.data.microstructure_metrics.map((item) => (
                    <option key={item} value={item}>
                      {titleCase(item)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Regional value</span>
                <select
                  value={valueMetric}
                  onChange={(event) => setValueMetric(event.target.value)}
                >
                  {catalog.data.data.value_metrics.map((item) => (
                    <option key={item} value={item}>
                      {titleCase(item)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control grow">
                <span>Diagnostic contrast</span>
                <select
                  value={contrast}
                  onChange={(event) => setContrast(event.target.value)}
                >
                  {CONTRASTS.map(([first, second]) => (
                    <option
                      key={`${first}-${second}`}
                      value={`${first} vs ${second}`}
                    >
                      {first} versus {second}
                    </option>
                  ))}
                </select>
              </label>
              <label className="control compact-number-control">
                <span>Top regions</span>
                <input
                  type="number"
                  min={5}
                  max={80}
                  step={5}
                  value={topN}
                  onChange={(event) =>
                    setTopN(
                      Math.max(
                        5,
                        Math.min(80, Number(event.target.value) || 20),
                      ),
                    )
                  }
                />
              </label>
            </div>
            <div className="definition-grid">
              <div className="definition-callout">
                <strong>COUPLING INDEX</strong>
                <span>
                  {catalog.data.data.definition.coupling_index}
                </span>
              </div>
              <div className="definition-callout">
                <strong>MICROSTRUCTURE / TOPOLOGY RATIO</strong>
                <span>
                  {catalog.data.data.definition.micro_topology_ratio}
                </span>
              </div>
              <div className="definition-callout">
                <strong>INFERENCE</strong>
                <span>{catalog.data.data.definition.inference}</span>
              </div>
            </div>
          </>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading
          index="02"
          title="AAL3 regional contrast ranking"
        />
        <div className="control-row wrap">
          <label className="toggle-control">
            <input
              type="checkbox"
              checked={nominalOnly}
              onChange={(event) => {
                setNominalOnly(event.target.checked);
                if (event.target.checked) setFdrOnly(false);
              }}
            />
            Nominal p &lt; 0.05 only
          </label>
          <label className="toggle-control">
            <input
              type="checkbox"
              checked={fdrOnly}
              onChange={(event) => {
                setFdrOnly(event.target.checked);
                if (event.target.checked) setNominalOnly(false);
              }}
            />
            BH-FDR q &lt; 0.05 only
          </label>
        </div>
        {ranking.isLoading ? (
          <Loading label="Computing subject-level regional ranking" />
        ) : ranking.isError ? (
          <ErrorState error={ranking.error} />
        ) : ranking.data ? (
          <>
            <div className="metric-strip">
              <div>
                <span>Regions tested</span>
                <strong>{ranking.data.data.rows.length}</strong>
              </div>
              <div>
                <span>Nominal p &lt; 0.05</span>
                <strong>{nominalCount}</strong>
              </div>
              <div>
                <span>BH-FDR q &lt; 0.05</span>
                <strong>{fdrCount}</strong>
              </div>
              <div>
                <span>Visible after filter</span>
                <strong>{filteredRows.length}</strong>
              </div>
            </div>
            {filteredRows.length ? (
              <div className="analysis-split">
                <NodeRankingPlot
                  rows={filteredRows}
                  groupA={groupA}
                  groupB={groupB}
                  topN={topN}
                />
                <DataTable
                  rows={filteredRows.slice(0, topN)}
                  compact
                  maxHeight={850}
                  caption="Selected regional ranking"
                />
              </div>
            ) : (
              <div className="mapping-warning">
                <strong>No regions pass the active filter.</strong> This is
                an explicit corrected null result for the selected
                specification; disable the filter to inspect the complete
                exploratory ranking.
              </div>
            )}
            <div className="analysis-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={!ranking.data.data.rows.length}
                onClick={() =>
                  downloadRows(
                    `coupling_aal_${topology}_${microstructure}_${valueMetric}_${groupA}_${groupB}.csv`,
                    ranking.data.data.rows,
                  )
                }
              >
                Download complete ranking
              </button>
            </div>
            <p className="provenance-line">
              {ranking.data.provenance.calculation}
            </p>
          </>
        ) : null}
      </section>

      <section className="panel">
        <SectionHeading index="OUTPUTS" title="Complete recorded outputs" />
        {artifacts.isLoading ? (
          <Loading label="Indexing coupling-AAL outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
