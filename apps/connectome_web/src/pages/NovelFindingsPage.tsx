import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Data } from "plotly.js";
import { getJson } from "../api";
import type {
  Artifact,
  NovelFindingsSummary,
  NoveltyCard,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { DataTable } from "../components/DataTable";
import { Plot } from "../components/Plot";
import { SectionHeading } from "../components/SectionHeading";

const FAMILY_COLORS: Record<string, string> = {
  microstructure: "#38bdf8",
  graph: "#a3e635",
  coupling: "#f5c842",
  age: "#a78bfa",
};

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function FindingsPlot({
  rows,
}: {
  rows: Record<string, unknown>[];
}) {
  const display = [...rows]
    .filter(
      (row) =>
        Number(row.q) > 0 &&
        Number.isFinite(Number(row.effect_cliffs_delta)),
    )
    .sort((first, second) => Number(first.q) - Number(second.q))
    .slice(0, 40)
    .reverse();
  const families = Array.from(
    new Set(display.map((row) => String(row.feature_family))),
  );
  const traces: Data[] = families.map((family) => {
    const familyRows = display.filter(
      (row) => String(row.feature_family) === family,
    );
    return {
      type: "scatter",
      mode: "markers",
      name: titleCase(family),
      x: familyRows.map((row) => -Math.log10(Number(row.q))),
      y: familyRows.map(
        (row) =>
          `${String(row.network)} · ${titleCase(String(row.metric))}`,
      ),
      marker: {
        color: FAMILY_COLORS[family] ?? "#94a3b8",
        size: familyRows.map(
          (row) =>
            8 + 18 * Math.min(1, Math.abs(Number(row.effect_cliffs_delta))),
        ),
        opacity: 0.82,
        line: { color: "#0b0f17", width: 1 },
      },
      customdata: familyRows.map((row) => [
        row.scheme,
        row.direction,
        row.effect_cliffs_delta,
        row.q,
        row.n,
      ]),
      hovertemplate:
        "%{y}<br>−log10(q)=%{x:.2f}" +
        "<br>scheme=%{customdata[0]}<br>%{customdata[1]}" +
        "<br>effect=%{customdata[2]:.3f}" +
        "<br>q=%{customdata[3]:.3g}<br>n=%{customdata[4]}" +
        "<extra></extra>",
    } as Data;
  });
  return (
    <Plot
      data={traces}
      layout={{
        height: 980,
        title: {
          text: "Top FDR-supported network findings",
          font: { size: 16 },
        },
        xaxis: {
          title: { text: "Statistical support (−log10 BH-FDR q)" },
          rangemode: "tozero",
        },
        yaxis: { automargin: true },
        legend: { orientation: "h" },
        margin: { l: 235 },
      }}
    />
  );
}

function FindingCard({ card, index }: { card: NoveltyCard; index: number }) {
  return (
    <article className="finding-card">
      <div className="finding-card-head">
        <span>{String(index + 1).padStart(2, "0")}</span>
        <div>
          <h3>{card.theme}</h3>
          <div className="finding-pills">
            <span>{titleCase(card.novelty_type)}</span>
            <span>Confidence: {card.confidence}</span>
          </div>
        </div>
      </div>
      <div className="finding-body">
        <div>
          <strong>Recorded finding</strong>
          <p>{card.our_finding}</p>
        </div>
        <div>
          <strong>Established context</strong>
          <p>{card.prior_literature}</p>
        </div>
        {card.frontier_context ? (
          <div>
            <strong>Frontier context</strong>
            <p>{card.frontier_context}</p>
          </div>
        ) : null}
        <div className="finding-claim">
          <strong>Bounded novelty hypothesis</strong>
          <p>{card.novelty_claim}</p>
        </div>
        <div className="finding-caveat">
          <strong>Interpretation boundary</strong>
          <p>{card.caveat}</p>
        </div>
      </div>
      {card.citations.length ? (
        <details className="definitions-box">
          <summary>
            Supporting literature ({card.citations.length} records)
          </summary>
          <ul>
            {card.citations.map((citation, citationIndex) => (
              <li key={`${citation.title}-${citationIndex}`}>
                <a
                  href={citation.url ?? citation.doi}
                  target="_blank"
                  rel="noreferrer"
                >
                  {citation.title}
                </a>{" "}
                ({citation.year ?? "year not recorded"};{" "}
                {citation.venue ?? "venue not recorded"};{" "}
                {citation.tier ?? "context"})
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </article>
  );
}

export function NovelFindingsPage({ section }: { section: Section }) {
  const [typeFilter, setTypeFilter] = useState("all");
  const findings = useQuery({
    queryKey: ["novel-findings-summary"],
    queryFn: ({ signal }) =>
      getJson<NovelFindingsSummary>("/findings/summary", signal),
  });
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(`/sections/${section.id}/artifacts`, signal),
  });
  const noveltyTypes = useMemo(
    () =>
      Array.from(
        new Set(
          (findings.data?.data.cards ?? []).map(
            (card) => card.novelty_type,
          ),
        ),
      ).sort(),
    [findings.data],
  );
  const visibleCards = useMemo(
    () =>
      (findings.data?.data.cards ?? []).filter(
        (card) =>
          typeFilter === "all" || card.novelty_type === typeFilter,
      ),
    [findings.data, typeFilter],
  );

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>
            Evidence-led synthesis of the recorded FDR-controlled
            network findings, with the observation, literature context,
            novelty hypothesis, and caveat kept explicitly separate.
          </p>
        </div>
        <div className="page-stat">
          <strong>{findings.data?.data.catalog.length ?? "—"}</strong>
          <span>FDR-supported catalog rows</span>
        </div>
      </div>

      {findings.isLoading ? (
        <Loading label="Loading findings synthesis" />
      ) : findings.isError ? (
        <ErrorState error={findings.error} />
      ) : findings.data ? (
        <>
          <div className="mapping-warning">
            <strong>Claim boundary:</strong>{" "}
            {findings.data.data.contract.claim_scope}{" "}
            {findings.data.data.contract.functional_boundary}
          </div>

          <section className="panel">
            <SectionHeading
              index="01"
              title="Literature-grounded finding cards"
            />
            <div className="control-row">
              <label className="control compact-control">
                <span>Novelty category</span>
                <select
                  value={typeFilter}
                  onChange={(event) => setTypeFilter(event.target.value)}
                >
                  <option value="all">All categories</option>
                  {noveltyTypes.map((item) => (
                    <option key={item} value={item}>
                      {titleCase(item)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="finding-stack">
              {visibleCards.map((card, index) => (
                <FindingCard
                  key={`${card.theme}-${index}`}
                  card={card}
                  index={index}
                />
              ))}
            </div>
            <p className="method-note">
              {findings.data.data.contract.literature_boundary}
            </p>
          </section>

          <section className="panel">
            <SectionHeading
              index="02"
              title="Full significant-findings catalog"
            />
            <div className="analysis-split">
              <FindingsPlot rows={findings.data.data.catalog} />
              <DataTable
                rows={findings.data.data.catalog}
                compact
                maxHeight={980}
                initialRows={60}
                caption="FDR-supported rows ranked by q"
              />
            </div>
            <div className="table-grid two">
              <DataTable
                rows={findings.data.data.family_counts}
                compact
                caption="Supported findings by scheme and family"
              />
              <DataTable
                rows={Object.entries(
                  findings.data.data.confidence_counts,
                ).map(([confidence, count]) => ({
                  confidence,
                  synthesis_cards: count,
                }))}
                compact
                caption="Synthesis-card confidence labels"
              />
            </div>
          </section>
        </>
      ) : null}

      <section className="panel">
        <SectionHeading index="OUTPUTS" title="Complete auditable outputs" />
        {artifacts.isLoading ? (
          <Loading label="Indexing findings outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
