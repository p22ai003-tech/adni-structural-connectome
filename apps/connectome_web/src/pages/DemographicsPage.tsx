import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJson } from "../api";
import type {
  Artifact,
  DemographicsSummary,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { DataTable } from "../components/DataTable";
import {
  CohortCountPlot,
  GroupDistributionPlot,
} from "../components/ScientificPlots";
import { SectionHeading } from "../components/SectionHeading";

const GROUPS = ["CN", "MCI", "AD"] as const;

export function DemographicsPage({ section }: { section: Section }) {
  const [clinicalIndex, setClinicalIndex] = useState(0);
  const [showSubjectRows, setShowSubjectRows] = useState(false);
  const summary = useQuery({
    queryKey: ["demographics-summary"],
    queryFn: ({ signal }) =>
      getJson<DemographicsSummary>("/demographics/summary", signal),
  });
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(`/sections/${section.id}/artifacts`, signal),
  });
  const clinical = summary.data?.data.clinical[clinicalIndex];
  const total =
    summary.data?.data.cohort.reduce(
      (sum, row) => sum + Number(row.n_subjects ?? 0),
      0,
    ) ?? 0;

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>{section.description}</p>
        </div>
        <div className="page-stat">
          <strong>{total || "—"}</strong>
          <span>analysis subjects</span>
        </div>
      </div>

      {summary.isLoading ? (
        <Loading label="Loading cohort contract" />
      ) : summary.isError ? (
        <ErrorState error={summary.error} />
      ) : summary.data ? (
        <>
          <section className="panel">
            <SectionHeading index="01" title="Cohort composition" />
            <div className="metric-strip">
              {GROUPS.map((group) => {
                const row = summary.data!.data.cohort.find(
                  (item) => String(item.group) === group,
                );
                return (
                  <div key={group}>
                    <span>{group}</span>
                    <strong>
                      {Number(row?.n_subjects ?? 0).toLocaleString()}
                    </strong>
                  </div>
                );
              })}
              <div>
                <span>Total</span>
                <strong>{total.toLocaleString()}</strong>
              </div>
            </div>
            <div className="plot-grid two">
              <CohortCountPlot rows={summary.data.data.cohort} />
              <GroupDistributionPlot
                rows={summary.data.data.age_rows}
                valueKey="age"
                title="Age distribution by diagnostic group"
                yTitle="Age (years)"
                height={390}
              />
            </div>
            <DataTable
              rows={summary.data.data.cohort}
              compact
              caption="Cohort and processing completeness"
            />
            <div className="analysis-actions">
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={showSubjectRows}
                  onChange={(event) =>
                    setShowSubjectRows(event.target.checked)
                  }
                />
                Show authenticated subject-level age rows
              </label>
            </div>
            {showSubjectRows ? (
              <DataTable
                rows={summary.data.data.age_rows}
                compact
                maxHeight={520}
                initialRows={100}
                caption="Subject-level demographic rows"
              />
            ) : null}
          </section>

          <section className="panel">
            <SectionHeading index="02" title="Age inference" />
            <div className="table-grid two">
              <DataTable
                rows={summary.data.data.age_descriptives}
                compact
                caption="Age descriptives"
              />
              <DataTable
                rows={summary.data.data.age_pairwise}
                compact
                caption="Stored age pairwise tests"
              />
            </div>
          </section>

          <section className="panel">
            <SectionHeading index="03" title="Clinical-score availability" />
            {summary.data.data.clinical.length ? (
              <>
                <label className="control compact-control">
                  <span>Clinical measure</span>
                  <select
                    value={clinicalIndex}
                    onChange={(event) =>
                      setClinicalIndex(Number(event.target.value))
                    }
                  >
                    {summary.data.data.clinical.map((item, index) => (
                      <option key={item.metric} value={index}>
                        {item.metric}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="table-grid two">
                  <DataTable
                    rows={clinical?.descriptives ?? []}
                    compact
                    caption={`${clinical?.metric ?? "Clinical measure"} descriptives`}
                  />
                  <DataTable
                    rows={clinical?.pairwise ?? []}
                    compact
                    caption={`${clinical?.metric ?? "Clinical measure"} pairwise tests`}
                  />
                </div>
              </>
            ) : (
              <div className="table-empty">
                No clinical-score inference tables are available.
              </div>
            )}
            <details className="definitions-box">
              <summary>Definitions and inferential scope</summary>
              <ul>
                {Object.entries(summary.data.data.definitions).map(
                  ([key, definition]) => (
                    <li key={key}>
                      <strong>{key.replaceAll("_", " ")}:</strong>{" "}
                      {definition}
                    </li>
                  ),
                )}
              </ul>
            </details>
          </section>
        </>
      ) : null}

      <section className="panel">
        <SectionHeading index="OUTPUTS" title="Source tables and figures" />
        {artifacts.isLoading ? (
          <Loading label="Indexing demographic outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
