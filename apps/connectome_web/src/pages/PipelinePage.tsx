import { useQuery } from "@tanstack/react-query";
import { formatNumber, getJson } from "../api";
import type { PipelineLegacy, PipelineRelease } from "../types";
import { DataTable } from "../components/DataTable";
import { ErrorState, Loading } from "../components/AsyncState";
import { SectionHeading } from "../components/SectionHeading";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function Freshness({
  generated,
  label,
}: {
  generated?: unknown;
  label: string;
}) {
  const date = generated ? new Date(String(generated)) : null;
  const valid = date && Number.isFinite(date.getTime());
  return (
    <div className="freshness">
      <span className="pulse-dot" />
      <span>
        {label}: {valid ? date.toLocaleString() : "timestamp unavailable"}
      </span>
    </div>
  );
}

export function PipelinePage() {
  const release = useQuery({
    queryKey: ["pipeline-release"],
    queryFn: ({ signal }) =>
      getJson<PipelineRelease>("/pipeline/release-status", signal),
    refetchInterval: 20_000,
    staleTime: 5_000,
  });
  const legacy = useQuery({
    queryKey: ["pipeline-legacy"],
    queryFn: ({ signal }) =>
      getJson<PipelineLegacy>("/pipeline/live-status", signal),
    refetchInterval: 20_000,
    staleTime: 5_000,
  });
  if (release.isLoading || legacy.isLoading) {
    return <Loading label="Reading pipeline ledgers" />;
  }
  if (release.isError) return <ErrorState error={release.error} />;
  if (legacy.isError) return <ErrorState error={legacy.error} />;
  if (!release.data || !legacy.data) return null;

  const liveSummary = asRecord(release.data.data.summary);
  const legacyStatus = asRecord(legacy.data.data.status);
  const overall = asRecord(legacyStatus.overall);
  const manifest = asRecord(legacy.data.data.manifest_summary);
  const manifestDensity = asRecord(manifest.density);
  const topology = asRecord(release.data.data.topology);
  const laneCounts = asRecord(topology.lane_counts);
  const laneRows = Object.entries(laneCounts).map(([lane, value]) => ({
    Lane: lane,
    ...asRecord(value),
  }));
  const cohort = asRecord(legacyStatus.cohort);
  const cohortRows = Object.entries(cohort).map(([group, value]) => ({
    Group: group,
    ...asRecord(value),
  }));
  const releaseStatus = String(liveSummary.status ?? "UNKNOWN");
  const releaseRows = Array.isArray(liveSummary.rows)
    ? (liveSummary.rows as Record<string, unknown>[])
    : [];
  const fullReleaseRow = releaseRows.find(
    (row) => String(row.processing_group) === "Full release total",
  );
  const pretractComplete = Number(
    fullReleaseRow?.pretract_completed_n ?? 0,
  );

  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">Operations · Read-only monitor</span>
          <h1>530-Subject Pipeline Monitor</h1>
          <p>
            Parallel view of the HCP379 recovery/release contract and the
            existing AAL3 production outputs. This page never starts, stops, or
            modifies a worker.
          </p>
        </div>
        <div
          className={`status-pill ${
            releaseStatus.includes("PASS") ? "success" : "warning"
          }`}
        >
          {releaseStatus.replaceAll("_", " ")}
        </div>
      </div>
      <div className="freshness-row">
        <Freshness
          generated={liveSummary.generated_utc}
          label="HCP379 snapshot"
        />
        <Freshness generated={legacyStatus.ts} label="AAL3 monitor" />
        <span>Auto-refresh: 20 seconds</span>
      </div>

      <section className="panel">
        <SectionHeading index={1} title="CURRENT COHORT SNAPSHOT" />
        <div className="metric-strip pipeline-metrics">
          <div>
            <span>HCP379 target</span>
            <strong>{formatNumber(liveSummary.total_n ?? 530, 0)}</strong>
          </div>
          <div>
            <span>Pretract complete</span>
            <strong>{formatNumber(pretractComplete, 0)}</strong>
          </div>
          <div>
            <span>Existing AAL3 complete</span>
            <strong>{formatNumber(overall.done, 0)}</strong>
          </div>
          <div>
            <span>Existing AAL3 median density</span>
            <strong>{formatNumber(overall.median_d, 3)}</strong>
          </div>
          <div>
            <span>AAL3 density ≥ 0.60</span>
            <strong>{formatNumber(manifestDensity.ge_0_60_n, 0)}</strong>
          </div>
        </div>
        <DataTable
          rows={release.data.data.group_rows}
          columns={[
            "processing_group",
            "target_n",
            "pretract_completed_n",
            "running_n",
            "corrective_hold_n",
            "waiting_n",
            "completion_fraction",
            "existing_aal3_166_median_density",
            "existing_hcp379_median_density",
            "corrected_hcp379_status",
          ]}
          maxHeight={460}
          initialRows={20}
          compact
          caption="HCP379 processing groups"
        />
      </section>

      <section className="panel">
        <SectionHeading index={2} title="RECOVERY LANES AND RELEASE GATES" />
        <div className="plot-grid two">
          <DataTable
            rows={release.data.data.ledger_summary}
            maxHeight={520}
            initialRows={100}
            compact
            caption="Current ledger states by lane"
          />
          <div>
            <DataTable
              rows={laneRows}
              columns={[
                "Lane",
                "source_n",
                "canary_overlap_excluded_n",
                "production_n",
                "pretract_ready_n",
              ]}
              maxHeight={260}
              initialRows={30}
              compact
              caption="Balanced release topology"
            />
            <div className="gate-list">
              {[
                [
                  "Exact cohort accounting",
                  topology.exact_15_plus_515_equals_530,
                ],
                ["Cohort uniform recipe", topology.cohort_uniform],
                ["Non-overwriting outputs", topology.non_overwriting],
                [
                  "All-nine canary required",
                  topology.requires_complete_hroi_all_nine_canary,
                ],
                [
                  "Final human review required",
                  topology.requires_final_hroi_human_review,
                ],
                ["Final release authorized", topology.final_release_authorized],
              ].map(([label, value]) => (
                <div className="gate-row" key={String(label)}>
                  <span
                    className={`gate-icon ${
                      value === true ? "pass" : value === false ? "hold" : ""
                    }`}
                  >
                    {value === true ? "✓" : value === false ? "–" : "?"}
                  </span>
                  <span>{String(label)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <SectionHeading index={3} title="EXISTING AAL3 PRODUCTION REFERENCE" />
        <DataTable
          rows={cohortRows}
          maxHeight={340}
          initialRows={10}
          compact
          caption="Completed matrices and density by diagnostic cohort"
        />
        <p className="method-note">
          This is the existing AAL3 production reference, not evidence that an
          HCP379 output has passed its all-nine release contract. The two
          pipelines remain separately labelled to prevent accidental mixing.
        </p>
      </section>
    </main>
  );
}
