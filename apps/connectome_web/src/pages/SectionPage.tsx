import { useQuery } from "@tanstack/react-query";
import { getJson } from "../api";
import type {
  AnalysisProfile,
  Artifact,
  Section,
} from "../types";
import { ArtifactBrowser } from "../components/ArtifactBrowser";
import { ErrorState, Loading } from "../components/AsyncState";
import { NodeAnalysisWorkspace } from "../components/NodeAnalysisWorkspace";
import { SectionHeading } from "../components/SectionHeading";
import { SubjectAnalysisWorkspace } from "../components/SubjectAnalysisWorkspace";

export function SectionPage({ section }: { section: Section }) {
  const artifacts = useQuery({
    queryKey: ["section-artifacts", section.id],
    queryFn: ({ signal }) =>
      getJson<Artifact[]>(
        `/sections/${section.id}/artifacts`,
        signal,
      ),
  });
  const profile = useQuery({
    queryKey: ["section-analysis-profile", section.id],
    queryFn: ({ signal }) =>
      getJson<AnalysisProfile>(
        `/sections/${section.id}/analysis/profile`,
        signal,
      ),
  });
  return (
    <main className="content-page">
      <div className="page-title">
        <div>
          <span className="eyebrow">{section.group}</span>
          <h1>{section.label}</h1>
          <p>{section.description}</p>
        </div>
        <div className="page-stat">
          <strong>{section.artifact_count.toLocaleString()}</strong>
          <span>auditable outputs</span>
        </div>
      </div>

      {profile.isLoading ? (
        <Loading label="Loading scientific analysis contract" />
      ) : profile.isError ? (
        <ErrorState error={profile.error} />
      ) : profile.data?.data.node.available ? (
        <NodeAnalysisWorkspace
          sectionId={section.id}
          metrics={profile.data.data.node.metrics}
        />
      ) : profile.data?.data.subject.available ? (
        <SubjectAnalysisWorkspace
          sectionId={section.id}
          metrics={profile.data.data.subject.metrics}
        />
      ) : (
        <section className="panel">
          <SectionHeading index="STATUS" title="Analysis interface" />
          <div className="mapping-warning">
            This section has no generic subject/node contract. Its specialised
            interface is being served from the audited outputs below; source
            files remain read-only.
          </div>
        </section>
      )}

      <section className="panel">
        <SectionHeading
          index="OUTPUTS"
          title="Figures, tables, and source notes"
        />
        <p className="section-intro">
          Files are read directly from the existing analysis tree. Expanding a
          card does not recalculate or overwrite the source artifact.
        </p>
        {artifacts.isLoading ? (
          <Loading label="Indexing section outputs" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} />
        ) : artifacts.data ? (
          <ArtifactBrowser artifacts={artifacts.data.data} />
        ) : null}
      </section>
    </main>
  );
}
