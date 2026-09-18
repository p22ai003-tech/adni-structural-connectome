import type { Section } from "../types";
import { SectionHeading } from "../components/SectionHeading";

export function HomePage({
  sections,
  cohort,
  navigate,
}: {
  sections: Section[];
  cohort: {
    subjects?: number;
    group_counts?: Record<string, number>;
    diffusivity_subjects?: number | null;
  };
  navigate: (path: string) => void;
}) {
  const groups = Array.from(new Set(sections.map((section) => section.group)));
  return (
    <main className="content-page home-page">
      <div className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Structural connectomics · AD continuum</span>
          <h1>
            From microstructure to network geometry,
            <em> one auditable analysis surface.</em>
          </h1>
          <p>
            A read-only React migration over the existing 530-subject
            connectome outputs, with subject-level provenance and the original
            statistical definitions preserved.
          </p>
          <div className="hero-actions">
            <button
              type="button"
              onClick={() => navigate("/section/lr-sr-analysis")}
            >
              Open LR–SR analysis
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => navigate("/pipeline")}
            >
              View pipeline monitor
            </button>
          </div>
        </div>
        <div className="cohort-orbit">
          <div className="orbit-core">
            <strong>{cohort.subjects ?? "—"}</strong>
            <span>subjects</span>
          </div>
          {["CN", "MCI", "AD"].map((group, index) => (
            <div
              className={`orbit-node orbit-${index + 1}`}
              key={group}
            >
              <strong>{cohort.group_counts?.[group] ?? "—"}</strong>
              <span>{group}</span>
            </div>
          ))}
        </div>
      </div>

      <section className="panel home-index">
        <SectionHeading index="INDEX" title="ANALYSIS FAMILIES" />
        <div className="section-card-grid">
          {groups.map((group) => {
            const groupSections = sections.filter(
              (section) => section.group === group,
            );
            return (
              <article className="section-family-card" key={group}>
                <span className="family-number">
                  {String(groups.indexOf(group) + 1).padStart(2, "0")}
                </span>
                <h3>{group}</h3>
                <ul>
                  {groupSections.map((section) => (
                    <li key={section.id}>
                      <button
                        type="button"
                        onClick={() =>
                          navigate(`/section/${section.id}`)
                        }
                      >
                        <span>{section.label}</span>
                        <small>
                          {section.artifact_count.toLocaleString()} outputs
                        </small>
                      </button>
                    </li>
                  ))}
                </ul>
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}
