import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJson } from "./api";
import type { HealthResponse, Section } from "./types";
import { ErrorState, Loading } from "./components/AsyncState";
import { HomePage } from "./pages/HomePage";
import { DemographicsPage } from "./pages/DemographicsPage";
import { CouplingAalPage } from "./pages/CouplingAalPage";
import { LrSrAnalysisPage, LrSrModelsPage } from "./pages/LrSrAnalysisPage";
import { MatrixViewerPage } from "./pages/MatrixViewerPage";
import { MlDiagnosticsPage } from "./pages/MlDiagnosticsPage";
import { NetworkAnalysisPage } from "./pages/NetworkAnalysisPage";
import { NovelFindingsPage } from "./pages/NovelFindingsPage";
import { PipelinePage } from "./pages/PipelinePage";
import { SectionPage } from "./pages/SectionPage";

interface CohortSummary {
  subjects: number;
  group_counts: Record<string, number>;
  columns: string[];
  diffusivity_subjects: number | null;
}

function currentRoute(): string {
  const value = window.location.hash.replace(/^#/, "");
  return value.startsWith("/") ? value : "/";
}

function App() {
  const [route, setRoute] = useState(currentRoute);
  const [mobileNav, setMobileNav] = useState(false);
  const sections = useQuery({
    queryKey: ["sections"],
    queryFn: ({ signal }) => getJson<Section[]>("/sections", signal),
  });
  const cohort = useQuery({
    queryKey: ["cohort-summary"],
    queryFn: ({ signal }) =>
      getJson<CohortSummary>("/cohort/summary", signal),
  });
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async ({ signal }) => {
      const response = await fetch("/api/v1/health/ready", { signal });
      if (!response.ok) throw new Error("API not ready");
      return response.json() as Promise<HealthResponse>;
    },
    refetchInterval: 30_000,
    retry: false,
  });
  useEffect(() => {
    const listener = () => {
      setRoute(currentRoute());
      setMobileNav(false);
      window.scrollTo({ top: 0, behavior: "instant" });
    };
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  const navigate = (path: string) => {
    window.location.hash = path;
  };
  const grouped = useMemo(() => {
    const result = new Map<string, Section[]>();
    sections.data?.data.forEach((section) => {
      const current = result.get(section.group) ?? [];
      current.push(section);
      result.set(section.group, current);
    });
    // synthetic front-end-only entry: the cognition-models workspace split out
    // of the LR-SR Analysis page (no backend section behind it)
    for (const [, items] of result) {
      const at = items.findIndex((s) => s.id === "lr-sr-analysis");
      if (at >= 0) {
        items.splice(at + 1, 0, {
          id: "lr-sr-models",
          label: "LR-SR Models",
          group: items[at].group,
          description: "Cognition models on the tiered connectome features",
          folders: [],
          subject_table: null,
          node_table: null,
          artifact_count: 0,
          metrics: { available: false, metrics: [] },
        });
        break;
      }
    }
    return result;
  }, [sections.data]);

  if (sections.isLoading || cohort.isLoading) {
    return (
      <div className="app-boot">
        <div className="brand-mark large">SC</div>
        <Loading label="Loading connectome contracts" />
      </div>
    );
  }
  if (sections.isError) return <ErrorState error={sections.error} />;
  if (cohort.isError) return <ErrorState error={cohort.error} />;
  if (!sections.data || !cohort.data) return null;

  const sectionId = route.startsWith("/section/")
    ? route.slice("/section/".length)
    : "";
  const selectedSection = sections.data.data.find(
    (section) => section.id === sectionId,
  );
  let page;
  if (route === "/") {
    page = (
      <HomePage
        sections={sections.data.data}
        cohort={cohort.data.data}
        navigate={navigate}
      />
    );
  } else if (
    route === "/section/lr-sr-analysis" ||
    route === "/lr-sr-analysis"
  ) {
    page = <LrSrAnalysisPage />;
  } else if (route === "/section/lr-sr-models") {
    page = <LrSrModelsPage />;
  } else if (
    route === "/section/sc-matrix-viewer" ||
    route === "/matrix"
  ) {
    page = <MatrixViewerPage />;
  } else if (
    selectedSection?.id === "demographics"
  ) {
    page = <DemographicsPage section={selectedSection} />;
  } else if (selectedSection?.id === "novel-findings") {
    page = <NovelFindingsPage section={selectedSection} />;
  } else if (selectedSection?.id === "coupling-aal") {
    page = <CouplingAalPage section={selectedSection} />;
  } else if (selectedSection?.id === "ml-diagnostics") {
    page = <MlDiagnosticsPage section={selectedSection} />;
  } else if (selectedSection?.id === "network-analysis") {
    page = <NetworkAnalysisPage section={selectedSection} />;
  } else if (route === "/pipeline") {
    page = <PipelinePage />;
  } else if (selectedSection) {
    page = <SectionPage section={selectedSection} />;
  } else {
    page = (
      <main className="content-page">
        <ErrorState error={new Error(`Unknown route: ${route}`)} />
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "mobile-open" : ""}`}>
        <button
          type="button"
          className="brand"
          onClick={() => navigate("/")}
        >
          <span className="brand-mark">SC</span>
          <span>
            <strong>Connectome</strong>
            <small>Observatory</small>
          </span>
        </button>
        <div className="migration-badge">
          <span>React preview</span>
          <small>Read-only shadow</small>
        </div>
        <nav className="sidebar-nav" aria-label="Analysis sections">
          <button
            type="button"
            className={route === "/" ? "active" : ""}
            onClick={() => navigate("/")}
          >
            <span className="nav-icon">⌂</span>
            Overview
          </button>
          {Array.from(grouped.entries()).map(([group, items]) => (
            <div className="nav-group" key={group}>
              <div className="nav-group-label">{group}</div>
              {items.map((section) => (
                <button
                  type="button"
                  key={section.id}
                  className={
                    sectionId === section.id ||
                    (section.id === "lr-sr-analysis" &&
                      route === "/lr-sr-analysis")
                      ? "active"
                      : ""
                  }
                  onClick={() => navigate(`/section/${section.id}`)}
                >
                  <span className="nav-dot" />
                  <span>{section.label}</span>
                  <small>{section.artifact_count}</small>
                </button>
              ))}
            </div>
          ))}
          <div className="nav-group">
            <div className="nav-group-label">Operations</div>
            <button
              type="button"
              className={route === "/pipeline" ? "active" : ""}
              onClick={() => navigate("/pipeline")}
            >
              <span className="nav-icon">↻</span>
              Pipeline monitor
            </button>
          </div>
        </nav>
        <div className="sidebar-footer">
          <a href="/" target="_blank" rel="noreferrer">
            Open Streamlit reference ↗
          </a>
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API contract ↗
          </a>
        </div>
      </aside>
      <div
        className={`mobile-scrim ${mobileNav ? "visible" : ""}`}
        onClick={() => setMobileNav(false)}
        aria-hidden="true"
      />
      <div className="workspace">
        <header className="topbar">
          <button
            className="mobile-menu"
            type="button"
            onClick={() => setMobileNav((value) => !value)}
            aria-label="Toggle navigation"
          >
            ☰
          </button>
          <div className="topbar-context">
            <span>AD structural-connectome programme</span>
            <strong>
              {cohort.data.data.subjects} subjects ·{" "}
              {cohort.data.data.diffusivity_subjects ?? "—"} diffusivity
            </strong>
          </div>
          <div className="topbar-actions">
            <span
              className={`api-state ${
                health.isSuccess ? "online" : "offline"
              }`}
            >
              <span />
              {health.isSuccess ? "API ready" : "API checking"}
            </span>
            <span className="schema-chip">schema v1.0</span>
          </div>
        </header>
        <div className="workspace-scroll">{page}</div>
      </div>
    </div>
  );
}

export default App;
