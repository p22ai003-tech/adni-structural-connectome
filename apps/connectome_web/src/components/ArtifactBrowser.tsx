import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  artifactDownloadUrl,
  getJson,
  humanBytes,
} from "../api";
import type { Artifact, TablePayload } from "../types";
import { DataTable } from "./DataTable";
import { ErrorState, Loading } from "./AsyncState";

function ArtifactCard({ artifact }: { artifact: Artifact }) {
  const [expanded, setExpanded] = useState(false);
  const table = useQuery({
    queryKey: ["artifact-table", artifact.id],
    queryFn: ({ signal }) =>
      getJson<TablePayload>(
        `/artifacts/${artifact.id}/table?limit=200`,
        signal,
      ),
    enabled: expanded && artifact.kind === "table",
  });
  const content = useQuery({
    queryKey: ["artifact-content", artifact.id],
    queryFn: ({ signal }) =>
      getJson<{
        artifact: Artifact;
        content: unknown;
      }>(`/artifacts/${artifact.id}/content`, signal),
    enabled:
      expanded && (artifact.kind === "note" || artifact.kind === "json"),
  });
  const downloadable = artifactDownloadUrl(artifact.id);
  const isImage = [".png", ".svg"].includes(artifact.suffix);

  return (
    <article className={`artifact-card ${expanded ? "expanded" : ""}`}>
      <button
        className="artifact-summary"
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className={`artifact-icon ${artifact.kind}`}>
          {isImage
            ? "◫"
            : artifact.kind === "table"
              ? "▦"
              : artifact.kind === "note"
                ? "¶"
                : "{}"}
        </span>
        <span className="artifact-title">
          <strong>{artifact.name}</strong>
          <small>
            {artifact.relative_path} · {humanBytes(artifact.size_bytes)}
          </small>
        </span>
        <span className="artifact-action">
          {expanded ? "Close" : "Inspect"}
        </span>
      </button>
      {expanded ? (
        <div className="artifact-body">
          {isImage ? (
            <a href={downloadable} target="_blank" rel="noreferrer">
              <img
                className="artifact-image"
                src={downloadable}
                alt={artifact.name}
                loading="lazy"
              />
            </a>
          ) : null}
          {artifact.suffix === ".pdf" ? (
            <div className="artifact-download-card">
              <p>
                PDF preview stays isolated from the application. Open the
                source document in a new tab.
              </p>
              <a href={downloadable} target="_blank" rel="noreferrer">
                Open PDF
              </a>
            </div>
          ) : null}
          {artifact.kind === "table" ? (
            table.isLoading ? (
              <Loading label="Loading table preview" />
            ) : table.isError ? (
              <ErrorState error={table.error} />
            ) : table.data ? (
              <DataTable
                rows={table.data.data.rows}
                columns={table.data.data.columns}
                initialRows={50}
                caption={`First ${table.data.data.rows.length.toLocaleString()} of ${table.data.data.total_rows.toLocaleString()} rows`}
              />
            ) : null
          ) : null}
          {artifact.kind === "note" || artifact.kind === "json" ? (
            content.isLoading ? (
              <Loading label="Loading artifact" />
            ) : content.isError ? (
              <ErrorState error={content.error} />
            ) : content.data ? (
              <pre className="artifact-text">
                {typeof content.data.data.content === "string"
                  ? content.data.data.content
                  : JSON.stringify(content.data.data.content, null, 2)}
              </pre>
            ) : null
          ) : null}
          <a
            className="artifact-download"
            href={downloadable}
            target="_blank"
            rel="noreferrer"
          >
            Download source
          </a>
        </div>
      ) : null}
    </article>
  );
}

export function ArtifactBrowser({
  artifacts,
}: {
  artifacts: Artifact[];
}) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("all");
  const [showAll, setShowAll] = useState(false);
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return artifacts.filter((artifact) => {
      const matchesKind =
        kind === "all" ||
        artifact.kind === kind ||
        (kind === "image" &&
          [".png", ".svg"].includes(artifact.suffix));
      const matchesQuery =
        !normalized ||
        artifact.name.toLowerCase().includes(normalized) ||
        artifact.relative_path.toLowerCase().includes(normalized);
      return matchesKind && matchesQuery;
    });
  }, [artifacts, kind, query]);
  const visible = showAll ? filtered : filtered.slice(0, 24);

  return (
    <div>
      <div className="artifact-toolbar">
        <label>
          <span>Find output</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search file or folder"
          />
        </label>
        <label>
          <span>Output type</span>
          <select
            value={kind}
            onChange={(event) => setKind(event.target.value)}
          >
            <option value="all">All outputs</option>
            <option value="image">Figures</option>
            <option value="table">Tables</option>
            <option value="note">Notes</option>
            <option value="json">JSON</option>
          </select>
        </label>
        <div className="artifact-count">
          {filtered.length.toLocaleString()} matched
        </div>
      </div>
      <div className="artifact-list">
        {visible.map((artifact) => (
          <ArtifactCard key={artifact.id} artifact={artifact} />
        ))}
      </div>
      {filtered.length > 24 ? (
        <button
          className="secondary-button"
          type="button"
          onClick={() => setShowAll((value) => !value)}
        >
          {showAll ? "Show first 24" : `Show all ${filtered.length}`}
        </button>
      ) : null}
    </div>
  );
}
