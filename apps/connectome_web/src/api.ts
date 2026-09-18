import type { ApiEnvelope } from "./types";

export const API_ROOT = "/api/v1";

export async function getJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<ApiEnvelope<T>> {
  const response = await fetch(`${API_ROOT}${path}`, {
    signal,
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
    },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Preserve the status-based message when a proxy returns HTML.
    }
    throw new Error(detail);
  }
  return (await response.json()) as ApiEnvelope<T>;
}

export function artifactDownloadUrl(artifactId: string): string {
  return `${API_ROOT}/artifacts/${encodeURIComponent(artifactId)}/download`;
}

export function formatNumber(
  value: unknown,
  digits = 3,
): string {
  if (value === null || value === undefined || value === "") return "NA";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (Math.abs(number) >= 100_000) {
    return number.toLocaleString(undefined, {
      maximumFractionDigits: 0,
    });
  }
  if (Math.abs(number) > 0 && Math.abs(number) < 0.001) {
    return number.toExponential(2);
  }
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

export function humanBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text =
    typeof value === "object" ? JSON.stringify(value) : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

export function downloadRows(
  filename: string,
  rows: Record<string, unknown>[],
): void {
  if (!rows.length) return;
  const columns = Array.from(
    new Set(rows.flatMap((row) => Object.keys(row))),
  );
  const content = [
    columns.map(csvCell).join(","),
    ...rows.map((row) =>
      columns.map((column) => csvCell(row[column])).join(","),
    ),
  ].join("\n");
  const url = URL.createObjectURL(
    new Blob([content], { type: "text/csv;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
