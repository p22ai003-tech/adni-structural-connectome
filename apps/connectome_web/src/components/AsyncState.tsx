import type { ReactNode } from "react";

export function Loading({ label = "Loading data" }: { label?: string }) {
  return (
    <div className="state-card loading-state" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({
  error,
  action,
}: {
  error: unknown;
  action?: ReactNode;
}) {
  const message =
    error instanceof Error ? error.message : "An unknown error occurred";
  return (
    <div className="state-card error-state" role="alert">
      <strong>Could not load this view</strong>
      <span>{message}</span>
      {action}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="state-card empty-state">{children}</div>;
}
