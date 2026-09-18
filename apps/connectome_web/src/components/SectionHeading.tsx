import type { ReactNode } from "react";

export function SectionHeading({
  index,
  title,
  aside,
}: {
  index?: number | string;
  title: string;
  aside?: ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        <span className="section-kicker">
          {index !== undefined ? `${index} · ` : ""}
        </span>
        <h2>{title}</h2>
      </div>
      {aside ? <div className="section-heading-aside">{aside}</div> : null}
    </div>
  );
}
