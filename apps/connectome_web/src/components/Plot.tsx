import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-cartesian-dist-min";
import type { Config, Data, Layout } from "plotly.js";

const PlotlyComponent = createPlotlyComponent(Plotly);

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: {
    color: "#0f172a",
    family: "Inter, ui-sans-serif, system-ui, sans-serif",
  },
  margin: { l: 60, r: 24, t: 48, b: 56 },
  xaxis: {
    gridcolor: "#e2e8f0",
    zerolinecolor: "#cbd5e1",
    linecolor: "#94a3b8",
  },
  yaxis: {
    gridcolor: "#e2e8f0",
    zerolinecolor: "#cbd5e1",
    linecolor: "#94a3b8",
  },
  hoverlabel: {
    bgcolor: "#ffffff",
    bordercolor: "#cbd5e1",
    font: { color: "#0f172a" },
  },
};

const BASE_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

export function Plot({
  data,
  layout,
  config,
  className,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
}) {
  const legendVisible = layout?.showlegend !== false;
  const legendOrientation = layout?.legend?.orientation ?? "h";
  const horizontalLegend =
    legendVisible && legendOrientation === "h";
  const requestedTop = Number(layout?.margin?.t ?? 0);
  // Horizontal legends and outside bar labels both occupy the band above the
  // plotting rectangle. Reserve enough space globally so page-level layouts
  // cannot place the legend on top of the first data labels.
  const safeTop = Math.max(
    requestedTop,
    horizontalLegend ? 132 : legendVisible ? 92 : 70,
  );
  return (
    <div className={`plot-shell ${className ?? ""}`}>
      <PlotlyComponent
        data={data}
        layout={{
          ...BASE_LAYOUT,
          ...layout,
          title: {
            x: 0.02,
            xanchor: "left",
            y: 0.98,
            yanchor: "top",
            pad: { b: 12 },
            ...(typeof layout?.title === "object"
              ? layout.title
              : layout?.title
                ? { text: layout.title }
                : {}),
          },
          legend: {
            orientation: "h",
            x: 0,
            xanchor: "left",
            y: horizontalLegend ? 1.06 : 1,
            yanchor: "bottom",
            ...layout?.legend,
          },
          margin: {
            ...BASE_LAYOUT.margin,
            ...layout?.margin,
            t: safeTop,
          },
          xaxis: {
            ...BASE_LAYOUT.xaxis,
            ...layout?.xaxis,
            automargin: true,
          },
          yaxis: {
            ...BASE_LAYOUT.yaxis,
            ...layout?.yaxis,
            automargin: true,
          },
          autosize: true,
        }}
        config={{ ...BASE_CONFIG, ...config }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
