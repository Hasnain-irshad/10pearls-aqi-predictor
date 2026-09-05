import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { api, STATIC_MODE } from "../api";

const CATEGORY_COLORS = {
  Good: "#00e400",
  Moderate: "#ffde33",
  USG: "#ff9933",
  Unhealthy: "#ff5050",
  "Very Unhealthy": "#b25aff",
  Hazardous: "#c81d3f",
};

const getCategoryColor = (category) => CATEGORY_COLORS[category] || "#94a3b8";

const tooltipStyle = {
  background: "rgba(8,11,16,0.92)",
  border: "1px solid rgba(255,255,255,0.18)",
  borderRadius: 10,
  color: "#f8fafc",
};

const tooltipItemStyle = {
  color: "#38bdf8",
};

function heatColor(value) {
  if (value > 0) return `rgba(239, 68, 68, ${Math.min(Math.abs(value) * 1.5, 1)})`;
  if (value < 0) return `rgba(100, 149, 237, ${Math.min(Math.abs(value) * 1.5, 1)})`;
  return "transparent";
}

const AnalyticsPage = () => {
  const [stats, setStats] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchAnalytics = async () => {
      try {
        setLoading(true);
        const [statsData, shapGlobalData] = await Promise.all([
          api.statistics(),
          api.shapGlobal(),
        ]);
        if (isMounted) {
          setStats(statsData);
          setShapData(shapGlobalData);
          setError(null);
        }
      } catch (err) {
        console.error("Error fetching analytics:", err);
        if (isMounted) setError("Failed to load analytics data.");
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchAnalytics();
    return () => {
      isMounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="analytics-page">
        <div className="notice">Loading analytics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-page">
        <div className="notice error">{error}</div>
      </div>
    );
  }

  if (!stats || !shapData) {
    return (
      <div className="analytics-page">
        <div className="notice error">No data available.</div>
      </div>
    );
  }

  // Pre-process SHAP data (Top 15)
  const topFeatures = shapData.features
    ? [...shapData.features]
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 15)
        .reverse()
    : [];

  return (
    <div className="analytics-page">
      <style>{`
        .analytics-page {
          padding: 24px;
          max-width: 1400px;
          margin: 0 auto;
        }
        .page-title {
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 8px;
          background: linear-gradient(to right, #38bdf8, #2dd4bf);
          -webkit-background-clip: text;
          color: transparent;
        }
        .analytics-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 24px;
          margin-top: 24px;
        }
        .full-width {
          grid-column: 1 / -1;
        }
        .panel-card {
          background: rgba(15, 23, 42, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 16px;
          padding: 20px;
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }
        .panel-header {
          margin-bottom: 16px;
        }
        .panel-title {
          font-size: 18px;
          font-weight: 600;
          color: #f8fafc;
          margin: 0;
        }
        .muted-line {
          font-size: 14px;
          color: #94a3b8;
          margin: 4px 0 0 0;
        }
        .heatmap-container {
          overflow-x: auto;
          padding-bottom: 8px;
        }
        .heatmap-grid {
          display: inline-grid;
          gap: 2px;
          border-radius: 8px;
          background: rgba(255,255,255,0.05);
          padding: 16px;
        }
        .heatmap-cell {
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 11px;
          border-radius: 4px;
          cursor: pointer;
          position: relative;
          color: transparent;
          transition: transform 0.1s, box-shadow 0.1s;
        }
        .heatmap-cell:hover {
          color: #fff;
          z-index: 10;
          transform: scale(1.2);
          box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .heatmap-cell.label {
          background: transparent !important;
          color: #cbd5e1;
          justify-content: flex-end;
          padding-right: 8px;
          width: auto;
          white-space: nowrap;
          font-weight: 500;
        }
        .heatmap-cell.label-top {
          justify-content: flex-start;
          align-items: flex-end;
          padding-right: 0;
          padding-bottom: 8px;
          writing-mode: vertical-rl;
          transform: rotate(180deg);
          height: 120px;
        }
        @media (max-width: 900px) {
          .analytics-grid {
            grid-template-columns: 1fr;
          }
          .full-width {
            grid-column: 1 / -1;
          }
        }
      `}</style>

      <div>
        <h1 className="page-title">Analytics & Insights</h1>
        <p className="muted-line">
          Explore data distributions, global features, and city-level trends.
        </p>
      </div>

      <div className="analytics-grid">
        {/* 1. Global SHAP Feature Importance */}
        <div className="panel-card full-width">
          <div className="panel-header">
            <h2 className="panel-title">🔬 Global Feature Importance (SHAP)</h2>
            <p className="muted-line">
              Top 15 most influential features globally across the XGBoost model
            </p>
          </div>
          <div style={{ width: "100%", height: 450 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={topFeatures}
                margin={{ top: 10, right: 30, left: 120, bottom: 20 }}
              >
                <defs>
                  <linearGradient id="colorShap" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#2dd4bf" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#38bdf8" stopOpacity={1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={true} vertical={false} />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis
                  dataKey="label"
                  type="category"
                  stroke="#94a3b8"
                  tick={{ fill: "#cbd5e1", fontSize: 12 }}
                  width={110}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  itemStyle={tooltipItemStyle}
                  cursor={{ fill: "rgba(255,255,255,0.05)" }}
                />
                <Bar dataKey="importance" fill="url(#colorShap)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. AQI Distribution */}
        <div className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">📊 Historical AQI Distribution</h2>
            <p className="muted-line">Histogram of AQI values from training data</p>
          </div>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={stats.distribution || []}
                margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorDist" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="#2dd4bf" stopOpacity={0.3} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="bin_start" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItemStyle} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
                
                <ReferenceLine x={50} stroke={CATEGORY_COLORS["Good"]} strokeDasharray="3 3" label={{ position: 'top', value: '50', fill: CATEGORY_COLORS["Good"], fontSize: 12 }} />
                <ReferenceLine x={100} stroke={CATEGORY_COLORS["Moderate"]} strokeDasharray="3 3" label={{ position: 'top', value: '100', fill: CATEGORY_COLORS["Moderate"], fontSize: 12 }} />
                <ReferenceLine x={150} stroke={CATEGORY_COLORS["USG"]} strokeDasharray="3 3" label={{ position: 'top', value: '150', fill: CATEGORY_COLORS["USG"], fontSize: 12 }} />
                <ReferenceLine x={200} stroke={CATEGORY_COLORS["Unhealthy"]} strokeDasharray="3 3" label={{ position: 'top', value: '200', fill: CATEGORY_COLORS["Unhealthy"], fontSize: 12 }} />
                <ReferenceLine x={300} stroke={CATEGORY_COLORS["Very Unhealthy"]} strokeDasharray="3 3" label={{ position: 'top', value: '300', fill: CATEGORY_COLORS["Very Unhealthy"], fontSize: 12 }} />
                
                <Bar dataKey="count" fill="url(#colorDist)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 7. AQI Category Breakdown */}
        <div className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">🥧 AQI Category Breakdown</h2>
            <p className="muted-line">Proportion of days spent in each health category</p>
          </div>
          <div style={{ width: "100%", height: 350 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
                <Pie
                  data={stats.category_distribution || []}
                  dataKey="count"
                  nameKey="category"
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  label={({ percentage }) => `${percentage.toFixed(1)}%`}
                  labelLine={false}
                >
                  {(stats.category_distribution || []).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color || getCategoryColor(entry.category)} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItemStyle} />
                <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: "14px", color: "#f8fafc" }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. City Air Quality Ranking */}
        <div className="panel-card full-width">
          <div className="panel-header">
            <h2 className="panel-title">🏙️ City Air Quality Ranking</h2>
            <p className="muted-line">Average AQI across all monitored cities</p>
          </div>
          <div style={{ width: "100%", height: 600 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={[...(stats.city_ranking || [])].sort((a, b) => a.mean_aqi - b.mean_aqi)}
                margin={{ top: 20, right: 40, left: 100, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={true} vertical={false} />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis dataKey="city" type="category" stroke="#94a3b8" width={90} tick={{ fill: "#cbd5e1" }} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
                <Bar dataKey="mean_aqi" radius={[0, 4, 4, 0]}>
                  {(stats.city_ranking || []).sort((a, b) => a.mean_aqi - b.mean_aqi).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getCategoryColor(entry.category)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 5. Seasonal Pattern */}
        <div className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">📅 Seasonal Pattern</h2>
            <p className="muted-line">Average AQI by month</p>
          </div>
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stats.seasonality || []} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorSeason" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#a78bfa" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="month_name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItemStyle} />
                <ReferenceLine y={100} stroke={CATEGORY_COLORS["Moderate"]} strokeDasharray="3 3" />
                <ReferenceLine y={150} stroke={CATEGORY_COLORS["USG"]} strokeDasharray="3 3" />
                <Area type="monotone" dataKey="mean_aqi" stroke="#a78bfa" strokeWidth={3} fillOpacity={1} fill="url(#colorSeason)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 6. Daily Pattern (Diurnal) */}
        <div className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">⏰ Daily Pattern (Diurnal)</h2>
            <p className="muted-line">Average AQI by hour of day</p>
          </div>
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stats.diurnal || []} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorDiurnal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="hour" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItemStyle} />
                <Area type="monotone" dataKey="mean_aqi" stroke="#2dd4bf" strokeWidth={3} fillOpacity={1} fill="url(#colorDiurnal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 3. Correlation Heatmap */}
        <div className="panel-card full-width">
          <div className="panel-header">
            <h2 className="panel-title">🔥 Correlation Heatmap</h2>
            <p className="muted-line">Pearson correlation between meteorological features and AQI</p>
          </div>
          
          <div className="heatmap-container">
            {stats.correlation && stats.correlation.features && stats.correlation.matrix ? (
              <div 
                className="heatmap-grid" 
                style={{ 
                  gridTemplateColumns: `auto repeat(${stats.correlation.features.length}, 40px)` 
                }}
              >
                {/* Top Headers */}
                <div className="heatmap-cell label"></div>
                {stats.correlation.features.map((f, i) => (
                  <div key={`col-${i}`} className="heatmap-cell label label-top" title={f}>
                    {f.length > 12 ? f.substring(0, 10) + '..' : f}
                  </div>
                ))}

                {/* Matrix Rows */}
                {stats.correlation.matrix.map((row, rowIndex) => (
                  <React.Fragment key={`row-${rowIndex}`}>
                    <div className="heatmap-cell label" title={stats.correlation.features[rowIndex]}>
                      {stats.correlation.features[rowIndex].length > 15 
                        ? stats.correlation.features[rowIndex].substring(0, 13) + '..' 
                        : stats.correlation.features[rowIndex]}
                    </div>
                    {row.map((val, colIndex) => (
                      <div
                        key={`cell-${rowIndex}-${colIndex}`}
                        className="heatmap-cell"
                        style={{ backgroundColor: heatColor(val) }}
                        title={`${stats.correlation.features[rowIndex]} × ${stats.correlation.features[colIndex]}\nCorrelation: ${val != null ? val.toFixed(3) : "N/A"}`}
                      >
                        {val != null ? val.toFixed(1) : "-"}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            ) : (
              <div className="muted-line">No correlation data available.</div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default AnalyticsPage;
