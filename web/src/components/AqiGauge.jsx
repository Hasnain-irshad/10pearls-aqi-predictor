import React, { useMemo } from 'react';
import { categoryFor } from '../aqi';

const intervals = [
  { max: 50, color: '#00e400', rangeSpan: 50, fracSpan: 0.1 },
  { max: 100, color: '#ffde33', rangeSpan: 50, fracSpan: 0.1 },
  { max: 150, color: '#ff9933', rangeSpan: 50, fracSpan: 0.1 },
  { max: 200, color: '#ff5050', rangeSpan: 50, fracSpan: 0.1 },
  { max: 300, color: '#b25aff', rangeSpan: 100, fracSpan: 0.2 },
  { max: 500, color: '#c81d3f', rangeSpan: 200, fracSpan: 0.4 },
];

function getFraction(value) {
  let val = Math.max(0, Math.min(500, value));
  let baseFrac = 0;
  let baseVal = 0;
  for (let i = 0; i < intervals.length; i++) {
    const { max, rangeSpan, fracSpan } = intervals[i];
    if (val <= max) {
      return baseFrac + ((val - baseVal) / rangeSpan) * fracSpan;
    }
    baseFrac += fracSpan;
    baseVal = max;
  }
  return 1;
}

const getCoordinatesForFraction = (f, cx, cy, r) => {
  const theta = Math.PI * (1 - f);
  const x = cx + r * Math.cos(theta);
  const y = cy - r * Math.sin(theta);
  return { x, y };
};

const AqiGauge = ({ value = 0, size = 240 }) => {
  const cx = 120;
  const cy = 110;
  const r = 90;
  
  const currentCategory = categoryFor(value) || {};
  const currentColor = currentCategory.color || '#8a97a8';
  
  const segments = useMemo(() => {
    const segs = [];
    let startFrac = 0;
    intervals.forEach((interval) => {
      const endFrac = startFrac + interval.fracSpan;
      const p1 = getCoordinatesForFraction(startFrac, cx, cy, r);
      const p2 = getCoordinatesForFraction(endFrac, cx, cy, r);
      const path = `M ${p1.x},${p1.y} A ${r},${r} 0 0,1 ${p2.x},${p2.y}`;
      segs.push({
        path,
        color: interval.color,
        isActive: value > (startFrac === 0 ? -1 : intervals[segs.length - 1].max) && value <= interval.max,
        max: interval.max
      });
      startFrac = endFrac;
    });
    return segs;
  }, [value, cx, cy, r]);

  const fraction = getFraction(value);
  const angle = -90 + fraction * 180;

  // Markers
  const markers = [0, 50, 100, 150, 200, 300, 500];

  return (
    <div 
      className="aqi-gauge-container" 
      role="img"
      aria-label={`Current AQI ${value}, ${currentCategory.name || "Unknown"}`}
      style={{ 
        width: size, 
        position: 'relative', 
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}
    >
      <svg width="100%" viewBox="0 0 240 140" style={{ display: 'block', overflow: 'visible' }}>
        <defs>
          <filter id="needle-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#000000" floodOpacity="0.5"/>
          </filter>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Background Track */}
        <path 
          d={`M ${cx - r},${cy} A ${r},${r} 0 0,1 ${cx + r},${cy}`} 
          fill="none" 
          stroke="rgba(255,255,255,0.06)" 
          strokeWidth="12" 
          strokeLinecap="round" 
        />

        {/* Segments */}
        {segments.map((seg, i) => (
          <path 
            key={i}
            d={seg.path} 
            fill="none" 
            stroke={seg.color} 
            strokeWidth="12" 
            strokeLinecap="butt"
            style={{
              filter: seg.isActive ? 'url(#glow)' : 'none',
              opacity: seg.isActive ? 1 : 0.6,
              transition: 'all 0.3s ease'
            }}
          />
        ))}

        {/* Scale Markers */}
        {markers.map((mark) => {
          const f = getFraction(mark);
          const { x, y } = getCoordinatesForFraction(f, cx, cy, r + 15);
          return (
            <text 
              key={mark} 
              x={x} 
              y={y} 
              fill="var(--muted, #8a97a8)" 
              fontSize="10" 
              textAnchor="middle" 
              alignmentBaseline="middle"
            >
              {mark}
            </text>
          );
        })}

        {/* Needle */}
        <g 
          style={{ 
            transform: `rotate(${angle}deg)`, 
            transformOrigin: `${cx}px ${cy}px`, 
            transition: 'transform 1s cubic-bezier(0.4, 0.0, 0.2, 1)' 
          }}
        >
          <polygon 
            points={`${cx - 4},${cy} ${cx + 4},${cy} ${cx},${cy - r + 5}`} 
            fill="white" 
            filter="url(#needle-shadow)" 
          />
          <circle cx={cx} cy={cy} r="6" fill="#1a202c" stroke="white" strokeWidth="2" filter="url(#needle-shadow)" />
        </g>
      </svg>
      
      {/* Center Display */}
      <div 
        style={{
          position: 'absolute',
          bottom: '0',
          width: '100%',
          textAlign: 'center',
          transform: 'translateY(-20px)'
        }}
      >
        <div style={{ fontSize: '36px', fontWeight: 'bold', color: 'white', lineHeight: '1' }}>
          {value}
        </div>
        <div style={{ fontSize: '10px', color: 'var(--muted, #8a97a8)', letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: '5px' }}>
          US EPA AQI
        </div>
        <div 
          style={{ 
            fontSize: '14px', 
            fontWeight: '600', 
            color: currentColor,
            marginTop: '4px',
            textShadow: `0 0 8px ${currentColor}66`
          }}
        >
          {currentCategory.name || 'Unknown'}
        </div>
      </div>
      
      <style>{`
        .aqi-gauge-container {
          background: var(--bg, transparent);
        }
      `}</style>
    </div>
  );
};

export default AqiGauge;
