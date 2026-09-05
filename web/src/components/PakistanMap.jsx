import { MapContainer, TileLayer, CircleMarker, Tooltip as LTooltip } from "react-leaflet";
import { colorFor, categoryFor } from "../aqi";

// Interactive map: one coloured circle per city, sized/coloured by current AQI.
export default function PakistanMap({ cities, selected, onSelect }) {
  const points = Object.entries(cities).map(([name, info]) => ({
    name,
    lat: info.lat,
    lon: info.lon,
    aqi: info.current.aqi,
  }));

  return (
    <div className="map-card">
      <h3>Live AQI map</h3>
      <MapContainer center={[30.4, 69.3]} zoom={5} className="map" scrollWheelZoom={false}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {points.map((p) => (
          <CircleMarker
            key={p.name}
            center={[p.lat, p.lon]}
            radius={p.name === selected ? 12 : 8}
            pathOptions={{
              color: p.name === selected ? "#fff" : colorFor(p.aqi),
              weight: p.name === selected ? 2 : 1,
              fillColor: colorFor(p.aqi),
              fillOpacity: 0.85,
            }}
            eventHandlers={{ click: () => onSelect(p.name) }}
          >
            <LTooltip>
              <b>{p.name}</b>: AQI {p.aqi} ({categoryFor(p.aqi).name})
            </LTooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}
