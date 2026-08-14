import { useMemo } from "react";

// Dropdown of cities grouped by province.
export default function CitySelect({ cities, value, onChange }) {
  const groups = useMemo(() => {
    const byProvince = {};
    for (const [name, info] of Object.entries(cities)) {
      (byProvince[info.province] ??= []).push(name);
    }
    for (const list of Object.values(byProvince)) list.sort();
    return byProvince;
  }, [cities]);

  return (
    <label className="city-select">
      <span>City</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {Object.entries(groups).map(([province, names]) => (
          <optgroup key={province} label={province}>
            {names.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}
