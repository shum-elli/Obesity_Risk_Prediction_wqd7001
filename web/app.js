const metrics = [
  ["accuracy", "Accuracy"],
  ["weighted_f1", "Weighted F1"],
  ["obesity_auc", "Obesity AUC"],
  ["overweight_or_obesity_auc", "At-risk AUC"]
];

let meta = null;
let eda = null;
let selectedMetric = "accuracy";
let predictTimer = null;

const formNode = document.querySelector("#questionnaireFields");
const statusNode = document.querySelector("#predictStatus");

function pct(value, digits = 1) {
  return `${value.toFixed(digits)}%`;
}

function niceLabel(value) {
  return String(value).replaceAll("_", " ");
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${path}`);
  }
  return data;
}

function defaultOption(options, preferred) {
  if (options.includes(preferred)) return preferred;
  return options[0];
}

function makeField(name, spec) {
  const label = document.createElement("label");
  label.className = "field";
  label.dataset.feature = name;

  const title = document.createElement("span");
  title.textContent = spec.label || niceLabel(name);
  label.appendChild(title);

  let input;
  if (spec.type === "select") {
    input = document.createElement("select");
    spec.options.forEach((optionValue) => {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = niceLabel(optionValue);
      input.appendChild(option);
    });
    const preferredDefaults = {
      Gender: "Female",
      family_history_with_overweight: "yes",
      FAVC: "yes",
      CAEC: "Sometimes",
      SMOKE: "no",
      SCC: "no",
      CALC: "Sometimes",
      MTRANS: "Public_Transportation"
    };
    input.value = defaultOption(spec.options, preferredDefaults[name] || spec.options[0]);
  } else {
    input = document.createElement("input");
    input.type = "range";
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = spec.default;
    const output = document.createElement("output");
    output.textContent = Number(input.value).toFixed(Number(spec.step) < 1 ? 1 : 0);
    input.addEventListener("input", () => {
      output.textContent = Number(input.value).toFixed(Number(spec.step) < 1 ? 1 : 0);
    });
    label.appendChild(output);
  }

  input.id = name;
  input.name = name;
  input.addEventListener("input", schedulePrediction);
  label.appendChild(input);

  if (spec.hint) {
    const hint = document.createElement("small");
    hint.textContent = spec.hint;
    label.appendChild(hint);
  }
  formNode.appendChild(label);
}

function buildQuestionnaire() {
  formNode.innerHTML = "";
  Object.entries(meta.feature_schema).forEach(([name, spec]) => makeField(name, spec));
}

function currentPayload() {
  const payload = { model: document.querySelector("#modelSelect").value };
  Object.keys(meta.feature_schema).forEach((name) => {
    const input = document.querySelector(`#${CSS.escape(name)}`);
    payload[name] = input.value;
  });
  return payload;
}

function riskBand(score) {
  if (score < 35) return "Lower risk profile";
  if (score < 65) return "Watch profile";
  return "Higher risk profile";
}

function topNonNormalClass(rows) {
  const nonNormalRows = rows
    .filter((row) => row.label !== "Normal_Weight")
    .sort((a, b) => b.probability_percent - a.probability_percent);
  return nonNormalRows[0] || rows[0];
}

function updateGauge(displayClass) {
  const score = displayClass.probability_percent;
  const angle = -90 + (score / 100) * 180;
  document.querySelector("#riskNeedle").style.transform = `rotate(${angle}deg)`;
  document.querySelector("#riskValue").textContent = pct(score, 0);
  document.querySelector("#riskLabel").textContent = riskBand(score);
  document.querySelector("#topClass").textContent = niceLabel(displayClass.label);
  document.querySelector("#topProb").textContent = pct(displayClass.probability_percent);
}

function renderClassBars(rows) {
  const node = document.querySelector("#classBars");
  node.innerHTML = "";
  rows.forEach((row, index) => {
    const element = document.createElement("div");
    element.className = "class-row";
    element.innerHTML = `
      <div class="class-name">${niceLabel(row.label)} <span class="mobile-value">${pct(row.probability_percent)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${row.probability_percent}%"></div></div>
      <div class="class-value">${pct(row.probability_percent)}</div>
    `;
    if (index === 0) element.setAttribute("aria-current", "true");
    node.appendChild(element);
  });
}

function driverItems(payload) {
  const activity = Number(payload.FAF);
  const water = Number(payload.CH2O);
  const screen = Number(payload.TUE);
  const items = [
    {
      title: "Family history",
      detail: payload.family_history_with_overweight === "yes" ? "Strong cohort signal in EDA" : "Lower observed obesity rate",
      tone: payload.family_history_with_overweight === "yes" ? "risk" : "protective"
    },
    {
      title: "Physical activity",
      detail: `${activity.toFixed(1)} on the 0 to 3 scale`,
      tone: activity >= 1.8 ? "protective" : activity < 0.7 ? "risk" : "watch"
    },
    {
      title: "Water intake",
      detail: `${water.toFixed(1)} on the 1 to 3 scale`,
      tone: water >= 2.4 ? "protective" : water < 1.6 ? "risk" : "watch"
    },
    {
      title: "Snacking",
      detail: niceLabel(payload.CAEC),
      tone: ["Frequently", "Always"].includes(payload.CAEC) ? "risk" : payload.CAEC === "no" ? "protective" : "watch"
    },
    {
      title: "Screen time",
      detail: `${screen.toFixed(1)} on the 0 to 2 scale`,
      tone: screen > 1.2 ? "risk" : screen < 0.4 ? "protective" : "watch"
    },
    {
      title: "High-calorie food",
      detail: payload.FAVC === "yes" ? "Frequent consumption marked yes" : "Marked no",
      tone: payload.FAVC === "yes" ? "risk" : "protective"
    },
    {
      title: "Meal pattern",
      detail: `${Number(payload.NCP).toFixed(1)} meals; vegetables ${Number(payload.FCVC).toFixed(1)}`,
      tone: "watch"
    }
  ];
  if (Object.hasOwn(payload, "MTRANS")) {
    items.splice(4, 0, {
      title: "Transport",
      detail: niceLabel(payload.MTRANS),
      tone: ["Walking", "Bike"].includes(payload.MTRANS) ? "protective" : "watch"
    });
  }
  return items;
}

function renderDrivers(payload) {
  const node = document.querySelector("#driverChips");
  node.innerHTML = "";
  driverItems(payload).forEach((item) => {
    const element = document.createElement("div");
    element.className = `driver-chip ${item.tone}`;
    element.innerHTML = `<strong>${item.title}</strong><span>${item.detail}</span>`;
    node.appendChild(element);
  });
}

function renderMetricTabs() {
  const node = document.querySelector("#metricTabs");
  node.innerHTML = "";
  metrics.forEach(([key, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(key === selectedMetric));
    button.addEventListener("click", () => {
      selectedMetric = key;
      renderMetricTabs();
      renderModelChart();
    });
    node.appendChild(button);
  });
}

function renderModelChart() {
  const node = document.querySelector("#modelChart");
  const rows = Object.entries(meta.models)
    .map(([name, data]) => ({ name, value: data.metrics[selectedMetric] }))
    .sort((a, b) => b.value - a.value);
  node.innerHTML = "";
  rows.forEach((row) => {
    const isBest = row.name === meta.best_model;
    const element = document.createElement("div");
    element.className = `model-row ${isBest ? "best" : ""}`;
    element.innerHTML = `
      <div class="model-name">${row.name}${isBest ? "<small>Default</small>" : ""} <span class="mobile-value">${row.value.toFixed(3)}</span></div>
      <div class="bar-track"><div class="model-fill" style="width:${row.value * 100}%"></div></div>
      <div class="class-value">${row.value.toFixed(3)}</div>
    `;
    node.appendChild(element);
  });
}

function renderContext() {
  document.querySelector("#rowCount").textContent = `${meta.rows.toLocaleString()} model rows`;
  document.querySelector("#bestModel").textContent = `Best model: ${meta.best_model}`;
  const modelSelect = document.querySelector("#modelSelect");
  modelSelect.innerHTML = "";
  Object.keys(meta.models).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === meta.best_model;
    modelSelect.appendChild(option);
  });
  modelSelect.addEventListener("input", () => {
    renderModelChart();
    schedulePrediction();
  });

  const node = document.querySelector("#contextStats");
  const classRows = Object.entries(eda.class_counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([name, count]) => `<div class="mini-class"><span>${niceLabel(name)}</span><strong>${count}</strong></div>`)
    .join("");

  node.innerHTML = `
    <div class="stat-tile"><strong>${eda.obesity_prevalence_percent.toFixed(1)}%</strong><span>Obesity prevalence in the engineered EDA summary.</span></div>
    <div class="stat-tile"><strong>${eda.rows.toLocaleString()}</strong><span>Original EDA records before duplicate handling in the latest model bundle.</span></div>
    <div class="stat-tile"><strong>${eda.average_bmi_by_transport.Walking.toFixed(1)}</strong><span>Average BMI for walking transport in EDA context. The latest prediction model excludes transportation as an input.</span></div>
    <div class="stat-tile"><strong>${eda.obesity_prevalence_by_family_history_percent.yes.toFixed(1)}%</strong><span>Obesity prevalence where family history is yes, versus ${eda.obesity_prevalence_by_family_history_percent.no.toFixed(1)}% where no.</span></div>
    <div class="stat-tile">${classRows}</div>
  `;
}

async function runPrediction() {
  const payload = currentPayload();
  statusNode.textContent = "Calling local model...";
  try {
    const result = await postJson("/api/predict", payload);
    updateGauge(topNonNormalClass(result.class_probabilities));
    renderClassBars(result.class_probabilities);
    renderDrivers(payload);
    statusNode.textContent = "Model prediction updated.";
  } catch (error) {
    statusNode.textContent = error.message;
  }
}

function schedulePrediction() {
  window.clearTimeout(predictTimer);
  predictTimer = window.setTimeout(runPrediction, 180);
}

function resetQuestionnaire() {
  buildQuestionnaire();
  schedulePrediction();
}

async function init() {
  [meta, eda] = await Promise.all([loadJson("/api/meta"), loadJson("/api/eda")]);
  renderContext();
  buildQuestionnaire();
  renderMetricTabs();
  renderModelChart();
  document.querySelector("#resetBtn").addEventListener("click", resetQuestionnaire);
  await runPrediction();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><section class="risk-stage"><h1>Visualization data could not load</h1><p>${error.message}</p><p>Start the model-backed server with: python web/server.py</p></section></main>`;
});
