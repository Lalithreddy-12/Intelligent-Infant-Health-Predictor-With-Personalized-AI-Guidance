// static/app.js
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const predictBtn = document.getElementById("predict");
  const clearBtn = document.getElementById("clear");
  const resultCard = document.getElementById("resultCard");
  const planContainer = document.getElementById("planContainer");
  const meterCircle = document.getElementById("meterCircle");
  const riskLabel = document.getElementById("riskLabel");
  const probLabel = document.getElementById("probLabel");
  const probBar = document.getElementById("probBar");
  const explainBtn = document.getElementById("explainBtn");
  const explainText = document.getElementById("explainText");
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");

  // Theme
  const applyTheme = (dark) => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    themeIcon.textContent = dark ? "☀️" : "🌙";
    localStorage.setItem("dm_theme", dark ? "dark" : "light");
  };
  applyTheme(localStorage.getItem("dm_theme") === "dark");
  themeToggle.addEventListener("click", () => applyTheme(!(localStorage.getItem("dm_theme") === "dark")));

  function setMeter(percent) {
    const circumference = 100;
    const dash = (percent / 100) * circumference;
    let color = "#10b981";
    if (percent >= 70) color = "#ef4444";
    else if (percent >= 40) color = "#f59e0b";
    meterCircle.style.stroke = color;
    meterCircle.setAttribute("stroke-dasharray", `${dash}, ${circumference}`);
  }

  async function showPlan(planObj) {
    planContainer.innerHTML = "";
    const years = planObj?.years || {};
    const expectedYears = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"];
    let delay = 0;
    for (const year of expectedYears) {
      const actions = Array.isArray(years[year]) && years[year].length ? years[year] : ["No data available"];
      const card = document.createElement("div");
      card.className = "plan-card bg-white dark:bg-gray-800 p-4 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 animate-fade-up";
      card.innerHTML = `
        <h4 class="font-semibold text-primary">${year}</h4>
        <ul class="mt-2 text-sm text-gray-700 dark:text-gray-200 space-y-1">
          ${actions.map(a => `<li>• ${a}</li>`).join('')}
        </ul>
      `;
      planContainer.appendChild(card);
      setTimeout(() => card.classList.add("in"), 80 + delay);
      delay += 120;
    }
  }

  clearBtn.addEventListener("click", () => {
    document.getElementById("birth_weight").value = "2.8";
    document.getElementById("maternal_age").value = "26";
    document.getElementById("immunized").value = "1";
    document.getElementById("nutrition").value = "60";
    document.getElementById("socioeconomic").value = "1";
    document.getElementById("prenatal_visits").value = "4";
    resultCard.classList.add("hidden");
    explainText.classList.add("hidden");
    explainText.innerHTML = "";
    explainText.dataset.text = "";
    planContainer.innerHTML = "";
  });

  // Toggle explain (robust)
  explainBtn.addEventListener("click", () => {
    if (explainText.classList.contains("hidden")) {
      explainText.classList.remove("hidden");
      // smooth reveal
      explainText.scrollIntoView({behavior: "smooth", block: "center"});
    } else {
      explainText.classList.add("hidden");
    }
  });

  // Helper: build friendly HTML if backend doesn't provide ui_html
  function buildFriendlyExplanation(expl) {
    try {
      if (!expl || !expl.feature_importance_human) return "<div>No explanation available.</div>";
      const fh = expl.feature_importance_human;
      const summary = expl.summary || "Main drivers";
      let html = `<div style="font-weight:600;margin-bottom:8px">${summary}</div><ul style="list-style:none;padding-left:0;margin:0">`;
      for (const it of fh) {
        const arrow = it.direction === "Increase" ? "↑" : (it.direction === "Decrease" ? "↓" : "→");
        const color = it.direction === "Increase" ? "#ef4444" : (it.direction === "Decrease" ? "#10b981" : "#6b7280");
        const pct = it.relative_importance_pct || 0;
        html += `<li style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(0,0,0,0.04)">
          <div style="max-width:78%">
            <div style="font-weight:600">${it.label} <span style="color:${color};margin-left:8px">${arrow}</span>
              <small style="font-weight:600;margin-left:8px">${it.magnitude_label} (${pct}%)</small>
            </div>
            <div style="font-size:13px;color:#374151;margin-top:6px">${it.recommendation || ""}</div>
          </div>
          <div style="text-align:right;min-width:40px"><div style="font-size:12px;color:${color};font-weight:700">${it.direction}</div></div>
        </li>`;
      }
      html += `</ul>`;
      if (expl.recommendations && expl.recommendations.length) {
        html += `<div style="margin-top:8px"><div style="font-weight:600;margin-bottom:6px">Suggested next steps</div><ul>`;
        for (const r of expl.recommendations.slice(0,4)) {
          html += `<li>• <strong>${r.feature}:</strong> ${r.recommendation}</li>`;
        }
        html += `</ul></div>`;
      }
      // tech details transform
      const techJson = JSON.stringify({ feature_contributions: expl.feature_contributions || [] }, null, 2);
      html += `<details style="margin-top:8px"><summary style="cursor:pointer;font-weight:600">Show technical details</summary><pre style="white-space:pre-wrap;padding:8px;background:#f9fafb;border-radius:6px;margin-top:6px">${techJson}</pre></details>`;
      return html;
    } catch (e) {
      console.error("buildFriendlyExplanation error", e);
      return "<div>Error building explanation</div>";
    }
  }

  // Main predict handler
  predictBtn.addEventListener("click", async () => {
    const payload = {
      birth_weight: parseFloat(document.getElementById("birth_weight").value || 0),
      maternal_age: parseFloat(document.getElementById("maternal_age").value || 0),
      immunized: parseInt(document.getElementById("immunized").value || 0),
      nutrition: parseFloat(document.getElementById("nutrition").value || 0),
      socioeconomic: parseInt(document.getElementById("socioeconomic").value || 0),
      prenatal_visits: parseFloat(document.getElementById("prenatal_visits").value || 0),
      debug: true
    };

    predictBtn.disabled = true;
    predictBtn.classList.add("opacity-70");
    predictBtn.innerHTML = `<svg class="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="white" stroke-width="4" fill="none"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path></svg> Processing`;

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const data = await resp.json();

      resultCard.classList.remove("hidden");

      const prob = (data.mortality_risk_probability != null) ? Number(data.mortality_risk_probability) * 100 : null;
      const pred = data.mortality_prediction;
      const riskText = (pred === 1 || (prob !== null && prob >= 60)) ? "High" : (prob !== null && prob >= 35 ? "Medium" : "Low");

      setMeter(prob || 0);
      riskLabel.textContent = `Risk: ${riskText}`;
      probLabel.textContent = `Probability: ${prob !== null ? prob.toFixed(1) + "%" : "N/A"}`;
      probBar.style.width = `${prob !== null ? Math.min(100, prob) : 0}%`;
      probBar.style.backgroundColor = (prob >= 70 ? "#ef4444" : prob >= 40 ? "#f59e0b" : "#10b981");

      // Explanation rendering logic
      const explanation = data.debug?.explanation || null;
      const hfRaw = data.debug?.hf_raw_truncated || null;
      // Preferred: UI HTML from server
      if (explanation && explanation.ui_html) {
        explainText.innerHTML = explanation.ui_html;
      } else if (explanation) {
        // Build friendly HTML client-side if server didn't provide ui_html
        explainText.innerHTML = buildFriendlyExplanation(explanation);
      } else if (hfRaw) {
        explainText.innerHTML = `<pre style="white-space:pre-wrap; padding:8px; background:#f9fafb; border-radius:6px">${hfRaw.slice(0,1200)}${hfRaw.length>1200?"…":""}</pre>`;
      } else {
        explainText.innerHTML = "<div>Top contributing features not available for this demo.</div>";
      }
      // hide by default, user clicks Explain to open
      explainText.classList.add("hidden");

      // show plan blocks
      await showPlan(data.survival_plan || { years: {} });

      // scroll to results
      resultCard.scrollIntoView({behavior: "smooth", block: "center"});
    } catch (err) {
      console.error(err);
      resultCard.classList.remove("hidden");
      planContainer.innerHTML = `<div class="col-span-1 p-4 text-sm text-red-600">Error: ${err.message}</div>`;
      explainText.dataset.text = "";
    } finally {
      predictBtn.disabled = false;
      predictBtn.classList.remove("opacity-70");
      predictBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 5v10m5-5H5"/></svg> Predict`;
    }
  });
});
