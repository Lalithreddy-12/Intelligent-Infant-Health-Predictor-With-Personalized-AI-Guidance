// static/app.js (FULL updated)
// Frontend for Child Mortality Prediction
// Requires: index.html elements with IDs used below and Tailwind available

const API_BASE = ""; // same-origin (if your backend runs at different origin, set full URL e.g. "http://127.0.0.1:8000")

// ---------- Helpers ----------
function authHeaders() { return { "Content-Type": "application/json" }; }
function nowIso() { return new Date().toISOString(); }
function escapeHtml(s) { if (s == null) return ""; return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;"); }

// wait for server session to be available after login/register
async function waitForServerSession(retries = 4, delayMs = 300) {
  for (let i = 0; i < retries; i++) {
    try {
      const r = await fetch((API_BASE || "") + "/api/auth/whoami", { credentials: 'include' });
      if (r.ok) {
        const js = await r.json().catch(() => ({}));
        if (js && js.logged_in) return js;
      }
    } catch (e) {
      // ignore and retry
    }
    await new Promise(res => setTimeout(res, delayMs));
  }
  return null;
}

// ---------- Local history helpers ----------
function pushLocalPrediction(entry) {
  const raw = localStorage.getItem("cm_local_history");
  let arr = [];
  if (raw) { try { arr = JSON.parse(raw); } catch (e) { arr = []; } }
  arr.unshift(entry);
  if (arr.length > 200) arr = arr.slice(0, 200);
  localStorage.setItem("cm_local_history", JSON.stringify(arr));
}
function getLocalPredictions() { const r = localStorage.getItem("cm_local_history"); if (!r) return []; try { return JSON.parse(r) } catch (e) { return [] } }
function clearLocalPredictions() { localStorage.removeItem("cm_local_history"); }

async function mergeLocalHistoryOnLogin() {
  const entries = getLocalPredictions();
  if (!entries?.length) return;
  try {
    const res = await fetch(API_BASE + "/api/history/merge", {
      method: "POST",
      headers: authHeaders(),
      credentials: "include",
      body: JSON.stringify({ entries })
    });
    if (res.ok) {
      clearLocalPredictions();
      // Remove any cached local copy of server history to protect privacy
      try { localStorage.removeItem("child_mortality_history_v1"); } catch (e) { }
      console.log("Merged local history to server");
    } else {
      console.warn("Merge failed", await res.text());
    }
  } catch (e) { console.warn("Merge error", e); }
}

// ---------- DOM refs ----------
const loginOnlyBtn = document.getElementById("loginOnlyBtn");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");

let loginModal = null; // will hold modal backdrop element

// ---------- Theme ----------
function applyTheme(dark) {
  if (dark) document.documentElement.classList.add("dark");
  else document.documentElement.classList.remove("dark");
  if (themeIcon) themeIcon.textContent = dark ? "🌙" : "☀️";
  localStorage.setItem("dm_theme", dark ? "dark" : "light");
}
(function initTheme() { applyTheme(localStorage.getItem("dm_theme") === "dark"); })();
themeToggle && themeToggle.addEventListener("click", () => applyTheme(!(localStorage.getItem("dm_theme") === "dark")));

// ---------- Modal creation (dark-safe Tailwind classes + show/hide password + register confirmation) ----------
function createLoginModal() {
  if (loginModal) return loginModal;

  const backdrop = document.createElement("div");
  backdrop.className = "cm-modal-backdrop fixed inset-0 z-50 flex items-center justify-center p-4";
  backdrop.style.background = "rgba(2,6,23,0.6)";

  const modal = document.createElement("div");
  modal.className = "cm-modal w-full max-w-md rounded-xl shadow-xl p-6 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-50";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");

  modal.innerHTML = `
    <div class="flex items-start justify-between">
      <div>
        <h3 id="cm-modal-title" class="text-lg font-semibold">Login</h3>
        <p id="cm-modal-sub" class="text-sm text-gray-600 dark:text-gray-300 mt-1">Sign in to sync & save your prediction history</p>
      </div>
      <button id="cm-close" class="text-gray-500 dark:text-gray-300 hover:text-gray-700 dark:hover:text-white">✖</button>
    </div>

    <div id="cm-form-login" class="mt-4 space-y-3">
      <div class="space-y-1">
        <label class="text-xs font-medium text-gray-700 dark:text-gray-300">Username</label>
        <div class="flex items-center gap-2">
          <input id="cm-user" autocomplete="username" class="flex-1 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 text-sm" placeholder="username" />
        </div>
      </div>

      <div class="space-y-1">
        <label class="text-xs font-medium text-gray-700 dark:text-gray-300">Password</label>
        <div class="relative">
          <input id="cm-pass" type="password" autocomplete="current-password" class="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 text-sm pr-10" placeholder="password" />
          <button type="button" class="cm-eye absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-300" data-target="cm-pass" aria-label="Toggle password visibility">👁️</button>
        </div>
      </div>

      <div class="flex items-center justify-between gap-3 pt-2">
        <button id="cm-login" class="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-primary to-secondary text-white font-semibold">Login</button>
        <button id="cm-show-register" class="flex-1 px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-sm">Register</button>
      </div>

      <p class="text-xs text-gray-600 dark:text-gray-300">Don't have an account? Click Register.</p>
    </div>

    <div id="cm-form-register" style="display:none" class="mt-4 space-y-3">
      <div class="space-y-1">
        <label class="text-xs font-medium text-gray-700 dark:text-gray-300">Username</label>
        <input id="cm-reg-user" autocomplete="username" class="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 text-sm" placeholder="choose username" />
      </div>

      <div class="space-y-1">
        <label class="text-xs font-medium text-gray-700 dark:text-gray-300">Password</label>
        <div class="relative">
          <input id="cm-reg-pass" type="password" autocomplete="new-password" class="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 text-sm pr-10" placeholder="password" />
          <button type="button" class="cm-eye absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-300" data-target="cm-reg-pass" aria-label="Toggle password visibility">👁️</button>
        </div>
      </div>

      <div class="space-y-1">
        <label class="text-xs font-medium text-gray-700 dark:text-gray-300">Confirm password</label>
        <div class="relative">
          <input id="cm-reg-pass2" type="password" autocomplete="new-password" class="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700 text-sm pr-10" placeholder="confirm password" />
          <button type="button" class="cm-eye absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-300" data-target="cm-reg-pass2" aria-label="Toggle password visibility">👁️</button>
        </div>
      </div>

      <div class="flex items-center justify-between gap-3 pt-2">
        <button id="cm-register" class="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-primary to-secondary text-white font-semibold">Register</button>
        <button id="cm-reg-back" class="flex-1 px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-sm">Back</button>
      </div>

      <p id="cm-reg-note" class="text-xs text-gray-600 dark:text-gray-300">Passwords must match to register.</p>
    </div>
  `;

  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  loginModal = backdrop;

  // close handlers
  modal.querySelector("#cm-close").addEventListener("click", closeLoginModal);
  backdrop.addEventListener("click", (ev) => { if (ev.target === backdrop) closeLoginModal(); });

  // toggle to register
  modal.querySelector("#cm-show-register").addEventListener("click", () => {
    modal.querySelector("#cm-form-login").style.display = "none";
    modal.querySelector("#cm-form-register").style.display = "block";
    modal.querySelector("#cm-modal-title").textContent = "Register";
  });
  modal.querySelector("#cm-reg-back").addEventListener("click", () => {
    modal.querySelector("#cm-form-register").style.display = "none";
    modal.querySelector("#cm-form-login").style.display = "block";
    modal.querySelector("#cm-modal-title").textContent = "Login";
  });

  // show/hide password inline eye buttons
  Array.from(modal.querySelectorAll('.cm-eye')).forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const ip = modal.querySelector('#' + targetId);
      if (!ip) return;
      if (ip.type === 'password') {
        ip.type = 'text';
        btn.textContent = '🙈';
      } else {
        ip.type = 'password';
        btn.textContent = '👁️';
      }
    });
  });

  // LOGIN action
  modal.querySelector("#cm-login").addEventListener("click", async () => {
    const u = modal.querySelector("#cm-user").value.trim();
    const p = modal.querySelector("#cm-pass").value;
    if (!u || !p) { alert("username & password required"); return; }
    const btn = modal.querySelector("#cm-login");
    btn.disabled = true; btn.textContent = "Logging in...";
    try {
      const res = await fetch(API_BASE + "/api/auth/login", {
        method: "POST",
        headers: authHeaders(),
        credentials: "include",
        body: JSON.stringify({ username: u, password: p })
      });
      const js = await res.json().catch(() => ({}));
      if (res.ok) {
        // Wait for the server to establish the session cookie (whoami) before merging history
        const who = await waitForServerSession(5, 300);
        if (!who) {
          // proceed but warn user
          alert('Login succeeded but server session not confirmed. History sync may fail until you reload.');
        }
        try { await mergeLocalHistoryOnLogin(); } catch (e) { console.warn('mergeLocalHistoryOnLogin failed', e); }
        setLoggedInUI(u, js.user_id);
        closeLoginModal();
        alert("Login successful");
        if (typeof loadHistory === "function") loadHistory(10, false);
      } else {
        alert("Login failed: " + (js.error || JSON.stringify(js) || res.statusText));
      }
    } catch (err) {
      alert("Login error: " + (err.message || err));
    } finally {
      btn.disabled = false; btn.textContent = "Login";
    }
  });

  // REGISTER action with password match validation
  modal.querySelector("#cm-register").addEventListener("click", async () => {
    const u = modal.querySelector("#cm-reg-user").value.trim();
    const p = modal.querySelector("#cm-reg-pass").value;
    const p2 = modal.querySelector("#cm-reg-pass2").value;
    if (!u || !p || !p2) { alert("username & both password fields required"); return; }
    if (p !== p2) { alert("Passwords do not match"); return; }
    const btn = modal.querySelector("#cm-register");
    btn.disabled = true; btn.textContent = "Registering...";
    try {
      const res = await fetch(API_BASE + "/api/auth/register", {
        method: "POST",
        headers: authHeaders(),
        credentials: "include",
        body: JSON.stringify({ username: u, password: p })
      });
      const js = await res.json().catch(() => ({}));
      if (res.ok) {
        const who = await waitForServerSession(5, 300);
        if (!who) alert('Registered but server session not confirmed. You may need to reload.');
        try { await mergeLocalHistoryOnLogin(); } catch (e) { console.warn('mergeLocalHistoryOnLogin failed', e); }
        setLoggedInUI(u, js.user_id);
        closeLoginModal();
        alert("Registered and logged in");
        if (typeof loadHistory === "function") loadHistory(10, false);
      } else {
        alert("Register failed: " + (js.error || JSON.stringify(js) || res.statusText));
      }
    } catch (err) {
      alert("Register error: " + (err.message || err));
    } finally {
      btn.disabled = false; btn.textContent = "Register";
    }
  });

  return loginModal;
}

function openLoginModal() {
  createLoginModal(); // ensure inputs are cleared each time modal opens
  clearAuthModalInputs();
  // Always default to the login view (avoid showing register if previously left open)
  try {
    const m = loginModal.querySelector('#cm-form-login');
    const r = loginModal.querySelector('#cm-form-register');
    if (m) m.style.display = 'block';
    if (r) r.style.display = 'none';
    const title = loginModal.querySelector('#cm-modal-title'); if (title) title.textContent = 'Login';
  } catch (e) { }
  loginModal.style.display = "flex";
  const uEl = loginModal.querySelector("#cm-user"); if (uEl) uEl.focus();
}
function closeLoginModal() { if (!loginModal) return; loginModal.style.display = "none"; clearAuthModalInputs(); }

// Clear auth modal inputs to avoid leaking previous user's credentials
function clearAuthModalInputs() {
  if (!loginModal) return;
  const ids = ['cm-user', 'cm-pass', 'cm-reg-user', 'cm-reg-pass', 'cm-reg-pass2'];
  ids.forEach(id => {
    const el = loginModal.querySelector('#' + id);
    if (el) { try { el.value = ''; } catch (e) { } }
    // reset eye button icon to default open-eye
    const eye = loginModal.querySelector('.cm-eye[data-target="' + id + '"]');
    if (eye) eye.textContent = '👁️';
  });
}

// Hook header login button (if exists)
if (loginOnlyBtn) loginOnlyBtn.addEventListener("click", openLoginModal);
else {
  const alt = document.getElementById("cm-login-btn");
  if (alt) alt.addEventListener("click", openLoginModal);
}

// ---------- Header logged-in UI (persist username) ----------
// Find the right-side header container reliably so adding the user box
// doesn't move the theme toggle. Prefer the known container with gap-3.
const headerRightContainer = (function () {
  const container = document.querySelector('header .max-w-7xl .flex.items-center.gap-3');
  if (container) return container;
  // fallback: the header inner wrapper
  return document.querySelector('header .max-w-7xl') || null;
})();

function setLoggedInUI(username, userId) {
  try {
    localStorage.setItem("cm_logged_in_user", username);
    if (userId) localStorage.setItem("cm_current_user_id", userId);
  } catch (e) { }
  // hide login-only button if present
  if (loginOnlyBtn) loginOnlyBtn.style.display = "none";

  // remove existing headerUserBox
  const prev = document.getElementById("headerUserBox");
  if (prev) prev.remove();

  const userDiv = document.createElement("div");
  userDiv.id = "headerUserBox";
  userDiv.className = "flex items-center gap-3 text-sm";

  userDiv.innerHTML = `
    <div class="text-xs text-gray-700 dark:text-gray-300"><strong>${escapeHtml(username)}</strong></div>
    <button id="cm-logout-btn" class="px-3 py-1 rounded-lg bg-gray-100 dark:bg-gray-700 text-sm">Logout</button>
  `;

  // append near header right area (if found) or to body as fallback
  // insert user box in the header right container in a stable position
  if (headerRightContainer) {
    // try to place before the login button (so theme toggle stays left-most)
    const loginBtn = headerRightContainer.querySelector('#loginOnlyBtn');
    if (loginBtn) headerRightContainer.insertBefore(userDiv, loginBtn);
    else headerRightContainer.appendChild(userDiv);
  } else {
    document.body.insertBefore(userDiv, document.body.firstChild);
  }

  document.getElementById("cm-logout-btn").addEventListener("click", async () => {
    try {
      await fetch(API_BASE + "/api/auth/logout", { method: "POST", credentials: "include" });
    } catch (e) { /* ignore */ }
    // Clear login flags and any cached server-side history so a subsequent user
    // on this browser cannot see the previous user's history.
    localStorage.removeItem("cm_logged_in_user");
    localStorage.removeItem("cm_current_user_id");
    try { localStorage.removeItem("child_mortality_history_v1"); } catch (e) { }
    userDiv.remove();
    // Clear any auth modal inputs so credentials don't persist in the DOM
    try { clearAuthModalInputs(); } catch (e) { }
    if (loginOnlyBtn) loginOnlyBtn.style.display = "";
    if (typeof loadHistory === "function") loadHistory(6, false);
  });
}

(function restoreHeaderLoginState() {
  const stored = localStorage.getItem("cm_logged_in_user");
  if (stored) setLoggedInUI(stored);
})();

// ---------- Prediction wiring (improved error display) ----------
const predictBtn = document.getElementById("predict");
const clearBtn = document.getElementById("clear");
const resultCard = document.getElementById("resultCard");
const planContainer = document.getElementById("planContainer");
// meterCircle removed
const riskLabel = document.getElementById("riskLabel");
const probLabel = document.getElementById("probLabel");
const probBar = document.getElementById("probBar");
const explainText = document.getElementById("explainText");
const historyContainer = document.getElementById("historyContainer");
const historyPlaceholder = document.getElementById("historyPlaceholder");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

// helpers


function saveHistory(entry) {
  try {
    const key = "child_mortality_history_v1";
    const raw = localStorage.getItem(key);
    const arr = raw ? JSON.parse(raw) : [];
    arr.unshift(entry);
    localStorage.setItem(key, JSON.stringify(arr.slice(0, 200)));
  } catch (e) { console.warn("Failed save history", e); }
}

// Load history (local + server merged if logged in)
const historySearchInput = document.getElementById("historySearchInput");
const historySearchBtn = document.getElementById("historySearchBtn");
const toggleSearchBtn = document.getElementById("toggleSearchBtn");
const historySearchContainer = document.getElementById("historySearchContainer");

// Toggle search visibility
if (toggleSearchBtn && historySearchContainer) {
  toggleSearchBtn.addEventListener("click", () => {
    const isHidden = historySearchContainer.classList.contains("hidden");
    if (isHidden) {
      historySearchContainer.classList.remove("hidden");
      if (historySearchInput) historySearchInput.focus();
    } else {
      historySearchContainer.classList.add("hidden");
    }
  });
}

// Wire search button
if (historySearchBtn) {
  const doSearch = () => {
    const val = historySearchInput ? historySearchInput.value.trim() : "";
    loadHistory(20, false, val);
  };
  historySearchBtn.addEventListener("click", doSearch);
  historySearchInput && historySearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });
}

// Wire refresh button to clear search
refreshHistoryBtn && refreshHistoryBtn.addEventListener("click", () => {
  if (historySearchInput) historySearchInput.value = "";
  loadHistory(20, false);
});


async function loadHistory(limit = 20, showDebug = false, searchTerm = "") {
  try {
    historyContainer && (historyContainer.innerHTML = "");
    const logged = !!localStorage.getItem('cm_logged_in_user');
    let serverSuccess = false;

    if (logged) {
      // Verify session is still active before attempting server fetch
      try {
        const whoResp = await fetch((API_BASE || "") + "/api/auth/whoami", { credentials: 'include' });
        const whoData = await whoResp.json().catch(() => ({}));
        if (!whoData.logged_in) {
          // Session expired, clear login flag
          localStorage.removeItem('cm_logged_in_user');
          localStorage.removeItem('cm_current_user_id');
          // Fall back to local history
        } else {
          // fetch from server
          let url = (API_BASE || "") + `/api/history?limit=${limit}&debug=${showDebug ? 'true' : 'false'}`;
          if (searchTerm) url += `&search=${encodeURIComponent(searchTerm)}`;

          const resp = await fetch(url, { credentials: 'include' });
          if (resp.ok) {
            const js = await resp.json().catch(() => null);
            const items = (js && js.history) ? js.history : [];
            if (!items.length) {
              historyContainer.innerHTML = `<div class="text-xs text-gray-500 dark:text-gray-400">No history yet. Run a prediction to save the first one.</div>`;
              updateClearButtonVisibility(false);
              return;
            }
            // Display server history
            for (const it of items) {
              const time = new Date(it.timestamp).toLocaleString();
              const prob = it.probability != null ? (it.probability * 100).toFixed(1) : "N/A";
              const risk = (it.prediction === 1 || (it.probability != null && it.probability * 100 >= 60)) ? "High" : (it.probability != null && it.probability * 100 >= 35 ? "Medium" : "Low");

              const rawName = it.patient_name;
              const isUnknown = !rawName || rawName === 'Unknown';
              const displayName = isUnknown ? 'Unknown' : rawName;

              const card = document.createElement("div");
              card.className = "p-3 rounded-lg bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-sm";
              card.innerHTML = `
                <div class="flex items-start justify-between gap-3">
                  <div style="min-width:0">
                    <div class="flex items-center gap-2">
                      <div class="font-semibold text-gray-800 dark:text-gray-200 truncate">${escapeHtml(displayName)}</div>
                      ${isUnknown ? `<button type="button" class="addNameBtn relative z-10 cursor-pointer px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs font-bold hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors uppercase tracking-wide border border-blue-200 dark:border-blue-700 shadow-sm">+ Add Name</button>` : ''}
                    </div>
                    <div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">${escapeHtml(time)}</div>
                    <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">Risk: ${escapeHtml(risk)} • Prob: ${prob}%</div>
                  </div>
                  <div class="flex items-center gap-2">
                    <button type="button" class="viewBtn px-2 py-1 rounded bg-gradient-to-r from-primary to-secondary text-white text-xs">View</button>
                    <button type="button" class="deleteBtn px-2 py-1 rounded bg-red-100 text-red-700 text-xs">Delete</button>
                  </div>
                </div>
              `;

              card.querySelector(".viewBtn").addEventListener("click", () => showHistoryModal(it, showDebug));

              // Add name handler (server)
              const addBtn = card.querySelector(".addNameBtn");
              if (addBtn) {
                addBtn.addEventListener("click", (e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  showEditNameDialog("", async (newName) => {
                    try {
                      const resp = await fetch((API_BASE || "") + `/api/history/${it.id}`, {
                        method: 'PUT',
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ patient_name: newName }),
                        credentials: 'include'
                      });
                      if (resp.ok) {
                        loadHistory(limit, showDebug, searchTerm);
                      } else {
                        alert("Update failed");
                      }
                    } catch (err) { alert("Error updating: " + err.message); }
                  });
                });
              }

              card.querySelector(".deleteBtn").addEventListener("click", async () => {
                if (!confirm('Delete this history entry from server?')) return;
                try {
                  const dresp = await fetch((API_BASE || "") + `/api/history/${it.id}`, { method: 'DELETE', credentials: 'include' });
                  if (dresp.ok) { alert('Deleted'); loadHistory(limit, showDebug, searchTerm); } else { alert('Delete failed'); }
                } catch (e) { alert('Delete error: ' + e.message); }
              });
              historyContainer.appendChild(card);
            }
            updateClearButtonVisibility(true);
            serverSuccess = true;
            return;
          }
        }
      } catch (e) {
        console.warn("Error checking session or fetching server history", e);
        // Fall through to local history below
      }
    }

    // Fall back to local history (if not logged in or server fetch failed)
    if (!serverSuccess) {
      const raw = localStorage.getItem("child_mortality_history_v1");
      let arr = raw ? JSON.parse(raw) : [];

      // Filter local items if search term exists
      if (searchTerm) {
        const lower = searchTerm.toLowerCase();
        arr = arr.filter(it => {
          const pName = it.patient_name || it.inputs?.patient_name || "";
          return pName.toLowerCase().includes(lower);
        });
      }

      const items = arr.slice(0, limit);
      if (!historyContainer) return;
      if (!items.length) {
        historyContainer.innerHTML = `<div class="text-xs text-gray-500 dark:text-gray-400">No history yet. Run a prediction to save the first one.</div>`;
        updateClearButtonVisibility(false);
        return;
      }
      for (const it of items) {
        const time = new Date(it.timestamp).toLocaleString();
        const prob = it.probability != null ? (it.probability * 100).toFixed(1) : "N/A";
        const risk = (it.prediction === 1 || (it.probability != null && it.probability * 100 >= 60)) ? "High" : (it.probability != null && it.probability * 100 >= 35 ? "Medium" : "Low");

        const rawName = it.patient_name || it.inputs?.patient_name;
        const isUnknown = !rawName || rawName === 'Unknown';
        const displayName = isUnknown ? 'Unknown' : rawName;

        const card = document.createElement("div");
        card.className = "p-3 rounded-lg bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 text-sm";
        card.innerHTML = `
          <div class="flex items-start justify-between gap-3">
            <div style="min-width:0">
              <div class="flex items-center gap-2">
                <div class="font-semibold text-gray-800 dark:text-gray-200 truncate">${escapeHtml(displayName)}</div>
                ${isUnknown ? `<button type="button" class="addNameBtn relative z-10 cursor-pointer px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs font-bold hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors uppercase tracking-wide border border-blue-200 dark:border-blue-700 shadow-sm">+ Add Name</button>` : ''}
              </div>
              <div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">${escapeHtml(time)}</div>
              <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">Risk: ${escapeHtml(risk)} • Prob: ${prob}%</div>
            </div>
            <div class="flex items-center gap-2">
              <button type="button" class="viewBtn px-2 py-1 rounded bg-gradient-to-r from-primary to-secondary text-white text-xs">View</button>
              <button type="button" class="deleteBtn px-2 py-1 rounded bg-red-100 text-red-700 text-xs">Delete</button>
            </div>
          </div>
        `;

        card.querySelector(".viewBtn").addEventListener("click", () => showHistoryModal(it, showDebug));

        // Add name handler (local)
        const addBtn = card.querySelector(".addNameBtn");
        if (addBtn) {
          addBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            showEditNameDialog("", (newName) => {
              try {
                const rawLocal = localStorage.getItem("child_mortality_history_v1");
                let list = rawLocal ? JSON.parse(rawLocal) : [];
                const idx = list.findIndex(x => x.timestamp === it.timestamp);
                if (idx !== -1) {
                  list[idx].patient_name = newName;
                  localStorage.setItem("child_mortality_history_v1", JSON.stringify(list));
                  loadHistory(limit, showDebug, searchTerm);
                }
              } catch (err) { alert("Error updating local history: " + err.message); }
            });
          });
        }

        card.querySelector(".deleteBtn").addEventListener("click", () => {
          if (!confirm('Delete this local history entry?')) return;
          const raw2 = localStorage.getItem('child_mortality_history_v1');
          let a2 = raw2 ? JSON.parse(raw2) : [];
          a2 = a2.filter(x => x.timestamp !== it.timestamp);
          localStorage.setItem('child_mortality_history_v1', JSON.stringify(a2));
          loadHistory(limit, showDebug);
        });
        historyContainer.appendChild(card);
      }
      updateClearButtonVisibility(true);
    }
  } catch (e) { console.error("loadHistory error", e); if (historyContainer) historyContainer.innerHTML = `<div class="text-xs text-red-600">Failed to load history.</div>`; }
}

// Custom minimal modal for editing name
function showEditNameDialog(currentName, onSave) {
  const backdrop = document.createElement("div");
  backdrop.className = "cm-modal-backdrop fixed inset-0 z-[60] flex items-center justify-center p-4";
  backdrop.style.background = "rgba(0,0,0,0.5)";

  const card = document.createElement("div");
  card.className = "bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 w-full max-w-sm transform transition-all scale-100";
  card.innerHTML = `
    <h3 class="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Add Patient Name</h3>
    <input type="text" id="editNameInput" class="w-full px-4 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-700 dark:text-white focus:ring-2 focus:ring-primary mb-4" placeholder="Enter name..." value="${escapeHtml(currentName || '')}" autocomplete="off" />
    <div class="flex justify-end gap-2">
      <button type="button" id="cancelEditBtn" class="px-4 py-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 font-medium text-sm">Cancel</button>
      <button type="button" id="saveEditBtn" class="px-4 py-2 rounded-lg bg-gradient-to-r from-primary to-secondary text-white font-medium text-sm shadow-lg hover:opacity-90">Save</button>
    </div>
  `;

  backdrop.appendChild(card);
  document.body.appendChild(backdrop);

  const input = card.querySelector("#editNameInput");
  input.focus();

  const close = () => { backdrop.remove(); };

  card.querySelector("#cancelEditBtn").addEventListener("click", close);
  backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });

  const save = () => {
    const val = input.value.trim();
    if (val) {
      onSave(val);
      close();
    } else {
      input.classList.add("border-red-500");
    }
  };

  card.querySelector("#saveEditBtn").addEventListener("click", save);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
}

function formatInputsHtml(inputs) {
  if (!inputs) return '';
  const labels = {
    birth_weight: "Birth Weight",
    maternal_age: "Maternal Age",
    immunized: "Immunized",
    nutrition: "Nutrition Score",
    socioeconomic: "Socioeconomic",
    prenatal_visits: "Prenatal Visits"
  };

  let html = '<div class="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">';
  for (const [k, v] of Object.entries(inputs)) {
    if (k === 'patient_name') continue; // Don't show name again
    let displayVal = v;
    if (k === 'immunized') displayVal = (v == 1 ? 'Yes' : 'No');
    if (k === 'socioeconomic') {
      if (v == 0) displayVal = 'Low'; else if (v == 1) displayVal = 'Middle'; else displayVal = 'High';
    }
    // Pretty label or fallback
    const label = labels[k] || k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    html += `
      <div class="flex flex-col">
        <span class="text-xs text-gray-500 dark:text-gray-400 font-semibold">${escapeHtml(label)}</span>
        <span class="text-gray-800 dark:text-gray-200">${escapeHtml(String(displayVal))}</span>
      </div>`;
  }
  html += '</div>';
  return html;
}

function formatPlanHtml(plan) {
  if (!plan) return '<div class="text-gray-500 italic">No plan available</div>';

  // If string (Low risk usually)
  if (typeof plan === 'string') {
    return `<div class="text-gray-700 dark:text-gray-300 leading-relaxed">${escapeHtml(plan)}</div>`;
  }

  // If JSON object (High risk)
  if (typeof plan === 'object') {
    let html = '<div class="space-y-4">';

    // 1. Show Years first
    const years = plan.years || plan; // Handle if wrapped in 'years' key
    const yearKeys = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"];

    let foundYears = false;

    // 1a. Handle simple message list (Low Risk)
    if (plan.message && Array.isArray(plan.message)) {
      foundYears = true;
      html += `
         <div class="border-l-4 border-success pl-4 py-1">
           <h4 class="font-bold text-gray-900 dark:text-gray-100 mb-2">General Recommendations</h4>
           <ul class="list-disc list-outside ml-4 space-y-1 text-gray-700 dark:text-gray-300">
             ${plan.message.map(step => `<li>${escapeHtml(step)}</li>`).join('')}
           </ul>
         </div>
       `;
    }

    // 1b. Handle Year-based keys (High Risk)
    for (const yk of yearKeys) {
      if (years[yk] && Array.isArray(years[yk])) {
        foundYears = true;
        html += `
             <div class="border-l-4 border-primary pl-4 py-1">
               <h4 class="font-bold text-gray-900 dark:text-gray-100 mb-2">${escapeHtml(yk)}</h4>
               <ul class="list-disc list-outside ml-4 space-y-1 text-gray-700 dark:text-gray-300">
                 ${years[yk].map(step => `<li>${escapeHtml(step)}</li>`).join('')}
               </ul>
             </div>
           `;
      }
    }

    // Fallback if no known keys found but it's an object
    if (!foundYears) {
      html += `<pre class="whitespace-pre-wrap text-xs bg-gray-50 dark:bg-gray-900 p-3 rounded font-mono">${escapeHtml(JSON.stringify(plan, null, 2))}</pre>`;
    }

    html += '</div>';
    return html;
  }

  return `<pre>${escapeHtml(String(plan))}</pre>`;
}

function showHistoryModal(item, debug = false) {
  const modal = document.createElement("div");
  modal.className = "cm-modal-backdrop fixed inset-0 z-50 flex items-center justify-center p-4";
  modal.style.background = "rgba(15, 23, 42, 0.65)";
  modal.style.backdropFilter = "blur(4px)";

  const rawName = item.patient_name || item.inputs?.patient_name;
  const isUnknown = !rawName || rawName === 'Unknown';
  const displayName = isUnknown ? 'Unknown Patient' : rawName;
  const dateStr = new Date(item.timestamp).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

  // Icons
  const iconUser = `<svg class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>`;
  const iconPlan = `<svg class="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`;
  const iconDetails = `<svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>`;

  modal.innerHTML = `
    <div class="cm-modal w-full max-w-2xl rounded-2xl shadow-2xl bg-white dark:bg-gray-900 overflow-hidden flex flex-col max-h-[90vh]">
      
      <!-- Header -->
      <div class="px-6 py-4 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between shrink-0">
        <div class="flex items-center gap-3">
          <div class="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700">
             ${iconUser}
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h3 id="modalPatientName" class="text-lg font-bold text-gray-900 dark:text-gray-100">${escapeHtml(displayName)}</h3>
              ${isUnknown ? `<button type="button" id="addNameBtn" class="px-2 py-0.5 rounded-md bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-[10px] font-bold uppercase tracking-wider hover:bg-blue-200 transition-colors">+ Add Name</button>` : ''}
            </div>
            <p class="text-xs text-gray-500 font-medium">${escapeHtml(dateStr)}</p>
          </div>
        </div>
        <button type="button" id="closeHistModal" class="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-gray-500">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>

      <!-- Scrollable Content -->
      <div class="p-6 overflow-y-auto custom-scrollbar space-y-6">
        
        <!-- Clinical Details -->
        <section>
           <div class="flex items-center gap-2 mb-3">
             ${iconDetails}
             <h4 class="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Clinical Profile</h4>
           </div>
           <div class="bg-gray-50/80 dark:bg-gray-800/40 p-4 rounded-xl border border-gray-100 dark:border-gray-800/60">
             ${formatInputsHtml(item.inputs)}
           </div>
        </section>

        <!-- Prediction Stats -->
        <section class="grid grid-cols-2 gap-4">
           <div class="flex flex-col items-center justify-center p-4 rounded-xl border border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-800 shadow-sm">
             <span class="text-xs font-bold uppercase text-gray-400 mb-1">Risk Level</span>
             <span class="text-xl font-black ${String(item.prediction) === '1' ? 'text-red-500' : 'text-green-500'}">
               ${String(item.prediction) === '1' ? 'High Risk' : 'Low Risk'}
             </span>
           </div>
           
           <div class="flex flex-col items-center justify-center p-4 rounded-xl border border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-800 shadow-sm">
             <span class="text-xs font-bold uppercase text-gray-400 mb-1">Probability</span>
             <span class="text-xl font-black text-gray-800 dark:text-gray-200">
               ${escapeHtml(String(item.probability))}
             </span>
           </div>
        </section>

        <!-- Plan -->
        <section>
           <div class="flex items-center gap-2 mb-3">
             ${iconPlan}
             <h4 class="text-sm font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">Survival & Care Plan</h4>
           </div>
           <div class="bg-white dark:bg-gray-800 border-l-0">
             ${formatPlanHtml(item.survival_plan)}
           </div>
        </section>

      </div>
      
      <!-- Footer Actions -->
      <div class="p-4 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50 flex justify-end gap-3 shrink-0">
        <button type="button" id="deleteHistBtn" class="px-4 py-2 rounded-lg bg-white dark:bg-gray-800 border border-red-200 dark:border-red-900/30 text-red-600 dark:text-red-400 text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors shadow-sm">Delete Entry</button>
        <button type="button" id="closeHistModalFooter" class="px-4 py-2 rounded-lg bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-sm font-bold hover:opacity-90 transition-opacity shadow-lg">Done</button>
      </div>

    </div>
  `;
  document.body.appendChild(modal);

  const closeFn = () => {
    modal.style.opacity = '0';
    modal.querySelector('.cm-modal').style.transform = 'scale(0.95)';
    setTimeout(() => modal.remove(), 200);
  };

  modal.querySelector("#closeHistModal").addEventListener("click", closeFn);
  modal.querySelector("#closeHistModalFooter").addEventListener("click", closeFn);
  modal.addEventListener("click", (ev) => { if (ev.target === modal) closeFn(); });

  // Delete Handler
  modal.querySelector("#deleteHistBtn").addEventListener("click", async () => {
    if (!confirm('Delete this history entry?')) return;
    // ... logic for delete ...
    try {
      if (!item.id) { // Local delete
        const raw = localStorage.getItem("child_mortality_history_v1");
        let list = raw ? JSON.parse(raw) : [];
        list = list.filter(x => x.timestamp !== item.timestamp);
        localStorage.setItem("child_mortality_history_v1", JSON.stringify(list));
        if (typeof loadHistory === 'function') loadHistory(20, false);
        closeFn();
      } else { // Server delete
        const resp = await fetch((API_BASE || "") + `/api/history/${item.id}`, { method: 'DELETE', credentials: 'include' });
        if (resp.ok) {
          if (typeof loadHistory === 'function') loadHistory(20, false);
          closeFn();
        } else { alert('Delete failed'); }
      }
    } catch (e) { alert('Error: ' + e.message); }
  });

  // Add Name Handler
  const addNameBtn = modal.querySelector("#addNameBtn");
  if (addNameBtn) {
    addNameBtn.addEventListener("click", () => {
      showEditNameDialog("", async (newName) => {
        const finalName = newName.trim();
        const logged = !!localStorage.getItem("cm_logged_in_user");
        try {
          if (logged) {
            if (!item.id) { alert("Cannot edit server item without ID"); return; }
            const resp = await fetch((API_BASE || "") + `/api/history/${item.id}`, {
              method: 'PUT',
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ patient_name: finalName }),
              credentials: 'include'
            });
            if (resp.ok) {
              item.patient_name = finalName;
              modal.querySelector("#modalPatientName").textContent = finalName;
              addNameBtn.remove();
              if (typeof loadHistory === 'function') loadHistory(20, false);
            }
          } else {
            const rawLocal = localStorage.getItem("child_mortality_history_v1");
            let list = rawLocal ? JSON.parse(rawLocal) : [];
            const idx = list.findIndex(x => x.timestamp === item.timestamp);
            if (idx !== -1) {
              list[idx].patient_name = finalName;
              localStorage.setItem("child_mortality_history_v1", JSON.stringify(list));
              item.patient_name = finalName;
              modal.querySelector("#modalPatientName").textContent = finalName;
              addNameBtn.remove();
              if (typeof loadHistory === 'function') loadHistory(20, false);
            }
          }
        } catch (e) { alert(e.message); }
      });
    });
  }
}



// show plan
async function showPlan(planObj) {
  if (!planContainer) return;
  planContainer.innerHTML = "";
  // If a low-risk plan was returned, it may include a message or general precautions key
  const risk = (planObj && planObj.risk_level) ? planObj.risk_level : null;
  if (risk === 'low') {
    const msg = planObj.message || planObj.general_precautions || [
      'Continue routine preventive care and regular health checkups',
      'Complete immunization schedule on schedule',
      'Ensure balanced nutrition and adequate feeding',
      'Maintain safe sleeping environment (back sleeping, firm surface)',
      'Practice good hygiene and sanitation habits',
      'Monitor for early signs of illness and seek prompt care',
      'Provide safe drinking water and proper sanitation',
      'Ensure adequate rest and physical activity for development'
    ];
    const card = document.createElement('div');
    card.className = 'plan-card bg-white dark:bg-gray-800 p-4 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 md:col-span-2';
    // If message is an array, render as list; if string, split into sentences/bullets
    let bullets = [];
    if (Array.isArray(msg)) bullets = msg;
    else if (typeof msg === 'string') bullets = msg.split(/\n|\.|;|\u2022/).map(s => s.trim()).filter(Boolean);
    if (!bullets.length) bullets = ['Continue routine preventive care and regular checkups.'];
    card.innerHTML = `<h4 class="font-semibold text-primary">General Precautions</h4><ul class="mt-2 text-sm text-gray-700 dark:text-gray-200 space-y-1">${bullets.map(a => `<li>• ${escapeHtml(a)}</li>`).join('')}</ul>`;
    planContainer.appendChild(card);
    return;
  }

  const years = planObj?.years || {};
  const expectedYears = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"];
  for (const year of expectedYears) {
    const actions = Array.isArray(years[year]) && years[year].length ? years[year] : [];
    const card = document.createElement("div");
    card.className = "plan-card bg-white dark:bg-gray-800 p-4 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700";
    if (actions.length) {
      card.innerHTML = `<h4 class="font-semibold text-primary">${escapeHtml(year)}</h4><ul class="mt-2 text-sm text-gray-700 dark:text-gray-200 space-y-1">${actions.map(a => `<li>• ${escapeHtml(a)}</li>`).join('')}</ul>`;
    } else {
      card.innerHTML = `<h4 class="font-semibold text-primary">${escapeHtml(year)}</h4><div class="mt-2 text-sm text-gray-500 dark:text-gray-400">No personalized actions for this age range.</div>`;
    }
    planContainer.appendChild(card);
  }
}

// Update Clear My History button visibility
function updateClearButtonVisibility(hasHistory) {
  try {
    if (!clearHistoryBtn) return;
    clearHistoryBtn.style.display = hasHistory ? '' : 'none';
    if (historyPlaceholder) {
      historyPlaceholder.style.display = hasHistory ? 'none' : '';
    }
  } catch (e) { }
}

// ---------- B/o Toggle Logic ----------
const isBabyOf = document.getElementById("isBabyOf");
const patientNameInput = document.getElementById("patient_name");
const nameLabel = document.getElementById("nameLabel");

if (isBabyOf && patientNameInput) {
  isBabyOf.addEventListener("change", () => {
    if (isBabyOf.checked) {
      if (nameLabel) nameLabel.textContent = "Mother's Name";
      patientNameInput.placeholder = "e.g. Nancy";
    } else {
      if (nameLabel) nameLabel.textContent = "Patient Name (Optional)";
      patientNameInput.placeholder = "e.g. Baby Lucy";
    }
  });
}

// ---------- Predict handler (improved debug + network error display) ----------
predictBtn && predictBtn.addEventListener("click", async () => {
  const rawName = document.getElementById("patient_name").value.trim();
  const usePrefix = document.getElementById("isBabyOf")?.checked;
  const finalName = (usePrefix && rawName) ? ("B/o " + rawName) : rawName;

  const payload = {
    birth_weight: parseFloat(document.getElementById("birth_weight").value || 0),
    maternal_age: parseFloat(document.getElementById("maternal_age").value || 0),
    immunized: parseInt(document.getElementById("immunized").value || 0),
    nutrition: parseFloat(document.getElementById("nutrition").value || 0),
    socioeconomic: parseInt(document.getElementById("socioeconomic").value || 0),
    prenatal_visits: parseFloat(document.getElementById("prenatal_visits").value || 0),
    patient_name: finalName,
    debug: true
  };

  predictBtn.disabled = true;
  const originalText = predictBtn.innerHTML;
  predictBtn.style.opacity = "0.6";
  predictBtn.textContent = "Processing...";

  try {
    const resp = await fetch((API_BASE || "") + "/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      credentials: "include"
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "(no text)");
      // Network or backend error — show visible message
      alert("Server responded with error: " + (txt || resp.statusText || resp.status));
      return;
    }

    const data = await resp.json().catch(() => null);
    if (!data) {
      alert("Failed to parse server response (not JSON). Check backend logs and console network tab.");
      return;
    }

    // Update UI
    resultCard && resultCard.classList.remove("hidden");
    const prob = (data.mortality_risk_probability != null) ? Number(data.mortality_risk_probability) * 100 : null;
    const pred = data.mortality_prediction;
    const riskText = (pred === 1 || (prob !== null && prob >= 60)) ? "High" : (prob !== null && prob >= 35 ? "Medium" : "Low");
    // setMeter removed
    if (riskLabel) riskLabel.textContent = `Risk: ${riskText}`;
    if (probLabel) probLabel.textContent = `Probability: ${prob !== null ? prob.toFixed(2) + "%" : "N/A"} `;
    if (probBar) probBar.style.width = `${prob !== null ? Math.min(100, prob) : 0}% `;

    // Display explanation if available (hide by default in details tag, user can click to expand)
    if (explainText) {
      if (data.debug && data.debug.explanation && data.debug.explanation.ui_html) {
        // Remove "Show technical details" section if present
        let html = data.debug.explanation.ui_html;
        html = html.replace(/<details[^>]*>.*?<\/details>/gs, ''); // Remove details tag
        explainText.innerHTML = html;
      } else if (data.debug && data.debug.explanation_summary) {
        explainText.innerHTML = escapeHtml(data.debug.explanation_summary);
      } else {
        explainText.innerHTML = "Analysis not available.";
      }
    }

    await showPlan(data.survival_plan || { years: {} });

    const historyEntry = {
      timestamp: nowIso(),
      inputs: payload,
      probability: (data.mortality_risk_probability != null) ? Number(data.mortality_risk_probability) : null,
      prediction: pred,
      survival_plan: data.survival_plan || null,
      debug: data.debug || null
    };

    // Only save to localStorage for anonymous users. For logged-in users
    // save to the server instead to avoid caching another user's history
    // locally and exposing it after logout.
    const logged = !!localStorage.getItem('cm_logged_in_user');
    if (!logged) {
      pushLocalPrediction({
        timestamp: historyEntry.timestamp,
        input_json: payload,
        prediction_result: historyEntry.prediction,
        survival_plan: historyEntry.survival_plan,
        explanation: historyEntry.debug
      });
      saveHistory(historyEntry);
    } else {
      try {
        const serverResp = await fetch((API_BASE || "") + "/api/history/merge", {
          method: "POST",
          headers: authHeaders(),
          credentials: "include",
          body: JSON.stringify({
            entries: [{
              timestamp: historyEntry.timestamp,
              input_json: payload,
              prediction: historyEntry.prediction,
              probability: historyEntry.probability,
              survival_plan: historyEntry.survival_plan,
              explanation: historyEntry.debug,
              patient_name: payload.patient_name
            }]
          })
        });
        if (!serverResp.ok) {
          console.warn("Failed to save prediction to server");
        }
      } catch (e) {
        console.warn("Error saving prediction to server", e);
      }
    }

    loadHistory(10, false);

  } catch (err) {
    console.error("Predict failed", err);
    alert("Predict failed: " + (err.message || err));
  } finally {
    predictBtn.disabled = false;
    predictBtn.style.opacity = "1";
    predictBtn.innerHTML = originalText || `Predict`;
  }
});

// clear handler
clearBtn && clearBtn.addEventListener("click", () => {
  try {
    const pName = document.getElementById("patient_name");
    const babyCheck = document.getElementById("isBabyOf");
    if (pName) { pName.value = ""; pName.placeholder = "e.g. Baby Doe"; }
    if (babyCheck) { babyCheck.checked = false; }
    const nLabel = document.getElementById("nameLabel");
    if (nLabel) nLabel.textContent = "Patient Name (Optional)";

    document.getElementById("birth_weight").value = "2.8";
    document.getElementById("maternal_age").value = "26";
    document.getElementById("immunized").value = "1";
    document.getElementById("nutrition").value = "60";
    document.getElementById("socioeconomic").value = "1";
    document.getElementById("prenatal_visits").value = "4";

    if (resultSection) resultSection.style.display = "none";
    if (explanationSection) explanationSection.classList.add("hidden");
    if (planContainer) planContainer.innerHTML = "";
    if (errorDiv) errorDiv.classList.add("hidden");
  } catch (e) {
    console.error("Error clearing form:", e);
  }
});

// initial history load
try { loadHistory(6, false); } catch (e) { /* ignore */ }

// Explanation is in a <details> tag now (collapsed by default), no need to hide it here
// The user clicks the summary to expand and see the explanation

// Clear My History handler (clears server history when logged in, local history when not)
async function clearMyHistory() {
  try {
    const logged = !!localStorage.getItem('cm_logged_in_user');
    if (logged) {
      if (!confirm('Clear all your server-side history? This cannot be undone.')) return;
      const resp = await fetch((API_BASE || "") + '/api/history', { method: 'DELETE', credentials: 'include' });
      if (resp.ok) {
        alert('Server history cleared.');
        if (typeof loadHistory === 'function') loadHistory(10, false);
      } else {
        const txt = await resp.text().catch(() => null);
        alert('Failed to clear server history: ' + (txt || resp.statusText || resp.status));
      }
    } else {
      if (!confirm('Clear local history stored in your browser?')) return;
      try { localStorage.removeItem('child_mortality_history_v1'); } catch (e) { }
      try { localStorage.removeItem('cm_local_history'); } catch (e) { }
      try { localStorage.removeItem('cm_local_history'); } catch (e) { }
      alert('Local history cleared.');
      if (typeof loadHistory === 'function') loadHistory(0, false);
    }
  } catch (e) {
    console.error('clearMyHistory error', e);
    alert('Failed to clear history: ' + (e.message || e));
  }
}

if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearMyHistory);

