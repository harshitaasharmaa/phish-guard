/**
 * PhishGuard Frontend Application Logic
 * Integrates with Flask /health and /predict endpoints.
 */

document.addEventListener("DOMContentLoaded", () => {
    const urlForm = document.getElementById("urlForm");
    const urlInput = document.getElementById("urlInput");
    const checkBtn = document.getElementById("checkBtn");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const loadingStep = document.getElementById("loadingStep");
    const errorBox = document.getElementById("errorBox");
    const errorTitle = document.getElementById("errorTitle");
    const errorMessage = document.getElementById("errorMessage");
    const resultBox = document.getElementById("resultBox");
    const systemStatus = document.getElementById("systemStatus");
    const statusText = document.getElementById("statusText");
    const samplePills = document.querySelectorAll(".pill-btn");

    // Initialize backend health check
    checkBackendHealth();

    // Sample pills click handler
    samplePills.forEach(pill => {
        pill.addEventListener("click", () => {
            const sampleUrl = pill.getAttribute("data-url");
            urlInput.value = sampleUrl;
            urlInput.focus();
            analyzeUrl(sampleUrl);
        });
    });

    // Form submit handler
    urlForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const urlValue = urlInput.value.trim();
        if (!urlValue) {
            showError("Input Required", "Please enter a valid website address to analyze.");
            return;
        }
        analyzeUrl(urlValue);
    });

    /**
     * Query /health endpoint to verify backend status
     */
    async function checkBackendHealth() {
        try {
            const resp = await fetch("/health");
            const data = await resp.json();

            if (resp.ok && data.status === "healthy") {
                systemStatus.className = "status-indicator online";
                statusText.textContent = `Model Online: ${data.model_type || "RandomForest"} (30 Features)`;
            } else {
                systemStatus.className = "status-indicator degraded";
                statusText.textContent = "Service Degraded";
            }
        } catch (err) {
            systemStatus.className = "status-indicator degraded";
            statusText.textContent = "Backend Offline";
        }
    }

    /**
     * Send URL to POST /predict
     */
    async function analyzeUrl(targetUrl) {
        // Reset states
        hideError();
        hideResults();
        setLoading(true);

        // Dynamic loading step text
        const stepInterval = simulateLoadingSteps();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ url: targetUrl })
            });

            const data = await response.json();
            clearInterval(stepInterval);
            setLoading(false);

            if (response.status === 422 && data.status === "incomplete_feature_extraction") {
                // Expected safe refusal: Lexical features extracted, but model prediction safely withheld
                renderIncompleteAnalysis(data);
            } else if (response.ok && data.status === "success") {
                // Complete prediction payload
                renderCompletePrediction(data);
            } else {
                // Error response (e.g. 400 Bad Request)
                showError(data.error || "Analysis Failed", data.message || "An unexpected error occurred.");
            }
        } catch (err) {
            clearInterval(stepInterval);
            setLoading(false);
            showError("Network Error", "Unable to communicate with the PhishGuard backend service. Please check your connection.");
        }
    }

    /**
     * Render Incomplete Analysis result (Requirement 8)
     * Clearly and honestly explains why prediction is withheld, displays the 8 extracted
     * lexical features, and outlines the 22 missing external dependencies.
     */
    function renderIncompleteAnalysis(data) {
        const extracted = data.extracted_lexical_features || {};
        const missingCats = data.missing_features_by_category || {};
        const summary = data.coverage_summary || {};

        let featuresHtml = "";
        for (const [feat, val] of Object.entries(extracted)) {
            let badgeClass = "safe";
            let badgeText = "Legitimate (1)";

            if (val === -1) {
                badgeClass = "phish";
                badgeText = "Phishing (-1)";
            } else if (val === 0) {
                badgeClass = "suspicious";
                badgeText = "Suspicious (0)";
            }

            featuresHtml += `
                <div class="feature-pill">
                    <span class="feat-name">${escapeHtml(feat)}</span>
                    <span class="feat-badge ${badgeClass}">${badgeText}</span>
                </div>
            `;
        }

        // Missing features HTML
        const catMap = {
            network_and_dns: { title: "Network, DNS & Certificates", count: 6 },
            webpage_content_dom: { title: "Webpage DOM & JavaScript", count: 11 },
            external_third_party_apis: { title: "Third-Party Intelligence APIs", count: 5 }
        };

        let missingCategoriesHtml = "";
        for (const [catKey, meta] of Object.entries(catMap)) {
            const list = missingCats[catKey] || [];
            const tags = list.map(item => `<span class="feature-tag">${escapeHtml(item)}</span>`).join("");
            missingCategoriesHtml += `
                <div class="category-card">
                    <div class="cat-header">
                        <h5>${meta.title}</h5>
                        <span class="cat-count">${list.length} features</span>
                    </div>
                    <div class="tag-cloud">
                        ${tags}
                    </div>
                </div>
            `;
        }

        resultBox.innerHTML = `
            <!-- Incomplete Feature Extraction Alert Banner -->
            <div class="status-banner warning">
                <div class="banner-icon">
                    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                </div>
                <div class="banner-body">
                    <h3>Safe Refusal Enforced: Feature Extraction Incomplete</h3>
                    <p>
                        The backend's Random Forest classifier requires <strong>30 pre-extracted website features</strong> in exact column order. 
                        Only <strong>8 lexical features</strong> could be reliably derived from this URL string alone without making risky network requests, 
                        probing DNS records, or downloading and parsing untrusted website HTML/JavaScript DOM.
                    </p>
                    <p>
                        <strong>Scientific Integrity Note:</strong> PhishGuard safely withholds a final classification rather than fabricating 
                        fake values for the 22 missing features, which would generate misleading and unverified predictions.
                    </p>
                    <div class="scanned-url-chip">
                        <span>Analyzed Target:</span>
                        <strong>${escapeHtml(data.url || "")}</strong>
                    </div>
                </div>
            </div>

            <!-- Feature Coverage Meter -->
            <div class="coverage-card">
                <div class="meter-header">
                    <span>Model Feature Coverage Audit</span>
                    <span>${summary.total_available || 8} / ${summary.total_required || 30} Features (${summary.coverage_percentage || "26.67%"})</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${summary.coverage_percentage || "26.67%"}"></div>
                </div>
                <div class="meter-legend">
                    <span>Available Lexical Features: ${summary.total_available || 8}</span>
                    <span>Unextracted External Features: ${summary.total_missing || 22}</span>
                </div>
            </div>

            <!-- Available Lexical Features -->
            <div class="features-block">
                <div class="features-block-header">
                    <h4>Calculated Lexical Features (From URL Alone)</h4>
                    <span class="badge-tag info">8 Features Computed</span>
                </div>
                <div class="features-grid">
                    ${featuresHtml}
                </div>
            </div>

            <!-- Missing Features Breakdown -->
            <div class="missing-block">
                <div class="features-block-header">
                    <h4>Features Requiring External Web Probes (Not Evaluated)</h4>
                    <span class="badge-tag info">22 Features Safeguarded</span>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.5rem;">
                    To protect users and servers from malicious website payloads, automated background crawling and script execution are disabled.
                </p>
                <div class="missing-categories">
                    ${missingCategoriesHtml}
                </div>
            </div>
        `;

        resultBox.classList.remove("hidden");
    }

    /**
     * Render Complete Prediction result (When all 30 features are supplied)
     */
    function renderCompletePrediction(data) {
        const isPhish = data.prediction === -1;
        const bannerClass = isPhish ? "danger" : "safe";
        const title = isPhish ? "Phishing Website Detected" : "Legitimate Website Verified";

        resultBox.innerHTML = `
            <div class="status-banner ${bannerClass}">
                <div class="banner-body">
                    <h3>${title}</h3>
                    <p>Model Confidence: <strong>${data.confidence}%</strong></p>
                    <p>Phishing Probability: ${(data.probabilities.phishing * 100).toFixed(2)}% | Legitimate Probability: ${(data.probabilities.legitimate * 100).toFixed(2)}%</p>
                </div>
            </div>
        `;
        resultBox.classList.remove("hidden");
    }

    /**
     * Helper: Simulate loading step text
     */
    function simulateLoadingSteps() {
        const steps = [
            "Validating URL syntax...",
            "Extracting 8 lexical features...",
            "Auditing 30-feature vector schema...",
            "Checking model dependency boundaries..."
        ];
        let idx = 0;
        loadingStep.textContent = steps[0];
        return setInterval(() => {
            idx = (idx + 1) % steps.length;
            loadingStep.textContent = steps[idx];
        }, 350);
    }

    function setLoading(isLoading) {
        if (isLoading) {
            loadingIndicator.classList.remove("hidden");
            checkBtn.disabled = true;
        } else {
            loadingIndicator.classList.add("hidden");
            checkBtn.disabled = false;
        }
    }

    function showError(title, msg) {
        errorTitle.textContent = title;
        errorMessage.textContent = msg;
        errorBox.classList.remove("hidden");
    }

    function hideError() {
        errorBox.classList.add("hidden");
    }

    function hideResults() {
        resultBox.classList.add("hidden");
        resultBox.innerHTML = "";
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
