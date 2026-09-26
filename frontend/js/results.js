/*
 * Rendering of MONAI + research results.
 */


function escapeHTML(value) {

    return String(value ?? "")
        .replace(
            /[&<>"']/g,
            character => ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                "\"": "&quot;",
                "'": "&#39;"
            })[character]
        );

}


/* ============================================================
   LOADING PROGRESS
============================================================ */

/*
 * Das Backend antwortet erst am Ende, es gibt keine echten
 * Zwischenstände. Der aktive Schritt wird deshalb anhand der
 * gemessenen Dauer geschätzt (GPU ~40 s, Amass + LLM ~10 s).
 * "until" = Sekunde, ab der der nächste Schritt als aktiv gilt.
 */
const LOADING_STEPS = [
    {
        name: "Uploading scan",
        detail: "",
        until: 3
    },
    {
        name: "Segmenting tumor regions",
        detail: "MONAI · GPU",
        until: 45
    },
    {
        name: "Searching medical literature",
        detail: "Amass",
        until: 50
    },
    {
        name: "Writing clinical summary",
        detail: "AI",
        until: Infinity
    }
];


// ab hier Hinweis auf GPU-Kaltstart
const SLOW_AFTER_SECONDS = 90;


let loadingTimer = null;


function stopLoadingTimer() {

    clearInterval(loadingTimer);

    loadingTimer = null;
}


function formatElapsed(seconds) {

    const minutes =
        Math.floor(seconds / 60);

    const rest =
        String(seconds % 60).padStart(2, "0");

    return `${minutes}:${rest}`;
}


function renderLoadingSteps(container, seconds) {

    // Dauert es sehr lange, ist fast immer der GPU-Kaltstart schuld:
    // dann Segmentierung als aktiv zeigen statt als erledigt.
    const activeIndex =
        seconds >= SLOW_AFTER_SECONDS
            ? 1
            : LOADING_STEPS.findIndex(step => seconds < step.until);


    container.querySelector(".loading-steps").innerHTML =
        LOADING_STEPS.map((step, index) => {

            const state =
                index < activeIndex
                    ? "done"
                    : index === activeIndex
                        ? "active"
                        : "pending";


            return `
                <li class="loading-step ${state}">

                    <span class="loading-marker">
                        ${state === "done" ? "✓" : ""}
                    </span>

                    <span class="loading-name">
                        ${escapeHTML(step.name)}
                    </span>

                    <small>
                        ${escapeHTML(step.detail)}
                    </small>

                </li>
            `;

        }).join("");


    container.querySelector(".loading-time").textContent =
        formatElapsed(seconds);


    container.querySelector(".loading-hint").textContent =
        seconds < SLOW_AFTER_SECONDS
            ? "Usually takes about a minute."
            : "Taking longer than usual – the GPU may be starting up " +
              "after a break. Please keep this page open.";
}


/**
 * Initial empty state.
 */
export function renderEmptyResult(container) {

    stopLoadingTimer();

    container.innerHTML = `
        <div class="empty-result">

            <div class="result-icon">
                AI
            </div>

            <h3>
                No analysis yet
            </h3>

            <p>
                Select an MRI and start the analysis
                to generate a MONAI segmentation.
            </p>

        </div>
    `;
}


/**
 * Loading state: Schritt-Liste mit Laufzeit.
 */
export function renderLoading(
    container,
    fileName = ""
) {

    stopLoadingTimer();


    container.innerHTML = `
        <div class="loading">

            <div class="loading-header">

                <strong>
                    Analyzing ${escapeHTML(fileName || "scan")}
                </strong>

                <span class="loading-time">
                    0:00
                </span>

            </div>


            <ol class="loading-steps"></ol>


            <p class="loading-hint"></p>

        </div>
    `;


    const startedAt =
        Date.now();


    const update = () =>
        renderLoadingSteps(
            container,
            Math.floor((Date.now() - startedAt) / 1000)
        );


    update();


    loadingTimer =
        setInterval(update, 1000);
}


/**
 * Render final NeuroScope result.
 *
 * Expected backend structure:
 *
 * {
 *     patient_id: "BRATS_457",
 *
 *     segmentation: {
 *         whole_tumor_ml: 83.9,
 *         tumor_core_ml: 3.5,
 *         enhancing_tumor_ml: 0.5,
 *         edema_ml: 80.4
 *     },
 *
 *     report_html: "<h1>Clinical Decision Support ...",
 *
 *     overview: {                       // null beim Fallback-Bericht
 *         summary: "...",
 *         primary_consideration: "...",
 *         primary_evidence: "...",
 *         next_steps: ["..."]
 *     },
 *
 *     evidence: [{ number, kind, title, url, meta, summary }],
 *
 *     overlay_image: "data:image/png;base64,...",
 *     overlay_slice: 77,
 *
 *     research: {
 *         summary: "...",
 *         papers: 3,
 *         trials: 2
 *     }
 * }
 */
export function renderAnalysisResult(
    container,
    result
) {

    stopLoadingTimer();


    const evidence =
        result.evidence || [];


    const patient =
        escapeHTML(
            result.patient_id || "MRI scan"
        );


    container.innerHTML = `

        <div class="result-summary">

            <h3>
                Tumor segmentation complete
            </h3>

            <p>
                ${patient}
            </p>

        </div>


        <div
            class="tabs"
            role="tablist"
        >

            ${createTab("overview", "Overview", true)}

            ${createTab("report", "Report", false)}

            ${createTab(
                "evidence",
                `Evidence (${evidence.length})`,
                false
            )}

        </div>


        <section
            class="tab-panel"
            id="tab-overview"
            role="tabpanel"
        >
            ${renderOverview(result)}
        </section>


        <section
            class="tab-panel"
            id="tab-report"
            role="tabpanel"
            hidden
        >
            ${renderReport(result.report_html)}
        </section>


        <section
            class="tab-panel"
            id="tab-evidence"
            role="tabpanel"
            hidden
        >
            ${renderResearch(result.research || null)}

            ${renderEvidenceList(evidence)}
        </section>


        <p class="result-disclaimer">
            NeuroScope AI is a research prototype.
            AI-generated segmentation must not be used
            as a standalone clinical diagnosis.
        </p>
    `;


    bindTabs(container);
}


/* ============================================================
   TABS
============================================================ */

function createTab(
    id,
    label,
    active
) {

    return `
        <button
            class="tab${active ? " active" : ""}"
            type="button"
            role="tab"
            data-tab="${id}"
            aria-controls="tab-${id}"
            aria-selected="${active}"
        >
            ${escapeHTML(label)}
        </button>
    `;
}


function bindTabs(container) {

    const tabs =
        container.querySelectorAll(".tab");


    tabs.forEach(tab => {

        tab.addEventListener(
            "click",
            () => {

                tabs.forEach(other => {

                    const active =
                        other === tab;


                    other.classList.toggle(
                        "active",
                        active
                    );


                    other.setAttribute(
                        "aria-selected",
                        String(active)
                    );


                    container.querySelector(
                        `#tab-${other.dataset.tab}`
                    ).hidden = !active;

                });

            }
        );

    });
}


/* ============================================================
   OVERVIEW TAB
============================================================ */

function renderOverview(result) {

    const segmentation =
        result.segmentation || {};

    const overview =
        result.overview || null;


    const metrics = `

        <p class="section-label">
            SEGMENTATION
        </p>


        <div class="metric-grid">

            ${createMetric(
                "Whole Tumor",
                segmentation.whole_tumor_ml
            )}

            ${createMetric(
                "Tumor Core",
                segmentation.tumor_core_ml
            )}

            ${createMetric(
                "Enhancing Tumor",
                segmentation.enhancing_tumor_ml
            )}

            ${createMetric(
                "Edema",
                segmentation.edema_ml
            )}

        </div>
    `;


    // Fallback-Bericht (LLM nicht erreichbar): keine Kurzfassung
    if (!overview) {

        return `
            ${metrics}

            <p class="overview-hint">
                See the <strong>Report</strong> tab
                for the full findings.
            </p>
        `;

    }


    const nextSteps =
        (overview.next_steps || [])
            .map(step => `<li>${escapeHTML(step)}</li>`)
            .join("");


    return `

        ${metrics}


        <p class="section-label overview-label">
            KEY FINDINGS
        </p>

        <p class="overview-text">
            ${escapeHTML(overview.summary)}
        </p>


        <div class="consideration-card">

            <span>
                Primary consideration
            </span>

            <strong>
                ${escapeHTML(overview.primary_consideration)}
            </strong>

            <p>
                ${escapeHTML(overview.primary_evidence)}
            </p>

        </div>


        <p class="section-label overview-label">
            SUGGESTED NEXT STEPS
        </p>

        <ul class="next-steps">
            ${nextSteps}
        </ul>


        <p class="overview-hint">
            Differential diagnosis and limitations:
            see the <strong>Report</strong> tab.
        </p>
    `;
}


/* ============================================================
   EVIDENCE TAB
============================================================ */

function safeHref(url) {

    return /^https?:\/\//i.test(url || "")
        ? escapeHTML(url)
        : null;
}


function renderEvidenceList(evidence) {

    if (!evidence.length) {
        return "";
    }


    const items =
        evidence.map(item => {

            const href =
                safeHref(item.url);


            const title =
                href
                    ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${escapeHTML(item.title)}</a>`
                    : escapeHTML(item.title);


            return `
                <li class="evidence-item">

                    <span class="evidence-number">
                        ${escapeHTML(item.number)}
                    </span>

                    <div>

                        <strong>
                            ${title}
                        </strong>

                        <small>
                            ${escapeHTML(item.meta)}
                        </small>

                        <p>
                            ${escapeHTML(item.summary)}
                        </p>

                    </div>

                </li>
            `;

        }).join("");


    return `
        <ol class="evidence-list">
            ${items}
        </ol>
    `;
}


function createMetric(
    name,
    value
) {

    const displayValue =
        value !== undefined &&
        value !== null
            ? `${escapeHTML(value)} ml`
            : "—";


    return `

        <div class="metric">

            <span>
                ${escapeHTML(name)}
            </span>

            <strong>
                ${displayValue}
            </strong>

        </div>
    `;
}


/**
 * Befundbericht vom Backend (backend/app.py, aus ml/report.py).
 * Das HTML wird serverseitig aus unseren eigenen Textbausteinen
 * erzeugt; die Patienten-ID ist dort auf sichere Zeichen reduziert.
 */
function renderReport(reportHTML) {

    if (!reportHTML) {
        return "";
    }


    return `

        <article class="report">
            ${reportHTML}
        </article>
    `;
}


function renderResearch(research) {

    if (!research) {

        return `
            <div class="research-card">

                <h4>
                    Research Evidence
                </h4>

                <p>
                    AMAAS research data has not
                    been loaded yet.
                </p>

            </div>
        `;

    }


    return `

        <div class="research-card">

            <h4>
                Research Evidence
            </h4>

            <p>
                ${escapeHTML(
                    research.summary ||
                    "Research evidence available."
                )}
            </p>

            <p style="margin-top:10px">

                Papers:
                <strong>
                    ${escapeHTML(
                        research.papers ?? 0
                    )}
                </strong>

                &nbsp;·&nbsp;

                Clinical trials:
                <strong>
                    ${escapeHTML(
                        research.trials ?? 0
                    )}
                </strong>

            </p>

        </div>
    `;
}