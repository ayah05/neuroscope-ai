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


/**
 * Initial empty state.
 */
export function renderEmptyResult(container) {

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
 * Loading state.
 */
export function renderLoading(container) {

    container.innerHTML = `
        <div class="loading">

            <span class="spinner"></span>

            Running MONAI segmentation...

        </div>
    `;
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
 *         enhancing_tumor_ml: 0.5
 *     },
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

    const segmentation =
        result.segmentation || {};

    const research =
        result.research || null;


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

        </div>


        ${renderResearch(research)}


        <p
            style="
                margin-top:20px;
                color:#8199a2;
                font-size:11px;
                line-height:1.5;
            "
        >
            NeuroScope AI is a research prototype.
            AI-generated segmentation must not be used
            as a standalone clinical diagnosis.
        </p>
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