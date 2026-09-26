/*
 * Rendering of the patient list and the patient context card.
 * Daten kommen aus patients.js (Format dort dokumentiert).
 */

import {
    escapeHTML
} from "./results.js";

import {
    ageFromBirthDate
} from "./patients.js";


const SEX_LABELS = {
    F: "Female",
    M: "Male",
    D: "Diverse"
};


/* ============================================================
   HELPERS
============================================================ */

function initials(name) {

    return String(name || "?")
        .split(/\s+/)
        .map(part => part[0] || "")
        .join("")
        .slice(0, 2)
        .toUpperCase();
}


function formatDate(isoDate) {

    const date =
        new Date(isoDate);


    if (Number.isNaN(date.getTime())) {
        return "—";
    }


    return date.toLocaleDateString(
        "en-GB",
        {
            day: "2-digit",
            month: "short",
            year: "numeric"
        }
    );
}


/**
 * Alter und Geschlecht als gut sichtbare Chips neben dem Namen.
 */
function demographicChips(patient) {

    const age =
        ageFromBirthDate(patient.birth_date);


    const sex =
        SEX_LABELS[patient.sex] || patient.sex;


    return [
        age !== null ? `${age} years` : null,
        sex
    ]
        .filter(Boolean)
        .map(text => `<span class="demo-chip">${escapeHTML(text)}</span>`)
        .join("");
}


function statusClass(status) {

    return {
        "Awaiting MRI": "status-awaiting",
        "MRI analyzed": "status-analyzed",
        "Follow-up": "status-followup"
    }[status] || "";
}


function listOrEmpty(items) {

    if (!items || !items.length) {
        return `<p class="history-empty">None documented</p>`;
    }


    return `
        <ul>
            ${items.map(item => `<li>${escapeHTML(item)}</li>`).join("")}
        </ul>
    `;
}


/* ============================================================
   PATIENT LIST
============================================================ */

/**
 * Liste der Patienten; Klick ruft onSelect(patient.id) auf.
 */
export function renderPatientList(
    container,
    patients,
    onSelect
) {

    if (!patients.length) {

        container.innerHTML = `
            <li class="patient-empty">
                No patients match your search.
            </li>
        `;

        return;
    }


    container.innerHTML =
        patients.map(patient => `

            <li>

                <button
                    class="patient-row"
                    type="button"
                    data-patient-id="${escapeHTML(patient.id)}"
                >

                    <span class="patient-avatar">
                        ${escapeHTML(initials(patient.name))}
                    </span>


                    <span class="patient-main">

                        <span class="patient-name-line">

                            <strong>
                                ${escapeHTML(patient.name)}
                            </strong>

                            ${demographicChips(patient)}

                        </span>

                        <small>
                            ${escapeHTML(patient.id)}
                        </small>

                    </span>


                    <span class="patient-complaint">
                        ${escapeHTML(patient.chief_complaint)}
                    </span>


                    <span class="patient-meta">

                        <span class="patient-status ${statusClass(patient.status)}">
                            ${escapeHTML(patient.status)}
                        </span>

                        <small>
                            Last visit ${escapeHTML(formatDate(patient.last_visit))}
                        </small>

                    </span>


                    <span
                        class="patient-chevron"
                        aria-hidden="true"
                    >
                        ›
                    </span>

                </button>

            </li>

        `).join("");


    container
        .querySelectorAll(".patient-row")
        .forEach(row => {

            row.addEventListener(
                "click",
                () => onSelect(row.dataset.patientId)
            );

        });
}


/* ============================================================
   PATIENT CONTEXT CARD
============================================================ */

/**
 * Karte über der MRI-Analyse. patient = null -> Hinweis ohne Kontext.
 * onChange wird bei "Change patient" / "Select patient" aufgerufen.
 */
export function renderPatientContext(
    container,
    patient,
    onChange
) {

    if (!patient) {

        container.innerHTML = `

            <div class="context-empty">

                <div>

                    <strong>
                        No patient selected
                    </strong>

                    <p>
                        The analysis runs without clinical history.
                    </p>

                </div>

                <button
                    class="secondary-button"
                    type="button"
                    data-action="change"
                >
                    Select patient
                </button>

            </div>
        `;

    }

    else {

        const imaging =
            (patient.prior_imaging || []).map(study =>
                `${formatDate(study.date)} · ${study.modality} – ${study.finding}`
            );


        container.innerHTML = `

            <div class="context-header">

                <span class="patient-avatar large">
                    ${escapeHTML(initials(patient.name))}
                </span>


                <div class="context-title">

                    <p class="panel-eyebrow">
                        PATIENT CONTEXT
                    </p>

                    <div class="patient-name-line">

                        <h2>
                            ${escapeHTML(patient.name)}
                        </h2>

                        ${demographicChips(patient)}

                    </div>

                    <small>
                        ${escapeHTML(patient.id)}
                        · Last visit ${escapeHTML(formatDate(patient.last_visit))}
                    </small>

                </div>


                <span class="patient-status ${statusClass(patient.status)}">
                    ${escapeHTML(patient.status)}
                </span>


                <button
                    class="secondary-button"
                    type="button"
                    data-action="change"
                >
                    Change patient
                </button>

            </div>


            <p class="context-complaint">
                <span>Chief complaint</span>
                ${escapeHTML(patient.chief_complaint)}
            </p>


            <div class="history-grid">

                <div>
                    <h3>Symptoms</h3>
                    ${listOrEmpty(patient.symptoms)}
                </div>

                <div>
                    <h3>Conditions</h3>
                    ${listOrEmpty(patient.conditions)}
                </div>

                <div>
                    <h3>Medications</h3>
                    ${listOrEmpty(patient.medications)}
                </div>

                <div>
                    <h3>Allergies</h3>
                    ${listOrEmpty(patient.allergies)}
                </div>

            </div>


            <details class="history-details">

                <summary>
                    Full history
                </summary>


                <div class="history-grid">

                    <div>
                        <h3>Prior imaging</h3>
                        ${listOrEmpty(imaging)}
                    </div>

                    <div>
                        <h3>Prior treatment</h3>
                        ${listOrEmpty(patient.prior_treatment)}
                    </div>

                    <div>
                        <h3>Family history</h3>
                        ${listOrEmpty(patient.family_history ? [patient.family_history] : [])}
                    </div>

                    <div>
                        <h3>Notes</h3>
                        ${listOrEmpty(patient.notes ? [patient.notes] : [])}
                    </div>

                </div>

            </details>
        `;

    }


    container
        .querySelector('[data-action="change"]')
        .addEventListener("click", onChange);
}
