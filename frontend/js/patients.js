/*
 * Patient data layer.
 *
 * Aktuell: erfundene DEMO-Patienten (keine echten Personen).
 * Später: die echten Datensätze kommen vom Backend. Dann nur die zwei
 * exportierten Funktionen unten auf fetch() umstellen, z.B.
 *
 *     GET /api/patients        -> Patient[]
 *     GET /api/patients/{id}   -> Patient
 *
 * Der Rest des Frontends benutzt ausschließlich listPatients() und
 * getPatient() und muss dafür nicht geändert werden.
 *
 * ------------------------------------------------------------
 * DATENFORMAT (Patient)
 * ------------------------------------------------------------
 * {
 *   id:              "P-1001",          // eindeutig, wird beim Upload
 *                                       // als patient_id mitgeschickt
 *   name:            "Maria Beispiel",
 *   birth_date:      "1971-03-14",      // ISO-Datum
 *   sex:             "F" | "M" | "D",
 *   status:          "Awaiting MRI" | "MRI analyzed" | "Follow-up",
 *   last_visit:      "2026-09-18",      // ISO-Datum
 *   chief_complaint: "Kurzbeschreibung der aktuellen Beschwerden",
 *   symptoms:        ["...", ...],
 *   conditions:      ["...", ...],      // Vorerkrankungen
 *   medications:     ["...", ...],
 *   allergies:       ["...", ...],
 *   prior_imaging:   [{ date: "2026-08-02", modality: "CT",
 *                       finding: "..." }, ...],
 *   prior_treatment: ["...", ...],      // leer = keine Vorbehandlung
 *   family_history:  "...",
 *   notes:           "..."              // freie Notiz des Arztes
 * }
 * Fehlende Angaben: leeres Array bzw. leerer String, nicht weglassen.
 */


const DEMO_PATIENTS = [
    {
        id: "P-1001",
        name: "Maria Beispiel",
        birth_date: "1971-03-14",
        sex: "F",
        status: "Awaiting MRI",
        last_visit: "2026-09-18",
        chief_complaint:
            "Progressive headaches over 6 weeks, new word-finding difficulties",
        symptoms: [
            "Morning headaches, worsening",
            "Word-finding difficulties",
            "One focal seizure (right arm) two weeks ago"
        ],
        conditions: [
            "Arterial hypertension",
            "Hypothyroidism"
        ],
        medications: [
            "Ramipril 5 mg",
            "Levothyroxine 75 µg",
            "Levetiracetam 500 mg (since seizure)"
        ],
        allergies: [
            "Penicillin"
        ],
        prior_imaging: [
            {
                date: "2026-09-05",
                modality: "CT head",
                finding: "Hypodense area, left hemisphere; MRI recommended"
            }
        ],
        prior_treatment: [],
        family_history: "No known brain tumors in the family",
        notes: "Referred by neurology for MRI work-up."
    },
    {
        id: "P-1002",
        name: "Thomas Muster",
        birth_date: "1958-11-02",
        sex: "M",
        status: "Follow-up",
        last_visit: "2026-09-10",
        chief_complaint:
            "Follow-up after resection and chemoradiation of a glioma",
        symptoms: [
            "Mild fatigue",
            "No new neurological deficits"
        ],
        conditions: [
            "Type 2 diabetes",
            "Glioma (histologically confirmed, 2025)"
        ],
        medications: [
            "Metformin 1000 mg",
            "Dexamethasone tapered off in 2026-06"
        ],
        allergies: [],
        prior_imaging: [
            {
                date: "2026-06-12",
                modality: "MRI",
                finding: "Post-operative changes, no clear progression"
            },
            {
                date: "2025-11-20",
                modality: "MRI",
                finding: "Enhancing lesion, right hemisphere, pre-operative"
            }
        ],
        prior_treatment: [
            "Subtotal resection (2025-12)",
            "Radiotherapy with concomitant temozolomide (2026-01 to 2026-02)"
        ],
        family_history: "Not documented",
        notes:
            "Distinguish progression vs. treatment-related changes " +
            "on follow-up imaging."
    },
    {
        id: "P-1003",
        name: "Lena Probe",
        birth_date: "1994-07-21",
        sex: "F",
        status: "Awaiting MRI",
        last_visit: "2026-09-22",
        chief_complaint:
            "First generalized seizure, otherwise healthy",
        symptoms: [
            "Generalized tonic-clonic seizure",
            "Occasional headaches"
        ],
        conditions: [],
        medications: [
            "Oral contraceptive"
        ],
        allergies: [],
        prior_imaging: [],
        prior_treatment: [],
        family_history: "Mother: epilepsy",
        notes: "First MRI after emergency admission."
    },
    {
        id: "P-1004",
        name: "Jonas Test",
        birth_date: "1966-01-30",
        sex: "M",
        status: "MRI analyzed",
        last_visit: "2026-09-01",
        chief_complaint:
            "Personality changes and gait unsteadiness reported by family",
        symptoms: [
            "Apathy, personality change",
            "Gait unsteadiness"
        ],
        conditions: [
            "Former smoker (30 pack-years)",
            "COPD"
        ],
        medications: [
            "Tiotropium inhaler"
        ],
        allergies: [
            "Iodinated contrast (mild rash)"
        ],
        prior_imaging: [
            {
                date: "2026-08-15",
                modality: "Chest X-ray",
                finding: "Unremarkable"
            }
        ],
        prior_treatment: [],
        family_history: "Father: lung cancer",
        notes:
            "Smoking history: consider systemic work-up depending " +
            "on imaging."
    }
];


/**
 * Alle Patienten (für die Übersicht).
 * Später: return (await fetch("/api/patients")).json();
 */
export async function listPatients() {

    return DEMO_PATIENTS;
}


/**
 * Ein Patient per ID, oder null.
 * Später: fetch(`/api/patients/${encodeURIComponent(id)}`)
 */
export async function getPatient(id) {

    return DEMO_PATIENTS.find(patient => patient.id === id) || null;
}


/**
 * Alter in Jahren aus dem Geburtsdatum.
 */
export function ageFromBirthDate(birthDate) {

    const birth =
        new Date(birthDate);


    if (Number.isNaN(birth.getTime())) {
        return null;
    }


    const today =
        new Date();


    let age =
        today.getFullYear() - birth.getFullYear();


    const birthdayPassed =
        today.getMonth() > birth.getMonth() ||
        (
            today.getMonth() === birth.getMonth() &&
            today.getDate() >= birth.getDate()
        );


    if (!birthdayPassed) {
        age -= 1;
    }


    return age;
}
