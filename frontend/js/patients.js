/*
 * Patient data layer.
 *
 * Die Patienten kommen vom Backend (backend/app.py):
 *
 *     GET /api/patients        -> Patient[]
 *     GET /api/patients/{id}   -> Patient
 *
 * Quelle: ein Ordner pro Patient in ml/data/real_patients/<ID>/ mit
 * flair/t1/t1ce/t2.nii.gz. Die Krankengeschichte liegt optional als
 * patient.json im selben Ordner (Felder siehe unten). Ohne patient.json
 * ist der Patient nur mit seiner ID und den Scans bekannt.
 *
 * Der Rest des Frontends benutzt ausschließlich listPatients() und
 * getPatient().
 *
 * ------------------------------------------------------------
 * DATENFORMAT (Patient)
 * ------------------------------------------------------------
 * {
 *   id:              "BRATS_001",       // = Ordnername
 *   name:            "Maria Beispiel",  // ohne patient.json = id
 *   birth_date:      "1971-03-14",      // ISO-Datum oder ""
 *   sex:             "F" | "M" | "D" | "",
 *   status:          "Scans available" | "Incomplete scans" | ...,
 *   last_visit:      "2026-09-18",      // ISO-Datum oder ""
 *   chief_complaint: "Kurzbeschreibung der aktuellen Beschwerden",
 *   symptoms:        ["...", ...],
 *   conditions:      ["...", ...],      // Vorerkrankungen
 *   medications:     ["...", ...],
 *   allergies:       ["...", ...],
 *   prior_imaging:   [{ date: "2026-08-02", modality: "CT",
 *                       finding: "..." }, ...],
 *   prior_treatment: ["...", ...],
 *   family_history:  "...",
 *   notes:           "...",
 *
 *   // vom Backend ergänzt, nicht Teil von patient.json:
 *   has_scans:        true,             // alle 4 Sequenzen vorhanden
 *   has_ground_truth: true,
 *   scans:            { flair: true, t1: true, t1ce: true, t2: true }
 * }
 * Fehlende Angaben: leeres Array bzw. leerer String.
 */


async function fetchJSON(url) {

    const response =
        await fetch(url);


    if (!response.ok) {

        throw new Error(
            `Could not load patients (HTTP ${response.status}).`
        );

    }


    return response.json();
}


/**
 * Alle Patienten (für die Übersicht).
 */
export async function listPatients() {

    return fetchJSON("/api/patients");
}


/**
 * Ein Patient per ID, oder null.
 */
export async function getPatient(id) {

    try {
        return await fetchJSON(
            `/api/patients/${encodeURIComponent(id)}`
        );
    }
    catch {
        return null;
    }
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
