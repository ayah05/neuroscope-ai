/*
 * NeuroScope API layer
 *
 * Frontend components should not call fetch() directly.
 * All backend communication goes through this file.
 */


const API_BASE_URL =
    window.NEUROSCOPE_API_URL || "";


/**
 * Sends MRI files to the NeuroScope backend:
 * one 4D scan, or the 4 sequence files (FLAIR, T1, T1ce, T2).
 * patientId (optional) verknüpft die Analyse mit dem Patienten.
 */
export async function analyzeMRI(files, patientId = null) {

    if (!API_BASE_URL) {
        throw new Error(
            "NeuroScope API endpoint is not configured."
        );
    }


    const formData = new FormData();

    files.forEach(file => {

        formData.append(
            "images",
            file
        );

    });


    if (patientId) {

        formData.append(
            "patient_id",
            patientId
        );

    }


    return postAnalysis(
        API_BASE_URL,
        formData
    );
}


/**
 * Analyzes the MRI scans stored on the server for this patient
 * (ml/data/real_patients/<id>/), no upload needed.
 */
export async function analyzePatient(patientId) {

    return postAnalysis(
        `/api/patients/${encodeURIComponent(patientId)}/analyze`
    );
}


async function postAnalysis(url, body = undefined) {

    const response = await fetch(
        url,
        {
            method: "POST",
            body
        }
    );


    if (!response.ok) {

        // FastAPI liefert Fehlertexte als {"detail": "..."},
        // Validierungsfehler (422) als {"detail": [{"msg": ...}, ...]}
        const errorBody =
            await response.json().catch(() => null);


        const detail =
            Array.isArray(errorBody?.detail)
                ? errorBody.detail
                    .map(item => item.msg)
                    .join("; ")
                : errorBody?.detail;


        throw new Error(
            detail ||
            `NeuroScope API returned HTTP ${response.status}`
        );

    }


    const data =
        await response.json();


    if (
        !data ||
        typeof data !== "object" ||
        Array.isArray(data)
    ) {

        throw new Error(
            "Invalid response from NeuroScope API."
        );

    }


    return data;
}
