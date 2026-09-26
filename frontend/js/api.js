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
 * patientId (optional) verknüpft die Analyse mit dem Patienten;
 * das Backend nutzt ihn noch nicht (Anbindung an Patientendaten folgt).
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


    const response = await fetch(
        API_BASE_URL,
        {
            method: "POST",
            body: formData
        }
    );


    if (!response.ok) {

        // FastAPI liefert Fehlertexte als {"detail": "..."},
        // Validierungsfehler (422) als {"detail": [{"msg": ...}, ...]}
        const body =
            await response.json().catch(() => null);


        const detail =
            Array.isArray(body?.detail)
                ? body.detail
                    .map(item => item.msg)
                    .join("; ")
                : body?.detail;


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