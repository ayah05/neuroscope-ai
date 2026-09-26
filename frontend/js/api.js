/*
 * NeuroScope API layer
 *
 * Frontend components should not call fetch() directly.
 * All backend communication goes through this file.
 */


const API_BASE_URL =
    window.NEUROSCOPE_API_URL || "";


/**
 * Sends an MRI file to the NeuroScope backend.
 */
export async function analyzeMRI(file) {

    if (!API_BASE_URL) {
        throw new Error(
            "NeuroScope API endpoint is not configured."
        );
    }


    const formData = new FormData();

    formData.append(
        "image",
        file
    );


    const response = await fetch(
        API_BASE_URL,
        {
            method: "POST",
            body: formData
        }
    );


    if (!response.ok) {

        throw new Error(
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