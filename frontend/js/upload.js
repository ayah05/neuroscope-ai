/*
 * File validation and browser preview helpers.
 */


const MAX_FILE_SIZE =
    50 * 1024 * 1024;


// Browser liefern für .nii.gz keinen verlässlichen MIME-Type,
// deshalb nur über die Dateiendung prüfen.
const SUPPORTED_EXTENSIONS =
    /\.nii(\.gz)?$/i;


/**
 * Validate uploaded MRI/image file.
 */
export function validateFile(file) {

    if (!file) {

        return {
            valid: false,
            error: "No file selected."
        };

    }


    const supported =
        SUPPORTED_EXTENSIONS.test(file.name);


    if (!supported) {

        return {
            valid: false,
            error:
                "Please select a NIfTI scan (.nii or .nii.gz)."
        };

    }


    if (file.size > MAX_FILE_SIZE) {

        return {
            valid: false,
            error:
                "The selected file is larger than 50 MB."
        };

    }


    return {
        valid: true,
        error: null
    };
}


/**
 * Determine whether file is a NIfTI scan
 * (no browser preview possible).
 */
export function isNifti(file) {

    return SUPPORTED_EXTENSIONS.test(file.name);
}


/**
 * Human-readable file size.
 */
export function formatFileSize(bytes) {

    const mb =
        bytes / 1024 / 1024;

    return `${mb.toFixed(1)} MB`;
}


/**
 * Generate browser preview URL.
 */
export function createPreviewURL(file) {

    if (isNifti(file)) {
        return null;
    }

    return URL.createObjectURL(file);
}


/**
 * Release old browser preview.
 */
export function releasePreviewURL(url) {

    if (url) {
        URL.revokeObjectURL(url);
    }
}