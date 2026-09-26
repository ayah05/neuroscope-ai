/*
 * File validation and browser preview helpers.
 */


const MAX_FILE_SIZE =
    50 * 1024 * 1024;


const SUPPORTED_TYPES = [
    "image/png",
    "image/jpeg",
    "application/dicom"
];


const SUPPORTED_EXTENSIONS =
    /\.(png|jpe?g|dcm)$/i;


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
        SUPPORTED_TYPES.includes(file.type) ||
        SUPPORTED_EXTENSIONS.test(file.name);


    if (!supported) {

        return {
            valid: false,
            error:
                "Please select a PNG, JPG or DICOM file."
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
 * Determine whether file is DICOM.
 */
export function isDicom(file) {

    return (
        file.type === "application/dicom" ||
        /\.dcm$/i.test(file.name)
    );
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

    if (isDicom(file)) {
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