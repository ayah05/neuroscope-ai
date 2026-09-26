import {
    analyzeMRI
} from "./api.js";


import {
    validateFile,
    isNifti,
    formatFileSize,
    createPreviewURL,
    releasePreviewURL
} from "./upload.js";


import {
    renderEmptyResult,
    renderLoading,
    renderAnalysisResult
} from "./results.js";


/* ============================================================
   DOM HELPERS
============================================================ */

const $ = id =>
    document.getElementById(id);


/* ============================================================
   APPLICATION STATE
============================================================ */

const state = {

    selectedFile: null,

    previewURL: null,

    viewer: {
        zoom: 1,
        rotation: 0
    },

    analysis: {
        loading: false,
        result: null
    }

};


/* ============================================================
   ERROR HANDLING
============================================================ */

function setError(message = "") {

    const errorBox =
        $("errorBox");


    errorBox.textContent =
        message;


    errorBox.style.display =
        message
            ? "block"
            : "none";
}


/* ============================================================
   VIEWER
============================================================ */

function updateTransform() {

    const {
        zoom,
        rotation
    } = state.viewer;


    $("scanImage").style.transform =
        `scale(${zoom}) rotate(${rotation}deg)`;
}


function resetViewerTransform() {

    state.viewer.zoom = 1;

    state.viewer.rotation = 0;

    updateTransform();
}


/* ============================================================
   FILE SELECTION
============================================================ */

function handleFile(file) {

    if (!file) {
        return;
    }


    setError("");


    const validation =
        validateFile(file);


    if (!validation.valid) {

        setError(
            validation.error
        );

        return;
    }


    releasePreviewURL(
        state.previewURL
    );


    state.selectedFile =
        file;


    state.previewURL =
        createPreviewURL(file);


    resetViewerTransform();


    renderEmptyResult(
        $("resultContent")
    );


    $("resultTag").textContent =
        "Ready";


    $("step2")
        .classList
        .remove("active");


    $("step3")
        .classList
        .remove("active");


    const nifti =
        isNifti(file);


    updateFileInformation(
        file,
        nifti
    );


    updatePreview(
        file,
        nifti
    );


    $("analyzeBtn").disabled =
        false;
}


/* ============================================================
   FILE INFORMATION
============================================================ */

function updateFileInformation(
    file,
    nifti
) {

    $("scanMeta").textContent =
        `${file.name} · ${formatFileSize(file.size)}`;


    $("imageTag").textContent =
        nifti
            ? "NIFTI"
            : getImageType(file);


    $("viewportLabel").textContent =
        nifti
            ? "NIFTI · PREVIEW AFTER ANALYSIS"
            : "MRI PREVIEW · LOCAL";
}


function getImageType(file) {

    if (file.type === "image/png") {
        return "PNG";
    }

    if (file.type === "image/jpeg") {
        return "JPG";
    }

    return "IMAGE";
}


/* ============================================================
   PREVIEW
============================================================ */

function updatePreview(
    file,
    nifti
) {

    const image =
        $("scanImage");


    $("scanEmpty").style.display =
        "none";


    if (nifti) {

        image.style.display =
            "none";


        image.removeAttribute(
            "src"
        );


        $("dicomMessage").style.display =
            "block";


        setViewerButtonsEnabled(
            false
        );

        return;
    }


    $("dicomMessage").style.display =
        "none";


    image.src =
        state.previewURL;


    image.style.display =
        "block";


    setViewerButtonsEnabled(
        true
    );
}


/* ============================================================
   VIEWER BUTTONS
============================================================ */

function setViewerButtonsEnabled(
    enabled
) {

    [
        "zoomOut",
        "zoomIn",
        "rotate"
    ].forEach(id => {

        $(id).disabled =
            !enabled;

    });
}


/* ============================================================
   SEGMENTATION VIEW
============================================================ */

function showSegmentation(result) {

    if (!result.overlay_image) {
        return;
    }


    const image =
        $("scanImage");


    image.src =
        result.overlay_image;


    image.alt =
        "MRI slice with MONAI tumor segmentation";


    image.style.display =
        "block";


    $("dicomMessage").style.display =
        "none";


    $("viewportLabel").textContent =
        `MONAI SEGMENTATION · AXIAL SLICE ${result.overlay_slice}`;


    resetViewerTransform();


    setViewerButtonsEnabled(
        true
    );
}


/* ============================================================
   ANALYSIS
============================================================ */

async function startAnalysis() {

    if (
        !state.selectedFile ||
        state.analysis.loading
    ) {
        return;
    }


    setError("");


    state.analysis.loading =
        true;


    $("analyzeBtn").disabled =
        true;


    $("resultTag").textContent =
        "Running";


    $("step2")
        .classList
        .add("active");


    renderLoading(
        $("resultContent")
    );


    try {

        const result =
            await analyzeMRI(
                state.selectedFile
            );


        state.analysis.result =
            result;


        $("resultTag").textContent =
            "Complete";


        $("step3")
            .classList
            .add("active");


        renderAnalysisResult(
            $("resultContent"),
            result
        );


        showSegmentation(
            result
        );

    }

    catch (error) {

        console.error(
            error
        );


        $("resultTag").textContent =
            "Error";


        setError(
            error?.message ||
            "MRI analysis failed."
        );


        renderEmptyResult(
            $("resultContent")
        );

    }

    finally {

        state.analysis.loading =
            false;


        $("analyzeBtn").disabled =
            false;
    }
}


/* ============================================================
   EVENT LISTENERS
============================================================ */

$("chooseBtn")
    .addEventListener(
        "click",
        () => $("fileInput").click()
    );


$("fileInput")
    .addEventListener(
        "change",
        event => {

            handleFile(
                event.target.files[0]
            );

        }
    );


/* Drag & Drop */

const dropzone =
    $("dropzone");


[
    "dragenter",
    "dragover"
].forEach(eventName => {

    dropzone.addEventListener(
        eventName,
        event => {

            event.preventDefault();

            dropzone.classList.add(
                "dragging"
            );

        }
    );

});


[
    "dragleave",
    "drop"
].forEach(eventName => {

    dropzone.addEventListener(
        eventName,
        event => {

            event.preventDefault();

            dropzone.classList.remove(
                "dragging"
            );

        }
    );

});


dropzone.addEventListener(
    "drop",
    event => {

        handleFile(
            event.dataTransfer.files[0]
        );

    }
);


/* Zoom */

$("zoomIn")
    .addEventListener(
        "click",
        () => {

            state.viewer.zoom =
                Math.min(
                    state.viewer.zoom + 0.25,
                    3
                );

            updateTransform();

        }
    );


$("zoomOut")
    .addEventListener(
        "click",
        () => {

            state.viewer.zoom =
                Math.max(
                    state.viewer.zoom - 0.25,
                    0.5
                );

            updateTransform();

        }
    );


/* Rotation */

$("rotate")
    .addEventListener(
        "click",
        () => {

            state.viewer.rotation =
                (
                    state.viewer.rotation + 90
                ) % 360;

            updateTransform();

        }
    );


/* Analyze */

$("analyzeBtn")
    .addEventListener(
        "click",
        startAnalysis
    );


/* ============================================================
   CLEANUP
============================================================ */

window.addEventListener(
    "pagehide",
    () => {

        releasePreviewURL(
            state.previewURL
        );

    }
);


/* ============================================================
   INITIAL STATE
============================================================ */

renderEmptyResult(
    $("resultContent")
);