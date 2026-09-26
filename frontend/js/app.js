import {
    analyzeMRI
} from "./api.js";


import {
    validateSelection,
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


import {
    listPatients,
    getPatient
} from "./patients.js";


import {
    renderPatientList,
    renderPatientContext
} from "./patientsView.js";


/* ============================================================
   DOM HELPERS
============================================================ */

const $ = id =>
    document.getElementById(id);


/* ============================================================
   APPLICATION STATE
============================================================ */

const state = {

    // alle Patienten (für Liste + Suche)
    patients: [],

    // ausgewählter Patient oder null (Analyse ohne Kontext)
    patient: null,

    // ein 4D-Scan oder 4 Sequenz-Dateien
    selectedFiles: [],

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

function handleFiles(fileList) {

    const files =
        Array.from(fileList || []);


    if (!files.length) {
        return;
    }


    setError("");


    const validation =
        validateSelection(files);


    if (!validation.valid) {

        setError(
            validation.error
        );

        return;
    }


    releasePreviewURL(
        state.previewURL
    );


    state.selectedFiles =
        files;


    // NIfTI hat keine Browser-Vorschau; relevant nur für Einzelbilder
    const file =
        files[0];


    state.previewURL =
        createPreviewURL(file);


    resetViewerTransform();


    renderEmptyResult(
        $("resultContent")
    );


    // Legende gehört zum alten Ergebnis
    $("segLegend").hidden =
        true;


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
        files,
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

function selectionLabel(files) {

    return files.length === 1
        ? files[0].name
        : `${files.length} MRI sequences`;
}


function updateFileInformation(
    files,
    nifti
) {

    const totalSize =
        files.reduce((sum, file) => sum + file.size, 0);


    const names =
        files.length === 1
            ? files[0].name
            : `${files.length} files: ${files.map(file => file.name).join(", ")}`;


    $("scanMeta").textContent =
        `${names} · ${formatFileSize(totalSize)}`;


    $("imageTag").textContent =
        nifti
            ? "NIFTI"
            : getImageType(files[0]);


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


    renderLegend(
        result.legend || []
    );
}


/**
 * Legende unter dem Bild. Farben kommen aus dem Backend
 * (SEGMENTATION_CLASSES), damit Bild und Legende übereinstimmen.
 */
function renderLegend(entries) {

    const list =
        $("segLegend");


    list.replaceChildren();


    entries.forEach(entry => {

        // nur echte Hex-Farben in style übernehmen
        const color =
            /^#[0-9a-f]{6}$/i.test(entry.color)
                ? entry.color
                : "#888888";


        const item =
            document.createElement("li");


        const swatch =
            document.createElement("span");

        swatch.className =
            "seg-swatch";

        swatch.style.background =
            color;


        const text =
            document.createElement("div");


        const name =
            document.createElement("strong");

        name.textContent =
            entry.name;


        const description =
            document.createElement("small");

        description.textContent =
            entry.description;


        text.append(
            name,
            description
        );


        const volume =
            document.createElement("span");

        volume.className =
            "seg-volume";

        volume.textContent =
            `${Number(entry.volume_ml).toFixed(1)} ml`;


        item.append(
            swatch,
            text,
            volume
        );


        list.append(item);

    });


    list.hidden =
        entries.length === 0;
}


/* ============================================================
   ANALYSIS
============================================================ */

async function startAnalysis() {

    if (
        !state.selectedFiles.length ||
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
        $("resultContent"),
        selectionLabel(state.selectedFiles)
    );


    try {

        const result =
            await analyzeMRI(
                state.selectedFiles,
                state.patient?.id
            );


        // Anzeige: Patientenname statt Dateiname
        if (state.patient) {

            result.patient_id =
                `${state.patient.name} · ${state.patient.id}`;

        }


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

            handleFiles(
                event.target.files
            );


            // gleiche Auswahl erneut wählbar machen
            event.target.value =
                "";

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

        handleFiles(
            event.dataTransfer.files
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
   VIEWS: PATIENTS <-> MRI ANALYSIS
============================================================ */

function showView(view) {

    const patientsActive =
        view === "patients";


    $("patientsView").hidden =
        !patientsActive;

    $("analysisView").hidden =
        patientsActive;


    $("navPatients").classList.toggle(
        "active",
        patientsActive
    );

    $("navAnalysis").classList.toggle(
        "active",
        !patientsActive
    );


    const crumbs =
        patientsActive
            ? ["Workspace", "Patients"]
            : state.patient
                ? ["Patients", state.patient.name, "MRI Analysis"]
                : ["Workspace", "MRI Analysis"];


    $("breadcrumb").replaceChildren(
        ...crumbs.flatMap((crumb, index) => {

            const text =
                document.createTextNode(crumb);

            if (index === 0) {
                return [text];
            }

            const separator =
                document.createElement("span");

            separator.textContent =
                "/";

            return [separator, text];

        })
    );


    window.scrollTo(0, 0);
}


/**
 * Alles zur vorherigen Analyse zurücksetzen (neuer Patient).
 */
function resetAnalysis() {

    releasePreviewURL(
        state.previewURL
    );


    state.previewURL =
        null;

    state.selectedFiles =
        [];

    state.analysis.result =
        null;


    setError("");


    renderEmptyResult(
        $("resultContent")
    );


    $("resultTag").textContent =
        "Ready";

    $("step2").classList.remove("active");

    $("step3").classList.remove("active");


    $("scanMeta").textContent =
        "No file selected";

    $("imageTag").textContent =
        "No scan";

    $("viewportLabel").textContent =
        "VIEWPORT 01";


    $("scanImage").style.display =
        "none";

    $("scanImage").removeAttribute(
        "src"
    );

    $("dicomMessage").style.display =
        "none";

    $("scanEmpty").style.display =
        "";

    $("segLegend").hidden =
        true;


    resetViewerTransform();

    setViewerButtonsEnabled(false);


    $("analyzeBtn").disabled =
        true;
}


function openAnalysis(patient) {

    // laufende Analyse würde sonst beim falschen Patienten landen
    if (state.analysis.loading) {

        setError(
            "Please wait until the current analysis has finished."
        );

        showView("analysis");

        return;
    }


    if (patient !== state.patient) {

        state.patient =
            patient;

        resetAnalysis();

    }


    renderPatientContext(
        $("patientContext"),
        state.patient,
        () => showView("patients")
    );


    showView("analysis");
}


async function selectPatient(id) {

    const patient =
        await getPatient(id);


    if (patient) {
        openAnalysis(patient);
    }
}


function filterPatients(query) {

    const needle =
        query.trim().toLowerCase();


    const matches =
        state.patients.filter(patient =>
            [
                patient.name,
                patient.id,
                patient.chief_complaint
            ]
                .join(" ")
                .toLowerCase()
                .includes(needle)
        );


    renderPatientList(
        $("patientList"),
        matches,
        selectPatient
    );
}


$("navPatients")
    .addEventListener(
        "click",
        () => showView("patients")
    );


$("navAnalysis")
    .addEventListener(
        "click",
        () => openAnalysis(state.patient)
    );


$("noPatientBtn")
    .addEventListener(
        "click",
        () => openAnalysis(null)
    );


$("patientSearch")
    .addEventListener(
        "input",
        event => filterPatients(event.target.value)
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


renderPatientContext(
    $("patientContext"),
    null,
    () => showView("patients")
);


state.patients =
    await listPatients();


filterPatients("");


showView("patients");