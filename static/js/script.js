/* ==============================================================================
   LEAF DISEASE AI - FRONTEND LOGIC
   Handles: drag & drop upload, image preview, AJAX prediction requests,
   animated rendering of results, and navigation interactions.
   ============================================================================== */

document.addEventListener("DOMContentLoaded", () => {

    /* ---------------------------------------------------------------------
       DOM ELEMENT REFERENCES
    --------------------------------------------------------------------- */
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropZoneContent = document.getElementById("dropZoneContent");
    const imagePreviewWrapper = document.getElementById("imagePreviewWrapper");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImageBtn");

    const cameraBtn = document.getElementById("cameraBtn");
    const cameraModalOverlay = document.getElementById("cameraModalOverlay");
    const cameraCloseBtn = document.getElementById("cameraCloseBtn");
    const cameraVideo = document.getElementById("cameraVideo");
    const cameraCanvas = document.getElementById("cameraCanvas");
    const cameraErrorMsg = document.getElementById("cameraErrorMsg");
    const captureShutterBtn = document.getElementById("captureShutterBtn");

    let activeCameraStream = null;

    const predictBtn = document.getElementById("predictBtn");
    const resetBtn = document.getElementById("resetBtn");
    const flashMessages = document.getElementById("flashMessages");

    const uploadCard = document.getElementById("uploadCard");
    const analyzingCard = document.getElementById("analyzingCard");
    const resultsSection = document.getElementById("resultsSection");

    const resultImage = document.getElementById("resultImage");
    const statusBadge = document.getElementById("statusBadge");
    const uploadAnotherBtn = document.getElementById("uploadAnotherBtn");
    const predictedDisease = document.getElementById("predictedDisease");
    const predictedCategory = document.getElementById("predictedCategory");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");
    const top3List = document.getElementById("top3List");

    const infoDescription = document.getElementById("infoDescription");
    const infoCauses = document.getElementById("infoCauses");
    const infoSymptoms = document.getElementById("infoSymptoms");
    const infoTreatment = document.getElementById("infoTreatment");
    const infoPrevention = document.getElementById("infoPrevention");

    const similarImagesGrid = document.getElementById("similarImagesGrid");

    const navToggle = document.getElementById("navToggle");
    const navLinks = document.getElementById("navLinks");
    const navLinkItems = document.querySelectorAll(".nav-link");

    let selectedFile = null;

    /* ---------------------------------------------------------------------
       ALLOWED FILE TYPES (mirrors backend validation)
    --------------------------------------------------------------------- */
    const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];
    const MAX_FILE_SIZE_MB = 8;

    /* =======================================================================
       DRAG & DROP HANDLERS
    ======================================================================= */

    dropZone.addEventListener("click", (e) => {
        // Avoid double-triggering when the Browse button itself is clicked
        if (e.target === browseBtn || browseBtn.contains(e.target)) return;
        if (!imagePreviewWrapper.style.display || imagePreviewWrapper.style.display === "none") {
            fileInput.click();
        }
    });

    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSelectedImage();
    });

    /* =======================================================================
       CAMERA CAPTURE
       Lets the farmer take a photo directly with their device camera as an
       alternative to browsing or dragging a file from storage.
    ======================================================================= */

    cameraBtn.addEventListener("click", openCamera);
    cameraCloseBtn.addEventListener("click", closeCamera);
    captureShutterBtn.addEventListener("click", capturePhotoFromCamera);

    cameraModalOverlay.addEventListener("click", (e) => {
        if (e.target === cameraModalOverlay) {
            closeCamera();
        }
    });

    /**
     * Open the camera modal and request access to the device camera.
     * Prefers the rear ("environment") camera on mobile devices, which is
     * more practical for photographing leaves in the field.
     */
    async function openCamera() {
        clearFlashMessages();
        cameraModalOverlay.style.display = "flex";
        cameraErrorMsg.style.display = "none";
        cameraVideo.style.display = "block";

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            cameraErrorMsg.style.display = "flex";
            cameraVideo.style.display = "none";
            return;
        }

        try {
            activeCameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: "environment" } },
                audio: false,
            });
            cameraVideo.srcObject = activeCameraStream;
        } catch (error) {
            // Common causes: permission denied, no camera device, or
            // insecure (non-HTTPS/non-localhost) context.
            cameraErrorMsg.style.display = "flex";
            cameraVideo.style.display = "none";
        }
    }

    /**
     * Stop all active camera tracks and hide the camera modal.
     */
    function closeCamera() {
        if (activeCameraStream) {
            activeCameraStream.getTracks().forEach((track) => track.stop());
            activeCameraStream = null;
        }
        cameraVideo.srcObject = null;
        cameraModalOverlay.style.display = "none";
    }

    /**
     * Snapshot the current video frame onto the hidden canvas, convert it
     * to a JPEG Blob, and feed it into the same handleFileSelection()
     * pipeline used for drag-and-drop/browse uploads.
     */
    function capturePhotoFromCamera() {
        if (!activeCameraStream) return;

        const width = cameraVideo.videoWidth;
        const height = cameraVideo.videoHeight;

        cameraCanvas.width = width;
        cameraCanvas.height = height;

        const context = cameraCanvas.getContext("2d");
        context.drawImage(cameraVideo, 0, 0, width, height);

        cameraCanvas.toBlob((blob) => {
            if (!blob) {
                showFlashMessage("Could not capture photo. Please try again.");
                return;
            }

            const timestamp = new Date().getTime();
            const capturedFile = new File([blob], `capture_${timestamp}.jpg`, {
                type: "image/jpeg",
            });

            handleFileSelection(capturedFile);
            closeCamera();
        }, "image/jpeg", 0.92);
    }

    /* =======================================================================
       FILE SELECTION & VALIDATION
    ======================================================================= */

    /**
     * Validate and preview a newly selected image file.
     * @param {File} file - The file selected via drag-drop or file input.
     */
    function handleFileSelection(file) {
        clearFlashMessages();

        // Validate file type
        if (!ALLOWED_TYPES.includes(file.type)) {
            showFlashMessage("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
            return;
        }

        // Validate file size
        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > MAX_FILE_SIZE_MB) {
            showFlashMessage(`File is too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`);
            return;
        }

        selectedFile = file;

        // Render image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.style.display = "none";
            imagePreviewWrapper.style.display = "inline-block";
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    /**
     * Reset the upload area back to its empty state.
     */
    function clearSelectedImage() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        imagePreviewWrapper.style.display = "none";
        dropZoneContent.style.display = "block";
        predictBtn.disabled = true;
    }

    /* =======================================================================
       FLASH MESSAGES
    ======================================================================= */

    /**
     * Display an error/info message above the action buttons.
     * @param {string} message - The message text to display.
     */
    function showFlashMessage(message) {
        flashMessages.innerHTML = `
            <div class="flash-message">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <span>${message}</span>
            </div>
        `;
    }

    function clearFlashMessages() {
        flashMessages.innerHTML = "";
    }

    /* =======================================================================
       RESET BUTTON
    ======================================================================= */

    /**
     * Reset the interface back to the upload screen, clearing any
     * previously selected image and hiding results/loading states.
     * Used by both the "Reset" button and the "Upload Another Image"
     * button shown alongside prediction results.
     */
    function resetToUploadScreen() {
        clearSelectedImage();
        clearFlashMessages();
        resultsSection.style.display = "none";
        analyzingCard.style.display = "none";
        uploadCard.style.display = "block";
        uploadCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    resetBtn.addEventListener("click", resetToUploadScreen);

    uploadAnotherBtn.addEventListener("click", resetToUploadScreen);

    /* =======================================================================
       PREDICT BUTTON -> AJAX REQUEST TO FLASK BACKEND
    ======================================================================= */

    predictBtn.addEventListener("click", async () => {
        if (!selectedFile) {
            showFlashMessage("Please select an image before predicting.");
            return;
        }

        clearFlashMessages();

        // Show "Analyzing Leaf..." loading state
        uploadCard.style.display = "none";
        resultsSection.style.display = "none";
        analyzingCard.style.display = "block";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Prediction failed. Please try again.");
            }

            renderResults(data);

        } catch (error) {
            analyzingCard.style.display = "none";
            uploadCard.style.display = "block";
            showFlashMessage(error.message || "Something went wrong. Please try again.");
        }
    });

    /* =======================================================================
       RESULT RENDERING
    ======================================================================= */

    /**
     * Populate the results section with data returned from the /predict
     * endpoint and reveal it with a smooth transition.
     * @param {object} data - JSON payload returned by the Flask backend.
     */
    function renderResults(data) {
        // Hide loading state
        analyzingCard.style.display = "none";
        resultsSection.style.display = "block";

        // --- Uploaded image + health status badge ---------------------------
        resultImage.src = data.uploaded_image;

        const isHealthy = data.health_status.status === "Healthy";
        statusBadge.textContent = data.health_status.label;
        statusBadge.className = "status-badge " + (isHealthy ? "healthy" : "diseased");

        // --- Prediction summary ----------------------------------------------
        predictedDisease.textContent = data.predicted_class;
        predictedCategory.textContent = data.disease_info.category
            ? `Category: ${data.disease_info.category}`
            : "";

        // --- Animated confidence bar ------------------------------------------
        const confidence = data.confidence.toFixed(2);
        confidenceValue.textContent = `${confidence} %`;
        confidenceBar.style.width = "0%";
        requestAnimationFrame(() => {
            setTimeout(() => {
                confidenceBar.style.width = `${confidence}%`;
            }, 100);
        });

        // --- Top 3 predictions --------------------------------------------------
        top3List.innerHTML = "";
        data.top3_predictions.forEach((pred, index) => {
            const item = document.createElement("div");
            item.className = "top3-item";
            item.style.animationDelay = `${index * 0.1}s`;
            item.innerHTML = `
                <div class="top3-item-header">
                    <span class="top3-item-name">${pred.class}</span>
                    <span class="top3-item-value">${pred.confidence.toFixed(2)} %</span>
                </div>
                <div class="top3-bar-track">
                    <div class="top3-bar-fill" style="width:0%;"></div>
                </div>
            `;
            top3List.appendChild(item);

            // Animate each bar shortly after insertion
            const barFill = item.querySelector(".top3-bar-fill");
            setTimeout(() => {
                barFill.style.width = `${pred.confidence}%`;
            }, 150 + index * 100);
        });

        // --- Disease information card -------------------------------------------
        infoDescription.textContent = data.disease_info.description;
        infoCauses.textContent = data.disease_info.causes;
        infoSymptoms.textContent = data.disease_info.symptoms;

        // --- Treatment & prevention card -----------------------------------------
        infoTreatment.textContent = data.disease_info.treatment;
        infoPrevention.textContent = data.disease_info.prevention;

        // --- Similar images gallery ------------------------------------------------
        similarImagesGrid.innerHTML = "";
        if (data.similar_images && data.similar_images.length > 0) {
            data.similar_images.forEach((imgUrl, index) => {
                const item = document.createElement("div");
                item.className = "similar-image-item";
                item.style.animationDelay = `${index * 0.08}s`;
                item.innerHTML = `<img src="${imgUrl}" alt="Similar leaf sample ${index + 1}" loading="lazy">`;
                similarImagesGrid.appendChild(item);
            });
        } else {
            similarImagesGrid.innerHTML = `<p class="no-images-msg">No sample images available for this class.</p>`;
        }

        // Smoothly scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    /* =======================================================================
       NAVIGATION BAR: MOBILE TOGGLE
    ======================================================================= */

    navToggle.addEventListener("click", () => {
        navLinks.classList.toggle("open");
    });

    navLinkItems.forEach((link) => {
        link.addEventListener("click", () => {
            navLinks.classList.remove("open");
            navLinkItems.forEach((l) => l.classList.remove("active"));
            link.classList.add("active");
        });
    });

    /* =======================================================================
    NAVIGATION BAR: ACTIVE LINK ON SCROLL
    ======================================================================= */

    const sections = document.querySelectorAll("section[id]");

    window.addEventListener("scroll", () => {
        let currentSectionId = "";

        sections.forEach((section) => {
            const sectionTop = section.offsetTop - 120;
            if (window.scrollY >= sectionTop) {
                currentSectionId = section.getAttribute("id");
            }
        });

        navLinkItems.forEach((link) => {
            link.classList.remove("active");
            if (link.getAttribute("href") === `#${currentSectionId}`) {
                link.classList.add("active");
            }
        });
    });

});