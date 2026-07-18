// static_app.js - UI controller for the fully client-side Vintage Cam website.

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('imageInput');
    const processBtn = document.getElementById('processBtn');
    const uploadForm = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const originalImg = document.getElementById('originalImg');
    const processedImg = document.getElementById('processedImg');
    const errorMsg = document.getElementById('errorMsg');
    const fileNameDisplay = document.getElementById('file-name-display');
    const filterOptions = document.querySelectorAll('.filter-option');
    const languageToggle = document.getElementById('languageToggle');
    const intensitySlider = document.getElementById('intensitySlider');
    const intensityValue = document.getElementById('intensityValue');
    const blendBaseImg = document.getElementById('blendBaseImg');
    const downloadBtn = document.getElementById('downloadBtn');

    let uploadedFile = null;
    let originalImageObject = new Image();

    // Slider live update
    function updateSlider() {
        const val = intensitySlider.value;
        intensityValue.textContent = val + '%';
        intensitySlider.style.setProperty('--val', val + '%');
        
        // Live Preview: Opacity of overlay filter
        if (processedImg) {
            processedImg.style.opacity = val / 100;
        }
    }
    intensitySlider.addEventListener('input', updateSlider);
    updateSlider(); // init

    // Bilingual Support
    const translations = {
        de: {
            title: "Vintage Cam by Enes",
            subtitle: "Verleihe deinen digitalen Bildern den authentischen Look analogen Films.",
            filter1: "Vintage Warm",
            filter2: "Moody Dark",
            filter3: "Fuji Chrome",
            filter4: "Cinestill Night",
            filter5: "Party Flash",
            presetLabel: "Film Simulation Recipes",
            portra400: "Kodak Portra 400",
            cinestill800t: "Cinestill 800T",
            hp5: "Ilford HP5",
            golddust: "Gold Dust",
            fujiLabel: "Fuji Originals",
            nostalgic_neg: "Nostalgic Neg",
            eterna_bb: "Eterna Bypass",
            classic_neg: "Classic Neg",
            intensityLabel: "Filter Intensität",
            selectBtn: "Bild auswählen",
            processBtn: "Film Entwickeln",
            loading: "Entwickle Filmstreifen...",
            exporting: "Exportiere in voller Auflösung...",
            original: "Original",
            processed: "Entwickelt (Analog-Look)",
            errorMissing: "Fehler bei der Bildverarbeitung.",
            downloadBtn: "Bild herunterladen"
        },
        en: {
            title: "Vintage Cam by Enes",
            subtitle: "Give your digital photos the authentic look of analog film.",
            filter1: "Vintage Warm",
            filter2: "Moody Dark",
            filter3: "Fuji Chrome",
            filter4: "Cinestill Night",
            filter5: "Party Flash",
            presetLabel: "Film Simulation Recipes",
            portra400: "Kodak Portra 400",
            cinestill800t: "Cinestill 800T",
            hp5: "Ilford HP5",
            golddust: "Gold Dust",
            fujiLabel: "Fuji Originals",
            nostalgic_neg: "Nostalgic Neg",
            eterna_bb: "Eterna Bypass",
            classic_neg: "Classic Neg",
            intensityLabel: "Filter Intensity",
            selectBtn: "Select Image",
            processBtn: "Develop Film",
            loading: "Developing film strip...",
            exporting: "Exporting in full resolution...",
            original: "Original",
            processed: "Processed (Analog Look)",
            errorMissing: "Error processing image.",
            downloadBtn: "Download Image"
        }
    };

    let currentLang = 'de';

    languageToggle.addEventListener('change', (e) => {
        currentLang = e.target.value;
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (translations[currentLang][key]) {
                if (el.tagName === 'INPUT' && el.type === 'button') {
                    el.value = translations[currentLang][key];
                } else {
                    el.textContent = translations[currentLang][key];
                }
            }
        });
    });

    // UI Feedback for selected filter
    filterOptions.forEach(option => {
        option.addEventListener('click', () => {
            filterOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
        });
    });

    // Handle File Selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadedFile = fileInput.files[0];
            processBtn.disabled = false;
            fileNameDisplay.textContent = uploadedFile.name;
            fileNameDisplay.style.display = 'block';
            
            // Create object URL to load image in background
            const objectUrl = URL.createObjectURL(uploadedFile);
            originalImageObject.src = objectUrl;
        } else {
            processBtn.disabled = true;
            fileNameDisplay.style.display = 'none';
            uploadedFile = null;
        }
    });

    // Form submission (Client-side Canvas Image Processing)
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!uploadedFile || !originalImageObject.src) return;

        const selectedFilter = document.querySelector('input[name="filter"]:checked').value;

        // UI state: loading
        processBtn.disabled = true;
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorMsg.classList.add('hidden');

        // Let the DOM render the spinner before starting the heavy canvas loop
        setTimeout(() => {
            try {
                // To keep the UI snappy, we do our screen preview processing on a downscaled version.
                // If the source image is larger than 1400px, downscale it for the screen view.
                const maxPreviewDim = 1400;
                let previewW = originalImageObject.naturalWidth;
                let previewH = originalImageObject.naturalHeight;
                
                if (previewW > maxPreviewDim || previewH > maxPreviewDim) {
                    if (previewW > previewH) {
                        previewH = Math.round((previewH * maxPreviewDim) / previewW);
                        previewW = maxPreviewDim;
                    } else {
                        previewW = Math.round((previewW * maxPreviewDim) / previewH);
                        previewH = maxPreviewDim;
                    }
                }
                
                // Create a temporary downscaled canvas
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = previewW;
                tempCanvas.height = previewH;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(originalImageObject, 0, 0, previewW, previewH);
                
                // Process the downscaled image
                const processedCanvas = document.createElement('canvas');
                processImageClientSide(tempCanvas, processedCanvas, selectedFilter);
                
                // Display the results
                const originalUrl = originalImageObject.src;
                originalImg.src = originalUrl;
                blendBaseImg.src = originalUrl;
                
                // Convert processed canvas to dataURL for display
                processedImg.src = processedCanvas.toDataURL('image/jpeg', 0.9);
                processedImg.style.opacity = intensitySlider.value / 100;
                
                // Show results UI
                resultsSection.classList.remove('hidden');
                
                // Hook download click
                downloadBtn.onclick = (event) => {
                    event.preventDefault();
                    downloadFullResBlended(selectedFilter);
                };
                
            } catch (err) {
                console.error(err);
                errorMsg.textContent = err.message || translations[currentLang].errorMissing;
                errorMsg.classList.remove('hidden');
            } finally {
                loading.classList.add('hidden');
                processBtn.disabled = false;
            }
        }, 80);
    });

    // Helper: full resolution blend and download
    function downloadFullResBlended(filterName) {
        // Show progress spinner for export
        const loadingText = loading.querySelector('p');
        const origText = loadingText.textContent;
        loadingText.textContent = translations[currentLang].exporting;
        loading.classList.remove('hidden');
        
        setTimeout(() => {
            try {
                // 1. Process original image at FULL resolution in the background
                const fullFilteredCanvas = document.createElement('canvas');
                processImageClientSide(originalImageObject, fullFilteredCanvas, filterName);
                
                // 2. Blend the original full-res and the filtered full-res on a final canvas
                const finalCanvas = document.createElement('canvas');
                finalCanvas.width = originalImageObject.naturalWidth;
                finalCanvas.height = originalImageObject.naturalHeight;
                const fCtx = finalCanvas.getContext('2d');
                
                // Draw full resolution original
                fCtx.drawImage(originalImageObject, 0, 0);
                
                // Blend full resolution filtered with selected opacity
                const opacity = intensitySlider.value / 100;
                fCtx.globalAlpha = opacity;
                fCtx.drawImage(fullFilteredCanvas, 0, 0);
                
                // 3. Trigger download
                const link = document.createElement('a');
                link.download = `vintage_${filterName}_intensity_${intensitySlider.value}.jpg`;
                link.href = finalCanvas.toDataURL('image/jpeg', 0.95);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
            } catch (err) {
                console.error(err);
                alert("Export error: " + err.message);
            } finally {
                // Hide spinner
                loading.classList.add('hidden');
                loadingText.textContent = origText;
            }
        }, 100);
    }
});
