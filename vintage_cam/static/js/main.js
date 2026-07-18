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

    // Slider live update
    function updateSlider() {
        const val = intensitySlider.value;
        intensityValue.textContent = val + '%';
        intensitySlider.style.setProperty('--val', val + '%');
        
        // Live Preview: Opacity des Filters anpassen
        if (processedImg) {
            processedImg.style.opacity = val / 100;
        }
    }
    intensitySlider.addEventListener('input', updateSlider);
    updateSlider(); // init

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

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            processBtn.disabled = false;
            fileNameDisplay.textContent = fileInput.files[0].name;
            fileNameDisplay.style.display = 'block';
        } else {
            processBtn.disabled = true;
            fileNameDisplay.style.display = 'none';
        }
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (fileInput.files.length === 0) return;

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('intensity', intensitySlider.value);

        const selectedFilter = document.querySelector('input[name="filter"]:checked').value;
        formData.append('filter', selectedFilter);

        processBtn.disabled = true;
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorMsg.classList.add('hidden');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                // Check if server sent a specific error, otherwise use general translation
                throw new Error(data.error || translations[currentLang].errorMissing);
            }

            originalImg.src = data.original_url;
            blendBaseImg.src = data.original_url;
            processedImg.src = data.processed_url;

            // Reset live opacity based on slider
            processedImg.style.opacity = intensitySlider.value / 100;

            // Set download link - override default behavior for live blending
            const filterName = document.querySelector('input[name="filter"]:checked').value;
            downloadBtn.onclick = (e) => {
                e.preventDefault();
                downloadBlendedImage(filterName);
            };

            resultsSection.classList.remove('hidden');

        } catch (error) {
            errorMsg.textContent = error.message;
            errorMsg.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
            processBtn.disabled = false;
        }
    });

    /**
     * Blendet Original und gefiltertes Bild auf einem Canvas zusammen
     * und startet den Download des resultierenden Bildes.
     */
    function downloadBlendedImage(filterName) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Verwende die natürliche Größe der Bilder
        canvas.width = blendBaseImg.naturalWidth;
        canvas.height = blendBaseImg.naturalHeight;
        
        // 1. Original Zeichnen
        ctx.drawImage(blendBaseImg, 0, 0);
        
        // 2. Filter-Overlay mit aktueller Opacity zeichnen
        const alpha = intensitySlider.value / 100;
        ctx.globalAlpha = alpha;
        ctx.drawImage(processedImg, 0, 0);
        
        // 3. Download triggern
        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
        const link = document.createElement('a');
        link.download = `vintage_${filterName}_intensity_${intensitySlider.value}.jpg`;
        link.href = dataUrl;
        link.click();
    }
});
