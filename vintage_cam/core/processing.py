import cv2
import numpy as np
import rawpy
from core.film_engine import FilmEngine, PRESETS
from core.fuji_engine import FujiEngine, FUJI_PRESETS
from core.party_flash import party_flash_engine

def apply_color_matrix(img_rgb, matrix):
    """Wendet eine 3x3 Farbkorrekturmatrix an."""
    matrix = np.array(matrix, dtype=np.float32)
    return cv2.transform(img_rgb, matrix)

def apply_grain(l_channel, intensity=0.04):
    """Generiert physikalisch motiviertes Filmkorn basierend auf dem L*-Kanal (Lab)."""
    gauss = np.random.normal(0, intensity * 255, l_channel.shape)
    # Poisson-ähnliche Komponente (proportional zur Wurzel des Signals)
    poisson_std = np.sqrt(l_channel.astype(np.float32) + 1.0) * (intensity * 3.0)
    poisson = np.random.normal(0, poisson_std)
    
    noisy_l = l_channel.astype(np.float32) + gauss + poisson
    return np.clip(noisy_l, 0, 255).astype(np.uint8)

def apply_adaptive_grain(l_channel, intensity=0.06):
    """Fügt Korn hinzu, das in den Mitteltönen am stärksten ist und in den Schatten/Lichtern abnimmt."""
    l_float = l_channel.astype(np.float32)
    # Parabolische Maske: 1.0 bei L=128, 0.0 bei L=0 und L=255
    # Mask = 1 - ((L - 128) / 128)^2
    mask = 1.0 - np.square((l_float - 128.0) / 128.0)
    
    # Rauschen generieren
    noise = np.random.normal(0, intensity * 255, l_channel.shape).astype(np.float32)
    
    # Rauschen mit der Maske modulieren
    adaptive_noise = noise * mask
    
    noisy_l = l_float + adaptive_noise
    return np.clip(noisy_l, 0, 255).astype(np.uint8)

def apply_s_curve_lut(img_rgb):
    """S-Kurve mit stark angehobenen Schwarztönen (Milky Blacks) und weichem Highlight-Roll-Off."""
    x = np.arange(256, dtype=np.float32) / 255.0
    
    # Sigmoid function für Kontrast
    # y = 1 / (1 + exp(-k * (x - x0)))
    k = 4.0   # Steepness (Kontrast)
    x0 = 0.5  # Midpoint
    
    s_curve = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    
    # Normalisieren der S-Kurve zurück auf 0..1
    s_min = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    s_max = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    s_curve = (s_curve - s_min) / (s_max - s_min)
    
    # Mapping auf neues Range [black_point, white_point]
    # Faded Blacks (ca. 25-30) -> Milchiges Dunkelgrau
    black_point = 28.0 / 255.0
    # Soft Highlight Roll-Off (ca. 240) -> Kein hartes Weiß
    white_point = 240.0 / 255.0
    
    lut = black_point + s_curve * (white_point - black_point)
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    
    return cv2.LUT(img_rgb, lut)


def apply_steep_s_curve(img_rgb):
    """Moderates S-Kurve für filmischen Kontrast mit leicht angehobenen Schwarztönen."""
    x = np.arange(256, dtype=np.float32) / 255.0
    
    # Moderateres Sigmoid (k=5) für weniger aggressiven Kontrast
    k = 5.0
    x0 = 0.5
    s_curve = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    
    # Normalisieren
    s_min = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    s_max = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    s_curve = (s_curve - s_min) / (s_max - s_min)
    
    # Leicht angehobener Schwarzpunkt (weiche Schatten statt Crushed Blacks)
    black_point = 12.0 / 255.0
    white_point = 245.0 / 255.0
    
    lut = black_point + s_curve * (white_point - black_point)
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    
    return cv2.LUT(img_rgb, lut)


def apply_teal_orange_grading(img_rgb):
    """Teal & Orange Color Grading im HSV-Raum.
    - Blautöne werden in Richtung Cyan (Teal) verschoben
    - Rot-/Orangetöne erhalten hohe Sättigung und Luminanz
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = cv2.split(img_hsv)

    # --- Teal: Blautöne (H ~100-130 in OpenCV 0-179) in Richtung Cyan schieben ---
    # Cyan liegt bei ~90, Blau bei ~120
    blue_mask = np.exp(-0.5 * np.square((H - 115.0) / 18.0))
    H = H - blue_mask * 12.0  # Hue Richtung Cyan (tiefer)
    # Sättigung in Teal-Bereich leicht erhöhen
    S = S + blue_mask * 20.0
    # Luminanz in Teal-Schatten leicht senken für filmischen Look
    V = V - blue_mask * 10.0

    # --- Orange: Warm-Töne (H ~5-20 in OpenCV) boosten ---
    # Orange zirkulär erfassen (nahe 0/179)
    h_shifted = np.mod(H + 90.0, 180.0) - 90.0
    orange_mask = np.exp(-0.5 * np.square((h_shifted - 0.0) / 15.0))
    # Starke Sättigungserhöhung für Orange/Rot
    S = S + orange_mask * 45.0
    # Luminanz leicht anheben für leuchtende Hauttöne & Lichter
    V = V + orange_mask * 15.0

    H = np.clip(H, 0, 179)
    S = np.clip(S, 0, 255)
    V = np.clip(V, 0, 255)

    img_hsv_out = cv2.merge((H, S, V)).astype(np.uint8)
    return cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)


def apply_halation(img_rgb, blur_radius=35, intensity=0.45):
    """Halation-Effekt: Lichthof um helle, warme Lichtquellen (Rot/Orange).
    - Isoliert helle warme Töne (hohe Rot-, niedrige Blau-Werte)
    - Wendet einen starken Gaußschen Weichzeichner an
    - Addiert diesen Glow-Layer über das Originalbild
    """
    img_float = img_rgb.astype(np.float32) / 255.0

    # Maske: Pixel mit hohem Rot, niedrigem Blau und hoher Gesamthelligkeit
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Warme, helle Pixel isolieren
    warm_mask = np.clip((r - b) * 2.5, 0.0, 1.0)   # Rot deutlich stärker als Blau
    bright_mask = np.clip((lum - 0.55) / 0.3, 0.0, 1.0)  # Nur helle Bereiche
    glow_mask = (warm_mask * bright_mask)[:, :, np.newaxis]

    # Nur die Rotkanal-dominierten hellen Pixel isolieren
    glow_source = img_float * glow_mask

    # Farbe des Glow: warmes Orange-Rot
    glow_colored = np.zeros_like(img_float)
    glow_colored[:, :, 0] = glow_source[:, :, 0]         # Rot behalten
    glow_colored[:, :, 1] = glow_source[:, :, 1] * 0.4   # Grün stark dämpfen
    glow_colored[:, :, 2] = glow_source[:, :, 2] * 0.1   # Blau fast entfernen

    # Gaußscher Weichzeichner (Kernelgröße muss ungerade sein)
    ksize = blur_radius * 2 + 1
    glow_blurred = cv2.GaussianBlur(glow_colored, (ksize, ksize), sigmaX=blur_radius * 0.5)

    # Glow über Originalbild addieren (Screen-ähnlich)
    result = img_float + glow_blurred * intensity
    return np.clip(result, 0.0, 1.0)


def apply_fine_luminance_grain(l_channel, intensity=0.025):
    """Sehr feines, scharfes Luminanz-Filmkorn das Details nicht zerstört.
    Rein gaussisches Rauschen mit niedriger Intensität, nur im L*-Kanal.
    """
    noise = np.random.normal(0, intensity * 255, l_channel.shape).astype(np.float32)
    noisy_l = l_channel.astype(np.float32) + noise
    return np.clip(noisy_l, 0, 255).astype(np.uint8)

def apply_color_adjustments(img_rgb):
    """Selektive Farbanpassung: Kühle Farben entsättigen, warme Töne sanft betonen."""
    # Konvertierung nach HSV (H: 0-179, S: 0-255, V: 0-255)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = cv2.split(img_hsv)
    
    # --- Entsättigung von kühlen Farben (Blau/Grün) ---
    # Blau/Grün liegt ungefähr zwischen H=40 und H=140
    # Wir erstellen eine weiche Maske um diesen Bereich
    cool_mask = np.exp(-0.5 * np.square((H - 90.0) / 40.0))
    # Reduziere Sättigung in diesem Bereich um bis zu 40%
    S = S * (1.0 - (cool_mask * 0.4))
    
    # --- Betonen von warmen Tönen (Orange/Rot/Gelb) ---
    # Warme Töne liegen um H=0/179 (Rot) bis H=30 (Gelb)
    # Da Hue zirkulär ist, zentrieren wir um 0:
    # H_shifted = (H + 90) % 180 - 90 bringt Rot in die Mitte
    h_shifted = np.mod(H + 90.0, 180.0) - 90.0
    warm_mask = np.exp(-0.5 * np.square(h_shifted / 20.0))
    # Erhöhe Sättigung sanft um bis zu 15%
    S = S * (1.0 + (warm_mask * 0.15))
    
    # --- Color Chrome (Luminanz abwärts bei hoher Sättigung) ---
    # Optionaler Chrome-Effekt für zusätzliche Tiefe
    saturation_mask = S / 255.0
    v_reduction = 1.0 - (saturation_mask * 0.15)
    V = V * v_reduction
    
    img_hsv_adjusted = cv2.merge((H, S, V))
    img_hsv_adjusted = np.clip(img_hsv_adjusted, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(img_hsv_adjusted, cv2.COLOR_HSV2RGB)

def apply_split_toning(img_rgb):
    """Schatten warm/grünlich (Vintage), Lichter weich warm."""
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Luminanz berechnen
    lum = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
    
    # Weiche Masken für Schatten und Lichter
    shadow_mask = np.clip(1.0 - (lum / 0.5), 0.0, 1.0) ** 1.5
    highlight_mask = np.clip((lum - 0.5) / 0.5, 0.0, 1.0) ** 1.5
    
    shadow_mask = np.dstack([shadow_mask]*3)
    highlight_mask = np.dstack([highlight_mask]*3)
    
    # RGB Offsets
    # Schatten: Warm/Grünlich-Braun (+Rot, +Grün, -Blau)
    shadow_tint = np.array([0.05, 0.04, -0.03], dtype=np.float32)
    
    # Lichter: Sanft warm (+Rot, leicht +Grün, -Blau)
    highlight_tint = np.array([0.04, 0.02, -0.04], dtype=np.float32)
    
    img_tinted = img_float + (shadow_mask * shadow_tint) + (highlight_mask * highlight_tint)
    return np.clip(img_tinted, 0.0, 1.0) * 255.0

def apply_vignette(img, intensity=0.5):
    """Fügt eine dunkle Vignette an den Rändern hinzu."""
    rows, cols = img.shape[:2]
    # Generiere Gaussian Kernels für X und Y
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    # Erzeuge die Vignette-Maske durch äußeres Produkt
    kernel = kernel_y * kernel_x.T
    # Normalisiere die Maske so, dass das Zentrum 1.0 ist
    mask = kernel / kernel.max()
    
    # Blende die Maske mit der gewünschten Intensität ein
    # intensity=0.0: keine Vignette, intensity=1.0: sehr starke Vignette
    mask = 1.0 - intensity * (1.0 - mask)
    
    # Maske auf 3 Kanäle erweitern, falls es ein Farbbild ist
    if len(img.shape) == 3:
         mask = np.dstack([mask] * 3)
         
    # Vignette anwenden
    vignetted_img = img.astype(np.float32) * mask
    return np.clip(vignetted_img, 0, 255).astype(np.uint8)

def apply_dehaze_approximation(img_rgb, amount=0.08):
    """Approximaive Dehaze-Funktion durch Kontrasterhöhung und leichtes Abdunkeln der Schatten."""
    # amount=0.08 entspricht etwa 8% Dehaze
    # Wir machen das Bild etwas kontrastreicher und ziehen die Schwarztöne leicht nach unten.
    
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Lineare Kontraststreckung (S-Kurve-artig, sehr leicht)
    # Wir zentrieren um 0.5 und multiplizieren mit (1 + amount)
    img_dehazed = (img_float - 0.5) * (1.0 + amount) + 0.5
    
    # Schatten leicht absenken (Schwarzpunkt verschieben)
    # Wir ziehen einen kleinen Wert ab, der proportional zur Menge ist
    black_point = 0.05 * amount
    img_dehazed = img_dehazed - black_point
    
    return np.clip(img_dehazed, 0.0, 1.0) * 255.0

def _apply_intensity_blend(orig_bgr: np.ndarray, filtered_bgr: np.ndarray,
                            intensity: float) -> np.ndarray:
    """Blendet Original und gefilteres Bild. intensity=1.0 → voll gefiltert."""
    if intensity >= 1.0:
        return filtered_bgr
    t = float(np.clip(intensity, 0.0, 1.0))
    blended = cv2.addWeighted(orig_bgr, 1.0 - t, filtered_bgr, t, 0)
    return blended


def process_vintage_film(image_path, output_path, filter_type='filter1', intensity=1.0):
    """Komplette Image Processing Pipeline abhängig vom gewählten Filter."""
    
    # 1. Bild Laden (RAW-Support oder Fallback JPEG/PNG)
    img_rgb = None
    is_raw = any(image_path.lower().endswith(ext) for ext in ['.dng', '.cr2', '.nef', '.arw'])
    
    if is_raw:
        try:
            with rawpy.imread(image_path) as raw:
                # Postprocessing des RAWs in RGB sRGB (8-bit)
                img_rgb = raw.postprocess()
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der RAW-Datei: {str(e)}")
    else:
        # Standard OpenCV Lese-Methode für JPEGs
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- PERFORMANZ OPTIMIERUNG ---
    # Python/NumPy braucht für Bilder mit 12-40 Megapixeln (z.B. iPhone) extrem lange für 
    # komplexe Matrix-Gleichungen. Wir skalieren das Bild auf Web-Größe (max. 1920px).
    h, w = img_rgb.shape[:2]
    max_dim = 1920.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Original in BGR speichern für späteres Intensity-Blending
    img_bgr_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # --- FILTER SPEZIFISCHE VERARBEITUNG ---
    if filter_type == 'filter1':
        # Filter 1: Bild Wärmer machen (ca. 15% mehr Rot, weniger Blau)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        warm_matrix = [
            [1.15, 0.0, 0.0],  # Red +15%
            [0.0, 1.05, 0.0],  # Green +5%
            [0.0, 0.0, 0.85]   # Blue -15%
        ]
        
        img_color = apply_color_matrix(img_float, warm_matrix)
        img_color = np.clip(img_color, 0.0, 1.0)
        img_processed_8u = (img_color * 255.0).astype(np.uint8)
        
    elif filter_type == 'filter2':
        # Filter 2: Originalwärme (nichts an Farben ändern), dunkle Vignette, ca 15% Dehaze
        img_float = img_rgb.astype(np.float32)
        
        # Dehaze anwenden (ca 15%)
        img_dehazed = apply_dehaze_approximation(img_float, amount=0.15)
        img_processed_8u = img_dehazed.astype(np.uint8)
        
        # Dunkle Vignette anwenden
        img_processed_8u = apply_vignette(img_processed_8u, intensity=0.6)
        
    elif filter_type == 'filter3':
        # Filter 3: Nostalgischer Fujifilm Vintage Look
        
        # 1. Selektive Farbanpassung (Kühle Töne entsättigen, warme Töne betonen)
        img_color_adjusted = apply_color_adjustments(img_rgb)
        
        # 2. Split Toning (Schatten warm/grünlich, Lichter warm)
        img_tinted = apply_split_toning(img_color_adjusted)
        
        # 3. S-Gradationskurve mit Faded Blacks und weichem Highlight-Roll-Off
        img_curve = apply_s_curve_lut(img_tinted.astype(np.uint8))
        
        img_processed_8u = img_curve

    elif filter_type == 'filter4':
        # Filter 4: Cinestill 800T – Filmischer Nacht-Look

        # 1. Steile S-Gradationskurve: hoher Kontrast, tief gedrückte Schwarztöne
        img_curve = apply_steep_s_curve(img_rgb)

        # 2. Teal & Orange Color Grading im HSV-Raum
        img_graded = apply_teal_orange_grading(img_curve)

        # 3. Halation-Effekt: Lichthof um helle, warme Lichtquellen
        img_halo_float = apply_halation(img_graded, blur_radius=35, intensity=0.45)
        img_processed_8u = np.clip(img_halo_float * 255.0, 0, 255).astype(np.uint8)

    elif filter_type == 'filter5':
        # Filter 5: Direct Flash / Party-Blitz
        # Wendet Smart/Radial Masking, Vignette, Kontrast und hartes Digitalrauschen an
        img_processed_8u = party_flash_engine.process(
            img_rgb,
            force_radial=False,
            # Hier könnten auch Parameter aus dem Frontend via processing.py übergeben werden:
            fg_exposure=1.4,
            bg_darken=0.2,
            grain_intensity=0.08
        )

    elif filter_type in PRESETS:
        # --- Film Simulation Engine (Preset-basiert) ---
        preset = PRESETS[filter_type]
        img_final_rgb = FilmEngine.process(img_rgb, preset)
        img_final_bgr = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)
        img_final_bgr = _apply_intensity_blend(img_bgr_orig, img_final_bgr, intensity)
        cv2.imwrite(output_path, img_final_bgr)
        return

    elif filter_type in FUJI_PRESETS:
        # --- Fuji Film Simulation Engine ---
        preset = FUJI_PRESETS[filter_type]
        img_final_rgb = FujiEngine.process(img_rgb, preset)
        img_final_bgr = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)
        img_final_bgr = _apply_intensity_blend(img_bgr_orig, img_final_bgr, intensity)
        cv2.imwrite(output_path, img_final_bgr)
        return

    else:
        # Fallback: unbekannter Filter
        img_processed_8u = img_rgb


    # Filmkorn im L*-Farbraum applizieren (für legacy filter1–4)
    img_lab = cv2.cvtColor(img_processed_8u, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)

    if filter_type == 'filter3':
        L_grained = apply_adaptive_grain(L, intensity=0.06)
    elif filter_type == 'filter4':
        L_grained = apply_fine_luminance_grain(L, intensity=0.025)
    elif filter_type == 'filter5':
        # Filter 5 wendet das Grain bereits drastischer mit Color-Noise im RGB Raum (PartyFlashEngine) an.
        # Daher hier kein L*-Grain mehr drüberlegen.
        L_grained = L
    else:
        L_grained = apply_grain(L, intensity=0.04)

    img_lab_grained = cv2.merge((L_grained, a, b))

    img_final_rgb = cv2.cvtColor(img_lab_grained, cv2.COLOR_LAB2RGB)
    img_final_bgr = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)
    img_final_bgr = _apply_intensity_blend(img_bgr_orig, img_final_bgr, intensity)
    cv2.imwrite(output_path, img_final_bgr)
