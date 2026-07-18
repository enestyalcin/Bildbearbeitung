import cv2
import numpy as np

import os

try:
    import mediapipe as mp
    try:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
    except AttributeError:
        import mediapipe.python.solutions.selfie_segmentation as mp_selfie_segmentation
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class PartyFlashEngine:
    """
    Simuliert einen 'Direct Flash / Party-Blitz'-Effekt:
    - Harter kamerainterner Blitz auf der Person (stark belichtet, hohe Kontraste).
    - Drastisch abfallendes Licht im Hintergrund (Crushed Blacks).
    - Dynamische Erkennung via Mediapipe (Smart Flash) oder radialer Fallback (Radial Flash).
    - Zusätzliche globale Effekte: Kühles Split-Toning (Party-Vibe) & hartes Rauschen (High ISO).
    """

    def __init__(self):
        self.segmentation = None
        if HAS_MEDIAPIPE:
             # Selfie Segmentation: model_selection=0 (General) oder 1 (Landscape/Fast)
             try:
                 self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
             except Exception as e:
                 print(f"[PartyFlash] Fehler beim Laden des MP Modells: {e}")
                 self.segmentation = None

    def process(self, img_rgb, 
                force_radial=False,
                
                # Foreground Settings (Blitz-Wirkung auf Person)
                fg_exposure=1.05,        # Viel sanftere Belichtung (war 1.4, zu hell)
                fg_contrast=1.1,         # Weniger starker Kontrast
                fg_highlights=0.01,      # Leichte Anhebung der Lichter
                fg_warmth=0.1,           # Warmer Blitzton
                
                # Background Settings (Lichtabfall)
                bg_darken=0.01,           # Abdunklung des Hintergrunds
                bg_contrast=1.2,         # Leichter Kontrast für Tiefen
                
                # Mask Settings
                mask_blur=21,            # Enge, reale Weichzeichnung (war 61)
                mask_expansion=0.015,    # 1.5% Erweiterung für leichten Licht-Spill (war 0.2 = Riesen-Halo)
                
                # Radial Fallback Settings
                radial_scale=0.8,        # Größe des Fallback-Blitzes
                
                # Global Effects
                shadow_warm_amount=0.1,  # Warme Vintage-Schatten statt kühlem Party-Vibe
                grain_intensity=0.06     # Härte des ISO-Rauschens
                ):
        """
        Wendet den Effekt auf ein RGB-Bild (0-255 uint8) an und liefert ein RGB-Bild (0-255 uint8) zurück.
        """
        img_float = img_rgb.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]
        
        mask = None
        use_smart = False
        
        # 1. Maske bestimmen: Smart Flash (Personensegmentierung, Prio 1)
        if self.segmentation and not force_radial:
            try:
                # ====== MASSIVE PERFORMANZ-OPTIMIERUNG ======
                # KI-Segmentierung braucht absolut keine 12+ Megapixel. 
                # Wir rechnen das Bild nur für die Masken-Erstellung radikal auf max. 384px runter.
                max_dim = 384.0
                scale = min(max_dim / w, max_dim / h)
                
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    img_small = img_rgb
                    
                # Bild für Mediapipe verarbeiten
                results = self.segmentation.process(img_small)
                
                if results.segmentation_mask is not None:
                    # Maske aus dem kleinen Bild berechnen
                    mask_small = results.segmentation_mask.astype(np.float32)
                    
                    # 1. Maske leicht erweitern für realistischen Blitz-Spill auf der kleinen Auflösung
                    if mask_expansion > 0:
                        s_h, s_w = mask_small.shape
                        k_size = int(max(s_h, s_w) * mask_expansion)
                        if k_size % 2 == 0: k_size += 1
                        if k_size > 0:
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                            mask_small = cv2.dilate(mask_small, kernel, iterations=1)
                    
                    # 2. Maske weichzeichnen (auf der kleinen Auflösung sehr schnell)
                    small_blur = int(mask_blur * scale)
                    if small_blur % 2 == 0: small_blur += 1
                    if small_blur > 0:
                        mask_small = cv2.GaussianBlur(mask_small, (small_blur, small_blur), 0)
                    
                    # Maske in Millisekunden wieder auf die Originalgröße von z.B. 4000x3000 hochskalieren
                    if scale < 1.0:
                         mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                         mask = mask_small
    
                    # Prüfen, ob eine Person signifikant erkannt wurde (> 1% des Bildes)
                    if np.mean(mask > 0.5) > 0.01:
                        use_smart = True
                        print("[PartyFlash] Smart Mask aktiviert (Person erkannt)")
                    else:
                        print("[PartyFlash] Smart Mask: Person zu klein/nicht erkannt")
                else:
                    print("[PartyFlash] Keine Maske von MediaPipe zurückgegeben")
            except Exception as e:
                 print(f"[PartyFlash] MediaPipe Error: {e}")
        else:
            print(f"[PartyFlash] MediaPipe nicht ausgeführt.")
        
        # 1b. Fallback: Radial Flash (falls keine Person erkannt oder force_radial=True)
        if not use_smart:
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            # Gaussian bell als radiale Vignetten-Maske
            sigma = min(h, w) * radial_scale
            dist_sq = (x - cx)**2 + (y - cy)**2
            mask = np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
            print("[PartyFlash] Fallback Radial Mask angewendet")
        
        # Wenn Fallback Radial genutzt wurde, muss hier noch geblurred werden
        if not use_smart and mask_blur > 0:
            if mask_blur % 2 == 0: mask_blur += 1
            mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), 0)
            
        # Optional: Den Smart Flash durch eine Radial-Maske begrenzen, damit die Füße bei 
        # Ganzkörper-Portraits abgedunkelt werden (Physikalischer Lichtabfall von Blitzen)
        if use_smart:
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            sigma = min(h, w) * 1.2  # Sehr sanfte Radial-Vignette über das ganze Bild
            dist_sq = (x - cx)**2 + (y - cy)**2
            vignette = np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
            mask = mask * vignette  # Freistellung * Vignette = Realistischerer Blitzabfall
            
        mask_3d = np.stack([mask]*3, axis=-1)
        
        # 2. Vordergrund (Person) bearbeiten (Methode A / B Blitz)
        # Exposure erhöhen
        fg = img_float * fg_exposure
        
        # Color Processing: Warme Blitzhauttöne
        # Rot anheben, Blau absenken
        fg[..., 0] = fg[..., 0] * (1.0 + fg_warmth)
        fg[..., 2] = fg[..., 2] * (1.0 - fg_warmth * 0.5)
        
        # Kontrast erhöhen (Zentriert um 0.5)
        fg = (fg - 0.5) * fg_contrast + 0.5
        fg = np.clip(fg, 0.0, 1.0)
        # Highlights hochziehen
        fg = fg + (fg ** 2) * fg_highlights
        fg = np.clip(fg, 0.0, 1.0)
        
        # 3. Hintergrund bearbeiten (Harter Lichtabfall)
        bg = img_float * bg_darken
        # Kontrast im Hintergrund für "Crushed Blacks"
        bg = (bg - 0.2) * bg_contrast + 0.2
        bg = np.clip(bg, 0.0, 1.0)
        
        # 4. Blending mit der Maske
        blended = fg * mask_3d + bg * (1.0 - mask_3d)
        
        # 5. Global Effect: Analog Film Split-Toning (Warme Vintage-Schatten)
        # Luminanz berechnen
        lum = 0.299 * blended[..., 0] + 0.587 * blended[..., 1] + 0.114 * blended[..., 2]
        # Schatten-Maske (inverse Luminanz)
        shadow_mask = np.clip(1.0 - (lum / 0.4), 0.0, 1.0) ** 1.5
        shadow_mask = np.stack([shadow_mask]*3, axis=-1)
        
        # Offset für warme, filmische Schatten: (+Rot, +Grün, -Blau)
        warm_tint = np.array([0.08, 0.04, -0.05], dtype=np.float32)
        blended = blended + (shadow_mask * warm_tint * shadow_warm_amount)
        
        # Leichte Halation (Glow um Lichter) global hinzufügen für Filmlook
        bright_mask = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)
        bright_mask = np.stack([bright_mask]*3, axis=-1)
        glow = cv2.GaussianBlur(blended * bright_mask, (51, 51), 0)
        
        # Glow (warmer) dazu addieren
        glow[..., 0] *= 1.2
        glow[..., 2] *= 0.8
        blended = blended + glow * 0.3
        
        blended = np.clip(blended, 0.0, 1.0)
        
        # 6. Global Effect: Hartes, digitales ISO-Rauschen
        # Einfacher Normalverteilungs-Rausch-Layer
        noise = np.random.normal(0, grain_intensity, blended.shape).astype(np.float32)
        blended = blended + noise
        blended = np.clip(blended, 0.0, 1.0)
        
        return (blended * 255.0).astype(np.uint8)

# Singleton Instance ähnlich wie FujiEngine
party_flash_engine = PartyFlashEngine()
