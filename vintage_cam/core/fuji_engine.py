"""
fuji_engine.py – Hochpräzise Fujifilm Film-Simulation Engine
=============================================================
Mathematische Approximation der Fujifilm-Sensor- und Film-Simulations-Logik.
Rein vektorisiert (cv2 + numpy), keine Pixel-Schleifen.

LUT-Einhängepunkte sind mit dem Kommentar  # [LUT_HOOK]  markiert.
Dort können später echte .cube-Dateien eingelesen und angewendet werden
(z.B. via colour-science oder eine eigene LUT-Interpolation).
"""

import cv2
import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# FUJI PRESETS  – direkt aus Kamera-Rezept-Screenshots abgelesen
# ---------------------------------------------------------------------------

FUJI_PRESETS: dict[str, dict] = {

    # ---- Nostalgic Negative ------------------------------------------------
    "nostalgic_neg": {
        "name":        "Nostalgic Neg",
        "base":        "nostalgic_neg",
        "wb":          (3, -4),      # R+3, B-4  → starke Erwärmung
        "highlight":   -1.0,
        "shadow":       1.0,
        "color":        1.0,
        "grain":       ("strong", "large"),
        "clarity":     -2.0,
        "chrome":      "strong",
        "blue_chrome": "weak",
    },

    # ---- Eterna Bleach Bypass ----------------------------------------------
    "eterna_bb": {
        "name":        "Eterna Bleach Bypass",
        "base":        "eterna_bb",
        "wb":          (2, -4),
        "highlight":    0.0,
        "shadow":       1.0,
        "color":       -1.0,
        "grain":       ("weak", "large"),
        "clarity":     -2.0,
        "chrome":      "off",
        "blue_chrome": "off",
    },

    # ---- Classic Negative --------------------------------------------------
    "classic_neg": {
        "name":        "Classic Negative",
        "base":        "classic_neg",
        "wb":          (-2, 1),
        "highlight":   -1.5,
        "shadow":       0.5,
        "color":        1.0,
        "grain":       ("strong", "small"),
        "clarity":     -2.0,
        "chrome":      "strong",
        "blue_chrome": "weak",
    },
}


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class FujiEngine:
    """
    Vektorisierte Fujifilm Film-Simulation Engine.
    Alle Methoden akzeptieren und geben uint8 RGB-Arrays zurück,
    sofern nicht anders angegeben.
    """

    # ------------------------------------------------------------------ #
    #  1. BASE SIMULATION                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_base_simulation(image: np.ndarray, sim_type: str) -> np.ndarray:
        """
        Basis-Color-Grading als Approximation der Fuji-Film-Simulation.

        # [LUT_HOOK] ──────────────────────────────────────────────────────
        # Hier kann eine echte .cube-LUT eingehängt werden:
        #   import colour
        #   lut = colour.io.read_LUT("Nostalgic_Neg.cube")
        #   img_linear = image / 255.0
        #   result = lut.apply(img_linear)
        #   return (np.clip(result, 0, 1) * 255).astype(np.uint8)
        # Solange keine LUT vorhanden, läuft die Approximation unten.
        # ─────────────────────────────────────────────────────────────────
        """
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, a, b_ch = cv2.split(img_lab)

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(img_hsv)

        if sim_type == "nostalgic_neg":
            # Warme Lichter: Lichter werden amber-verschoben (a+ = mehr Rot,  b+ = mehr Gelb)
            lum_mask = np.clip((L - 100.0) / 80.0, 0.0, 1.0)      # nur Lichter
            a  += lum_mask * 6.0    # Amber/Rot in Lichtern
            b_ch += lum_mask * 8.0  # Gelb-Stich in Lichtern

            # Sattes Rot: Rot-Hues (H nahe 0/179) gesättigt
            h_shifted = np.mod(H + 90.0, 180.0) - 90.0
            red_mask = np.exp(-0.5 * np.square(h_shifted / 12.0))
            S += red_mask * 25.0

            img_lab = cv2.merge((L, a, b_ch))
            img = cv2.cvtColor(np.clip(img_lab, 0, 255).astype(np.uint8),
                               cv2.COLOR_LAB2RGB)
            # Sättigungs-Boost aus HSV
            img_hsv2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            Hh, Ss, Vv = cv2.split(img_hsv2)
            h2 = np.mod(Hh + 90.0, 180.0) - 90.0
            rm = np.exp(-0.5 * np.square(h2 / 12.0))
            Ss = np.clip(Ss + rm * 25.0, 0, 255)
            img_hsv2 = cv2.merge((Hh, Ss, Vv)).astype(np.uint8)
            return cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2RGB)

        elif sim_type == "eterna_bb":
            # Bleach Bypass: globale Sättigung durch RGB-Graustufen-Mix erzielen
            # (sicherer als Lab-Skalierung, kein Farbstich)
            desat_factor = 0.45   # 0 = vollgrau, 1 = original

            img_f = image.astype(np.float32)
            # Perceptual Grayscale (Luminanz)
            gray = (0.299 * img_f[:, :, 0] +
                    0.587 * img_f[:, :, 1] +
                    0.114 * img_f[:, :, 2])
            gray3 = np.stack([gray, gray, gray], axis=2)

            # Bild mit Graustufen mischen (Desaturation)
            img_desat = img_f * desat_factor + gray3 * (1.0 - desat_factor)

            # Luminanzkontrast erhöhen via Lab L-Kanal
            img_desat_u8 = np.clip(img_desat, 0, 255).astype(np.uint8)
            lab2 = cv2.cvtColor(img_desat_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
            L2, a2, b2 = cv2.split(lab2)
            L2 = np.clip(128.0 + (L2 - 128.0) * 1.25, 0, 255)
            lab2 = cv2.merge((L2, a2, b2))
            return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8),
                                cv2.COLOR_LAB2RGB)

        elif sim_type == "classic_neg":
            # Schatten leicht in Magenta: a+ in Schatten
            shadow_mask = np.clip(1.0 - (L / 80.0), 0.0, 1.0)
            a += shadow_mask * 5.0    # Magenta in Schatten

            # Blautöne nach Cyan ziehen: blaue Hues (H ~110–130) nach Cyan (H ~90)
            blue_mask = np.exp(-0.5 * np.square((H - 118.0) / 15.0))
            H_shifted = H - blue_mask * 14.0   # Hue → Cyan

            H_shifted = np.clip(H_shifted, 0, 179)
            S_new = np.clip(S + blue_mask * 15.0, 0, 255)
            img_hsv_new = cv2.merge((H_shifted, S_new, V)).astype(np.uint8)
            img_from_hsv = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2RGB)

            img_lab2 = cv2.cvtColor(img_from_hsv, cv2.COLOR_RGB2LAB).astype(np.float32)
            L2, a2, b2 = cv2.split(img_lab2)
            sm = np.clip(1.0 - (L2 / 80.0), 0.0, 1.0)
            a2 += sm * 5.0
            img_lab2 = cv2.merge((L2, a2, b2))
            return cv2.cvtColor(np.clip(img_lab2, 0, 255).astype(np.uint8),
                                cv2.COLOR_LAB2RGB)

        # Fallback: kein Base-Grading
        return image

    # ------------------------------------------------------------------ #
    #  2. WHITE BALANCE SHIFT                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_white_balance_shift(image: np.ndarray,
                                   r_shift: float, b_shift: float) -> np.ndarray:
        """
        Fuji-Logik: R/B-Kanal-Multiplikation.
        r_shift/b_shift: -9..+9 → Faktor 0.75..1.25
        R+3 & B-4 = starke Erwärmung.
        """
        img = image.astype(np.float32)
        r_factor = 1.0 + r_shift * (0.25 / 9.0)
        b_factor = 1.0 + b_shift * (0.25 / 9.0)
        img[:, :, 0] = np.clip(img[:, :, 0] * r_factor, 0, 255)   # R
        img[:, :, 2] = np.clip(img[:, :, 2] * b_factor, 0, 255)   # B
        return img.astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  3. TONE CURVE (Fuji-Logik, Spline auf L*-Kanal)                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_tone_curve(image: np.ndarray,
                          highlight: float, shadow: float) -> np.ndarray:
        """
        Spline-Gradationskurve im L*ab-Kanal.

        Fuji-Logik:
          shadow > 0  → Schatten dunkler (Kontrast ↑),  < 0 → liftet Schatten
          highlight < 0 → Lichter weicher / Roll-off,   > 0 → Lichter heller

        Schrittweite 0.5 (Werte wie -1.5 sind möglich).
        """
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, a, b = cv2.split(img_lab)

        # Schwarzpunkt: shadow > 0 senkt, < 0 hebt
        black_pt = np.clip((-shadow / 4.0) * (40.0 / 255.0) * 255.0,
                            -20.0, 20.0)
        # Weißpunkt: highlight < 0 senkt, > 0 hebt
        white_pt = np.clip(235.0 + highlight * (15.0 / 2.0),
                            195.0, 255.0)

        # Spline: 6 Kontrollpunkte für weiche Fuji-typische Kurve
        xs = np.array([0.0,   30.0,  80.0, 128.0,  200.0, 255.0])
        ys = np.array([
            np.clip(black_pt,         0, 40),          # Schwarz
            np.clip(30.0 + black_pt * 0.5, 0, 55),    # Tiefe Schatten
            np.clip(78.0 + black_pt * 0.2, 0, 100),   # Schatten
            128.0,                                      # Mitteltöne (fix)
            np.clip(white_pt - 35.0, 150, 220),        # Lichter
            np.clip(white_pt, 200, 255),               # Weiß
        ])
        ys = np.clip(ys, 0, 255)

        cs = CubicSpline(xs, ys, bc_type="natural")
        x_lut = np.arange(256, dtype=np.float32)
        lut = np.clip(cs(x_lut), 0, 255).astype(np.uint8)

        L_mapped = cv2.LUT(L.astype(np.uint8), lut).astype(np.float32)
        img_lab = cv2.merge((L_mapped, a, b))
        return cv2.cvtColor(np.clip(img_lab, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------ #
    #  4. COLOR CHROME  (global + blue)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_color_chrome(image: np.ndarray,
                            global_chrome: str, blue_chrome: str) -> np.ndarray:
        """
        Fuji Color Chrome Effect: Luminanz-Abdunklung bei hoher Sättigung.
        - global_chrome 'strong': bis -22% Luminanz bei voll gesättigten Pixeln
        - blue_chrome   'weak' : ebenso, aber nur auf blaue Hues, Faktor ×0.45
        """
        if global_chrome == "off" and blue_chrome == "off":
            return image

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(img_hsv)
        sat_norm = S / 255.0   # 0..1

        if global_chrome in ("strong", "weak"):
            factor = 0.22 if global_chrome == "strong" else 0.12
            V = V * (1.0 - sat_norm * factor)

        if blue_chrome in ("strong", "weak"):
            # Nur auf blaue Hues (H ~100-130 in OpenCV 0-179)
            blue_factor = 0.22 if blue_chrome == "strong" else 0.10
            blue_mask = np.exp(-0.5 * np.square((H - 115.0) / 20.0))
            V = V * (1.0 - sat_norm * blue_mask * blue_factor)

        V = np.clip(V, 0, 255)
        img_hsv = cv2.merge((H, S, V)).astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # ------------------------------------------------------------------ #
    #  5. CLARITY (Local Contrast Reduction → Fuji Bloom / Weichheit)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_clarity(image: np.ndarray, clarity_val: float) -> np.ndarray:
        """
        Clarity < 0: Local Contrast Reduction – typischer Fuji 'Bloom'-Effekt.
        Implementierung: Bild = Original − blend*(Hochpass)
        wobei Hochpass = Original − GaussBlur(radius_groß)

        clarity_val: -4..+4 (Fuji-Skala), hier auf praktische Werte gemappt.
        """
        if clarity_val == 0:
            return image

        img_f = image.astype(np.float32)
        # Großer Gaußscher Blur extrahiert die Tieffrequenz-Ebene
        sigma = 25.0
        low_freq = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
        high_freq = img_f - low_freq   # Hochpass (Details + Mikrotexturen)

        # blend_factor: negatives clarity → Hochpass subtrahieren
        # -2 → blend ≈ -0.25 (Details werden sanft unterdrückt)
        blend = clarity_val * 0.125

        result = img_f + blend * high_freq
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  6. GRAIN (Poisson-inspiriert, luminanz-adaptiv)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_grain(image: np.ndarray,
                     size: str, strength: str) -> np.ndarray:
        """
        Poisson-inspiriertes, luminanz-adaptives Filmkorn im L*-Kanal.
        Korn ist am stärksten in Mitteltönen (parabolische Luminanz-Maske).

        size:     'small' | 'large'
        strength: 'off' | 'weak' | 'strong'
        """
        if strength == "off":
            return image

        amp = 0.028 if strength == "weak" else 0.058

        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(img_lab)
        L_f = L.astype(np.float32)

        # Luminanz-Maske: parabolisch, Maximum bei L=128
        lum_mask = 1.0 - np.square((L_f - 128.0) / 128.0)

        # Basis-Gauss-Rauschen
        noise = np.random.normal(0.0, amp * 255.0, L_f.shape).astype(np.float32)

        # Poisson-Komponente: std proportional zu √L (echte Photon-Shot-Noise-Näherung)
        poisson_std = np.sqrt(np.clip(L_f, 0, 255)) * (amp * 0.8)
        noise += np.random.normal(0.0, 1.0, L_f.shape).astype(np.float32) * poisson_std

        if size == "large":
            # Leichter Blur macht einzelne Körner größer/weicher
            noise = cv2.GaussianBlur(noise, (3, 3), sigmaX=1.4)

        L_noisy = np.clip(L_f + noise * lum_mask, 0, 255).astype(np.uint8)
        img_lab = cv2.merge((L_noisy, a, b))
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------ #
    #  MAIN PIPELINE                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def process(cls, image_rgb: np.ndarray, preset: dict) -> np.ndarray:
        """
        Führt die vollständige Fuji-Film-Simulation in der korrekten Reihenfolge aus.

        Pipeline-Reihenfolge (entspricht Fuji-Kamera-Prozessor):
          1. Base Simulation  (Film-Charakter)
          2. White Balance    (Farbtemperatur)
          3. Tone Curve       (Gradationskennlinie)
          4. Color Chrome     (Sättigungs-Abdunklung)
          5. Clarity          (Lokaler Kontrast / Bloom)
          6. Grain            (Filmkorn, letzter Schritt)
        """
        img = image_rgb.copy()

        # 1. Base Film Simulation
        img = cls.apply_base_simulation(img, preset["base"])

        # 2. White Balance Shift
        wb_r, wb_b = preset.get("wb", (0, 0))
        img = cls.apply_white_balance_shift(img, wb_r, wb_b)

        # 3. Tone Curve (Spline auf L*-Kanal)
        img = cls.apply_tone_curve(
            img,
            highlight=preset.get("highlight", 0.0),
            shadow=preset.get("shadow", 0.0)
        )

        # 4. Globale Sättigung (color param)
        color_val = preset.get("color", 0.0)
        if color_val != 0.0:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            Hh, Ss, Vv = cv2.split(img_hsv)
            factor = 1.0 + color_val * (0.55 / 4.0)
            Ss = np.clip(Ss * factor, 0, 255)
            img_hsv = cv2.merge((Hh, Ss, Vv)).astype(np.uint8)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # 5. Color Chrome Effect
        img = cls.apply_color_chrome(
            img,
            global_chrome=preset.get("chrome", "off"),
            blue_chrome=preset.get("blue_chrome", "off")
        )

        # 6. Clarity (Fuji Bloom in Mitteltönen)
        img = cls.apply_clarity(img, preset.get("clarity", 0.0))

        # 7. Grain (letzter Schritt, auf finales Bild)
        grain_strength, grain_size = preset.get("grain", ("off", "small"))
        img = cls.apply_grain(img, size=grain_size, strength=grain_strength)

        return img
