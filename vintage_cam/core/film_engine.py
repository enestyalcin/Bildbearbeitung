"""
film_engine.py – Preset-driven Film Simulation Engine
======================================================
Bildet die Logik von Fujifilm-Kamerasensoren und analogen Filmemulsionen
mathematisch nach. Jedes Preset ist ein einfaches Dictionary –
neue Film-Looks durch Hinzufügen eines neuen Dicts erzeugbar.
"""

import cv2
import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# PRESETS  – alle Werte entsprechen Fuji-Kameraeinstellungen
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    # ---- Kodak Portra 400 ------------------------------------------------
    "portra400": {
        "name":          "Kodak Portra 400",
        "mode":          "color",
        "highlight":     -1,     # -4..+4  (neg = soft roll-off)
        "shadow":        -2,     # -4..+4  (neg = lifted/faded blacks)
        "color":         +2,     # -4..+4  globale Sättigung
        "sharpness":     -2,     # -4..+4  (neg = optische Weichheit)
        "wb_r":          +1,     # -9..+9  Rotkanal-Verschiebung
        "wb_b":          -6,     # -9..+9  Blaukanal-Verschiebung
        "grain_size":    "large",
        "grain_strength":"strong",
        "color_chrome":  "strong",
        "halation":      False,
    },

    # ---- Cinestill 800T --------------------------------------------------
    "cinestill800t": {
        "name":          "Cinestill 800T",
        "mode":          "color",
        "highlight":     -2,
        "shadow":        +1,
        "color":         -1,
        "sharpness":     -2,
        "wb_r":          -3,
        "wb_b":          +1,
        "grain_size":    "large",
        "grain_strength":"strong",
        "color_chrome":  "strong",
        "halation":      True,
    },

    # ---- Ilford HP5 (S/W) ------------------------------------------------
    "hp5": {
        "name":          "Ilford HP5",
        "mode":          "bw",
        "highlight":      0,
        "shadow":        +2,
        "color":          0,
        "sharpness":     -1,
        "wb_r":           0,
        "wb_b":           0,
        "grain_size":    "large",
        "grain_strength":"strong",
        "color_chrome":  "off",
        "halation":      False,
    },

    # ---- Gold Dust (Classic Neg) -----------------------------------------
    "golddust": {
        "name":          "Gold Dust",
        "mode":          "color",
        "highlight":     -1,     # Tone Curve S: -1
        "shadow":         0,
        "color":         +2,     # Color: +2
        "sharpness":     +4,     # Sharpness: +4
        "wb_r":          -2,     # WB 7500k R:-2 B:-2
        "wb_b":          -2,
        "grain_size":    "small",
        "grain_strength":"strong",
        "color_chrome":  "strong",
        "halation":      False,
    },
}


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class FilmEngine:
    """Vektorisierte Bildverarbeitungs-Pipeline für Film-Simulation."""

    # ---- Tone Curve --------------------------------------------------------
    @staticmethod
    def apply_tone_curve(image: np.ndarray, highlight_val: float, shadow_val: float) -> np.ndarray:
        """
        Erzeugt eine Spline-Gradationskurve dynamisch aus den Preset-Werten.

        shadow_val  (-4..+4): negativ = Schwarzpunkt anheben (Faded Blacks)
                              positiv = Schwarzpunkt senken  (Crushed Blacks)
        highlight_val (-4..+4): negativ = Weißpunkt senken   (Soft Roll-Off)
                                positiv = Weißpunkt anheben  (Boosted Highlights)
        """
        # Stützpunkte der Kurve im normalisierten Raum [0, 1]
        # Schwarzpunkt-Mapping: shadow_val=-4 → 50/255 geliftet, +4 → 0 gecrusht
        black_point = np.clip((-shadow_val / 4.0) * (50.0 / 255.0), 0.0, 55.0 / 255.0)
        # Weißpunkt-Mapping: highlight_val=-4 → 200/255, +4 → 255/255
        white_point = np.clip(0.94 + highlight_val * (0.06 / 4.0), 0.78, 1.0)

        # 5 Stützpunkte für eine natürliche S-Form
        xs = np.array([0.0,   0.15,                0.5,  0.85,         1.0])
        ys = np.array([
            black_point,                            # Schwarzpunkt
            black_point + 0.12,                     # Schatten (leicht)
            0.5,                                    # Mitteltöne (unveränderт)
            white_point - 0.10,                     # Lichter (leicht)
            white_point,                            # Weißpunkt
        ])
        ys = np.clip(ys, 0.0, 1.0)

        cs = CubicSpline(xs, ys, bc_type='natural')
        x_lut = np.linspace(0.0, 1.0, 256)
        lut = np.clip(cs(x_lut) * 255.0, 0, 255).astype(np.uint8)

        return cv2.LUT(image, lut)

    # ---- White Balance Shift -----------------------------------------------
    @staticmethod
    def apply_wb_shift(image: np.ndarray, r_shift: float, b_shift: float) -> np.ndarray:
        """
        Verändert Farbtemperatur durch Skalierung der Rot- und Blaukanäle.
        r_shift/b_shift: -9..+9  →  Multiplikator: 1.0 ± 0.25
        """
        img = image.astype(np.float32)
        r_factor = 1.0 + r_shift * (0.25 / 9.0)
        b_factor = 1.0 + b_shift * (0.25 / 9.0)
        img[:, :, 0] = np.clip(img[:, :, 0] * r_factor, 0, 255)  # R
        img[:, :, 2] = np.clip(img[:, :, 2] * b_factor, 0, 255)  # B
        return img.astype(np.uint8)

    # ---- Color Chrome -------------------------------------------------------
    @staticmethod
    def apply_color_chrome(image: np.ndarray, strength: str) -> np.ndarray:
        """
        Fuji Color Chrome Effect: Abdunkeln hochgesättigter Pixelbereiche.
        Stärke: 'off' | 'weak' | 'strong'
        """
        if strength == "off":
            return image

        factor = 0.12 if strength == "weak" else 0.22

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(img_hsv)

        sat_mask = S / 255.0          # 0..1
        V = V * (1.0 - sat_mask * factor)

        img_hsv = cv2.merge((H, S, np.clip(V, 0, 255)))
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # ---- Global Saturation --------------------------------------------------
    @staticmethod
    def adjust_saturation(image: np.ndarray, color_val: float) -> np.ndarray:
        """
        Skaliert die globale Sättigung.
        color_val: -4..+4  →  Faktor 0.4..1.6
        """
        if color_val == 0:
            return image

        factor = 1.0 + color_val * (0.6 / 4.0)   # -4→0.4, 0→1.0, +4→1.6

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(img_hsv)
        S = np.clip(S * factor, 0, 255)
        img_hsv = cv2.merge((H, S, V)).astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # ---- Film Grain ---------------------------------------------------------
    @staticmethod
    def apply_grain(image: np.ndarray, size: str, strength: str) -> np.ndarray:
        """
        Luminanz-adaptives synthetisches Filmkorn im L*ab-Farbraum.
        size:     'small' | 'large'
        strength: 'off' | 'weak' | 'strong'
        """
        if strength == "off":
            return image

        # Amplitude
        amp = 0.025 if strength == "weak" else 0.055

        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(img_lab)
        L_float = L.astype(np.float32)

        # Parabolische Luminanz-Maske: stärkstes Korn bei L=128
        lum_mask = 1.0 - np.square((L_float - 128.0) / 128.0)

        # Basis-Rauschen
        noise = np.random.normal(0.0, amp * 255.0, L_float.shape).astype(np.float32)

        if size == "large":
            # Leichter Blur auf das Rauschen simuliert größeres Korn
            noise = cv2.GaussianBlur(noise, (3, 3), sigmaX=1.2)

        L_noisy = np.clip(L_float + noise * lum_mask, 0, 255).astype(np.uint8)
        img_lab = cv2.merge((L_noisy, a, b))
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    # ---- Optical Softening + Halation ---------------------------------------
    @staticmethod
    def apply_optical_softening(image: np.ndarray, sharpness_val: float,
                                 halation: bool = False) -> np.ndarray:
        """
        sharpness_val < 0: leichter Gaußscher Weichzeichner (optische Unschärfe)
        sharpness_val > 0: Unsharp Mask für mehr Schärfe
        halation=True: roter Lichthof-Glow über hellen Lichtquellen (Cinestill)
        """
        result = image.copy().astype(np.float32)

        if sharpness_val < 0:
            # Optische Weichheit: Stärke proportional zum negativen Wert
            sigma = abs(sharpness_val) * 0.4
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma).astype(np.float32)
            blend = abs(sharpness_val) / 4.0   # 0.25 bei -1, 0.5 bei -2
            result = result * (1.0 - blend) + blurred * blend

        elif sharpness_val > 0:
            # Unsharp Mask für zusätzliche Schärfe
            sigma = 1.0
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma).astype(np.float32)
            amount = sharpness_val * 0.3
            result = result + amount * (result - blurred)

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Halation: warmer roter Glow um helle Lichtquellen (Cinestill 800T)
        if halation:
            img_f = result.astype(np.float32) / 255.0
            r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b

            warm_mask = np.clip((r - b) * 2.0, 0.0, 1.0)
            bright_mask = np.clip((lum - 0.60) / 0.25, 0.0, 1.0)
            glow_mask = (warm_mask * bright_mask)[:, :, np.newaxis]
            glow_src = img_f * glow_mask

            glow_col = np.zeros_like(img_f)
            glow_col[:, :, 0] = glow_src[:, :, 0]
            glow_col[:, :, 1] = glow_src[:, :, 1] * 0.35
            glow_col[:, :, 2] = glow_src[:, :, 2] * 0.08

            glow_blurred = cv2.GaussianBlur(glow_col, (71, 71), sigmaX=18.0)
            img_f = np.clip(img_f + glow_blurred * 0.5, 0.0, 1.0)
            result = (img_f * 255.0).astype(np.uint8)

        return result

    # ---- B&W Conversion -----------------------------------------------------
    @staticmethod
    def to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
        """
        Konvertiert in Graustufen mit filmischen Kanaldgewichten (ähnlich Ilford HP5).
        Gibt ein 3-Kanal-Bild zurück (RGB, aber grau).
        """
        # HP5 hat eine leichte Rot-Empfindlichkeit → klassische panchro Gewichte
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = np.dot(image.astype(np.float32), weights)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ---- Main Pipeline ------------------------------------------------------
    @classmethod
    def process(cls, image_rgb: np.ndarray, preset: dict) -> np.ndarray:
        """
        Führt die vollständige Film-Simulation-Pipeline aus.

        Args:
            image_rgb: Eingabebild als numpy uint8 RGB-Array
            preset:    Preset-Dictionary (aus PRESETS)

        Returns:
            Verarbeitetes RGB uint8 Array
        """
        img = image_rgb.copy()

        # 0. S/W-Konvertierung (vor allem anderen)
        if preset.get("mode") == "bw":
            img = cls.to_grayscale_rgb(img)

        # 1. White Balance Shift
        img = cls.apply_wb_shift(img, preset.get("wb_r", 0), preset.get("wb_b", 0))

        # 2. Tone Curve (Gradationskennlinie via Spline)
        img = cls.apply_tone_curve(img, preset.get("highlight", 0), preset.get("shadow", 0))

        # 3. Sättigung anpassen (nur bei Farbe sinnvoll)
        if preset.get("mode") != "bw":
            img = cls.adjust_saturation(img, preset.get("color", 0))

        # 4. Color Chrome Effect (Fuji-typische Sättigungs-Abdunklung)
        if preset.get("mode") != "bw":
            img = cls.apply_color_chrome(img, preset.get("color_chrome", "off"))

        # 5. Optische Weichheit / Schärfe + Halation
        img = cls.apply_optical_softening(
            img,
            preset.get("sharpness", 0),
            halation=preset.get("halation", False)
        )

        # 6. Filmkorn (letzter Schritt für maximale Authentizität)
        img = cls.apply_grain(
            img,
            size=preset.get("grain_size", "small"),
            strength=preset.get("grain_strength", "off")
        )

        return img
