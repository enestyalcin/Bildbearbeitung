// canvas_filters.js - Client-side Film Simulation Engine using HTML5 Canvas
// Recreates the vintage cam image processing algorithms in pure JavaScript.

// RGB to HSV Conversion
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, v = max;
  const d = max - min;
  s = max === 0 ? 0 : d / max;
  if (max === min) {
    h = 0; // achromatic
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h * 360, s * 255, v * 255];
}

// HSV to RGB Conversion
function hsvToRgb(h, s, v) {
  h /= 360; s /= 255; v /= 255;
  let r, g, b;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Cubic Spline LUT Generator
function getCubicSplineLUT(shadow_val, highlight_val) {
  // shadow_val (-4..+4): neg = lift black point, pos = lower black point
  // highlight_val (-4..+4): neg = lower white point, pos = lift white point
  const black_point = Math.min(55/255, Math.max(0, (-shadow_val / 4.0) * (50.0 / 255.0)));
  const white_point = Math.min(1.0, Math.max(0.78, 0.94 + highlight_val * (0.06 / 4.0)));

  const xs = [0.0, 0.15, 0.5, 0.85, 1.0];
  const ys = [
    black_point,
    Math.min(1.0, Math.max(0.0, black_point + 0.12)),
    0.5,
    Math.min(1.0, Math.max(0.0, white_point - 0.10)),
    white_point
  ];

  // Natural Cubic Spline Interpolation
  const n = xs.length;
  const h = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i+1] - xs[i];

  const alpha = new Array(n - 1);
  alpha[0] = 0;
  for (let i = 1; i < n - 1; i++) {
    alpha[i] = (3.0 / h[i]) * (ys[i+1] - ys[i]) - (3.0 / h[i-1]) * (ys[i] - ys[i-1]);
  }

  const l = new Array(n);
  const mu = new Array(n);
  const z = new Array(n);
  l[0] = 1.0; mu[0] = 0.0; z[0] = 0.0;

  for (let i = 1; i < n - 1; i++) {
    l[i] = 2.0 * (xs[i+1] - xs[i-1]) - h[i-1] * mu[i-1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
  }

  l[n-1] = 1.0; z[n-1] = 0.0;
  const b = new Array(n);
  const c = new Array(n);
  const d = new Array(n);
  c[n-1] = 0.0;

  for (let j = n - 2; j >= 0; j--) {
    c[j] = z[j] - mu[j] * c[j+1];
    b[j] = (ys[j+1] - ys[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0;
    d[j] = (c[j+1] - c[j]) / (3.0 * h[j]);
  }

  // Generate 256 LUT values
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    const x = i / 255.0;
    let idx = 0;
    for (let k = 0; k < n - 1; k++) {
      if (x >= xs[k] && x <= xs[k+1]) {
        idx = k;
        break;
      }
    }
    const dx = x - xs[idx];
    const y = ys[idx] + b[idx]*dx + c[idx]*dx*dx + d[idx]*dx*dx*dx;
    lut[i] = Math.min(255, Math.max(0, Math.round(y * 255)));
  }
  return lut;
}

// Preset Definitions
const PRESETS = {
  // Classic filters
  filter1: { // Vintage Warm
    name: "Vintage Warm", mode: "color", wb_r: 3, wb_b: -4, highlight: -1, shadow: -1, color: 1.0, color_chrome: "strong", grain_size: "large", grain_strength: "weak", vignette: 0.25, base_sim: "warm"
  },
  filter2: { // Moody Dark
    name: "Moody Dark", mode: "color", wb_r: -2, wb_b: 3, highlight: 0.5, shadow: 1.5, color: -1.0, color_chrome: "off", grain_size: "small", grain_strength: "weak", vignette: 0.6, base_sim: "dark"
  },
  filter3: { // Fuji Chrome
    name: "Fuji Chrome", mode: "color", wb_r: -1, wb_b: -2, highlight: 1.0, shadow: 0.5, color: 0.5, color_chrome: "strong", grain_size: "large", grain_strength: "strong", vignette: 0.35, base_sim: "fuji_chrome"
  },
  filter4: { // Cinestill Night
    name: "Cinestill Night", mode: "color", wb_r: -3, wb_b: 4, highlight: -1.5, shadow: 1.0, color: -0.5, color_chrome: "strong", grain_size: "large", grain_strength: "strong", vignette: 0.45, halation: true, base_sim: "cinestill"
  },
  filter5: { // Party Flash
    name: "Party Flash", mode: "color", wb_r: 1, wb_b: -2, highlight: 2.0, shadow: 2.0, color: 0.8, color_chrome: "strong", grain_size: "small", grain_strength: "strong", vignette: 0.8, flash_glow: true, base_sim: "party_flash"
  },
  // Film recipes
  portra400: {
    name: "Kodak Portra 400", mode: "color", wb_r: 1, wb_b: -6, highlight: -1, shadow: -2, color: 2, color_chrome: "strong", grain_size: "large", grain_strength: "strong", vignette: 0.15, base_sim: "portra"
  },
  cinestill800t: {
    name: "Cinestill 800T", mode: "color", wb_r: -3, wb_b: 1, highlight: -2, shadow: 1, color: -1, color_chrome: "strong", grain_size: "large", grain_strength: "strong", vignette: 0.3, halation: true, base_sim: "cinestill"
  },
  hp5: {
    name: "Ilford HP5", mode: "bw", wb_r: 0, wb_b: 0, highlight: 0, shadow: 2, color: 0, color_chrome: "off", grain_size: "large", grain_strength: "strong", vignette: 0.35, base_sim: "mono"
  },
  golddust: {
    name: "Gold Dust", mode: "color", wb_r: -2, wb_b: -2, highlight: -1, shadow: 0, color: 2, color_chrome: "strong", grain_size: "small", grain_strength: "strong", vignette: 0.25, base_sim: "golddust"
  },
  // Fuji originals
  nostalgic_neg: {
    name: "Nostalgic Neg", mode: "color", wb_r: 3, wb_b: -4, highlight: -1.0, shadow: 1.0, color: 1.0, color_chrome: "strong", grain_size: "large", grain_strength: "strong", vignette: 0.25, base_sim: "nostalgic_neg"
  },
  eterna_bb: {
    name: "Eterna Bypass", mode: "color", wb_r: 2, wb_b: -4, highlight: 0.0, shadow: 1.0, color: -1.0, color_chrome: "off", grain_size: "large", grain_strength: "weak", vignette: 0.3, base_sim: "eterna_bb"
  },
  classic_neg: {
    name: "Classic Neg", mode: "color", wb_r: -2, wb_b: 1, highlight: -1.5, shadow: 0.5, color: 1.0, color_chrome: "strong", grain_size: "small", grain_strength: "strong", vignette: 0.35, base_sim: "classic_neg"
  }
};

// Apply Pixel Manipulations
function applyPixelAdjustments(imgData, preset) {
  const data = imgData.data;
  const len = data.length;
  
  // Pre-calculate White Balance factors
  const r_factor = 1.0 + preset.wb_r * (0.25 / 9.0);
  const b_factor = 1.0 + preset.wb_b * (0.25 / 9.0);
  
  // Pre-calculate S-Curve LUT
  const lut = getCubicSplineLUT(preset.shadow, preset.highlight);
  
  // Color Chrome Strength Factor
  const chrome_factor = preset.color_chrome === "strong" ? 0.22 : (preset.color_chrome === "weak" ? 0.12 : 0.0);
  
  // Saturation factor
  const sat_scale = 1.0 + preset.color * (0.6 / 4.0); // color_val is -4..+4, mapped to 0.4..1.6

  for (let i = 0; i < len; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];
    
    // 0. Grayscale Conversion (B&W mode)
    if (preset.mode === "bw") {
      const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      r = gray; g = gray; b = gray;
    }
    
    // 1. White Balance Shift
    r = Math.min(255, Math.max(0, r * r_factor));
    b = Math.min(255, Math.max(0, b * b_factor));
    
    // 2. Tone Curve LUT
    r = lut[Math.round(r)];
    g = lut[Math.round(g)];
    b = lut[Math.round(b)];
    
    // 3. Color Adjustments and Base Simulations
    if (preset.mode !== "bw") {
      // Convert to HSV for selective color editing
      let [h, s, v] = rgbToHsv(r, g, b);
      
      // Saturation boost
      s = Math.min(255, Math.max(0, s * sat_scale));
      
      // Fuji-specific Color Chrome Effect (darken highly saturated pixels)
      if (chrome_factor > 0) {
        const sat_mask = s / 255.0;
        v *= (1.0 - sat_mask * chrome_factor);
      }
      
      // Presets special color gradings
      if (preset.base_sim === "warm") {
        // Teal-Orange approximation: shift blue towards cyan, boost warm skin tones
        const blue_mask = Math.exp(-0.5 * Math.pow((h - 230) / 36, 2));
        h -= blue_mask * 24.0; // shift to teal (cyan-blue)
        s += blue_mask * 20.0;
        
        const warm_mask = Math.exp(-0.5 * Math.pow(((h + 180) % 360 - 180) / 30, 2));
        s += warm_mask * 45.0;
        v += warm_mask * 15.0;
      } 
      else if (preset.base_sim === "dark") {
        // Cool shadows, high contrast, low saturation
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        const shadow_mask = Math.max(0, Math.min(1.0, 1.0 - lum / 100));
        h = (h + shadow_mask * 10) % 360; // shift shadows cooler
        s *= (1.0 - shadow_mask * 0.2); // desaturate shadows
      }
      else if (preset.base_sim === "fuji_chrome") {
        // Deep greens and rich contrast (Fuji Chrome look)
        if (h > 60 && h < 160) { // Green tones
          s *= 1.25;
          v *= 0.9;
        }
      }
      else if (preset.base_sim === "nostalgic_neg") {
        // Warm amber highlights (shift light pixels red-yellow)
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (lum > 150) {
          const lum_mask = (lum - 150) / 105;
          // Add amber: boost Red & Green, drop Blue
          r += lum_mask * 10;
          g += lum_mask * 5;
          b -= lum_mask * 8;
        }
        // Boost saturated reds
        const red_diff = (h + 180) % 360 - 180;
        const red_mask = Math.exp(-0.5 * Math.pow(red_diff / 24, 2));
        s += red_mask * 30.0;
      }
      else if (preset.base_sim === "eterna_bb") {
        // Bleach Bypass: desaturate heavily and boost contrast
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        r = r * 0.45 + gray * 0.55;
        g = g * 0.45 + gray * 0.55;
        b = b * 0.45 + gray * 0.55;
        
        // Boost contrast on L
        r = 128 + (r - 128) * 1.3;
        g = 128 + (g - 128) * 1.3;
        b = 128 + (b - 128) * 1.3;
      }
      else if (preset.base_sim === "classic_neg") {
        // Magenta shadows, cyan blues
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        const shadow_mask = Math.max(0, Math.min(1.0, 1.0 - lum / 80));
        // Add magenta: boost Red/Blue, reduce Green in shadows
        r += shadow_mask * 8.0;
        b += shadow_mask * 10.0;
        g -= shadow_mask * 4.0;
        
        // Shift blue towards cyan
        const blue_mask = Math.exp(-0.5 * Math.pow((h - 236) / 30, 2));
        h -= blue_mask * 28.0; // blue towards cyan
        s += blue_mask * 15.0;
      }
      else if (preset.base_sim === "golddust") {
        // Warm gold wash, vintage glow
        const warm_mask = Math.exp(-0.5 * Math.pow(((h + 180) % 360 - 180) / 45, 2));
        h = (h - 5 + 360) % 360; // warm shift
        s += warm_mask * 35.0;
        v *= 1.05;
      }
      
      const [newR, newG, newB] = hsvToRgb(h, s, v);
      r = newR; g = newG; b = newB;
    }
    
    // Clamp values
    data[i] = Math.min(255, Math.max(0, r));
    data[i + 1] = Math.min(255, Math.max(0, g));
    data[i + 2] = Math.min(255, Math.max(0, b));
  }
}

// Apply Halation Glow (Cinestill / Filter 4)
function applyHalation(canvas, intensity = 0.45) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  // Create offscreen canvas for glow mask
  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = w;
  glowCanvas.height = h;
  const gCtx = glowCanvas.getContext('2d');
  
  // Draw base image to extract highlights
  gCtx.drawImage(canvas, 0, 0);
  const imgData = gCtx.getImageData(0, 0, w, h);
  const data = imgData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i+1];
    const b = data[i+2];
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    
    // Warme helle Pixel isolieren (Red deutlich höher als Blue, Gesamthelligkeit > 0.55)
    const warm_mask = Math.max(0.0, Math.min(1.0, (r - b) / 255.0 * 2.5));
    const bright_mask = Math.max(0.0, Math.min(1.0, (lum - 0.55) / 0.3));
    const mask = warm_mask * bright_mask;
    
    // Glow color: intense warm orange-red
    data[i] = r * mask;
    data[i+1] = g * mask * 0.4;
    data[i+2] = b * mask * 0.1;
    data[i+3] = 255;
  }
  gCtx.putImageData(imgData, 0, 0);
  
  // Draw mask back onto original canvas using blur filter and screen composite mode
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  ctx.globalAlpha = intensity;
  // Blur scale based on image dimensions
  const blurRadius = Math.max(8, Math.round(Math.min(w, h) * 0.015));
  ctx.filter = `blur(${blurRadius}px)`;
  ctx.drawImage(glowCanvas, 0, 0);
  ctx.restore();
}

// Apply Vignette (Edges darkening)
function applyVignette(canvas, intensity) {
  if (intensity <= 0) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  ctx.save();
  ctx.globalCompositeOperation = 'multiply';
  
  const outerRadius = Math.max(w, h) * 0.75;
  const innerRadius = Math.min(w, h) * 0.2;
  const grad = ctx.createRadialGradient(
    w / 2, h / 2, innerRadius,
    w / 2, h / 2, outerRadius
  );
  
  grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
  const darkValue = Math.round(255 * (1.0 - intensity * 0.85));
  grad.addColorStop(1, `rgba(${darkValue}, ${darkValue}, ${darkValue}, 1)`);
  
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

// Apply Film Grain (Tiled noise)
function applyGrain(canvas, strength = "off", size = "small") {
  if (strength === "off") return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  // Generate neutral-gray tileable noise canvas (128 = transparent in overlay/soft-light)
  const tileSize = 256;
  const tileCanvas = document.createElement('canvas');
  tileCanvas.width = tileSize;
  tileCanvas.height = tileSize;
  const tCtx = tileCanvas.getContext('2d');
  const tImgData = tCtx.createImageData(tileSize, tileSize);
  const tData = tImgData.data;
  
  const noiseAmp = strength === "strong" ? 42 : 22;
  
  for (let i = 0; i < tData.length; i += 4) {
    // Random gaussian noise approximation
    const noise = ((Math.random() + Math.random() + Math.random() - 1.5) / 1.5) * noiseAmp;
    const val = 128 + noise;
    tData[i] = val;
    tData[i+1] = val;
    tData[i+2] = val;
    tData[i+3] = 255;
  }
  tCtx.putImageData(tImgData, 0, 0);
  
  // If large grain size, blur the noise tile slightly to clump pixels together
  if (size === "large") {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = tileSize;
    tempCanvas.height = tileSize;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.filter = 'blur(1px)';
    tempCtx.drawImage(tileCanvas, 0, 0);
    tCtx.clearRect(0, 0, tileSize, tileSize);
    tCtx.drawImage(tempCanvas, 0, 0);
  }
  
  // Tile the noise canvas onto main canvas using Overlay composite
  ctx.save();
  ctx.globalCompositeOperation = 'overlay';
  ctx.globalAlpha = strength === "strong" ? 0.35 : 0.22;
  const pattern = ctx.createPattern(tileCanvas, 'repeat');
  ctx.fillStyle = pattern;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

// Flash simulation (bright circle at center)
function applyFlashGlow(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  
  const grad = ctx.createRadialGradient(
    w/2, h/2, 0,
    w/2, h/2, Math.max(w, h) * 0.65
  );
  grad.addColorStop(0, 'rgba(255, 253, 240, 0.45)'); // warm center light
  grad.addColorStop(0.3, 'rgba(255, 250, 230, 0.25)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

// Master Processing Function
function processImageClientSide(sourceImage, destCanvas, filterName) {
  const preset = PRESETS[filterName] || PRESETS.filter1;
  const ctx = destCanvas.getContext('2d');
  
  // Set dimensions
  destCanvas.width = sourceImage.naturalWidth || sourceImage.width;
  destCanvas.height = sourceImage.naturalHeight || sourceImage.height;
  
  // 1. Draw base image
  ctx.drawImage(sourceImage, 0, 0);
  
  // 2. Perform pixel manipulation
  const imgData = ctx.getImageData(0, 0, destCanvas.width, destCanvas.height);
  applyPixelAdjustments(imgData, preset);
  ctx.putImageData(imgData, 0, 0);
  
  // 3. Draw flash overlay if active
  if (preset.flash_glow) {
    applyFlashGlow(destCanvas);
  }
  
  // 4. Draw halation glow if active
  if (preset.halation) {
    applyHalation(destCanvas, 0.5);
  }
  
  // 5. Draw vignette
  if (preset.vignette > 0) {
    applyVignette(destCanvas, preset.vignette);
  }
  
  // 6. Draw film grain
  if (preset.grain_strength && preset.grain_strength !== "off") {
    applyGrain(destCanvas, preset.grain_strength, preset.grain_size);
  }
}
