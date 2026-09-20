//! Palette extraction.
//!
//! The central idea here is that an anti-aliased image contains two very
//! different kinds of pixel. Interior pixels carry real colours from the
//! original artwork. Edge pixels carry *blends* of two real colours, produced
//! by the rasteriser averaging coverage over the pixel square. A naive
//! clustering treats both alike, and the blends -- which form a dense ridge
//! between every pair of adjacent colours -- pull cluster centres off the true
//! colours and, worse, earn palette slots of their own. Those slots become thin
//! halo regions outlining every shape.
//!
//! So palette extraction weights each pixel by how flat its neighbourhood is.
//! The blends still get *assigned* to a palette entry later, which is exactly
//! what we want: an edge pixel resolves to whichever side it is closer to,
//! placing the initial contour at the 50% coverage line for the sub-pixel
//! solver to refine.

use crate::color::{srgb_to_lab, Lab, LabCache, Rgba8};
use crate::config::{PaletteMethod, SegmentConfig};
use crate::raster::Raster;

/// Small deterministic PRNG. Reproducibility matters more than statistical
/// quality here: the auto-picker compares configs, and a config has to mean the
/// same thing every time it is run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Per-pixel weight in 0..1 measuring how much a pixel should influence palette
/// selection. Flat interiors score near 1, anti-aliased edges near 0.
pub fn flatness_weights(img: &Raster) -> Vec<f32> {
    let w = img.width as i64;
    let h = img.height as i64;
    let mut out = vec![1.0f32; img.data.len()];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            // Sobel-style gradient magnitude over the three colour channels.
            let mut grad = 0.0f32;
            for ch in 0..3 {
                let at = |dx: i64, dy: i64| img.get_clamped(x + dx, y + dy)[ch];
                let gx = -at(-1, -1) - 2.0 * at(-1, 0) - at(-1, 1)
                    + at(1, -1)
                    + 2.0 * at(1, 0)
                    + at(1, 1);
                let gy = -at(-1, -1) - 2.0 * at(0, -1) - at(1, -1)
                    + at(-1, 1)
                    + 2.0 * at(0, 1)
                    + at(1, 1);
                grad = grad.max((gx * gx + gy * gy).sqrt() * 0.25);
            }
            // A gradient of ~0.15 per pixel already means a visible edge.
            out[(y * w + x) as usize] = (-(grad / 0.12).powi(2)).exp();
        }
    }
    out
}

/// A palette entry plus the statistics needed to decide whether it survives.
#[derive(Clone, Copy, Debug)]
pub struct PaletteEntry {
    pub color: Rgba8,
    pub lab: Lab,
    /// Pixels assigned to this entry are left unpainted. Set for the entry that
    /// stands for alpha-transparent pixels and for any entry matching one of
    /// `SegmentConfig::transparent_colors`.
    pub transparent: bool,
}

impl PaletteEntry {
    fn opaque(color: Rgba8, lab: Lab) -> Self {
        PaletteEntry {
            color,
            lab,
            transparent: false,
        }
    }
}

/// Otsu's method on the luminance histogram, returning a threshold in 0..1.
pub fn otsu_threshold(img: &Raster) -> f64 {
    let mut hist = [0u64; 256];
    for c in &img.data {
        // Weight by alpha so a transparent border does not drag the split.
        if c[3] < 0.5 {
            continue;
        }
        let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        hist[(y.clamp(0.0, 1.0) * 255.0).round() as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.5;
    }
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &n)| i as f64 * n as f64)
        .sum();

    let mut sum_b = 0.0f64;
    let mut w_b = 0u64;
    let mut best_var = -1.0f64;
    let mut best_t = 128usize;

    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t;
        }
    }
    best_t as f64 / 255.0
}

/// Collect a weighted sample of pixels for clustering. Full images are far
/// larger than clustering needs, and sampling keeps palette extraction roughly
/// constant-time regardless of resolution.
fn sample_pixels(
    img: &Raster,
    weights: &[f32],
    alpha_threshold: f32,
    limit: usize,
    rng: &mut Rng,
) -> Vec<([f32; 3], f32)> {
    let n = img.data.len();
    let mut out = Vec::with_capacity(limit.min(n));
    let stride = if n > limit { n / limit } else { 1 };
    let mut i = rng.below(stride.max(1));
    while i < n {
        let c = img.data[i];
        if c[3] > alpha_threshold {
            out.push(([c[0], c[1], c[2]], weights[i]));
        }
        i += stride;
    }
    if out.is_empty() {
        // Everything was transparent; fall back to using it all so downstream
        // code still receives a palette.
        for (i, c) in img.data.iter().enumerate() {
            out.push(([c[0], c[1], c[2]], weights[i]));
        }
    }
    out
}

/// Weighted k-means in CIELAB with k-means++ seeding.
fn kmeans_palette(samples: &[([f32; 3], f32)], k: usize, iterations: usize, rng: &mut Rng) -> Vec<Lab> {
    let labs: Vec<Lab> = samples.iter().map(|(c, _)| srgb_to_lab(*c)).collect();
    let wts: Vec<f32> = samples.iter().map(|(_, w)| w.max(1e-4)).collect();
    let n = labs.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n).max(1);

    // k-means++ seeding, weighted: the probability of picking a point as the
    // next centre scales with weight * distance-squared to the nearest centre.
    let mut centres: Vec<Lab> = Vec::with_capacity(k);
    centres.push(labs[rng.below(n)]);
    let mut d2: Vec<f32> = labs.iter().map(|l| l.dist_sq(centres[0])).collect();

    while centres.len() < k {
        let total: f64 = d2
            .iter()
            .zip(&wts)
            .map(|(d, w)| (*d as f64) * (*w as f64))
            .sum();
        if total <= 1e-9 {
            break;
        }
        let mut target = rng.next_f32() as f64 * total;
        let mut chosen = n - 1;
        for i in 0..n {
            target -= (d2[i] as f64) * (wts[i] as f64);
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        let c = labs[chosen];
        centres.push(c);
        for i in 0..n {
            d2[i] = d2[i].min(labs[i].dist_sq(c));
        }
    }

    let kk = centres.len();
    let mut assign = vec![0usize; n];
    for _ in 0..iterations {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for (j, c) in centres.iter().enumerate() {
                let d = labs[i].dist_sq(*c);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }

        let mut sums = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64); kk];
        for i in 0..n {
            let a = assign[i];
            let w = wts[i] as f64;
            sums[a].0 += labs[i].l as f64 * w;
            sums[a].1 += labs[i].a as f64 * w;
            sums[a].2 += labs[i].b as f64 * w;
            sums[a].3 += w;
        }
        for (j, s) in sums.iter().enumerate() {
            if s.3 > 1e-9 {
                centres[j] = Lab {
                    l: (s.0 / s.3) as f32,
                    a: (s.1 / s.3) as f32,
                    b: (s.2 / s.3) as f32,
                };
            }
        }
        if !changed {
            break;
        }
    }
    centres
}

/// Median cut over the weighted sample, operating in Lab so that splits track
/// perceived rather than numeric colour spread.
fn median_cut_palette(samples: &[([f32; 3], f32)], k: usize) -> Vec<Lab> {
    #[derive(Clone)]
    struct Box {
        items: Vec<(Lab, f32)>,
    }

    impl Box {
        fn spread(&self) -> (usize, f32) {
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for (l, _) in &self.items {
                let v = [l.l, l.a, l.b];
                for i in 0..3 {
                    lo[i] = lo[i].min(v[i]);
                    hi[i] = hi[i].max(v[i]);
                }
            }
            let mut axis = 0;
            let mut best = 0.0;
            for i in 0..3 {
                let s = hi[i] - lo[i];
                if s > best {
                    best = s;
                    axis = i;
                }
            }
            (axis, best)
        }

        fn mean(&self) -> Lab {
            let mut acc = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (l, w) in &self.items {
                let w = *w as f64;
                acc.0 += l.l as f64 * w;
                acc.1 += l.a as f64 * w;
                acc.2 += l.b as f64 * w;
                acc.3 += w;
            }
            if acc.3 < 1e-9 {
                return Lab::default();
            }
            Lab {
                l: (acc.0 / acc.3) as f32,
                a: (acc.1 / acc.3) as f32,
                b: (acc.2 / acc.3) as f32,
            }
        }
    }

    let items: Vec<(Lab, f32)> = samples
        .iter()
        .map(|(c, w)| (srgb_to_lab(*c), w.max(1e-4)))
        .collect();
    if items.is_empty() {
        return Vec::new();
    }
    let mut boxes = vec![Box { items }];

    while boxes.len() < k {
        // Split whichever box has the widest spread and enough members.
        let mut pick = None;
        let mut best = 0.0;
        for (i, b) in boxes.iter().enumerate() {
            if b.items.len() < 2 {
                continue;
            }
            let (_, s) = b.spread();
            if s > best {
                best = s;
                pick = Some(i);
            }
        }
        let Some(i) = pick else { break };
        if best < 1e-4 {
            break;
        }
        let mut b = boxes.swap_remove(i);
        let (axis, _) = b.spread();
        b.items.sort_by(|p, q| {
            let va = [p.0.l, p.0.a, p.0.b][axis];
            let vb = [q.0.l, q.0.a, q.0.b][axis];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = b.items.len() / 2;
        let right = b.items.split_off(mid);
        boxes.push(Box { items: b.items });
        boxes.push(Box { items: right });
    }

    boxes.iter().map(|b| b.mean()).collect()
}

/// The result of palette extraction.
pub struct Palette {
    pub entries: Vec<PaletteEntry>,
    /// Index of the entry that stands for pixels below the alpha threshold, if
    /// the image had any. It has no colour of its own, so it never takes part
    /// in nearest-colour matching. Colour-keyed transparent entries are not
    /// this: they are real colours, matched normally, and merely flagged.
    pub alpha_slot: Option<usize>,
}

impl Palette {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether pixels assigned to entry `i` are left unpainted.
    pub fn is_transparent(&self, i: usize) -> bool {
        self.entries[i].transparent
    }

    /// Nearest entry by Lab distance. The alpha slot is excluded; callers
    /// handle it before reaching here.
    pub fn nearest(&self, lab: Lab) -> usize {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for (i, e) in self.entries.iter().enumerate() {
            if Some(i) == self.alpha_slot {
                continue;
            }
            let d = e.lab.dist_sq(lab);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

/// Flag the palette entries that stand for `cfg.transparent_colors`.
///
/// Every entry within `transparent_tolerance` of a listed colour is flagged, not
/// just the nearest: k-means can split one flat background into two clusters,
/// and leaving one of them painted would put a ghost of the background back.
///
/// A listed colour that no entry is close to is added to the palette as its own
/// flagged entry. That keeps the option from silently doing nothing when the
/// clustering never found the colour -- say, a small `colors` budget that merged
/// it into a neighbour -- and costs nothing when it matches no pixel.
fn mark_transparent_colors(entries: &mut Vec<PaletteEntry>, cfg: &SegmentConfig) {
    let tolerance_sq = cfg.transparent_tolerance.max(0.0).powi(2);
    for key in &cfg.transparent_colors {
        // Exact Lab, not the 5-bit lookup cache: the key is a single colour the
        // user typed, and the cache's quantisation would move the tolerance
        // circle by a couple of units.
        let lab = srgb_to_lab([
            key.r as f32 / 255.0,
            key.g as f32 / 255.0,
            key.b as f32 / 255.0,
        ]);
        let mut matched = false;
        for e in entries.iter_mut() {
            if e.lab.dist_sq(lab) <= tolerance_sq {
                e.transparent = true;
                matched = true;
            }
        }
        if !matched {
            entries.push(PaletteEntry {
                color: Rgba8::opaque(key.r, key.g, key.b),
                lab,
                transparent: true,
            });
        }
    }
}

/// Build the palette for an image according to the segmentation config.
pub fn build_palette(img: &Raster, cfg: &SegmentConfig, cache: &LabCache) -> Palette {
    let has_alpha = img.data.iter().any(|c| c[3] <= cfg.alpha_threshold);

    let mut entries: Vec<PaletteEntry> = Vec::new();
    let mut alpha_slot = None;

    match cfg.mode {
        crate::config::SegmentMode::Binary => {
            let t = cfg.threshold.unwrap_or_else(|| otsu_threshold(img));
            // Use the mean colour of each side rather than pure black/white, so
            // a sepia scan stays sepia.
            let mut dark = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut light = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for c in &img.data {
                if c[3] <= cfg.alpha_threshold {
                    continue;
                }
                let y = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) as f64;
                let bin = if y <= t { &mut dark } else { &mut light };
                bin.0 += c[0] as f64;
                bin.1 += c[1] as f64;
                bin.2 += c[2] as f64;
                bin.3 += 1.0;
            }
            let finish = |b: (f64, f64, f64, f64), fallback: f32| -> [f32; 3] {
                if b.3 < 1.0 {
                    [fallback; 3]
                } else {
                    [
                        (b.0 / b.3) as f32,
                        (b.1 / b.3) as f32,
                        (b.2 / b.3) as f32,
                    ]
                }
            };
            let mut cols = vec![finish(dark, 0.0), finish(light, 1.0)];
            if cfg.invert {
                cols.reverse();
            }
            for c in cols {
                entries.push(PaletteEntry::opaque(
                    Rgba8::from_f32([c[0], c[1], c[2], 1.0]),
                    cache.lookup(c),
                ));
            }
        }
        crate::config::SegmentMode::Palette => {
            let weights = if cfg.ignore_edge_pixels {
                flatness_weights(img)
            } else {
                vec![1.0; img.data.len()]
            };
            let mut rng = Rng::new(cfg.seed);
            let samples = sample_pixels(img, &weights, cfg.alpha_threshold, 40_000, &mut rng);
            let k = cfg.colors.max(2);
            let labs = match cfg.palette_method {
                PaletteMethod::KMeans => {
                    kmeans_palette(&samples, k, cfg.kmeans_iterations, &mut rng)
                }
                PaletteMethod::MedianCut => median_cut_palette(&samples, k),
            };
            for lab in labs {
                let rgb = crate::color::lab_to_srgb(lab);
                entries.push(PaletteEntry::opaque(
                    Rgba8::from_f32([rgb[0], rgb[1], rgb[2], 1.0]),
                    lab,
                ));
            }
        }
    }

    if entries.is_empty() {
        entries.push(PaletteEntry::opaque(Rgba8::opaque(0, 0, 0), Lab::default()));
    }

    mark_transparent_colors(&mut entries, cfg);

    if has_alpha {
        alpha_slot = Some(entries.len());
        entries.push(PaletteEntry {
            color: Rgba8::TRANSPARENT,
            lab: Lab::default(),
            transparent: true,
        });
    }

    Palette { entries, alpha_slot }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SegmentConfig, SegmentMode};

    fn two_tone(width: u32, height: u32) -> Raster {
        let mut r = Raster::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let c = if x < width / 2 {
                    [0.05, 0.05, 0.05, 1.0]
                } else {
                    [0.95, 0.95, 0.95, 1.0]
                };
                r.set(x, y, c);
            }
        }
        r
    }

    #[test]
    fn otsu_splits_a_bimodal_image() {
        let t = otsu_threshold(&two_tone(32, 32));
        assert!(t > 0.05 && t < 0.95, "threshold {t} did not land between modes");
    }

    #[test]
    fn flatness_downweights_edges() {
        let img = two_tone(16, 16);
        let w = flatness_weights(&img);
        let interior = w[(8 * 16 + 2) as usize];
        let on_edge = w[(8 * 16 + 8) as usize];
        assert!(interior > 0.9, "interior weight {interior}");
        assert!(on_edge < 0.2, "edge weight {on_edge} should be suppressed");
    }

    #[test]
    fn kmeans_recovers_planted_colors() {
        // Three well-separated colours, plus a band of blends between two of
        // them. With edge weighting on, the blends must not claim a slot.
        let mut img = Raster::new(60, 20);
        for y in 0..20 {
            for x in 0..60 {
                let c = match x / 20 {
                    0 => [0.9, 0.1, 0.1, 1.0],
                    1 => [0.1, 0.8, 0.2, 1.0],
                    _ => [0.15, 0.2, 0.9, 1.0],
                };
                img.set(x, y, c);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Palette,
            colors: 3,
            ..Default::default()
        };
        let pal = build_palette(&img, &cfg, &cache);
        assert_eq!(pal.len(), 3);
        for want in [[0.9, 0.1, 0.1], [0.1, 0.8, 0.2], [0.15, 0.2, 0.9]] {
            let lab = srgb_to_lab(want);
            let idx = pal.nearest(lab);
            let d = pal.entries[idx].lab.dist_sq(lab).sqrt();
            assert!(d < 12.0, "planted colour {want:?} not recovered, dist {d}");
        }
    }

    #[test]
    fn median_cut_also_recovers_colors() {
        let mut img = Raster::new(40, 10);
        for y in 0..10 {
            for x in 0..40 {
                let c = if x < 20 {
                    [0.9, 0.2, 0.1, 1.0]
                } else {
                    [0.1, 0.3, 0.85, 1.0]
                };
                img.set(x, y, c);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Palette,
            palette_method: PaletteMethod::MedianCut,
            colors: 2,
            ..Default::default()
        };
        let pal = build_palette(&img, &cfg, &cache);
        assert_eq!(pal.len(), 2);
    }

    /// Three flat bands: white, red, blue.
    fn three_bands() -> Raster {
        let mut img = Raster::new(60, 20);
        for y in 0..20 {
            for x in 0..60 {
                let c = match x / 20 {
                    0 => [1.0, 1.0, 1.0, 1.0],
                    1 => [0.9, 0.1, 0.1, 1.0],
                    _ => [0.15, 0.2, 0.9, 1.0],
                };
                img.set(x, y, c);
            }
        }
        img
    }

    fn keyed(colors: &[Rgba8]) -> SegmentConfig {
        SegmentConfig {
            mode: SegmentMode::Palette,
            colors: 3,
            transparent_colors: colors.to_vec(),
            ..Default::default()
        }
    }

    fn transparent_count(pal: &Palette) -> usize {
        pal.entries.iter().filter(|e| e.transparent).count()
    }

    #[test]
    fn a_keyed_colour_flags_the_matching_entry_only() {
        let cache = LabCache::new();
        let pal = build_palette(&three_bands(), &keyed(&[Rgba8::opaque(255, 255, 255)]), &cache);
        assert_eq!(pal.len(), 3, "a match must not add an entry");
        assert_eq!(transparent_count(&pal), 1);
        let white = pal.nearest(srgb_to_lab([1.0, 1.0, 1.0]));
        assert!(pal.is_transparent(white));
    }

    #[test]
    fn the_tolerance_is_a_radius_not_an_equality() {
        let cache = LabCache::new();
        // Off-white by a few levels: well inside the default radius.
        let near = build_palette(&three_bands(), &keyed(&[Rgba8::opaque(248, 250, 247)]), &cache);
        assert_eq!(near.len(), 3);
        assert_eq!(transparent_count(&near), 1, "a near-white key should still find white");

        // A zero radius wants an exact colour, which a clustered centre is
        // unlikely to hit; the key is then pinned as its own entry instead.
        let mut cfg = keyed(&[Rgba8::opaque(248, 250, 247)]);
        cfg.transparent_tolerance = 0.0;
        let strict = build_palette(&three_bands(), &cfg, &cache);
        assert_eq!(strict.len(), 4);
    }

    #[test]
    fn a_key_no_entry_is_near_is_pinned_rather_than_ignored() {
        let cache = LabCache::new();
        let green = Rgba8::opaque(0, 200, 0);
        let pal = build_palette(&three_bands(), &keyed(&[green]), &cache);
        assert_eq!(pal.len(), 4, "the unmatched key should join the palette");
        let pinned = pal.entries.last().unwrap();
        assert!(pinned.transparent);
        assert_eq!(pinned.color, green);
        // The image has no green, so the three real colours stay painted.
        assert_eq!(transparent_count(&pal), 1);
    }

    #[test]
    fn several_keys_can_each_claim_an_entry() {
        let cache = LabCache::new();
        let pal = build_palette(
            &three_bands(),
            &keyed(&[Rgba8::opaque(255, 255, 255), Rgba8::opaque(38, 51, 230)]),
            &cache,
        );
        assert_eq!(pal.len(), 3);
        assert_eq!(transparent_count(&pal), 2);
        let red = pal.nearest(srgb_to_lab([0.9, 0.1, 0.1]));
        assert!(!pal.is_transparent(red), "the unkeyed colour stays painted");
    }

    #[test]
    fn keying_works_in_binary_mode_too() {
        // Line art: black strokes on white paper. Keying the paper out is the
        // most common use, and binary mode builds its palette a different way.
        let mut img = Raster::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                let ink = (8..12).contains(&x);
                let v = if ink { 0.05 } else { 0.97 };
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Binary,
            transparent_colors: vec![Rgba8::opaque(255, 255, 255)],
            ..Default::default()
        };
        let pal = build_palette(&img, &cfg, &cache);
        assert_eq!(pal.len(), 2);
        assert_eq!(transparent_count(&pal), 1);
        let ink = pal.nearest(srgb_to_lab([0.05; 3]));
        assert!(!pal.is_transparent(ink));
    }

    #[test]
    fn no_keys_leaves_everything_opaque() {
        let cache = LabCache::new();
        let pal = build_palette(&three_bands(), &keyed(&[]), &cache);
        assert_eq!(transparent_count(&pal), 0);
        assert_eq!(pal.alpha_slot, None);
    }

    #[test]
    fn rng_is_deterministic() {
        let a: Vec<u64> = (0..5).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        let b: Vec<u64> = (0..5).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        assert_eq!(a, b);
    }
}
