//! Automatic algorithm selection.
//!
//! Choosing tracing parameters by hand means guessing, then squinting at the
//! result. This module does the obvious thing instead: it actually runs the
//! candidates and measures them.
//!
//! The search is in three parts.
//!
//! 1. **Profile** the image cheaply -- how many colours, is it anti-aliased, is
//!    it photographic, is it pixel art. This decides which family of settings is
//!    even worth trying, so the search is not spent tracing a photograph as
//!    two-colour line art.
//! 2. **Seed** a spread of candidate configurations and score every one by a
//!    full round trip.
//! 3. **Refine** the best few by coordinate descent over the parameters that
//!    matter most, keeping any change that improves the ranking.
//!
//! The ranking is the criterion stated up front, applied literally: among
//! candidates that hit the fidelity target, the one with the fewest control
//! points wins. Fidelity is a gate, not something traded away for compactness.
//! Only when nothing reaches the target does ranking fall back to whichever
//! candidate got closest.
//!
//! On large inputs the search runs on a downscaled copy for speed, and the
//! winner is then re-traced and re-scored at full resolution, so every number
//! reported is measured on the real image.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::color::{LabCache, Rgba8};
use crate::config::{
    PaletteMethod, Preset, SegmentMode, VectorizeConfig, DEFAULT_TRANSPARENT_TOLERANCE,
};
use crate::raster::Raster;
use crate::score::{score, ScoreConfig, ScoreReport};
use crate::segment::scoring_reference;
use crate::vectorize::{vectorize_with_cache, TraceStats};

/// Cheap measurements that decide which settings are worth trying.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ImageProfile {
    pub width: u32,
    pub height: u32,
    /// Distinct colours at 5 bits per channel.
    pub unique_colors: usize,
    pub has_alpha: bool,
    /// Fraction of pixels sitting on a strong gradient.
    pub edge_density: f64,
    /// Of the pixels on an edge, the fraction that are genuine blends of their
    /// two neighbours. Near zero means hard pixel edges -- pixel art, a
    /// screenshot, or an already-posterised image. High means anti-aliased
    /// artwork, where sub-pixel reconstruction has real data to work with.
    pub antialiasing: f64,
    /// Fraction of pixels whose 3x3 neighbourhood is perfectly uniform.
    pub flatness: f64,
    pub photographic: bool,
    pub pixel_art: bool,
    pub bilevel: bool,
}

/// Profile an image. Deliberately cheap: this runs before any tracing.
pub fn profile(img: &Raster) -> ImageProfile {
    let w = img.width;
    let h = img.height;
    let mut p = ImageProfile {
        width: w,
        height: h,
        has_alpha: img.has_alpha(),
        ..Default::default()
    };
    if w == 0 || h == 0 {
        return p;
    }

    // Sample rather than scan when the image is large; these are statistics,
    // not exact counts.
    let total = (w as usize) * (h as usize);
    let stride = (total / 200_000).max(1);

    let mut seen: HashSet<u32> = HashSet::new();
    let mut i = 0usize;
    while i < total {
        let c = img.data[i];
        let key = (((c[0] * 31.0) as u32) << 10)
            | (((c[1] * 31.0) as u32) << 5)
            | ((c[2] * 31.0) as u32);
        seen.insert(key);
        i += stride;
    }
    p.unique_colors = seen.len();

    let mut edge_px = 0usize;
    let mut blend_px = 0usize;
    let mut flat_px = 0usize;
    let mut sampled = 0usize;

    let luma = |c: [f32; 4]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

    let ystep = ((h as usize / 400).max(1)) as u32;
    for y in (1..h.saturating_sub(1)).step_by(ystep as usize) {
        for x in 1..w.saturating_sub(1) {
            sampled += 1;
            let c = img.get(x, y);
            let l = img.get(x - 1, y);
            let r = img.get(x + 1, y);
            let u = img.get(x, y - 1);
            let d = img.get(x, y + 1);

            let gx = (luma(r) - luma(l)).abs();
            let gy = (luma(d) - luma(u)).abs();
            let grad = gx.max(gy);

            if grad < 0.004 {
                // Uniform enough to call flat.
                let same = |a: [f32; 4], b: [f32; 4]| {
                    (a[0] - b[0]).abs() < 0.004
                        && (a[1] - b[1]).abs() < 0.004
                        && (a[2] - b[2]).abs() < 0.004
                };
                if same(c, l) && same(c, r) && same(c, u) && same(c, d) {
                    flat_px += 1;
                }
            }

            if grad > 0.12 {
                edge_px += 1;
                // Is this pixel a blend of the two flanking colours, or equal to
                // one of them? Take whichever axis carries the stronger
                // gradient.
                let (a, b) = if gx >= gy { (l, r) } else { (u, d) };
                let mut num = 0.0f32;
                let mut den = 0.0f32;
                let mut resid = 0.0f32;
                for k in 0..3 {
                    let diff = b[k] - a[k];
                    num += (c[k] - a[k]) * diff;
                    den += diff * diff;
                }
                if den > 1e-6 {
                    let t = num / den;
                    for k in 0..3 {
                        let e = c[k] - (a[k] + (b[k] - a[k]) * t);
                        resid += e * e;
                    }
                    // Strictly between the two, and actually on the line
                    // joining them.
                    if (0.12..=0.88).contains(&t) && resid.sqrt() < 0.08 {
                        blend_px += 1;
                    }
                }
            }
        }
    }

    let sampled = sampled.max(1);
    p.edge_density = edge_px as f64 / sampled as f64;
    p.flatness = flat_px as f64 / sampled as f64;
    p.antialiasing = if edge_px == 0 {
        0.0
    } else {
        blend_px as f64 / edge_px as f64
    };

    // "Photographic" here means continuous-tone: no flat areas to segment into
    // regions, so a large palette is needed to avoid banding. Colour count
    // alone is a poor test -- a smooth two-hue gradient has only a few hundred
    // distinct colours yet is every bit as continuous as a photograph, while a
    // noisy flat-colour illustration has many colours and is not. Requiring
    // both a broad palette and an absence of flat areas separates them.
    p.photographic = p.unique_colors > 256 && p.flatness < 0.25;
    // Hard edges plus a small palette plus large flat areas: no anti-aliasing
    // to invert, and every pixel edge is intentional.
    //
    // The cut sits at 0.15 because a correctly anti-aliased edge only ever has
    // about one blended pixel for every two saturated ones either side of it,
    // so even clean artwork scores around 0.25-0.4 rather than near 1. Hard
    // pixel edges score essentially zero, leaving a wide margin.
    p.pixel_art = p.antialiasing < 0.15 && p.unique_colors <= 256 && p.flatness > 0.45;
    p.bilevel = p.unique_colors <= 8;
    p
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoConfig {
    pub score: ScoreConfig,
    /// Seed candidates evaluated in the first stage.
    pub max_seeds: usize,
    /// How many of the best seeds get refined.
    pub refine_top: usize,
    /// Coordinate-descent sweeps over the parameter axes.
    pub refine_rounds: usize,
    /// Run the search on a copy no larger than this on its longest side. The
    /// winner is always re-measured at full resolution.
    pub search_max_dimension: Option<u32>,
    /// Hard cap on control points. `None` derives one from the image size; see
    /// [`derive_point_budget`].
    pub point_budget: Option<usize>,
    /// Colours to leave unpainted in every candidate; see
    /// [`SegmentConfig::transparent_colors`](crate::config::SegmentConfig::transparent_colors).
    /// This is a requirement of the output rather than a setting to search
    /// over, so it is stamped onto every candidate and each one is scored
    /// against an image with those colours removed.
    #[serde(default)]
    pub transparent_colors: Vec<Rgba8>,
    #[serde(default = "default_transparent_tolerance")]
    pub transparent_tolerance: f32,
}

fn default_transparent_tolerance() -> f32 {
    DEFAULT_TRANSPARENT_TOLERANCE
}

impl Default for AutoConfig {
    fn default() -> Self {
        AutoConfig {
            score: ScoreConfig::default(),
            max_seeds: 12,
            refine_top: 2,
            refine_rounds: 2,
            search_max_dimension: Some(640),
            point_budget: None,
            transparent_colors: Vec::new(),
            transparent_tolerance: DEFAULT_TRANSPARENT_TOLERANCE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub label: String,
    pub config: VectorizeConfig,
    pub report: ScoreReport,
    pub stats: TraceStats,
}

impl Candidate {
    pub fn points(&self) -> usize {
        self.report.complexity.total_points()
    }
}

#[derive(Clone, Debug)]
pub struct AutoResult {
    pub profile: ImageProfile,
    /// Every candidate tried, best first.
    pub candidates: Vec<Candidate>,
    pub elapsed_ms: f64,
    /// True when the search ran on a downscaled copy.
    pub searched_downscaled: bool,
}

impl AutoResult {
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.first()
    }
}

/// A generous but finite cap on control points for an image of this size.
///
/// Some images -- anything noisy, and anything with a smooth gradient -- simply
/// cannot be matched by flat vector regions. Left alone, a search that ranks on
/// accuracy will happily spend a hundred thousand control points chasing
/// individual noise pixels, because each one does nudge the score up. The
/// result is a file nobody wants: enormous, unopenable in an editor, and no
/// better looking.
///
/// Beyond this budget a result has stopped being a vectorisation, so it is
/// ranked below every result that stayed within it. The cap grows with the
/// square root of the pixel count, because detail worth tracing scales with the
/// linear size of the image rather than its area.
pub fn derive_point_budget(width: u32, height: u32) -> usize {
    let px = (width as f64) * (height as f64);
    (24.0 * px.sqrt() + 1000.0) as usize
}

/// How many percentage points of match each doubling of control points must
/// earn to be worth taking, when the fidelity target is out of reach.
///
/// Tuned against the benchmark. At 0.25 the search still refused absurd trades
/// like thirty-five times the points for a quarter of a point of match, but
/// took milder ones it should not have: on a flat-colour logo it picked
/// 28,000 points over 1,900 for under one point of match, because fifteen
/// times the points cost it only about a point of score. At 0.5 a few hundred
/// points for half a point of match on clean artwork is still worth it, and
/// that trade is not.
const POINT_DISCOUNT: f64 = 0.5;

/// Accuracy discounted by what it cost to buy.
///
/// Used only when nothing reaches the target. A plain "closest match wins"
/// rule has no sense of proportion: faced with an image it cannot match, it
/// takes every trade that nudges the score, however many control points that
/// costs. The logarithm is the right shape because control-point cost is felt
/// multiplicatively -- going from 40 points to 80 is a real change in how
/// editable a result is, while 4000 to 4040 is not.
fn discounted_match(c: &Candidate, reference: f64) -> f64 {
    let ratio = c.points() as f64 / reference.max(1.0);
    c.report.match_pct - POINT_DISCOUNT * (1.0 + ratio).log2()
}

/// The ranking rule, stated once.
///
/// Fidelity is a gate rather than a tradeable quantity: a result a viewer can
/// see is wrong does not become acceptable by being compact. Among results that
/// clear the gate, fewest control points wins, because every remaining
/// difference is invisible and only editability and file size are left to
/// distinguish them.
///
/// The point budget is applied first, as a separate gate. Its job is to stop
/// the fallback branch -- used when *nothing* reaches the target -- from
/// rewarding a candidate that bought a couple of points of accuracy with tens
/// of thousands of control points.
fn better(a: &Candidate, b: &Candidate, target: f64, budget: usize) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let a_ok = a.points() <= budget;
    let b_ok = b.points() <= budget;
    match (a_ok, b_ok) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        // Both over budget: fall through and rank them normally, so that a
        // pathological image still yields the least-bad answer rather than an
        // arbitrary one.
        _ => {}
    }

    let am = a.report.meets(target);
    let bm = b.report.meets(target);
    match (am, bm) {
        (true, true) => a
            .points()
            .cmp(&b.points())
            // Tie-break toward the more accurate result.
            .then(
                b.report
                    .match_pct
                    .partial_cmp(&a.report.match_pct)
                    .unwrap_or(Ordering::Equal),
            ),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => {
            // Nothing can reach the target, so accuracy is discounted by what
            // it cost rather than pursued at any price.
            let reference = (budget as f64 / 32.0).max(32.0);
            discounted_match(b, reference)
                .partial_cmp(&discounted_match(a, reference))
                .unwrap_or(Ordering::Equal)
                .then(a.points().cmp(&b.points()))
        }
    }
}

/// Candidate configurations to try first, ordered by how likely the profile
/// says they are to suit the image.
fn seed_configs(p: &ImageProfile, max: usize) -> Vec<(String, VectorizeConfig)> {
    let mut out: Vec<(String, VectorizeConfig)> = Vec::new();

    let push = |label: &str, cfg: VectorizeConfig, out: &mut Vec<(String, VectorizeConfig)>| {
        out.push((label.to_string(), cfg));
    };

    if p.pixel_art {
        push("pixel-art", Preset::PixelArt.config(), &mut out);
    }

    if p.bilevel {
        push("black-and-white", Preset::BlackAndWhite.config(), &mut out);
    }

    // A spread of palette sizes is the single most important axis, so it is
    // sampled broadly rather than guessed at.
    let color_steps: &[usize] = if p.photographic {
        &[16, 24, 32, 48, 64]
    } else if p.unique_colors < 32 {
        &[2, 3, 4, 6, 8, 12]
    } else {
        &[4, 6, 8, 12, 16, 24, 32]
    };

    for &k in color_steps {
        let mut cfg = if p.photographic {
            Preset::Photo.config()
        } else if k <= 6 {
            Preset::Logo.config()
        } else if k <= 16 {
            Preset::Clipart.config()
        } else {
            Preset::Artwork.config()
        };
        cfg.segment.mode = SegmentMode::Palette;
        cfg.segment.colors = k;
        push(&format!("palette-{k}"), cfg, &mut out);
    }

    if !p.bilevel && !p.photographic {
        push("black-and-white", Preset::BlackAndWhite.config(), &mut out);
    }

    // Applied once, to every seed, rather than inside one branch: on hard-edged
    // input there is no anti-aliasing to invert and no noise to smooth, so
    // every candidate should be tracing the pixel grid exactly.
    if p.pixel_art {
        for (_, cfg) in out.iter_mut() {
            cfg.subpixel.enabled = false;
            cfg.corners.smoothing = 0.0;
            cfg.corners.noise_adaptive = false;
            cfg.fit.tolerance = cfg.fit.tolerance.min(0.1);
        }
    }

    out.truncate(max.max(1));
    out
}

/// Loosest curve tolerance, in pixels, that refinement will try.
const MAX_TOLERANCE: f64 = 3.0;

/// Parameter neighbourhoods explored during refinement, most valuable first.
fn refinement_axes(cfg: &VectorizeConfig) -> Vec<(&'static str, Vec<VectorizeConfig>)> {
    let mut axes: Vec<(&'static str, Vec<VectorizeConfig>)> = Vec::new();

    // Tolerance trades points against accuracy more directly than anything
    // else, so it is swept first and most widely.
    //
    // The sweep has to reach well past a pixel. Hard-edged input has one-pixel
    // stair-steps, and while the tolerance is below roughly 0.7px the fitter
    // must give every step its own line segment: a heart-and-leaf logo took
    // 3,200 points at 0.1px, 815 at 0.5px and 177 at 1.0px, for the same match.
    // Presets start at 0.3-0.5px, so multipliers stopping near 2x never got
    // there.
    let tol = cfg.fit.tolerance;
    let mut tolerances: Vec<f64> = Vec::new();
    for m in [0.35, 0.6, 0.85, 1.3, 1.8, 2.5, 3.5, 5.0] {
        let t = (tol * m).clamp(0.02, MAX_TOLERANCE);
        // Large multipliers all clamp to the same ceiling; trace it once.
        if !tolerances.iter().any(|&u| (u - t).abs() < 1e-9) {
            tolerances.push(t);
        }
    }
    axes.push((
        "tolerance",
        tolerances
            .into_iter()
            .map(|t| {
                let mut c = cfg.clone();
                c.fit.tolerance = t;
                c
            })
            .collect(),
    ));

    if cfg.segment.mode == SegmentMode::Palette {
        let k = cfg.segment.colors;
        axes.push((
            "colors",
            [-16i64, -8, -4, -2, -1, 1, 2, 4, 8, 16]
                .iter()
                .filter_map(|d| {
                    let nk = k as i64 + d;
                    if nk < 2 || nk > 128 {
                        return None;
                    }
                    let mut c = cfg.clone();
                    c.segment.colors = nk as usize;
                    Some(c)
                })
                .collect(),
        ));
        axes.push((
            "palette-method",
            vec![{
                let mut c = cfg.clone();
                c.segment.palette_method = match cfg.segment.palette_method {
                    PaletteMethod::KMeans => PaletteMethod::MedianCut,
                    PaletteMethod::MedianCut => PaletteMethod::KMeans,
                };
                c
            }],
        ));
        axes.push((
            "recolor",
            vec![{
                let mut c = cfg.clone();
                c.output.recolor_regions = !cfg.output.recolor_regions;
                c
            }],
        ));
    }

    axes.push((
        "smoothing",
        [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|s| {
                let mut c = cfg.clone();
                c.corners.smoothing = *s;
                c
            })
            .collect(),
    ));

    axes.push((
        "corner-threshold",
        [35.0, 50.0, 65.0, 80.0]
            .iter()
            .map(|t| {
                let mut c = cfg.clone();
                c.corners.threshold_deg = *t;
                c
            })
            .collect(),
    ));

    axes.push((
        "arcs",
        vec![
            {
                let mut c = cfg.clone();
                c.fit.allow_arcs = true;
                c.fit.emit_true_arcs = true;
                c
            },
            {
                let mut c = cfg.clone();
                c.fit.allow_arcs = true;
                c.fit.emit_true_arcs = false;
                c
            },
            {
                let mut c = cfg.clone();
                c.fit.allow_arcs = false;
                c
            },
        ],
    ));

    axes.push((
        "despeckle",
        [0u32, 2, 6, 12, 24]
            .iter()
            .map(|d| {
                let mut c = cfg.clone();
                c.segment.despeckle_area = *d;
                c
            })
            .collect(),
    ));

    axes
}

/// Trace and score one configuration.
fn evaluate(
    img: &Raster,
    label: &str,
    cfg: &VectorizeConfig,
    score_cfg: &ScoreConfig,
    cache: &LabCache,
) -> Option<Candidate> {
    let traced = vectorize_with_cache(img, cfg, cache);
    let reference = scoring_reference(img, &cfg.segment, cache);
    let report = score(&reference, &traced.image, score_cfg).ok()?;
    Some(Candidate {
        label: label.to_string(),
        config: cfg.clone(),
        report,
        stats: traced.stats,
    })
}

/// Run the full search. `on_progress` is called as `(done, total)` and may be
/// invoked from several threads.
pub fn auto_select(
    img: &Raster,
    cfg: &AutoConfig,
    on_progress: &(dyn Fn(usize, usize) + Sync),
) -> AutoResult {
    let start = Instant::now();
    let prof = profile(img);

    let search_img = match cfg.search_max_dimension {
        Some(d) => img.fit_within(d),
        None => img.clone(),
    };
    let searched_downscaled =
        search_img.width != img.width || search_img.height != img.height;

    let mut seeds = seed_configs(&prof, cfg.max_seeds);
    // Refinement clones from these, so stamping the seeds is enough for every
    // candidate the search ever produces to carry the keyed colours.
    for (_, c) in seeds.iter_mut() {
        c.segment.transparent_colors = cfg.transparent_colors.clone();
        c.segment.transparent_tolerance = cfg.transparent_tolerance;
    }
    // Rough total for the progress bar: seeds, plus each refined candidate's
    // axis sweep. It is an estimate, and the reporter clamps to it.
    let est_total = seeds.len() + cfg.refine_top * cfg.refine_rounds * 12;
    let done = AtomicUsize::new(0);
    let tick = || {
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        on_progress(n.min(est_total), est_total);
    };

    let cache = LabCache::new();
    let mut candidates: Vec<Candidate> = seeds
        .par_iter()
        .filter_map(|(label, c)| {
            let r = evaluate(&search_img, label, c, &cfg.score, &cache);
            tick();
            r
        })
        .collect();

    if candidates.is_empty() {
        return AutoResult {
            profile: prof,
            candidates,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            searched_downscaled,
        };
    }

    let target = cfg.score.target_match;
    let budget = cfg
        .point_budget
        .unwrap_or_else(|| derive_point_budget(img.width, img.height));
    candidates.sort_by(|a, b| better(a, b, target, budget));

    // --- Refinement: coordinate descent from each of the best few seeds.
    let mut refined: Vec<Candidate> = Vec::new();
    let starting: Vec<Candidate> = candidates
        .iter()
        .take(cfg.refine_top.max(1))
        .cloned()
        .collect();

    for seed in starting {
        let mut current = seed.clone();
        for _ in 0..cfg.refine_rounds.max(1) {
            let mut improved_this_round = false;
            for (axis, variants) in refinement_axes(&current.config) {
                let label = format!("{}+{}", seed.label, axis);
                let mut results: Vec<Candidate> = variants
                    .par_iter()
                    .filter_map(|c| {
                        let r = evaluate(&search_img, &label, c, &cfg.score, &cache);
                        tick();
                        r
                    })
                    .collect();
                refined.append(&mut results.clone());
                results.sort_by(|a, b| better(a, b, target, budget));
                if let Some(best) = results.first() {
                    if better(best, &current, target, budget) == std::cmp::Ordering::Less {
                        current = best.clone();
                        improved_this_round = true;
                    }
                }
            }
            if !improved_this_round {
                break;
            }
        }
        refined.push(current);
    }

    candidates.append(&mut refined);
    // Identical configurations can be reached by different paths; keep one of
    // each so the report is not padded with duplicates.
    candidates.sort_by(|a, b| better(a, b, target, budget));
    candidates.dedup_by(|a, b| {
        a.points() == b.points()
            && (a.report.match_pct - b.report.match_pct).abs() < 1e-9
            && a.config.segment.colors == b.config.segment.colors
            && (a.config.fit.tolerance - b.config.fit.tolerance).abs() < 1e-12
    });

    // Re-measure at full resolution: numbers chosen on a downscaled proxy are
    // fine for ranking but must not be reported as if they described the real
    // image.
    if searched_downscaled {
        let rescored: Vec<Candidate> = candidates
            .par_iter()
            .take(8)
            .filter_map(|c| evaluate(img, &c.label, &c.config, &cfg.score, &cache))
            .collect();
        if !rescored.is_empty() {
            candidates = rescored;
            candidates.sort_by(|a, b| better(a, b, target, budget));
        }
    }

    AutoResult {
        profile: prof,
        candidates,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        searched_downscaled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Raster {
        let mut r = Raster::new(w, h);
        for y in 0..h {
            for x in 0..w {
                r.set(x, y, c);
            }
        }
        r
    }

    /// Hard-edged checkerboard blocks: the signature of pixel art.
    fn pixel_art(block: u32) -> Raster {
        let mut img = Raster::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let on = ((x / block) + (y / block)) % 2 == 0;
                let v = if on { 0.15 } else { 0.85 };
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        img
    }

    /// A disc with analytically correct anti-aliasing.
    fn aa_disc() -> Raster {
        let mut img = solid(64, 64, [1.0, 1.0, 1.0, 1.0]);
        for y in 0..64 {
            for x in 0..64 {
                let d = ((x as f64 + 0.5 - 32.0).powi(2) + (y as f64 + 0.5 - 32.0).powi(2)).sqrt();
                let cov = ((20.0 - d) + 0.5).clamp(0.0, 1.0) as f32;
                img.set(x, y, [1.0 - cov * 0.9, 1.0 - cov * 0.8, 1.0 - cov * 0.2, 1.0]);
            }
        }
        img
    }

    #[test]
    fn profile_spots_pixel_art() {
        let p = profile(&pixel_art(8));
        assert!(p.pixel_art, "{p:?}");
        assert!(p.antialiasing < 0.08, "aa {}", p.antialiasing);
        assert!(!p.photographic);
    }

    #[test]
    fn profile_spots_antialiasing() {
        let p = profile(&aa_disc());
        assert!(
            p.antialiasing > 0.2,
            "anti-aliased disc scored aa {}",
            p.antialiasing
        );
        assert!(!p.pixel_art, "{p:?}");
    }

    #[test]
    fn profile_spots_a_photograph() {
        // Smooth noise in all three channels: many colours, nothing flat.
        let mut img = Raster::new(96, 96);
        for y in 0..96 {
            for x in 0..96 {
                let fx = x as f32 / 96.0;
                let fy = y as f32 / 96.0;
                img.set(
                    x,
                    y,
                    [
                        (fx * 6.0).sin() * 0.5 + 0.5,
                        (fy * 7.3 + 1.0).sin() * 0.5 + 0.5,
                        ((fx + fy) * 5.1).cos() * 0.5 + 0.5,
                        1.0,
                    ],
                );
            }
        }
        let p = profile(&img);
        assert!(p.photographic, "{p:?}");
        assert!(!p.bilevel);
    }

    #[test]
    fn continuous_tone_is_told_apart_from_noisy_flat_art() {
        // A smooth gradient has few distinct colours but nothing flat, and
        // needs a large palette. Noisy flat-colour art has plenty of colours
        // and also nothing flat, but a large palette would only chase noise.
        // Colour count alone cannot separate them.
        let mut gradient = Raster::new(96, 96);
        for y in 0..96u32 {
            for x in 0..96u32 {
                let fx = x as f32 / 95.0;
                let fy = y as f32 / 95.0;
                gradient.set(x, y, [0.15 + 0.7 * fx, 0.25 + 0.5 * fy, 0.85 - 0.5 * fx, 1.0]);
            }
        }
        let gp = profile(&gradient);
        assert!(gp.photographic, "smooth gradient not seen as continuous tone: {gp:?}");

        let mut noisy = solid(96, 96, [0.85, 0.85, 0.85, 1.0]);
        for y in 20..70 {
            for x in 20..70 {
                noisy.set(x, y, [0.2, 0.3, 0.7, 1.0]);
            }
        }
        let noisy = crate::synth::add_noise(&noisy, 0.04, 99);
        let np = profile(&noisy);
        assert!(
            !np.photographic,
            "noisy flat-colour art was mistaken for continuous tone: {np:?}"
        );
    }

    #[test]
    fn profile_spots_bilevel_art() {
        let mut img = solid(64, 64, [1.0, 1.0, 1.0, 1.0]);
        for y in 10..50 {
            for x in 10..30 {
                img.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        let p = profile(&img);
        assert!(p.bilevel, "{p:?}");
    }

    #[test]
    fn ranking_gates_on_fidelity_then_minimises_points() {
        let mk = |match_pct: f64, points: usize| {
            let mut c = Candidate {
                label: String::new(),
                config: VectorizeConfig::default(),
                report: ScoreReport {
                    match_pct,
                    ..Default::default()
                },
                stats: TraceStats::default(),
            };
            c.report.complexity.anchors = points;
            c
        };
        let target = 99.0;
        let budget = 100_000;

        // Both clear the gate: fewer points wins even though it is less exact.
        let lean = mk(99.1, 50);
        let fat = mk(99.9, 500);
        assert_eq!(better(&lean, &fat, target, budget), std::cmp::Ordering::Less);

        // Only one clears the gate: it wins regardless of cost.
        let accurate = mk(99.2, 5000);
        let cheap = mk(98.9, 10);
        assert_eq!(better(&accurate, &cheap, target, budget), std::cmp::Ordering::Less);

        // Neither clears it: closest to the target wins.
        let near = mk(98.5, 900);
        let far = mk(90.0, 10);
        assert_eq!(better(&near, &far, target, budget), std::cmp::Ordering::Less);
    }

    #[test]
    fn point_budget_stops_the_search_chasing_noise() {
        let mk = |match_pct: f64, points: usize| {
            let mut c = Candidate {
                label: String::new(),
                config: VectorizeConfig::default(),
                report: ScoreReport {
                    match_pct,
                    ..Default::default()
                },
                stats: TraceStats::default(),
            };
            c.report.complexity.anchors = points;
            c
        };
        let target = 99.0;
        let budget = 5_000;

        // On an image nothing can match, a slightly better score bought with an
        // absurd number of points must lose. Without the budget this is exactly
        // the trade the fallback branch used to make.
        let bloated = mk(66.4, 36_350);
        let sane = mk(58.7, 900);
        assert_eq!(better(&sane, &bloated, target, budget), std::cmp::Ordering::Less);

        // The budget never overrides the fidelity gate for in-budget results.
        let good = mk(99.5, 4_000);
        let small = mk(80.0, 100);
        assert_eq!(better(&good, &small, target, budget), std::cmp::Ordering::Less);

        // If everything is over budget, ranking still picks the least bad.
        let a = mk(70.0, 50_000);
        let b = mk(60.0, 60_000);
        assert_eq!(better(&a, &b, target, budget), std::cmp::Ordering::Less);
    }

    #[test]
    fn fifteen_times_the_points_is_not_worth_under_a_point_of_match() {
        // Real numbers from a flat-colour logo where the target was out of
        // reach. All three are inside the budget, so only the fallback ranking
        // is in play, and it used to pick the 28,058-point trace.
        let mk = |match_pct: f64, points: usize| {
            let mut c = Candidate {
                label: String::new(),
                config: VectorizeConfig::default(),
                report: ScoreReport {
                    match_pct,
                    ..Default::default()
                },
                stats: TraceStats::default(),
            };
            c.report.complexity.anchors = points;
            c
        };
        let target = 99.0;
        let budget = derive_point_budget(1254, 1254);

        let lean = mk(96.52, 1_934);
        let middling = mk(97.09, 17_984);
        let bloated = mk(97.45, 28_058);
        assert!(bloated.points() <= budget);
        for other in [&middling, &bloated] {
            assert_eq!(
                better(&lean, other, target, budget),
                std::cmp::Ordering::Less,
                "{} points beat {}",
                other.points(),
                lean.points()
            );
        }
    }

    #[test]
    fn refinement_tolerances_reach_past_a_pixel_without_repeats() {
        // Presets start below a pixel, and hard-edged art only collapses its
        // stair-steps into a few segments once tolerance clears one.
        let cfg = Preset::Logo.config();
        let axes = refinement_axes(&cfg);
        let (_, variants) = axes.iter().find(|(n, _)| *n == "tolerance").unwrap();
        let tols: Vec<f64> = variants.iter().map(|c| c.fit.tolerance).collect();
        assert!(
            tols.iter().any(|&t| t >= 1.0),
            "tolerance sweep tops out at {:?}",
            tols
        );
        assert!(tols.iter().all(|&t| t <= MAX_TOLERANCE));
        for (i, a) in tols.iter().enumerate() {
            for b in &tols[i + 1..] {
                assert!((a - b).abs() > 1e-9, "duplicate tolerance in {tols:?}");
            }
        }

        // A start already near the ceiling must not waste evaluations on
        // several copies of it.
        let mut hi = cfg.clone();
        hi.fit.tolerance = 2.5;
        let axes = refinement_axes(&hi);
        let (_, variants) = axes.iter().find(|(n, _)| *n == "tolerance").unwrap();
        let ceiling = variants
            .iter()
            .filter(|c| (c.fit.tolerance - MAX_TOLERANCE).abs() < 1e-9)
            .count();
        assert_eq!(ceiling, 1);
    }

    #[test]
    fn budget_scales_with_image_size() {
        let small = derive_point_budget(100, 100);
        let large = derive_point_budget(2000, 2000);
        assert!(small > 1000, "budget {small} too tight for a small image");
        assert!(large > small * 4, "{large} vs {small}");
        // A 180x140 image must allow a normal trace but not tens of thousands
        // of points.
        let b = derive_point_budget(180, 140);
        assert!((2_000..12_000).contains(&b), "budget {b}");
    }

    #[test]
    fn seeds_respond_to_the_profile() {
        let photo = ImageProfile {
            photographic: true,
            unique_colors: 50_000,
            ..Default::default()
        };
        let seeds = seed_configs(&photo, 12);
        assert!(seeds.iter().all(|(l, _)| l != "black-and-white"));
        assert!(seeds.iter().any(|(_, c)| c.segment.colors >= 32));

        let art = ImageProfile {
            pixel_art: true,
            unique_colors: 16,
            bilevel: false,
            ..Default::default()
        };
        let seeds = seed_configs(&art, 12);
        assert_eq!(seeds[0].0, "pixel-art");
        assert!(seeds.iter().all(|(_, c)| !c.subpixel.enabled));
    }

    #[test]
    fn auto_select_finds_a_good_config_for_a_simple_logo() {
        let mut img = solid(96, 96, [1.0, 1.0, 1.0, 1.0]);
        for y in 20..76 {
            for x in 20..76 {
                img.set(x, y, [0.85, 0.15, 0.2, 1.0]);
            }
        }
        for y in 34..62 {
            for x in 34..62 {
                img.set(x, y, [0.15, 0.3, 0.8, 1.0]);
            }
        }
        let cfg = AutoConfig {
            max_seeds: 5,
            refine_top: 1,
            refine_rounds: 1,
            search_max_dimension: None,
            ..Default::default()
        };
        let res = auto_select(&img, &cfg, &|_, _| {});
        let best = res.best().expect("a candidate");
        assert!(
            best.report.match_pct >= 99.0,
            "best only reached {:.2}% ({} points)",
            best.report.match_pct,
            best.points()
        );
        // Three nested rectangles is a handful of corners, not hundreds.
        assert!(
            best.points() < 80,
            "best used {} points: {:?}",
            best.points(),
            best.report.complexity
        );
        // Ranking must actually be applied.
        let budget = derive_point_budget(img.width, img.height);
        for w in res.candidates.windows(2) {
            assert_ne!(better(&w[1], &w[0], 99.0, budget), std::cmp::Ordering::Less);
        }
    }

    #[test]
    fn progress_is_reported() {
        let img = solid(48, 48, [0.4, 0.6, 0.2, 1.0]);
        let seen = std::sync::Mutex::new(Vec::new());
        let cfg = AutoConfig {
            max_seeds: 3,
            refine_top: 1,
            refine_rounds: 1,
            search_max_dimension: None,
            ..Default::default()
        };
        auto_select(&img, &cfg, &|done, total| {
            seen.lock().unwrap().push((done, total));
        });
        let seen = seen.into_inner().unwrap();
        assert!(!seen.is_empty(), "no progress callbacks");
        assert!(seen.iter().all(|(d, t)| d <= t));
    }

    #[test]
    fn keyed_colours_reach_every_candidate_and_are_not_scored_as_errors() {
        // Key out black, which the scorer's white background would otherwise
        // count as a total mismatch for every pixel the trace leaves empty.
        let mut img = solid(64, 64, [0.0, 0.0, 0.0, 1.0]);
        for y in 16..48 {
            for x in 16..48 {
                img.set(x, y, [0.9, 0.7, 0.1, 1.0]);
            }
        }
        let black = Rgba8::opaque(0, 0, 0);
        let cfg = AutoConfig {
            max_seeds: 4,
            refine_top: 1,
            refine_rounds: 1,
            search_max_dimension: None,
            transparent_colors: vec![black],
            ..Default::default()
        };
        let res = auto_select(&img, &cfg, &|_, _| {});
        assert!(!res.candidates.is_empty());
        for c in &res.candidates {
            assert_eq!(
                c.config.segment.transparent_colors,
                vec![black],
                "candidate {:?} lost the keyed colour",
                c.label
            );
        }
        let best = res.best().unwrap();
        assert!(
            best.report.match_pct >= 99.0,
            "a correctly keyed trace should match its reference; got {:.2}%",
            best.report.match_pct
        );
        assert_eq!(best.report.complexity.shapes, 1, "only the yellow square is painted");
    }
}
