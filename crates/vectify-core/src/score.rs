//! Round-trip scoring: raster -> vector -> raster, then measure.
//!
//! A tracer can only be judged by what it actually produces, so scoring here
//! goes all the way out to a real SVG file and renders it with a real SVG
//! renderer. Nothing is graded against the tracer's own internal contours,
//! which would let a fit mark its own homework.
//!
//! The headline metric is deliberately **not** RMSE. RMSE answers "how
//! different are these numbers", when the question a user is actually asking is
//! "can I see the difference". Those come apart badly: a one-level shift across
//! a large flat area is invisible but moves RMSE a lot, while a hard edge
//! displaced by a pixel is glaring and moves it very little.
//!
//! So the headline is the **percentage of pixels within a just-noticeable
//! colour difference**, measured with CIEDE2000. A "99% match" then means
//! something concrete and checkable: 99 in 100 pixels are perceptually
//! indistinguishable from the original. RMSE, PSNR and SSIM are reported
//! alongside, because they are the numbers people expect to see and they are
//! genuinely useful for spotting different failure modes.

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::color::{delta_e2000, srgb_to_lab};
use crate::config::{OutputConfig, VectorFormat};
use crate::export;
use crate::model::{Complexity, VectorImage};
use crate::raster::Raster;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreConfig {
    /// CIEDE2000 difference at or below which a pixel counts as matching.
    /// 1.0 is the classic just-noticeable-difference; 2.0 is a common
    /// "acceptable in practice" bound and is the default here.
    pub delta_e_threshold: f32,
    /// Render at this multiple of the native resolution and box-filter down.
    /// 1 compares like for like and is the honest default.
    pub supersample: u32,
    /// Background that both images are composited over before comparison, so
    /// transparency is judged consistently rather than ignored.
    pub background: [f32; 3],
    /// Match percentage a result must reach to count as acceptable.
    pub target_match: f64,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        ScoreConfig {
            delta_e_threshold: 2.0,
            supersample: 1,
            background: [1.0, 1.0, 1.0],
            target_match: 99.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ScoreReport {
    /// Percentage of pixels within the perceptual threshold. The headline.
    pub match_pct: f64,
    pub mean_delta_e: f64,
    pub p95_delta_e: f64,
    pub max_delta_e: f64,
    /// Root mean squared error over RGB, on a 0-255 scale.
    pub rmse: f64,
    /// Peak signal-to-noise ratio in dB, capped at 99 for an exact match.
    pub psnr: f64,
    /// Structural similarity on luma, in 0..1.
    pub ssim: f64,
    pub complexity: Complexity,
    /// Size of the serialised vector file in bytes.
    pub output_bytes: usize,
    pub render_ms: f64,
    pub compare_ms: f64,
}

impl ScoreReport {
    pub fn meets(&self, target: f64) -> bool {
        self.match_pct >= target
    }

    /// A single number in 0..100 for ranking and display.
    ///
    /// Fidelity decays exponentially in the *visible error rate* relative to
    /// the target, so falling short of the target is punished sharply while
    /// exceeding it yields only gentle further gain. Compactness is a
    /// saturating function of point count measured against a reference, so
    /// halving the points helps but never at the expense of a visibly worse
    /// image.
    pub fn composite(&self, target: f64, reference_points: f64) -> f64 {
        let miss = (100.0 - self.match_pct).max(0.0) / 100.0;
        let miss_target = ((100.0 - target).max(0.01)) / 100.0;
        let fidelity = (-miss / miss_target).exp();
        let r = reference_points.max(1.0);
        let compactness = r / (r + self.complexity.total_points() as f64);
        100.0 * fidelity * compactness
    }
}

/// Render a vector document to a raster, through a real SVG renderer.
pub fn render(image: &VectorImage, supersample: u32) -> Result<Raster> {
    let ss = supersample.max(1);
    let out_cfg = OutputConfig {
        format: VectorFormat::Svg,
        // Render from full-precision geometry: the exported file's own rounding
        // is already baked in by the pipeline, and re-rounding here would be
        // measuring the wrong thing.
        precision: 6,
        ..Default::default()
    };
    let svg = export::svg::write(image, &out_cfg);
    rasterize_svg(&svg, ss)
}

/// Rasterise SVG source at `supersample` times its declared size, then reduce.
pub fn rasterize_svg(svg: &str, supersample: u32) -> Result<Raster> {
    let ss = supersample.max(1);
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(svg.as_bytes(), &opt).context("parsing generated SVG")?;
    let size = tree.size();
    let w = ((size.width() * ss as f32).round() as u32).max(1);
    let h = ((size.height() * ss as f32).round() as u32).max(1);

    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| anyhow!("cannot allocate a {w}x{h} pixmap"))?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(ss as f32, ss as f32),
        &mut pixmap.as_mut(),
    );

    // tiny-skia stores premultiplied alpha; undo it to match Raster's straight
    // alpha convention.
    let mut out = Raster::new(w, h);
    for (i, px) in pixmap.pixels().iter().enumerate() {
        let c = px.demultiply();
        out.data[i] = [
            c.red() as f32 / 255.0,
            c.green() as f32 / 255.0,
            c.blue() as f32 / 255.0,
            c.alpha() as f32 / 255.0,
        ];
    }
    Ok(if ss > 1 { out.downsample(ss) } else { out })
}

fn gaussian_kernel(sigma: f64, radius: usize) -> Vec<f32> {
    let mut k: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-(x * x) / (2.0 * sigma * sigma)).exp() as f32
        })
        .collect();
    let sum: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

/// Separable Gaussian blur with edge clamping.
fn blur(src: &[f32], w: usize, h: usize, kernel: &[f32]) -> Vec<f32> {
    let r = (kernel.len() - 1) / 2;
    let mut tmp = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let sx = (x as isize + i as isize - r as isize).clamp(0, w as isize - 1) as usize;
                acc += src[y * w + sx] * k;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let sy = (y as isize + i as isize - r as isize).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * k;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Gaussian-windowed SSIM on luma, following Wang et al. with the standard
/// 11-tap sigma 1.5 window and C1/C2 constants.
pub fn ssim(a: &Raster, b: &Raster) -> f64 {
    let w = a.width.min(b.width) as usize;
    let h = a.height.min(b.height) as usize;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let luma = |r: &Raster| -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let c = r.get(x as u32, y as u32);
                v.push((0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * 255.0);
            }
        }
        v
    };
    let x = luma(a);
    let y = luma(b);

    let k = gaussian_kernel(1.5, 5);
    let mu_x = blur(&x, w, h, &k);
    let mu_y = blur(&y, w, h, &k);
    let xx: Vec<f32> = x.iter().map(|v| v * v).collect();
    let yy: Vec<f32> = y.iter().map(|v| v * v).collect();
    let xy: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
    let s_xx = blur(&xx, w, h, &k);
    let s_yy = blur(&yy, w, h, &k);
    let s_xy = blur(&xy, w, h, &k);

    const C1: f32 = 6.5025; // (0.01 * 255)^2
    const C2: f32 = 58.5225; // (0.03 * 255)^2

    let mut total = 0.0f64;
    for i in 0..w * h {
        let ux = mu_x[i];
        let uy = mu_y[i];
        let vx = s_xx[i] - ux * ux;
        let vy = s_yy[i] - uy * uy;
        let vxy = s_xy[i] - ux * uy;
        let num = (2.0 * ux * uy + C1) * (2.0 * vxy + C2);
        let den = (ux * ux + uy * uy + C1) * (vx + vy + C2);
        total += (num / den) as f64;
    }
    (total / (w * h) as f64).clamp(-1.0, 1.0)
}

/// Compare two rasters of equal size.
pub fn compare(original: &Raster, rendered: &Raster, cfg: &ScoreConfig) -> ScoreReport {
    let a = original.composite_over(cfg.background);
    let b = rendered.composite_over(cfg.background);
    let n = a.data.len().min(b.data.len());

    // CIEDE2000 per pixel is the expensive part of scoring; it parallelises
    // perfectly.
    let mut deltas: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = a.data[i];
            let q = b.data[i];
            delta_e2000(
                srgb_to_lab([p[0], p[1], p[2]]),
                srgb_to_lab([q[0], q[1], q[2]]),
            )
        })
        .collect();

    let matched = deltas
        .iter()
        .filter(|d| **d <= cfg.delta_e_threshold)
        .count();
    let mean_delta_e = deltas.iter().map(|d| *d as f64).sum::<f64>() / n.max(1) as f64;
    let max_delta_e = deltas.iter().fold(0.0f32, |m, d| m.max(*d)) as f64;
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let p95_delta_e = if n == 0 {
        0.0
    } else {
        deltas[((n as f64 * 0.95) as usize).min(n - 1)] as f64
    };

    let mut sq = 0.0f64;
    for i in 0..n {
        for c in 0..3 {
            let d = (a.data[i][c] - b.data[i][c]) as f64 * 255.0;
            sq += d * d;
        }
    }
    let rmse = (sq / (n.max(1) * 3) as f64).sqrt();
    let psnr = if rmse < 1e-9 {
        99.0
    } else {
        (20.0 * (255.0 / rmse).log10()).min(99.0)
    };

    ScoreReport {
        match_pct: 100.0 * matched as f64 / n.max(1) as f64,
        mean_delta_e,
        p95_delta_e,
        max_delta_e,
        rmse,
        psnr,
        ssim: ssim(&a, &b),
        ..Default::default()
    }
}

/// Full round trip: export, render, compare, and record complexity.
pub fn score(original: &Raster, image: &VectorImage, cfg: &ScoreConfig) -> Result<ScoreReport> {
    let t = Instant::now();
    let rendered = render(image, cfg.supersample)?;
    let render_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut report = compare(original, &rendered, cfg);
    report.compare_ms = t.elapsed().as_secs_f64() * 1000.0;
    report.render_ms = render_ms;
    report.complexity = image.complexity();
    report.output_bytes = export::svg::write(image, &OutputConfig::default()).len();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;
    use crate::vectorize::vectorize;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Raster {
        let mut r = Raster::new(w, h);
        for y in 0..h {
            for x in 0..w {
                r.set(x, y, c);
            }
        }
        r
    }

    #[test]
    fn identical_images_score_perfectly() {
        let a = solid(16, 16, [0.3, 0.5, 0.7, 1.0]);
        let r = compare(&a, &a, &ScoreConfig::default());
        assert!((r.match_pct - 100.0).abs() < 1e-9);
        assert!(r.rmse < 1e-9);
        assert!(r.psnr >= 99.0);
        assert!(r.ssim > 0.999);
        assert!(r.max_delta_e < 1e-4);
    }

    #[test]
    fn a_visibly_different_image_scores_badly() {
        let a = solid(16, 16, [0.0, 0.0, 0.0, 1.0]);
        let b = solid(16, 16, [1.0, 1.0, 1.0, 1.0]);
        let r = compare(&a, &b, &ScoreConfig::default());
        assert!(r.match_pct < 1.0);
        assert!(r.ssim < 0.5);
        assert!(r.mean_delta_e > 50.0);
    }

    #[test]
    fn an_imperceptible_shift_still_counts_as_a_match() {
        // One level of 8-bit difference is far below the perceptual threshold.
        // RMSE notices it; the headline metric correctly does not.
        let a = solid(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let b = solid(16, 16, [0.5 + 1.0 / 255.0, 0.5, 0.5, 1.0]);
        let r = compare(&a, &b, &ScoreConfig::default());
        assert!(r.match_pct > 99.9, "match {}", r.match_pct);
        assert!(r.rmse > 0.0, "rmse should still register the difference");
    }

    #[test]
    fn renders_a_traced_square_back_to_the_original() {
        let mut img = solid(64, 64, [1.0, 1.0, 1.0, 1.0]);
        for y in 16..48 {
            for x in 16..48 {
                img.set(x, y, [0.1, 0.2, 0.8, 1.0]);
            }
        }
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 2;
        let traced = vectorize(&img, &cfg);
        let report = score(&img, &traced.image, &ScoreConfig::default()).expect("score");
        assert!(
            report.match_pct > 99.0,
            "hard-edged square round-tripped at only {:.2}%",
            report.match_pct
        );
        assert!(report.complexity.total_points() > 0);
        assert!(report.output_bytes > 0);
    }

    #[test]
    fn rasterizes_svg_at_the_declared_size() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"12\" \
                   viewBox=\"0 0 20 12\"><rect width=\"20\" height=\"12\" fill=\"#ff0000\"/></svg>";
        let r = rasterize_svg(svg, 1).expect("render");
        assert_eq!((r.width, r.height), (20, 12));
        let c = r.get(10, 6);
        assert!(c[0] > 0.99 && c[1] < 0.01, "got {:?}", c);
    }

    #[test]
    fn supersampling_yields_the_same_output_size() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"12\" \
                   viewBox=\"0 0 20 12\"><circle cx=\"10\" cy=\"6\" r=\"5\" fill=\"#0000ff\"/></svg>";
        let a = rasterize_svg(svg, 1).expect("1x");
        let b = rasterize_svg(svg, 3).expect("3x");
        assert_eq!((a.width, a.height), (b.width, b.height));
    }

    #[test]
    fn composite_rewards_fewer_points_at_equal_fidelity() {
        let mut lean = ScoreReport {
            match_pct: 99.5,
            ..Default::default()
        };
        lean.complexity.anchors = 40;
        let mut fat = lean;
        fat.complexity.anchors = 400;
        assert!(lean.composite(99.0, 100.0) > fat.composite(99.0, 100.0));
    }

    #[test]
    fn composite_prefers_fidelity_over_compactness() {
        // A result that misses the target must lose to one that meets it, even
        // if the accurate one costs several times more points.
        let mut accurate = ScoreReport {
            match_pct: 99.5,
            ..Default::default()
        };
        accurate.complexity.anchors = 500;
        let mut sloppy = ScoreReport {
            match_pct: 92.0,
            ..Default::default()
        };
        sloppy.complexity.anchors = 50;
        assert!(accurate.composite(99.0, 100.0) > sloppy.composite(99.0, 100.0));
    }
}
