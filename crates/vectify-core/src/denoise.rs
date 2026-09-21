//! Edge-preserving smoothing for the scoring reference.
//!
//! A JPEG's blocking and noise are error in the eyes of the scorer, but they are
//! not error the tracer could or should reproduce: a flat region traced as one
//! flat colour is *correct*, and the scorer would still charge it for every
//! wobble in the source. Smoothing the original first removes that false error.
//!
//! It has to be an edge-preserving filter. A box or Gaussian blur averages
//! every neighbour equally, so it softens the very edges the tracer is judged
//! on and makes any trace look better by making the reference blurrier. The
//! bilateral filter weights each neighbour by distance *and* by colour
//! similarity, so pixels across a real edge barely contribute and the edge
//! survives, while noise within a flat area is averaged away.
//!
//! Similarity is measured in CIELAB, where a distance means about the same
//! thing everywhere, so one `range_sigma` behaves alike in shadows and highlights.
//! The averaging itself is done on the gamma-encoded values, to match how the
//! rest of the engine blends.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::color::srgb_to_lab;
use crate::raster::Raster;

pub const DEFAULT_RADIUS: u32 = 3;
pub const DEFAULT_RANGE_SIGMA: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BilateralConfig {
    /// Window radius in pixels. The spatial falloff is tied to it, so a larger
    /// radius also smooths over a wider area rather than only reaching further.
    pub radius: u32,
    /// Colour distance (CIELAB) at which a neighbour's weight has fallen to 61%.
    /// Differences well under this are averaged together; differences several
    /// times over it are left alone. It must sit above the noise being removed
    /// and below the contrast of the edges being kept.
    pub range_sigma: f32,
}

impl Default for BilateralConfig {
    fn default() -> Self {
        BilateralConfig {
            radius: DEFAULT_RADIUS,
            range_sigma: DEFAULT_RANGE_SIGMA,
        }
    }
}

/// Bilateral-filter the colour channels. Alpha is passed through untouched, so
/// callers that care about transparency should composite first.
///
/// Neighbours outside the image are skipped rather than clamped, which keeps the
/// border from over-weighting its own edge pixels.
pub fn bilateral(src: &Raster, cfg: &BilateralConfig) -> Raster {
    let w = src.width as usize;
    let h = src.height as usize;
    if w == 0 || h == 0 || cfg.radius == 0 || cfg.range_sigma <= 0.0 {
        return src.clone();
    }

    let r = cfg.radius as isize;
    let side = (2 * r + 1) as usize;
    let sigma_s = (cfg.radius as f32 / 2.0).max(0.5);
    let spatial: Vec<f32> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp())
        .collect();
    let inv_2sr2 = 1.0 / (2.0 * cfg.range_sigma * cfg.range_sigma);

    let lab: Vec<[f32; 3]> = src
        .data
        .par_iter()
        .map(|c| {
            let l = srgb_to_lab([c[0], c[1], c[2]]);
            [l.l, l.a, l.b]
        })
        .collect();

    let mut out = Raster::new(src.width, src.height);
    out.data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, px) in row.iter_mut().enumerate() {
                let centre = lab[y * w + x];
                let mut acc = [0.0f32; 3];
                let mut total = 0.0f32;
                for dy in -r..=r {
                    let sy = y as isize + dy;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for dx in -r..=r {
                        let sx = x as isize + dx;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let i = sy as usize * w + sx as usize;
                        let n = lab[i];
                        let d2 = (n[0] - centre[0]).powi(2)
                            + (n[1] - centre[1]).powi(2)
                            + (n[2] - centre[2]).powi(2);
                        let k = spatial[(dy + r) as usize * side + (dx + r) as usize]
                            * (-d2 * inv_2sr2).exp();
                        let c = src.data[i];
                        acc[0] += c[0] * k;
                        acc[1] += c[1] * k;
                        acc[2] += c[2] * k;
                        total += k;
                    }
                }
                // The centre pixel always contributes with weight 1, so this
                // cannot divide by zero.
                let a = src.data[y * w + x][3];
                *px = [acc[0] / total, acc[1] / total, acc[2] / total, a];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Left half dark, right half light, with a checkerboard wobble of `noise`
    /// on top: a hard edge at x = w/2 plus the high-frequency noise a JPEG adds.
    fn noisy_step(w: u32, h: u32, noise: f32) -> Raster {
        let mut r = Raster::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = if x < w / 2 { 0.2 } else { 0.8 };
                let wobble = if (x + y) % 2 == 0 { noise } else { -noise };
                let v = base + wobble;
                r.set(x, y, [v, v, v, 1.0]);
            }
        }
        r
    }

    fn spread(r: &Raster, xs: std::ops::Range<u32>) -> f32 {
        let vals: Vec<f32> = (0..r.height)
            .flat_map(|y| xs.clone().map(move |x| (x, y)))
            .map(|(x, y)| r.get(x, y)[0])
            .collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt()
    }

    #[test]
    fn flat_regions_are_left_alone() {
        let mut r = Raster::new(12, 12);
        for c in r.data.iter_mut() {
            *c = [0.3, 0.5, 0.7, 1.0];
        }
        let out = bilateral(&r, &BilateralConfig::default());
        for (a, b) in r.data.iter().zip(&out.data) {
            for i in 0..4 {
                assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} became {b:?}");
            }
        }
    }

    #[test]
    fn noise_is_removed_but_the_edge_survives() {
        let img = noisy_step(32, 16, 0.01);
        let out = bilateral(&img, &BilateralConfig::default());

        // Well clear of the edge, the wobble should be largely gone.
        let before = spread(&img, 2..12);
        let after = spread(&out, 2..12);
        assert!(after < before * 0.3, "noise {before:.4} only fell to {after:.4}");

        // The pixels either side of the step must still be on their own side.
        // A box blur of the same size would drag both to about 0.5.
        let dark = out.get(15, 8)[0];
        let light = out.get(16, 8)[0];
        assert!(dark < 0.25, "dark side of the edge blurred to {dark:.3}");
        assert!(light > 0.75, "light side of the edge blurred to {light:.3}");
    }

    #[test]
    fn a_zero_radius_is_the_identity() {
        let img = noisy_step(8, 8, 0.05);
        let out = bilateral(
            &img,
            &BilateralConfig {
                radius: 0,
                range_sigma: 4.0,
            },
        );
        assert_eq!(img.data, out.data);
    }

    #[test]
    fn alpha_is_passed_through() {
        let mut img = noisy_step(8, 8, 0.01);
        img.set(3, 3, [0.5, 0.5, 0.5, 0.25]);
        let out = bilateral(&img, &BilateralConfig::default());
        assert_eq!(out.get(3, 3)[3], 0.25);
    }

    #[test]
    fn tiny_and_degenerate_sizes_do_not_panic() {
        for (w, h) in [(0, 0), (1, 1), (1, 9), (9, 1)] {
            let img = Raster::new(w, h);
            let out = bilateral(&img, &BilateralConfig::default());
            assert_eq!((out.width, out.height), (w, h));
        }
    }
}
