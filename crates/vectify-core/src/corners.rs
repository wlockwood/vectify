//! Corner detection and artefact discrimination.
//!
//! A tracer has to answer one question at every vertex: is this bend something
//! the artist drew, or something the encoder did to them? Get it wrong in one
//! direction and every logo comes out with rounded, mushy corners. Get it wrong
//! in the other and JPEG ringing and pixel stair-steps are faithfully preserved
//! as hundreds of spurious hard vertices.
//!
//! A single smoothing filter cannot tell these apart, because at the finest
//! scale they look identical: a 45-degree staircase turns by 90 degrees at
//! every step, exactly like the corner of a square.
//!
//! They separate immediately under **scale**. Measure the turn over a wider and
//! wider neighbourhood:
//!
//! - A real corner keeps turning by the same angle at every scale. The corner
//!   of a square is 90 degrees whether you look one pixel out or eight.
//! - A staircase, or a burst of compression noise, turns sharply at one pixel
//!   and not at all at eight, because the wobble averages out.
//!
//! So a vertex is a corner only if it is sharp *persistently*, across scales.
//! The same measurement, read the other way, gives a per-contour noise estimate
//! -- how much bend exists at fine scale that has vanished by coarse scale --
//! and that drives how hard each contour is smoothed. Clean contours are left
//! alone; noisy ones are unified. The filter goes where the noise is instead of
//! being applied blindly to the whole canvas.

use crate::config::CornerConfig;
use crate::geom::Point;
use crate::topology::Topology;

/// Cyclic or clamped index into a contour.
struct Indexer {
    n: usize,
    closed: bool,
}

impl Indexer {
    /// Number of distinct vertices. A closed contour repeats its first point at
    /// the end, so that duplicate is excluded.
    fn distinct(&self) -> usize {
        if self.closed {
            self.n - 1
        } else {
            self.n
        }
    }

    fn at(&self, i: usize, off: isize) -> Option<usize> {
        if self.closed {
            let m = self.distinct() as isize;
            if m <= 0 {
                return None;
            }
            Some((((i as isize + off) % m) + m) as usize % self.distinct())
        } else {
            let v = i as isize + off;
            if v < 0 || v >= self.n as isize {
                None
            } else {
                Some(v as usize)
            }
        }
    }
}

/// Absolute turn angle at vertex `i` measured over a neighbourhood of `s`
/// vertices either side. `None` when the contour is too short to look that far.
fn turn_at_scale(pts: &[Point], ix: &Indexer, i: usize, s: usize) -> Option<f64> {
    let a = ix.at(i, -(s as isize))?;
    let b = ix.at(i, s as isize)?;
    let back = pts[i] - pts[a];
    let fwd = pts[b] - pts[i];
    if back.length_sq() < 1e-18 || fwd.length_sq() < 1e-18 {
        return None;
    }
    let u = back.normalized();
    let v = fwd.normalized();
    Some(u.cross(v).atan2(u.dot(v)).abs())
}

/// What the multi-scale analysis found on one contour.
pub struct Analysis {
    /// Vertex indices that are hard corners, ascending.
    pub corners: Vec<usize>,
    /// How much of this contour's fine-scale bending is not present at coarse
    /// scale, in 0..1. Near 0 for a cleanly reconstructed edge, near 1 for a
    /// raw pixel staircase or heavy compression noise.
    pub noise: f64,
}

/// Analyse one contour. `closed` marks a loop whose final point repeats its
/// first.
pub fn analyze_contour(pts: &[Point], closed: bool, cfg: &CornerConfig) -> Analysis {
    let n = pts.len();
    let ix = Indexer { n, closed };
    let m = ix.distinct();
    if m < 3 {
        let corners = if closed { Vec::new() } else { vec![0, n - 1] };
        return Analysis { corners, noise: 0.0 };
    }

    let threshold = cfg.threshold_deg.to_radians();
    let mut scales: Vec<usize> = cfg.scales.iter().copied().filter(|&s| s >= 1).collect();
    if scales.is_empty() {
        scales.push(1);
    }
    scales.sort_unstable();
    let coarsest = *scales.last().unwrap();

    let mut candidates: Vec<(f64, usize)> = Vec::new();
    let mut noise_acc = 0.0;
    let mut noise_n = 0usize;

    // Interior vertices only for an open arc: its endpoints are junctions and
    // are handled separately.
    let range: Vec<usize> = if closed {
        (0..m).collect()
    } else {
        (1..n.saturating_sub(1)).collect()
    };

    for &i in &range {
        let mut hits = 0usize;
        let mut tested = 0usize;
        let mut strength = 0.0;
        for &s in &scales {
            let Some(t) = turn_at_scale(pts, &ix, i, s) else { continue };
            tested += 1;
            strength += t;
            if t >= threshold {
                hits += 1;
            }
        }
        // Persistence is only meaningful if several scales actually fit. Within
        // a couple of vertices of an open arc's end, the wide neighbourhoods
        // run off the edge and only the finest scale can be measured -- and the
        // finest scale alone calls every staircase step a corner. Demand enough
        // scales to tell the difference. Vertices that close to a junction are
        // anchored by the junction itself anyway.
        let min_scales = scales.len().min(3);
        if tested >= min_scales && (hits as f64 / tested as f64) >= cfg.persistence {
            candidates.push((strength / tested as f64, i));
        }

        // Noise: bend present at one pixel that has vanished by the coarsest
        // scale. A staircase scores near 1, a genuine corner near 0 because it
        // bends just as much at both scales.
        if let (Some(fine), Some(coarse)) = (
            turn_at_scale(pts, &ix, i, 1),
            turn_at_scale(pts, &ix, i, coarsest),
        ) {
            noise_acc += ((fine - coarse) / std::f64::consts::FRAC_PI_2).clamp(0.0, 1.0);
            noise_n += 1;
        }
    }

    // Non-maximum suppression: a sharp corner usually trips the test at its
    // immediate neighbours too, and keeping all of them would spend three
    // anchors where one belongs.
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut accepted: Vec<usize> = Vec::new();
    for (_, i) in candidates {
        let clash = accepted.iter().any(|&j| {
            let d = if closed {
                let raw = (i as isize - j as isize).abs() as usize;
                raw.min(m - raw)
            } else {
                (i as isize - j as isize).unsigned_abs()
            };
            d <= cfg.suppress_radius
        });
        if !clash {
            accepted.push(i);
        }
    }

    if !closed {
        // Junction endpoints always anchor: three regions meet there, and
        // rounding that point would pull all of them apart.
        accepted.push(0);
        accepted.push(n - 1);
    }
    accepted.sort_unstable();
    accepted.dedup();

    Analysis {
        corners: accepted,
        noise: if noise_n == 0 {
            0.0
        } else {
            noise_acc / noise_n as f64
        },
    }
}

/// Taubin smoothing over a run of vertices, endpoints held fixed.
///
/// Plain Laplacian smoothing shrinks: run it on a circle and the circle gets
/// smaller every pass. Taubin alternates a positive shrinking step with a
/// slightly larger negative inflating step, so noise still decays but the
/// overall size is preserved -- which matters here because a shrunken contour
/// is a direct, measurable accuracy loss at round-trip time.
fn taubin_run(pts: &mut [Point], i0: usize, i1: usize, passes: usize) {
    if i1 <= i0 + 1 || passes == 0 {
        return;
    }
    const LAMBDA: f64 = 0.50;
    const MU: f64 = -0.53;
    for _ in 0..passes {
        for &factor in &[LAMBDA, MU] {
            let src: Vec<Point> = pts[i0..=i1].to_vec();
            for k in 1..src.len() - 1 {
                let lap = (src[k - 1] + src[k + 1]) * 0.5 - src[k];
                pts[i0 + k] = src[k] + lap * factor;
            }
        }
    }
}

/// Cyclic Taubin smoothing for a closed contour with no corners at all.
fn taubin_loop(pts: &mut [Point], passes: usize) {
    let n = pts.len();
    if n < 4 || passes == 0 {
        return;
    }
    let m = n - 1;
    const LAMBDA: f64 = 0.50;
    const MU: f64 = -0.53;
    for _ in 0..passes {
        for &factor in &[LAMBDA, MU] {
            let src: Vec<Point> = pts[..m].to_vec();
            for k in 0..m {
                let lap = (src[(k + m - 1) % m] + src[(k + 1) % m]) * 0.5 - src[k];
                pts[k] = src[k] + lap * factor;
            }
        }
    }
    pts[n - 1] = pts[0];
}

/// Detect corners on every arc and apply noise-proportional smoothing between
/// them. Corner vertices and junction endpoints never move.
pub fn analyze(topo: &mut Topology, cfg: &CornerConfig) {
    for arc in topo.arcs.iter_mut() {
        if arc.points.len() < 3 {
            arc.corners = (0..arc.points.len()).collect();
            continue;
        }
        // A border arc is a straight run along the image edge by construction.
        if arc.on_border {
            let n = arc.points.len();
            arc.corners = vec![0, n - 1];
            continue;
        }

        let closed = arc.loop_arc;
        let analysis = analyze_contour(&arc.points, closed, cfg);

        // How hard to smooth. With noise adaptation on, a contour that the
        // sub-pixel pass already reconstructed cleanly is left untouched, while
        // a raw staircase gets the full configured strength.
        let factor = if cfg.noise_adaptive {
            analysis.noise
        } else {
            1.0
        };
        let passes = (cfg.smoothing * factor * 5.0).round().max(0.0) as usize;

        if passes > 0 {
            if closed && analysis.corners.is_empty() {
                taubin_loop(&mut arc.points, passes);
            } else if closed {
                // Smooth each run between consecutive corners, wrapping through
                // the seam.
                let c = &analysis.corners;
                for w in 0..c.len() {
                    let a = c[w];
                    let b = c[(w + 1) % c.len()];
                    if b > a {
                        taubin_run(&mut arc.points, a, b, passes);
                    } else {
                        // The run that crosses the seam: rotate it into a
                        // temporary buffer so the smoother sees it contiguous.
                        let m = arc.points.len() - 1;
                        let mut buf: Vec<Point> = Vec::new();
                        let mut k = a;
                        loop {
                            buf.push(arc.points[k]);
                            if k == b {
                                break;
                            }
                            k = (k + 1) % m;
                        }
                        let len = buf.len();
                        if len > 2 {
                            taubin_run(&mut buf, 0, len - 1, passes);
                            let mut k = a;
                            for p in buf {
                                arc.points[k] = p;
                                if k == b {
                                    break;
                                }
                                k = (k + 1) % m;
                            }
                            let first = arc.points[0];
                            let last = arc.points.len() - 1;
                            arc.points[last] = first;
                        }
                    }
                }
            } else {
                let c = &analysis.corners;
                for w in c.windows(2) {
                    taubin_run(&mut arc.points, w[0], w[1], passes);
                }
            }
        }

        arc.corners = analysis.corners;
    }

    // Smoothing may have nudged the first or last vertex of a closed arc;
    // re-pin every endpoint to its shared junction node so arcs stay welded.
    let nodes = topo.nodes.clone();
    for arc in topo.arcs.iter_mut() {
        let n = arc.points.len();
        if arc.start_node != crate::topology::NO_NODE {
            arc.points[0] = nodes[arc.start_node as usize];
        }
        if arc.end_node != crate::topology::NO_NODE {
            arc.points[n - 1] = nodes[arc.end_node as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::pt;

    fn cfg() -> CornerConfig {
        CornerConfig::default()
    }

    /// An L-shaped path with one unambiguous 90-degree corner.
    fn ell() -> Vec<Point> {
        let mut v = Vec::new();
        for i in 0..12 {
            v.push(pt(i as f64, 0.0));
        }
        for i in 1..12 {
            v.push(pt(11.0, i as f64));
        }
        v
    }

    /// A hard-edged 45-degree staircase: alternating unit steps right and down.
    /// Every vertex turns by 90 degrees at scale 1, yet the path is straight.
    fn staircase() -> Vec<Point> {
        let mut v = vec![pt(0.0, 0.0)];
        for i in 0..14 {
            v.push(pt(i as f64 + 1.0, i as f64));
            v.push(pt(i as f64 + 1.0, i as f64 + 1.0));
        }
        v
    }

    #[test]
    fn finds_a_real_corner() {
        let pts = ell();
        let a = analyze_contour(&pts, false, &cfg());
        // Endpoints plus the corner at index 11.
        assert!(
            a.corners.contains(&11),
            "expected a corner at the bend, got {:?}",
            a.corners
        );
        assert_eq!(a.corners.len(), 3, "corners: {:?}", a.corners);
    }

    #[test]
    fn rejects_staircase_steps_as_corners() {
        // This is the discrimination that separates this engine from a
        // single-threshold smoother. Each step turns a full 90 degrees at the
        // finest scale, so any scale-blind detector marks all 28 of them.
        let pts = staircase();
        let a = analyze_contour(&pts, false, &cfg());
        let interior: Vec<usize> = a
            .corners
            .iter()
            .copied()
            .filter(|&i| i != 0 && i != pts.len() - 1)
            .collect();
        assert!(
            interior.is_empty(),
            "staircase steps were mistaken for corners: {:?}",
            interior
        );
    }

    #[test]
    fn staircase_scores_as_noisy_and_clean_line_does_not() {
        let noisy = analyze_contour(&staircase(), false, &cfg());
        assert!(noisy.noise > 0.5, "staircase noise {} too low", noisy.noise);

        let straight: Vec<Point> = (0..24).map(|i| pt(i as f64 * 0.7, i as f64 * 0.3)).collect();
        let clean = analyze_contour(&straight, false, &cfg());
        assert!(clean.noise < 0.02, "clean line noise {} too high", clean.noise);
    }

    #[test]
    fn a_smooth_circle_has_no_corners() {
        let n = 64;
        let mut pts: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                pt(20.0 + 15.0 * t.cos(), 20.0 + 15.0 * t.sin())
            })
            .collect();
        pts.push(pts[0]);
        let a = analyze_contour(&pts, true, &cfg());
        assert!(a.corners.is_empty(), "circle got corners {:?}", a.corners);
        assert!(a.noise < 0.02, "circle noise {}", a.noise);
    }

    #[test]
    fn a_square_loop_finds_exactly_four_corners() {
        let mut pts = Vec::new();
        let side = 14;
        for i in 0..side {
            pts.push(pt(i as f64, 0.0));
        }
        for i in 0..side {
            pts.push(pt(side as f64, i as f64));
        }
        for i in 0..side {
            pts.push(pt(side as f64 - i as f64, side as f64));
        }
        for i in 0..side {
            pts.push(pt(0.0, side as f64 - i as f64));
        }
        pts.push(pts[0]);
        let a = analyze_contour(&pts, true, &cfg());
        assert_eq!(a.corners.len(), 4, "corners: {:?}", a.corners);
    }

    #[test]
    fn taubin_preserves_circle_radius() {
        // Plain Laplacian smoothing would visibly shrink this; the whole reason
        // for Taubin is that shrinkage is a measurable accuracy loss.
        let n = 80;
        let mut pts: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                pt(30.0 * t.cos(), 30.0 * t.sin())
            })
            .collect();
        pts.push(pts[0]);
        taubin_loop(&mut pts, 5);
        let mean_r: f64 = pts[..n].iter().map(|p| p.length()).sum::<f64>() / n as f64;
        assert!(
            (mean_r - 30.0).abs() < 0.2,
            "radius drifted to {mean_r} after smoothing"
        );
    }

    #[test]
    fn smoothing_flattens_a_staircase() {
        let mut pts = staircase();
        let before = analyze_contour(&pts, false, &cfg());
        let n = pts.len();
        taubin_run(&mut pts, 0, n - 1, 5);
        let after = analyze_contour(&pts, false, &cfg());
        assert!(
            after.noise < before.noise * 0.5,
            "noise {} did not drop from {}",
            after.noise,
            before.noise
        );
        // And the run should now sit close to the ideal 45-degree line.
        let max_dev = pts[2..n - 2]
            .iter()
            .map(|p| (p.x - p.y).abs() / std::f64::consts::SQRT_2)
            .fold(0.0f64, f64::max);
        assert!(max_dev < 0.45, "max deviation from the true line: {max_dev}");
    }
}
