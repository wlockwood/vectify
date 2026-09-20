//! Sub-pixel edge reconstruction.
//!
//! A tracer that snaps to pixel boundaries can never be more accurate than half
//! a pixel, and the error is *structured* rather than random: it shows up as
//! stair-stepping on near-axis edges and as lumpiness on curves. Anti-aliasing,
//! though, is not noise to be smoothed away. It is a record of where the
//! original shape actually was.
//!
//! When a rasteriser draws a shape, each edge pixel is set to the area-weighted
//! average of the colours either side of the boundary. That average is
//! invertible. Given the two flanking colours we recover the coverage fraction
//! from a blended pixel, and given the coverage fraction and the edge
//! orientation we recover the signed distance from the pixel centre to the
//! original mathematical edge -- exactly, in closed form, not as an
//! approximation.
//!
//! The geometry: a unit square cut by a half-plane at signed distance `t` from
//! the square's centre, with the boundary normal at angle `theta` to the axes,
//! has covered area
//!
//! ```text
//!   A(t) = 0                              t <= -h
//!        = (t + h)^2 / (2 c s)            -h <= t <= -d
//!        = 1/2 + t / c                    -d <= t <= d
//!        = 1 - (h - t)^2 / (2 c s)         d <= t <= h
//!        = 1                              t >= h
//! ```
//!
//! where `c = cos theta`, `s = sin theta` folded into `[0, 45]` degrees,
//! `h = (c + s) / 2` and `d = (c - s) / 2`. This module inverts that piecewise
//! function.

use crate::color::Rgba8;
use crate::config::SubpixelConfig;
use crate::geom::Point;
use crate::raster::Raster;
use crate::segment::{Segmentation, OUTSIDE};
use crate::topology::Topology;

/// Fraction of a unit pixel covered by a half-plane whose boundary lies at
/// signed distance `t` from the pixel centre, measured along unit normal
/// `(nx, ny)`. The forward direction of the inversion below; exposed because
/// synthesising known-good anti-aliased test images needs exactly this.
pub fn distance_to_coverage(t: f64, nx: f64, ny: f64) -> f64 {
    let c = nx.abs().max(ny.abs());
    let s = nx.abs().min(ny.abs());
    let h = (c + s) * 0.5;
    let d = (c - s) * 0.5;
    if t <= -h {
        return 0.0;
    }
    if t >= h {
        return 1.0;
    }
    if s < 1e-12 {
        return (0.5 + t / c.max(1e-12)).clamp(0.0, 1.0);
    }
    if t <= -d {
        (t + h) * (t + h) / (2.0 * c * s)
    } else if t <= d {
        0.5 + t / c
    } else {
        1.0 - (h - t) * (h - t) / (2.0 * c * s)
    }
}

/// Inverse of [`distance_to_coverage`]: the signed distance from a pixel centre
/// to the edge that would produce coverage `a`.
///
/// This is the single most important function in the engine. Everything else is
/// bookkeeping around the fact that this inversion is exact.
pub fn coverage_to_distance(a: f64, nx: f64, ny: f64) -> f64 {
    let c = nx.abs().max(ny.abs());
    let s = nx.abs().min(ny.abs());
    let h = (c + s) * 0.5;
    let a = a.clamp(0.0, 1.0);
    if s < 1e-12 {
        return (a - 0.5) * c.max(1e-12);
    }
    // Coverage at the knee where the cut stops clipping a corner triangle and
    // starts clipping a trapezoid.
    let knee = s / (2.0 * c);
    if a < knee {
        (2.0 * c * s * a).sqrt() - h
    } else if a <= 1.0 - knee {
        (a - 0.5) * c
    } else {
        h - (2.0 * c * s * (1.0 - a)).sqrt()
    }
}

/// Premultiplied RGBA. Coverage blending is linear in premultiplied space even
/// when alpha varies, so a shape fading out over a transparent background
/// inverts with the same maths as an opaque one.
#[inline]
fn premul(c: [f32; 4]) -> [f64; 4] {
    let a = c[3] as f64;
    [c[0] as f64 * a, c[1] as f64 * a, c[2] as f64 * a, a]
}

#[inline]
fn premul8(c: Rgba8) -> [f64; 4] {
    premul(c.to_f32())
}

#[inline]
fn sub4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline]
fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Local tangent direction at vertex `i`, estimated over a window.
///
/// A single-step tangent is useless here: the raw contour is a staircase of
/// axis-aligned unit steps, so consecutive steps only ever point along an axis.
/// Widening the window recovers the true direction of a slanted edge. Near a
/// sharp turn a wide window would instead cut the corner, so the window shrinks
/// when it detects one.
fn local_tangent(pts: &[Point], i: usize, closed: bool, max_k: usize) -> Point {
    let n = pts.len();
    if n < 2 {
        return Point::new(1.0, 0.0);
    }
    let idx = |k: isize| -> Option<usize> {
        if closed {
            // Closed rings repeat their first point at the end; step over it.
            let m = (n - 1) as isize;
            Some((((i as isize + k) % m) + m) as usize % (n - 1))
        } else {
            let v = i as isize + k;
            if v < 0 || v >= n as isize {
                None
            } else {
                Some(v as usize)
            }
        }
    };

    let mut k = max_k.max(1);
    while k > 1 {
        let (Some(a), Some(b)) = (idx(-(k as isize)), idx(k as isize)) else {
            k -= 1;
            continue;
        };
        let back = pts[i] - pts[a];
        let fwd = pts[b] - pts[i];
        if back.length_sq() < 1e-18 || fwd.length_sq() < 1e-18 {
            k -= 1;
            continue;
        }
        let cosang = back.normalized().dot(fwd.normalized());
        // ~60 degrees. Beyond that, treat it as a corner and tighten the window.
        if cosang > 0.5 {
            break;
        }
        k -= 1;
    }

    for kk in (1..=k).rev() {
        let a = idx(-(kk as isize));
        let b = idx(kk as isize);
        match (a, b) {
            (Some(a), Some(b)) => {
                let d = pts[b] - pts[a];
                if d.length_sq() > 1e-18 {
                    return d.normalized();
                }
            }
            _ => continue,
        }
    }
    // Fall back to whichever neighbouring step exists.
    if i + 1 < n {
        (pts[i + 1] - pts[i]).normalized()
    } else {
        (pts[i] - pts[i - 1]).normalized()
    }
}

/// Estimate how far vertex `v` should move along `normal` to land on the true
/// edge between colours `pm_l` (left) and `pm_r` (right).
///
/// Returns `None` when the local pixels carry no usable coverage evidence: a
/// genuinely hard edge, a third colour intruding, or too little contrast. In
/// all of those cases the crack position is already the best answer available
/// and the vertex should stay put.
fn solve_offset(
    img: &Raster,
    v: Point,
    normal: Point,
    pm_l: [f64; 4],
    pm_r: [f64; 4],
    cfg: &SubpixelConfig,
) -> Option<f64> {
    let diff = sub4(pm_l, pm_r);
    let denom = dot4(diff, diff);
    if denom < (cfg.min_contrast as f64).powi(2) {
        return None;
    }

    // The four pixels sharing this lattice corner. An edge passing through the
    // vertex must clip at least one of them.
    let bx = v.x.round() as i64;
    let by = v.y.round() as i64;
    let mut samples: Vec<f64> = Vec::with_capacity(4);

    for (ox, oy) in [(-1i64, -1i64), (0, -1), (-1, 0), (0, 0)] {
        let px = bx + ox;
        let py = by + oy;
        if px < 0 || py < 0 || px >= img.width as i64 || py >= img.height as i64 {
            continue;
        }
        let pm = premul(img.get(px as u32, py as u32));
        let rel = sub4(pm, pm_r);
        let u = dot4(rel, diff) / denom;

        // Reject pixels that are not on the line between the two colours: a
        // third region leaking in, or noise. Without this guard a junction
        // between three colours drags edges toward whichever colour happens to
        // be nearby.
        let resid = {
            let proj = [
                pm_r[0] + diff[0] * u,
                pm_r[1] + diff[1] * u,
                pm_r[2] + diff[2] * u,
                pm_r[3] + diff[3] * u,
            ];
            let e = sub4(pm, proj);
            dot4(e, e).sqrt()
        };
        if resid > 0.18 {
            continue;
        }
        // Fully covered pixels carry no positional information; they only say
        // the edge is somewhere beyond them.
        if !(0.02..=0.98).contains(&u) {
            continue;
        }

        let t = coverage_to_distance(u, normal.x, normal.y);
        // The edge's closest approach to this pixel centre.
        let centre = Point::new(px as f64 + 0.5, py as f64 + 0.5);
        let edge_pt = centre - normal * t;
        samples.push((edge_pt - v).dot(normal));
    }

    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples.len() / 2;
    let s = if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    };
    Some(s.clamp(-cfg.max_shift, cfg.max_shift))
}

/// Move every contour vertex onto the edge the anti-aliasing implies.
///
/// Junction points are handled once, as shared nodes, so the three or more arcs
/// meeting there stay welded together.
pub fn refine(topo: &mut Topology, seg: &Segmentation, img: &Raster, cfg: &SubpixelConfig) {
    if !cfg.enabled {
        return;
    }

    let region_pm = |r: u32| -> Option<[f64; 4]> {
        if r == OUTSIDE {
            None
        } else {
            Some(premul8(seg.region_color[r as usize]))
        }
    };

    // Accumulated displacement for each shared junction node, plus a count.
    let mut node_disp: Vec<Point> = vec![Point::ZERO; topo.nodes.len()];
    let mut node_hits: Vec<u32> = vec![0; topo.nodes.len()];
    // Interior displacements, one vector per arc.
    let mut arc_disp: Vec<Vec<Point>> = Vec::with_capacity(topo.arcs.len());

    for arc in &topo.arcs {
        let n = arc.points.len();
        let mut disp = vec![Point::ZERO; n];
        if arc.on_border {
            arc_disp.push(disp);
            continue;
        }
        let (Some(pm_l), Some(pm_r)) = (region_pm(arc.left), region_pm(arc.right)) else {
            arc_disp.push(disp);
            continue;
        };

        let closed = arc.loop_arc;
        let orig = &arc.points;
        let mut cur = orig.clone();
        // Per-vertex bookkeeping the smoothing pass needs: the normal the solve
        // used, and whether there was any evidence to solve from at all.
        let mut normals = vec![Point::ZERO; n];
        let mut solved = vec![false; n];

        // Alternate between measuring orientation and solving position. The
        // first round measures orientation from the raw staircase, which is
        // only accurate for edges near 45 degrees; later rounds measure it from
        // a contour that is already close to the true edge.
        for round in 0..cfg.iterations.max(1) {
            // Round one has to span several staircase steps to see the slope at
            // all. Once the contour is straight, a tighter window tracks real
            // curvature better.
            let window = if round == 0 { 5 } else { 3 };
            let mut next = cur.clone();
            for i in 0..n {
                if closed && i == n - 1 {
                    continue;
                }
                let tangent = local_tangent(&cur, i, closed, window);
                let normal = tangent.perp();
                if normal.length_sq() < 1e-18 {
                    next[i] = orig[i];
                    continue;
                }
                normals[i] = normal;
                // Solve from the original lattice vertex every round. Feeding
                // the moved point back in would let vertices creep along the
                // contour instead of only across it.
                match solve_offset(img, orig[i], normal, pm_l, pm_r, cfg) {
                    Some(s) => {
                        next[i] = orig[i] + normal * s;
                        solved[i] = true;
                    }
                    None => {
                        next[i] = orig[i];
                        solved[i] = false;
                    }
                }
            }
            if closed {
                next[n - 1] = next[0];
                normals[n - 1] = normals[0];
                solved[n - 1] = solved[0];
            }
            cur = next;
        }

        // Smooth the *refined positions*, never the displacements.
        //
        // This distinction matters more than it looks. Along a slanted edge the
        // correction each staircase vertex needs is a sawtooth, and blurring a
        // sawtooth flattens it -- which throws away precisely the signal the
        // solve just recovered. The refined positions, by contrast, already lie
        // on the reconstructed edge, so smoothing them damps jitter where
        // individual solves disagreed without pulling the edge off its true
        // position.
        //
        // Three constraints keep this from doing harm:
        //
        // * only vertices the solve actually moved take part, and only their
        //   solved neighbours contribute -- otherwise a vertex with no evidence
        //   gets dragged by its neighbours despite the crack already being the
        //   correct answer;
        // * the correction is projected onto the vertex normal, so points may
        //   only move *across* the edge, never slide along it;
        // * high-curvature vertices are skipped outright.
        //
        // The last two exist because a corner is exactly where an unconstrained
        // blur does its worst: averaging a right-angle vertex with its two
        // perpendicular neighbours cuts the corner off. A quarter-pixel bite
        // there is enough for the fitter to then re-fit the whole side as a
        // line through the shortened ends, shifting an entire straight edge.
        for _ in 0..cfg.smooth_passes {
            let src = cur.clone();
            let m = if closed { n - 1 } else { n };
            let neighbours = |i: usize| -> Option<(usize, usize)> {
                if closed {
                    Some(((i + m - 1) % m, (i + 1) % m))
                } else if i >= 1 && i + 1 < n {
                    Some((i - 1, i + 1))
                } else {
                    // Endpoints are shared junction nodes, reconciled below.
                    None
                }
            };
            for i in 0..m {
                if !solved[i] {
                    continue;
                }
                let Some((a, b)) = neighbours(i) else { continue };
                if !solved[a] || !solved[b] {
                    continue;
                }
                // Skip corners: a bend this sharp is geometry, not jitter.
                let back = src[i] - src[a];
                let fwd = src[b] - src[i];
                if back.length_sq() > 1e-18 && fwd.length_sq() > 1e-18 {
                    let cosang = back.normalized().dot(fwd.normalized());
                    if cosang < std::f64::consts::FRAC_1_SQRT_2 {
                        continue;
                    }
                }
                let target = (src[a] + src[i] * 2.0 + src[b]) * 0.25;
                let delta = target - src[i];
                let nrm = normals[i];
                cur[i] = src[i] + nrm * delta.dot(nrm);
            }
            if closed {
                cur[n - 1] = cur[0];
            }
        }

        for i in 0..n {
            disp[i] = cur[i] - orig[i];
        }

        arc_disp.push(disp);
    }

    // Endpoints vote on where their shared junction should go.
    for (ai, arc) in topo.arcs.iter().enumerate() {
        if arc.on_border {
            continue;
        }
        let d = &arc_disp[ai];
        let n = d.len();
        for (node, disp) in [(arc.start_node, d[0]), (arc.end_node, d[n - 1])] {
            if node == crate::topology::NO_NODE {
                continue;
            }
            node_disp[node as usize] += disp;
            node_hits[node as usize] += 1;
        }
    }

    let w = seg.width as f64;
    let h = seg.height as f64;
    // Nodes that any border arc touches are pinned to the image edge; they may
    // slide along it but never off it, and image corners do not move at all.
    let mut pinned: Vec<u8> = vec![0; topo.nodes.len()];
    for arc in &topo.arcs {
        if !arc.on_border {
            continue;
        }
        for node in [arc.start_node, arc.end_node] {
            if node != crate::topology::NO_NODE {
                pinned[node as usize] = 1;
            }
        }
    }

    for (i, node) in topo.nodes.iter_mut().enumerate() {
        if node_hits[i] == 0 {
            continue;
        }
        let mut d = node_disp[i] / node_hits[i] as f64;
        if pinned[i] == 1 {
            let on_v = node.x <= 0.0 || node.x >= w;
            let on_h = node.y <= 0.0 || node.y >= h;
            if on_v && on_h {
                continue; // image corner
            } else if on_v {
                d.x = 0.0;
            } else if on_h {
                d.y = 0.0;
            }
        }
        *node += d;
        node.x = node.x.clamp(0.0, w);
        node.y = node.y.clamp(0.0, h);
    }

    // Apply: interior vertices take their own displacement, endpoints adopt the
    // shared node position so that arcs meeting at a junction cannot separate.
    for (ai, arc) in topo.arcs.iter_mut().enumerate() {
        let d = &arc_disp[ai];
        let n = arc.points.len();
        for i in 1..n.saturating_sub(1) {
            arc.points[i] += d[i];
            arc.points[i].x = arc.points[i].x.clamp(0.0, w);
            arc.points[i].y = arc.points[i].y.clamp(0.0, h);
        }
        if arc.start_node != crate::topology::NO_NODE {
            arc.points[0] = topo.nodes[arc.start_node as usize];
        }
        if arc.end_node != crate::topology::NO_NODE {
            arc.points[n - 1] = topo.nodes[arc.end_node as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::LabCache;
    use crate::config::{SegmentConfig, SegmentMode};
    use crate::segment::segment;
    use crate::topology;

    #[test]
    fn coverage_roundtrip_is_exact() {
        for angle_deg in [0.0f64, 10.0, 22.5, 30.0, 45.0, 60.0, 75.0, 90.0, 135.0] {
            let th: f64 = angle_deg.to_radians();
            let (nx, ny) = (th.cos(), th.sin());
            let h = (nx.abs() + ny.abs()) * 0.5;
            let mut t = -h + 1e-6;
            while t < h {
                let a = distance_to_coverage(t, nx, ny);
                let back = coverage_to_distance(a, nx, ny);
                assert!(
                    (back - t).abs() < 1e-9,
                    "angle {angle_deg}: t={t} -> a={a} -> {back}"
                );
                t += 0.01;
            }
        }
    }

    #[test]
    fn coverage_endpoints_and_midpoint() {
        assert!((distance_to_coverage(0.0, 1.0, 0.0) - 0.5).abs() < 1e-12);
        assert!((distance_to_coverage(0.0, 0.7071, 0.7071) - 0.5).abs() < 1e-6);
        assert_eq!(distance_to_coverage(-1.0, 1.0, 0.0), 0.0);
        assert_eq!(distance_to_coverage(1.0, 1.0, 0.0), 1.0);
    }

    /// Render an analytically anti-aliased half-plane: every pixel gets exactly
    /// the coverage a correct rasteriser would produce.
    fn render_halfplane(w: u32, h: u32, nx: f64, ny: f64, offset: f64) -> Raster {
        let mut img = Raster::new(w, h);
        let n = Point::new(nx, ny).normalized();
        for y in 0..h {
            for x in 0..w {
                let centre = Point::new(x as f64 + 0.5, y as f64 + 0.5);
                // Signed distance from the edge line to this pixel centre.
                let t = centre.dot(n) - offset;
                let cov = distance_to_coverage(t, n.x, n.y);
                let v = cov as f32; // white on the +n side, black on the other
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        img
    }

    /// Reconstruct a straight edge and report mean absolute positional error in
    /// pixels, along with the error the raw crack contour would have had.
    fn edge_error(nx: f64, ny: f64, offset: f64, enabled: bool) -> f64 {
        let (w, h) = (48u32, 48u32);
        let img = render_halfplane(w, h, nx, ny, offset);
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Binary,
            despeckle_area: 0,
            ..Default::default()
        };
        let seg = segment(&img, &cfg, &cache);
        let mut topo = topology::build(&seg);
        let sp = SubpixelConfig {
            enabled,
            ..Default::default()
        };
        refine(&mut topo, &seg, &img, &sp);

        let n = Point::new(nx, ny).normalized();
        let mut total = 0.0;
        let mut count = 0usize;
        for arc in &topo.arcs {
            if arc.on_border {
                continue;
            }
            // Skip endpoints: they are pinned to the image border.
            if arc.points.len() < 5 {
                continue;
            }
            for p in &arc.points[2..arc.points.len() - 2] {
                total += (p.dot(n) - offset).abs();
                count += 1;
            }
        }
        assert!(count > 0, "no interior contour vertices found");
        total / count as f64
    }

    #[test]
    fn subpixel_recovers_axis_aligned_edges() {
        // A vertical edge at x = 20.3. Snapping to the pixel grid can only ever
        // answer 20.0, an error of 0.3px; the inversion should do far better.
        let raw = edge_error(1.0, 0.0, 20.3, false);
        let refined = edge_error(1.0, 0.0, 20.3, true);
        assert!(raw > 0.2, "raw crack error unexpectedly small: {raw}");
        assert!(
            refined < 0.02,
            "refined error {refined} should be near-exact (raw was {raw})"
        );
    }

    #[test]
    fn subpixel_recovers_slanted_edges() {
        for (deg, offset) in [(15.0f64, 18.7f64), (30.0, 22.1), (45.0, 25.4), (63.0, 19.9)] {
            let th: f64 = deg.to_radians();
            let (nx, ny) = (th.cos(), th.sin());
            let raw = edge_error(nx, ny, offset, false);
            let refined = edge_error(nx, ny, offset, true);
            // The inversion is exact for a straight edge, so the only residue
            // is the tangent estimate and the light position smoothing. A
            // hundredth of a pixel leaves room for both without letting a real
            // regression through: the raw contour sits around 0.3px.
            assert!(
                refined < 0.01,
                "{deg} deg: refined error {refined} too large (raw {raw})"
            );
            assert!(
                refined < raw * 0.1,
                "{deg} deg: refinement did not clearly beat the raw contour ({refined} vs {raw})"
            );
        }
    }

    #[test]
    fn subpixel_leaves_hard_edges_alone() {
        // A hard-edged image carries no coverage information. The crack is
        // already the right answer and must not be perturbed.
        let mut img = Raster::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let v = if x < 16 { 0.0 } else { 1.0 };
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Binary,
            despeckle_area: 0,
            ..Default::default()
        };
        let seg = segment(&img, &cfg, &cache);
        let mut topo = topology::build(&seg);
        refine(&mut topo, &seg, &img, &SubpixelConfig::default());
        for arc in &topo.arcs {
            if arc.on_border {
                continue;
            }
            for p in &arc.points {
                assert!((p.x - 16.0).abs() < 1e-9, "hard edge moved to {:?}", p);
            }
        }
    }

    #[test]
    fn refinement_does_not_round_off_corners() {
        // Regression: the smoothing pass used to average each contour vertex
        // with its neighbours unconditionally. On a hard-edged square that bit
        // a quarter-pixel out of all four corners, and the fitter then re-fitted
        // each side as a straight line through the shortened ends -- shifting
        // every edge inward and costing about three percent of the round-trip
        // score on a shape that should be pixel-exact.
        let mut img = Raster::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, [1.0, 1.0, 1.0, 1.0]);
            }
        }
        for y in 16..48 {
            for x in 16..48 {
                img.set(x, y, [0.1, 0.2, 0.8, 1.0]);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Palette,
            colors: 2,
            despeckle_area: 0,
            ..Default::default()
        };
        let seg = segment(&img, &cfg, &cache);
        let mut topo = topology::build(&seg);
        refine(&mut topo, &seg, &img, &SubpixelConfig::default());

        for arc in &topo.arcs {
            if arc.on_border {
                continue;
            }
            for p in &arc.points {
                // Every vertex of the square's contour is an integer lattice
                // point and must still be one.
                assert!(
                    (p.x - p.x.round()).abs() < 1e-9 && (p.y - p.y.round()).abs() < 1e-9,
                    "hard-edged contour vertex drifted to {:?}",
                    p
                );
                assert!(
                    (16.0..=48.0).contains(&p.x) && (16.0..=48.0).contains(&p.y),
                    "vertex {:?} left the square's boundary",
                    p
                );
            }
            // The four corners must still be present, unrounded.
            for corner in [(16.0, 16.0), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0)] {
                let want = Point::new(corner.0, corner.1);
                assert!(
                    arc.points.iter().any(|p| p.dist(want) < 1e-9),
                    "corner {:?} was rounded away",
                    want
                );
            }
        }
    }

    #[test]
    fn junctions_stay_welded() {
        // Three colour bands meeting the image edge. After refinement the arcs
        // meeting at each junction must still share an identical endpoint.
        let mut img = Raster::new(36, 12);
        for y in 0..12 {
            for x in 0..36 {
                let c = match x / 12 {
                    0 => [0.9, 0.1, 0.1, 1.0],
                    1 => [0.1, 0.85, 0.15, 1.0],
                    _ => [0.1, 0.15, 0.9, 1.0],
                };
                img.set(x, y, c);
            }
        }
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: SegmentMode::Palette,
            colors: 3,
            despeckle_area: 0,
            ..Default::default()
        };
        let seg = segment(&img, &cfg, &cache);
        let mut topo = topology::build(&seg);
        refine(&mut topo, &seg, &img, &SubpixelConfig::default());

        for arc in &topo.arcs {
            let n = arc.points.len();
            if arc.start_node != crate::topology::NO_NODE {
                assert!(arc.points[0].dist(topo.nodes[arc.start_node as usize]) < 1e-12);
            }
            if arc.end_node != crate::topology::NO_NODE {
                assert!(arc.points[n - 1].dist(topo.nodes[arc.end_node as usize]) < 1e-12);
            }
        }
    }
}
