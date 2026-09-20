//! Primitive fitting: turning a refined contour into the fewest curve
//! commands that still hold it within tolerance.
//!
//! The usual output of a pixel tracer is a long chain of tiny cubic segments.
//! It hits the tolerance, but the file is bloated and the result is miserable
//! to edit: a circle drawn as two hundred Bezier knots cannot be resized or
//! nudged without destroying it.
//!
//! This fitter is greedy about span length and cheap about primitives. For each
//! run between corners it tries, in ascending order of cost:
//!
//! 1. a straight line, which costs **zero** control handles;
//! 2. a single cubic Bezier, two handles;
//! 3. a true circular arc, two handles but able to span far more than one
//!    cubic can, so a quarter-circle stays one command instead of three;
//!
//! and only splits the run in half when none of those fit. Because the widest
//! span is always attempted first, the result is close to the minimum number of
//! primitives for the given tolerance.
//!
//! The cubic fit itself is Schneider's least-squares method with
//! Newton-Raphson reparameterisation, constrained to match the tangent
//! directions at both ends so that segments join smoothly instead of kinking.

use crate::config::FitConfig;
use crate::geom::{
    arc_to_cubics, cubic_tangent, dist_point_segment, eval_cubic, Point, Seg,
};

/// Remove consecutive duplicate points, which otherwise produce zero-length
/// tangents and NaNs downstream.
pub fn dedupe(pts: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::with_capacity(pts.len());
    for &p in pts {
        if out.last().is_none_or(|l| l.dist_sq(p) > 1e-20) {
            out.push(p);
        }
    }
    out
}

/// One-sided tangent at index `i`, looking `forward` or backward, averaged over
/// a few vertices so that a single noisy step does not set the direction.
fn one_sided_tangent(pts: &[Point], i: usize, forward: bool) -> Point {
    let n = pts.len();
    let mut acc = Point::ZERO;
    let mut weight = 0.0;
    for k in 1..=3usize {
        let j = if forward {
            if i + k >= n {
                break;
            }
            i + k
        } else {
            if i < k {
                break;
            }
            i - k
        };
        let d = pts[j] - pts[i];
        if d.length_sq() < 1e-18 {
            continue;
        }
        // Nearer samples describe the direction at `i` more faithfully.
        let w = 1.0 / k as f64;
        acc += d.normalized() * w;
        weight += w;
    }
    if weight <= 0.0 || acc.length_sq() < 1e-18 {
        // Degenerate: fall back to the chord across whatever exists.
        let j = if forward { n - 1 } else { 0 };
        let d = pts[j] - pts[i];
        return d.normalized();
    }
    acc.normalized()
}

/// Smooth tangent through an interior point, used where a run is split for
/// tolerance rather than at a real corner. Keeping both sides parallel here is
/// what stops the split from showing up as a visible kink.
fn centre_tangent(pts: &[Point], i: usize) -> Point {
    let a = one_sided_tangent(pts, i, false);
    let b = one_sided_tangent(pts, i, true);
    let d = b - a;
    if d.length_sq() < 1e-18 {
        b
    } else {
        d.normalized()
    }
}

fn bezier_basis(u: f64) -> [f64; 4] {
    let mu = 1.0 - u;
    [
        mu * mu * mu,
        3.0 * mu * mu * u,
        3.0 * mu * u * u,
        u * u * u,
    ]
}

/// Chord-length parameterisation, normalised to 0..1.
fn chord_params(pts: &[Point]) -> Vec<f64> {
    let mut u = Vec::with_capacity(pts.len());
    u.push(0.0);
    let mut total = 0.0;
    for i in 1..pts.len() {
        total += pts[i].dist(pts[i - 1]);
        u.push(total);
    }
    if total <= 1e-12 {
        let n = pts.len().max(2) - 1;
        return (0..pts.len()).map(|i| i as f64 / n as f64).collect();
    }
    for v in u.iter_mut() {
        *v /= total;
    }
    u
}

/// Schneider's least-squares cubic with both end tangents fixed. Solves only
/// for the two handle lengths.
fn fit_cubic_with_tangents(pts: &[Point], u: &[f64], t1: Point, t2: Point) -> (Point, Point) {
    let p0 = pts[0];
    let p3 = pts[pts.len() - 1];
    let default_alpha = p0.dist(p3) / 3.0;

    let mut c00 = 0.0;
    let mut c01 = 0.0;
    let mut c11 = 0.0;
    let mut x0 = 0.0;
    let mut x1 = 0.0;

    for (i, &p) in pts.iter().enumerate() {
        let b = bezier_basis(u[i]);
        let a0 = t1 * b[1];
        let a1 = t2 * b[2];
        c00 += a0.dot(a0);
        c01 += a0.dot(a1);
        c11 += a1.dot(a1);
        // Residual after accounting for the fixed endpoints.
        let tmp = p - (p0 * (b[0] + b[1]) + p3 * (b[2] + b[3]));
        x0 += a0.dot(tmp);
        x1 += a1.dot(tmp);
    }

    let det = c00 * c11 - c01 * c01;
    let (mut alpha_l, mut alpha_r) = if det.abs() < 1e-12 {
        (default_alpha, default_alpha)
    } else {
        (
            (x0 * c11 - x1 * c01) / det,
            (c00 * x1 - c01 * x0) / det,
        )
    };

    // Negative or vanishing handles fold the curve back on itself. Wolberg's
    // standard remedy: fall back to the chord-thirds heuristic.
    let eps = p0.dist(p3) * 1e-6;
    if !alpha_l.is_finite() || alpha_l < eps {
        alpha_l = default_alpha;
    }
    if !alpha_r.is_finite() || alpha_r < eps {
        alpha_r = default_alpha;
    }
    (p0 + t1 * alpha_l, p3 + t2 * alpha_r)
}

/// Largest distance from any sample to the curve at its own parameter, and the
/// index where it occurs.
fn cubic_max_error(pts: &[Point], u: &[f64], p0: Point, p1: Point, p2: Point, p3: Point) -> (f64, usize) {
    let mut worst = 0.0;
    let mut at = pts.len() / 2;
    for (i, &p) in pts.iter().enumerate() {
        let d = eval_cubic(p0, p1, p2, p3, u[i]).dist(p);
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// One Newton-Raphson step per sample, moving each parameter toward the closest
/// point on the curve. Without this, chord-length parameters alone leave a fit
/// noticeably worse than the curve is capable of.
fn reparameterise(pts: &[Point], u: &[f64], p0: Point, p1: Point, p2: Point, p3: Point) -> Vec<f64> {
    pts.iter()
        .enumerate()
        .map(|(i, &p)| {
            let t = u[i];
            let q = eval_cubic(p0, p1, p2, p3, t);
            let d1 = cubic_tangent(p0, p1, p2, p3, t);
            // Second derivative of the cubic.
            let d2 = (p2 - p1 * 2.0 + p0) * (6.0 * (1.0 - t)) + (p3 - p2 * 2.0 + p1) * (6.0 * t);
            let diff = q - p;
            let num = diff.dot(d1);
            let den = d1.dot(d1) + diff.dot(d2);
            if den.abs() < 1e-12 {
                t
            } else {
                (t - num / den).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Algebraic circle fit (Kasa), on mean-centred data for conditioning.
/// Returns centre and radius.
fn fit_circle(pts: &[Point]) -> Option<(Point, f64)> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let mean = pts.iter().fold(Point::ZERO, |a, &b| a + b) / n as f64;
    let (mut sxx, mut syy, mut sxy, mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &p in pts {
        let x = p.x - mean.x;
        let y = p.y - mean.y;
        let z = x * x + y * y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-12 {
        return None; // collinear
    }
    let d = (-sxz * syy + syz * sxy) / det;
    let e = (-syz * sxx + sxz * sxy) / det;
    let f = -sz / n as f64;
    let cx = -d / 2.0;
    let cy = -e / 2.0;
    let r2 = cx * cx + cy * cy - f;
    if r2 <= 0.0 {
        return None;
    }
    Some((Point::new(cx + mean.x, cy + mean.y), r2.sqrt()))
}

/// Try to describe `pts` as a single circular arc within `tol`.
fn try_arc(pts: &[Point], tol: f64) -> Option<Seg> {
    let (centre, r) = fit_circle(pts)?;
    // An enormous radius means the run is effectively straight, which the line
    // test already handles more cheaply and more stably.
    let span = pts[0].dist(pts[pts.len() - 1]);
    if r > span * 200.0 || r < 1e-6 {
        return None;
    }
    // Radial deviation is a necessary condition and cheap to reject on.
    for p in pts {
        if (p.dist(centre) - r).abs() > tol {
            return None;
        }
    }

    // Total swept angle, unwrapped along the run, gives both flags.
    let angle = |p: &Point| (p.y - centre.y).atan2(p.x - centre.x);
    let mut total = 0.0;
    for w in pts.windows(2) {
        let mut d = angle(&w[1]) - angle(&w[0]);
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        total += d;
    }
    // A run that wraps almost all the way round cannot be one SVG arc.
    if total.abs() >= std::f64::consts::TAU * 0.97 || total.abs() < 1e-6 {
        return None;
    }

    let seg = Seg::Arc {
        r,
        large: total.abs() > std::f64::consts::PI,
        sweep: total > 0.0,
        to: pts[pts.len() - 1],
    };

    // Verify by reconstruction rather than trusting the flag logic: build the
    // arc exactly as an exporter would and confirm it still passes through the
    // samples. A flag error is otherwise silent and produces a wildly wrong
    // curve.
    let Seg::Arc { r, large, sweep, to } = seg else {
        unreachable!()
    };
    let cubics = arc_to_cubics(pts[0], r, large, sweep, to);
    if cubics.is_empty() {
        return None;
    }
    let mut poly = vec![pts[0]];
    let mut cur = pts[0];
    for (c1, c2, end) in &cubics {
        for k in 1..=16 {
            poly.push(eval_cubic(cur, *c1, *c2, *end, k as f64 / 16.0));
        }
        cur = *end;
    }
    for p in pts {
        let mut best = f64::INFINITY;
        for w in poly.windows(2) {
            best = best.min(dist_point_segment(*p, w[0], w[1]));
        }
        if best > tol {
            return None;
        }
    }
    Some(seg)
}

/// Fit the run `pts[i0..=i1]`, appending commands to `out`.
#[allow(clippy::too_many_arguments)]
fn fit_run(
    pts: &[Point],
    i0: usize,
    i1: usize,
    t_left: Point,
    t_right: Point,
    cfg: &FitConfig,
    depth: usize,
    out: &mut Vec<Seg>,
) {
    let span = &pts[i0..=i1];
    if span.len() < 2 {
        return;
    }
    let p0 = span[0];
    let p3 = span[span.len() - 1];

    if span.len() == 2 {
        out.push(Seg::Line { to: p3 });
        return;
    }

    // 1. Straight line: the cheapest possible answer.
    if cfg.allow_lines {
        let worst = span
            .iter()
            .map(|p| dist_point_segment(*p, p0, p3))
            .fold(0.0f64, f64::max);
        if worst <= cfg.tolerance {
            out.push(Seg::Line { to: p3 });
            return;
        }
    }

    // 2. One cubic, refined by reparameterisation.
    let mut u = chord_params(span);
    let mut best: Option<(f64, Point, Point)> = None;
    for _ in 0..=cfg.reparam_iterations {
        let (p1, p2) = fit_cubic_with_tangents(span, &u, t_left, t_right);
        let (err, _) = cubic_max_error(span, &u, p0, p1, p2, p3);
        if best.is_none_or(|b| err < b.0) {
            best = Some((err, p1, p2));
        }
        if err <= cfg.tolerance {
            break;
        }
        u = reparameterise(span, &u, p0, p1, p2, p3);
    }
    if let Some((err, p1, p2)) = best {
        if err <= cfg.tolerance {
            out.push(Seg::Cubic { c1: p1, c2: p2, to: p3 });
            return;
        }
    }

    // 3. A circular arc, which can hold a far wider sweep than one cubic.
    if cfg.allow_arcs {
        if let Some(seg) = try_arc(span, cfg.tolerance) {
            if cfg.emit_true_arcs {
                out.push(seg);
            } else if let Seg::Arc { r, large, sweep, to } = seg {
                let mut cur = p0;
                for (c1, c2, end) in arc_to_cubics(p0, r, large, sweep, to) {
                    out.push(Seg::Cubic { c1, c2, to: end });
                    cur = end;
                }
                let _ = cur;
            }
            return;
        }
    }

    // 4. Split at the worst point and fit both halves, keeping the tangent
    // continuous across the join so the split does not become a visible kink.
    if depth >= cfg.max_depth {
        for p in &span[1..] {
            out.push(Seg::Line { to: *p });
        }
        return;
    }

    let split = {
        let (_, at) = {
            let (p1, p2) = fit_cubic_with_tangents(span, &u, t_left, t_right);
            cubic_max_error(span, &u, p0, p1, p2, p3)
        };
        // Keep at least one interior point on each side.
        (i0 + at).clamp(i0 + 1, i1 - 1)
    };

    let tc = centre_tangent(pts, split);
    fit_run(pts, i0, split, t_left, -tc, cfg, depth + 1, out);
    fit_run(pts, split, i1, tc, t_right, cfg, depth + 1, out);
}

/// Fit one contour, splitting at the supplied corner indices.
///
/// `closed` marks a loop whose last point repeats its first; such a contour may
/// legitimately have no corners at all, in which case it is fitted as a single
/// smooth closed curve.
pub fn fit_contour(pts: &[Point], corners: &[usize], closed: bool, cfg: &FitConfig) -> Vec<Seg> {
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut breaks: Vec<usize> = corners.iter().copied().filter(|&c| c < n).collect();
    breaks.sort_unstable();
    breaks.dedup();

    if closed && breaks.is_empty() {
        // Smooth closed loop: one run from the seam back to itself, with the
        // tangent at the seam taken cyclically so the loop closes without a
        // kink.
        let seam = if n >= 3 {
            (pts[1] - pts[n - 2]).normalized()
        } else {
            (pts[n - 1] - pts[0]).normalized()
        };
        fit_run(pts, 0, n - 1, seam, -seam, cfg, 0, &mut out);
        if let Some(last) = out.last_mut() {
            last.set_end(pts[0]);
        }
        return out;
    }

    if breaks.first() != Some(&0) {
        breaks.insert(0, 0);
    }
    if breaks.last() != Some(&(n - 1)) {
        breaks.push(n - 1);
    }

    for w in breaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b <= a {
            continue;
        }
        let t_left = one_sided_tangent(pts, a, true);
        let t_right = one_sided_tangent(pts, b, false);
        fit_run(pts, a, b, t_left, t_right, cfg, 0, &mut out);
    }
    out
}

/// Total off-curve handles in a command list.
pub fn count_handles(segs: &[Seg]) -> usize {
    segs.iter().map(|s| s.handles()).sum()
}

/// Drop corners that turn out not to be needed.
///
/// The detector errs slightly toward keeping corners, because a lost corner is
/// far more visible than a spare one. This pass takes each interior corner out
/// in turn and refits the merged run; if the result still meets tolerance and
/// uses fewer commands, the corner was not earning its keep.
pub fn prune_corners(pts: &[Point], corners: &[usize], closed: bool, cfg: &FitConfig) -> Vec<usize> {
    let _ = closed;
    if !cfg.merge_pass || corners.len() <= 2 {
        return corners.to_vec();
    }
    let mut kept: Vec<usize> = corners.to_vec();
    let mut i = 1;
    while i + 1 < kept.len() {
        // Removing a corner only changes the two runs either side of it, so
        // only those are refitted. Refitting the whole contour for every
        // candidate made this cubic in contour length, which cost seconds on a
        // multi-megapixel image.
        let (a, c, b) = (kept[i - 1], kept[i], kept[i + 1]);
        if b <= a {
            i += 1;
            continue;
        }

        let left = fit_span(pts, a, c, cfg);
        let right = fit_span(pts, c, b, cfg);
        let with_corner = (left.len() + right.len(), count_handles(&left) + count_handles(&right));

        let merged = fit_span(pts, a, b, cfg);
        let without = (merged.len(), count_handles(&merged));

        if without < with_corner && max_deviation(&pts[a..=b], &merged) <= cfg.tolerance {
            kept.remove(i);
        } else {
            i += 1;
        }
    }
    kept
}

/// Fit a single run between two indices, with one-sided end tangents.
fn fit_span(pts: &[Point], a: usize, b: usize, cfg: &FitConfig) -> Vec<Seg> {
    let mut out = Vec::new();
    if b <= a || b >= pts.len() {
        return out;
    }
    let t_left = one_sided_tangent(pts, a, true);
    let t_right = one_sided_tangent(pts, b, false);
    fit_run(pts, a, b, t_left, t_right, cfg, 0, &mut out);
    out
}

/// Independent check that a command list stays within `tol` of the samples.
/// Deliberately measured against the flattened curve rather than reusing the
/// fitter's own parameterisation, so that a fit cannot pass by grading itself.
pub fn fit_is_within(pts: &[Point], segs: &[Seg], tol: f64) -> bool {
    max_deviation(pts, segs) <= tol
}

/// Largest distance from any sample point to the fitted curve.
pub fn max_deviation(pts: &[Point], segs: &[Seg]) -> f64 {
    if pts.is_empty() || segs.is_empty() {
        return 0.0;
    }
    // Flatten adaptively and far finer than any tolerance we check against.
    // A fixed subdivision count would charge long, strongly curved segments for
    // the chord error of the measuring stick rather than of the fit.
    let sub = crate::geom::SubPath {
        start: pts[0],
        segs: segs.to_vec(),
        closed: false,
    };
    let poly = sub.flatten(1e-3);
    let mut worst: f64 = 0.0;
    for p in pts {
        let mut best = f64::INFINITY;
        for w in poly.windows(2) {
            best = best.min(dist_point_segment(*p, w[0], w[1]));
            if best <= 1e-12 {
                break;
            }
        }
        worst = worst.max(best);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::pt;

    fn cfg() -> FitConfig {
        FitConfig::default()
    }

    #[test]
    fn a_straight_run_becomes_one_line_with_no_handles() {
        let pts: Vec<Point> = (0..40).map(|i| pt(i as f64 * 0.5, i as f64 * 0.25)).collect();
        let segs = fit_contour(&pts, &[0, pts.len() - 1], false, &cfg());
        assert_eq!(segs.len(), 1, "{:?}", segs);
        assert_eq!(count_handles(&segs), 0);
        assert!(matches!(segs[0], Seg::Line { .. }));
    }

    #[test]
    fn a_quarter_circle_costs_one_segment() {
        let n = 50;
        let pts: Vec<Point> = (0..=n)
            .map(|i| {
                let t = (i as f64 / n as f64) * std::f64::consts::FRAC_PI_2;
                pt(40.0 * t.cos(), 40.0 * t.sin())
            })
            .collect();
        let segs = fit_contour(&pts, &[0, pts.len() - 1], false, &cfg());
        assert_eq!(segs.len(), 1, "quarter circle took {} segments", segs.len());
        assert!(max_deviation(&pts, &segs) < cfg().tolerance);
    }

    #[test]
    fn a_full_circle_is_compact_and_accurate() {
        // The benchmark case for control-point economy: a circle should cost a
        // handful of segments, not one per sampled pixel.
        let n = 240;
        let mut pts: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                pt(100.0 + 60.0 * t.cos(), 100.0 + 60.0 * t.sin())
            })
            .collect();
        pts.push(pts[0]);
        let segs = fit_contour(&pts, &[], true, &cfg());
        assert!(
            segs.len() <= 8,
            "circle took {} segments: {:?}",
            segs.len(),
            segs.len()
        );
        assert!(
            max_deviation(&pts, &segs) < cfg().tolerance,
            "deviation {}",
            max_deviation(&pts, &segs)
        );
    }

    #[test]
    fn true_arcs_are_more_compact_than_cubics() {
        let n = 120;
        let pts: Vec<Point> = (0..=n)
            .map(|i| {
                let t = (i as f64 / n as f64) * std::f64::consts::PI * 1.5;
                pt(50.0 * t.cos(), 50.0 * t.sin())
            })
            .collect();
        let mut c = cfg();
        c.emit_true_arcs = true;
        let with_arcs = fit_contour(&pts, &[0, pts.len() - 1], false, &c);
        c.emit_true_arcs = false;
        c.allow_arcs = false;
        let without = fit_contour(&pts, &[0, pts.len() - 1], false, &c);
        assert!(
            with_arcs.len() < without.len(),
            "arcs {} vs cubics {}",
            with_arcs.len(),
            without.len()
        );
        assert!(max_deviation(&pts, &with_arcs) < c.tolerance);
    }

    #[test]
    fn corners_are_preserved_as_separate_runs() {
        let mut pts: Vec<Point> = (0..20).map(|i| pt(i as f64, 0.0)).collect();
        pts.extend((1..20).map(|i| pt(19.0, i as f64)));
        let corner = 19;
        let segs = fit_contour(&pts, &[0, corner, pts.len() - 1], false, &cfg());
        assert_eq!(segs.len(), 2);
        assert_eq!(count_handles(&segs), 0, "an L should cost two lines");
    }

    #[test]
    fn a_wiggly_curve_stays_within_tolerance() {
        let pts: Vec<Point> = (0..200)
            .map(|i| {
                let x = i as f64 * 0.4;
                pt(x, 20.0 * (x * 0.08).sin() + 6.0 * (x * 0.31).cos())
            })
            .collect();
        let segs = fit_contour(&pts, &[0, pts.len() - 1], false, &cfg());
        let dev = max_deviation(&pts, &segs);
        assert!(dev <= cfg().tolerance, "deviation {dev}");
        assert!(segs.len() < 24, "took {} segments", segs.len());
    }

    #[test]
    fn tighter_tolerance_costs_more_segments() {
        let pts: Vec<Point> = (0..160)
            .map(|i| {
                let x = i as f64 * 0.5;
                pt(x, 14.0 * (x * 0.09).sin())
            })
            .collect();
        let mut loose = cfg();
        loose.tolerance = 1.5;
        let mut tight = cfg();
        tight.tolerance = 0.05;
        let a = fit_contour(&pts, &[0, pts.len() - 1], false, &loose);
        let b = fit_contour(&pts, &[0, pts.len() - 1], false, &tight);
        assert!(a.len() <= b.len(), "loose {} tight {}", a.len(), b.len());
        assert!(max_deviation(&pts, &a) <= loose.tolerance);
        assert!(max_deviation(&pts, &b) <= tight.tolerance);
    }

    #[test]
    fn pruning_drops_a_corner_that_is_not_really_there() {
        // A smooth arc with a spurious corner marked in the middle of it.
        let n = 60;
        let pts: Vec<Point> = (0..=n)
            .map(|i| {
                let t = (i as f64 / n as f64) * std::f64::consts::FRAC_PI_2;
                pt(40.0 * t.cos(), 40.0 * t.sin())
            })
            .collect();
        let corners = vec![0, n / 2, n];
        let pruned = prune_corners(&pts, &corners, false, &cfg());
        assert_eq!(pruned, vec![0, n], "spurious corner survived: {:?}", pruned);
    }

    #[test]
    fn pruning_keeps_a_corner_that_is_really_there() {
        let mut pts: Vec<Point> = (0..20).map(|i| pt(i as f64, 0.0)).collect();
        pts.extend((1..20).map(|i| pt(19.0, i as f64)));
        let corners = vec![0, 19, pts.len() - 1];
        let pruned = prune_corners(&pts, &corners, false, &cfg());
        assert_eq!(pruned, corners, "a real corner was pruned away");
    }

    #[test]
    fn pruning_a_long_contour_is_not_quadratic() {
        // Regression guard. Pruning used to refit the entire contour, twice,
        // for every candidate corner, and verify it with an all-pairs distance
        // check -- cubic in contour length overall. On a multi-megapixel image
        // that cost seconds per trace.
        //
        // This contour is long with many corners, which is exactly the shape
        // that triggered it. Under the old implementation this single call took
        // longer than the whole rest of the suite; the assertion below is
        // generous enough not to be flaky but far below that.
        let n = 4000;
        let pts: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.05;
                // A wandering path with frequent genuine direction changes.
                pt(t, 8.0 * (t * 0.7).sin() + 3.0 * (t * 2.3).cos())
            })
            .collect();
        let corners: Vec<usize> = (0..n).step_by(10).collect();

        let start = std::time::Instant::now();
        let pruned = prune_corners(&pts, &corners, false, &cfg());
        let elapsed = start.elapsed();

        assert!(pruned.len() <= corners.len());
        assert!(!pruned.is_empty());
        assert!(
            elapsed.as_millis() < 3000,
            "pruning {n} points with {} corners took {elapsed:?}",
            corners.len()
        );
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert!(fit_contour(&[], &[], false, &cfg()).is_empty());
        assert!(fit_contour(&[pt(1.0, 1.0)], &[], false, &cfg()).is_empty());
        let two = vec![pt(0.0, 0.0), pt(1.0, 1.0)];
        assert_eq!(fit_contour(&two, &[0, 1], false, &cfg()).len(), 1);
        let same = vec![pt(2.0, 2.0); 6];
        let _ = fit_contour(&same, &[0, 5], false, &cfg());
    }
}
