//! Basic 2D geometry: points, vectors, and the vector path model.
//!
//! All coordinates are in *lattice space*: the top-left corner of the image is
//! (0, 0) and pixel (x, y) occupies the unit square [x, x+1] x [y, y+1], so its
//! centre sits at (x + 0.5, y + 0.5). Keeping tracing in lattice space rather
//! than pixel-centre space means the crack edges between pixels fall on
//! integers, which is what the topology pass wants.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub const fn pt(x: f64, y: f64) -> Point {
    Point { x, y }
}

impl Point {
    pub const ZERO: Point = Point { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn dot(self, o: Point) -> f64 {
        self.x * o.x + self.y * o.y
    }

    /// 2D cross product: the z component of the equivalent 3D cross product.
    pub fn cross(self, o: Point) -> f64 {
        self.x * o.y - self.y * o.x
    }

    pub fn dist(self, o: Point) -> f64 {
        (self - o).length()
    }

    pub fn dist_sq(self, o: Point) -> f64 {
        (self - o).length_sq()
    }

    pub fn normalized(self) -> Point {
        let l = self.length();
        if l < 1e-12 {
            Point::ZERO
        } else {
            Point { x: self.x / l, y: self.y / l }
        }
    }

    /// Rotate by 90 degrees. In a y-down coordinate system this yields the
    /// leftward normal of a direction vector.
    pub fn perp(self) -> Point {
        Point { x: self.y, y: -self.x }
    }

    pub fn lerp(self, o: Point, t: f64) -> Point {
        Point {
            x: self.x + (o.x - self.x) * t,
            y: self.y + (o.y - self.y) * t,
        }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Round to a fixed number of decimal places. Emitting geometry through
    /// this keeps shared borders byte-identical between adjacent shapes.
    pub fn round_to(self, decimals: u32) -> Point {
        let f = 10f64.powi(decimals as i32);
        Point {
            x: (self.x * f).round() / f,
            y: (self.y * f).round() / f,
        }
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, o: Point) -> Point {
        Point { x: self.x + o.x, y: self.y + o.y }
    }
}

impl AddAssign for Point {
    fn add_assign(&mut self, o: Point) {
        self.x += o.x;
        self.y += o.y;
    }
}

impl Sub for Point {
    type Output = Point;
    fn sub(self, o: Point) -> Point {
        Point { x: self.x - o.x, y: self.y - o.y }
    }
}

impl Mul<f64> for Point {
    type Output = Point;
    fn mul(self, s: f64) -> Point {
        Point { x: self.x * s, y: self.y * s }
    }
}

impl Div<f64> for Point {
    type Output = Point;
    fn div(self, s: f64) -> Point {
        Point { x: self.x / s, y: self.y / s }
    }
}

impl Neg for Point {
    type Output = Point;
    fn neg(self) -> Point {
        Point { x: -self.x, y: -self.y }
    }
}

/// Perpendicular distance from p to the infinite line through a and b.
pub fn dist_point_line(p: Point, a: Point, b: Point) -> f64 {
    let d = b - a;
    let l = d.length();
    if l < 1e-12 {
        return p.dist(a);
    }
    ((p - a).cross(d) / l).abs()
}

/// Distance from p to the finite segment a-b.
pub fn dist_point_segment(p: Point, a: Point, b: Point) -> f64 {
    let d = b - a;
    let l2 = d.length_sq();
    if l2 < 1e-24 {
        return p.dist(a);
    }
    let t = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    p.dist(a + d * t)
}

/// Signed turn angle at b along a -> b -> c, in radians within [-pi, pi].
pub fn turn_angle(a: Point, b: Point, c: Point) -> f64 {
    let u = b - a;
    let v = c - b;
    if u.length_sq() < 1e-24 || v.length_sq() < 1e-24 {
        return 0.0;
    }
    u.cross(v).atan2(u.dot(v))
}

/// Shoelace area of a closed polygon. Our left-hand-rule contour tracer
/// produces negative area for outer boundaries in y-down space, positive for
/// holes.
pub fn polygon_area(pts: &[Point]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        acc += a.cross(b);
    }
    acc * 0.5
}

// ---------------------------------------------------------------------------
// Path model
// ---------------------------------------------------------------------------

/// One drawing command. The start point is implicit: the previous command's
/// end, or the subpath start.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Seg {
    Line { to: Point },
    Cubic { c1: Point, c2: Point, to: Point },
    /// A true circular arc, following the SVG elliptical-arc convention with
    /// equal radii and no rotation.
    Arc { r: f64, large: bool, sweep: bool, to: Point },
}

impl Seg {
    pub fn end(&self) -> Point {
        match *self {
            Seg::Line { to } => to,
            Seg::Cubic { to, .. } => to,
            Seg::Arc { to, .. } => to,
        }
    }

    pub fn set_end(&mut self, p: Point) {
        match self {
            Seg::Line { to } => *to = p,
            Seg::Cubic { to, .. } => *to = p,
            Seg::Arc { to, .. } => *to = p,
        }
    }

    /// Number of off-curve control handles this command contributes.
    pub fn handles(&self) -> usize {
        match self {
            Seg::Line { .. } => 0,
            Seg::Cubic { .. } => 2,
            // An arc lands in an editor as one node with an implied handle
            // pair, so count it as 2 to keep comparisons against cubic output
            // honest rather than flattering the arc fitter.
            Seg::Arc { .. } => 2,
        }
    }

    /// Reverse this command, given the point it originally started from.
    pub fn reversed(&self, from: Point) -> Seg {
        match *self {
            Seg::Line { .. } => Seg::Line { to: from },
            Seg::Cubic { c1, c2, .. } => Seg::Cubic { c1: c2, c2: c1, to: from },
            Seg::Arc { r, large, sweep, .. } => Seg::Arc {
                r,
                large,
                sweep: !sweep,
                to: from,
            },
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SubPath {
    pub start: Point,
    pub segs: Vec<Seg>,
    pub closed: bool,
}

impl SubPath {
    pub fn new(start: Point) -> Self {
        SubPath { start, segs: Vec::new(), closed: false }
    }

    pub fn end(&self) -> Point {
        self.segs.last().map(|s| s.end()).unwrap_or(self.start)
    }

    /// On-curve anchor points. A closed subpath's last anchor coincides with
    /// its start, so it is not counted twice.
    pub fn anchors(&self) -> usize {
        if self.closed {
            self.segs.len()
        } else {
            self.segs.len() + 1
        }
    }

    pub fn handles(&self) -> usize {
        self.segs.iter().map(|s| s.handles()).sum()
    }

    /// Flatten to a polyline for scoring, DXF export and previews.
    pub fn flatten(&self, tol: f64) -> Vec<Point> {
        let mut out = vec![self.start];
        let mut cur = self.start;
        for seg in &self.segs {
            match *seg {
                Seg::Line { to } => {
                    out.push(to);
                    cur = to;
                }
                Seg::Cubic { c1, c2, to } => {
                    flatten_cubic(cur, c1, c2, to, tol, &mut out);
                    cur = to;
                }
                Seg::Arc { r, large, sweep, to } => {
                    for c in arc_to_cubics(cur, r, large, sweep, to) {
                        flatten_cubic(cur, c.0, c.1, c.2, tol, &mut out);
                        cur = c.2;
                    }
                    cur = to;
                }
            }
        }
        out
    }
}

pub fn eval_cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: f64) -> Point {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let c = 3.0 * mt * t * t;
    let d = t * t * t;
    Point {
        x: a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        y: a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    }
}

pub fn cubic_tangent(p0: Point, p1: Point, p2: Point, p3: Point, t: f64) -> Point {
    let mt = 1.0 - t;
    (p1 - p0) * (3.0 * mt * mt) + (p2 - p1) * (6.0 * mt * t) + (p3 - p2) * (3.0 * t * t)
}

fn flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tol: f64, out: &mut Vec<Point>) {
    // Wang's formula for the subdivision count that keeps the polyline within
    // tol of the curve.
    let d1 = (p0 - p1 * 2.0 + p2).length();
    let d2 = (p1 - p2 * 2.0 + p3).length();
    let m = d1.max(d2);
    let n = if m <= 0.0 {
        1
    } else {
        (((6.0 * m) / (8.0 * tol.max(1e-6))).sqrt().ceil() as usize).clamp(1, 256)
    };
    for i in 1..=n {
        out.push(eval_cubic(p0, p1, p2, p3, i as f64 / n as f64));
    }
}

/// Convert a circular arc into up to four cubic Beziers, each spanning at most
/// a quarter turn.
pub fn arc_to_cubics(
    from: Point,
    r: f64,
    large: bool,
    sweep: bool,
    to: Point,
) -> Vec<(Point, Point, Point)> {
    let chord = to - from;
    let d = chord.length();
    let mut out = Vec::new();
    if d < 1e-12 {
        return out;
    }
    let r = r.abs().max(d / 2.0);
    let h = (r * r - d * d / 4.0).max(0.0).sqrt();
    let mid = from.lerp(to, 0.5);
    // The perpendicular offset selects which of the two candidate centres the
    // large/sweep flag combination asks for. Both candidates put the endpoints
    // on a circle of radius r, so only the flag semantics distinguish them --
    // which is why a sign error here is invisible to a "points lie on the
    // circle" check and has to be tested against the sweep directly.
    // Derived from the SVG endpoint-to-centre conversion: the offset is
    // positive along perp(chord) exactly when the two flags agree.
    let n = chord.perp().normalized();
    let sign = if large == sweep { 1.0 } else { -1.0 };
    let centre = mid + n * (h * sign);

    let a0 = (from.y - centre.y).atan2(from.x - centre.x);
    let a1 = (to.y - centre.y).atan2(to.x - centre.x);
    let mut sweep_angle = a1 - a0;
    // In y-down space the SVG sweep flag corresponds to increasing angle.
    if sweep && sweep_angle < 0.0 {
        sweep_angle += std::f64::consts::TAU;
    }
    if !sweep && sweep_angle > 0.0 {
        sweep_angle -= std::f64::consts::TAU;
    }

    let n_seg = ((sweep_angle.abs() / std::f64::consts::FRAC_PI_2).ceil() as usize).max(1);
    let delta = sweep_angle / n_seg as f64;
    let k = (4.0 / 3.0) * (delta / 4.0).tan();
    let mut theta = a0;
    for _ in 0..n_seg {
        let t1 = theta + delta;
        let p0 = Point::new(centre.x + r * theta.cos(), centre.y + r * theta.sin());
        let p3 = Point::new(centre.x + r * t1.cos(), centre.y + r * t1.sin());
        let d0 = Point::new(-theta.sin(), theta.cos()) * (r * k);
        let d1 = Point::new(-t1.sin(), t1.cos()) * (r * k);
        out.push((p0 + d0, p3 - d1, p3));
        theta = t1;
    }
    if let Some(last) = out.last_mut() {
        last.2 = to;
    }
    out
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub fn empty() -> Self {
        Rect {
            x0: f64::INFINITY,
            y0: f64::INFINITY,
            x1: f64::NEG_INFINITY,
            y1: f64::NEG_INFINITY,
        }
    }

    pub fn add(&mut self, p: Point) {
        self.x0 = self.x0.min(p.x);
        self.y0 = self.y0.min(p.y);
        self.x1 = self.x1.max(p.x);
        self.y1 = self.y1.max(p.y);
    }

    pub fn width(&self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }

    pub fn height(&self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_ring_winding_is_negative() {
        // The crack tracer emits a unit pixel as this ring; the sign convention
        // that outer rings are negative underpins hole detection.
        let ring = [pt(1.0, 0.0), pt(0.0, 0.0), pt(0.0, 1.0), pt(1.0, 1.0)];
        assert!(polygon_area(&ring) < 0.0);
    }

    #[test]
    fn arc_endpoints_are_exact() {
        let from = pt(0.0, 0.0);
        let to = pt(10.0, 10.0);
        let cubics = arc_to_cubics(from, 8.0, false, true, to);
        assert!(!cubics.is_empty());
        let end = cubics.last().unwrap().2;
        assert!(end.dist(to) < 1e-9);
    }

    /// Both candidate centres put the endpoints on a circle of the right
    /// radius, so checking "the points lie on a circle" cannot detect a flag or
    /// sign error. This drives a known arc through the conversion and checks it
    /// traces the arc that was actually asked for -- right centre, right way
    /// round, right side.
    #[test]
    fn arc_reconstructs_the_arc_that_was_requested() {
        let centre = pt(12.0, -5.0);
        let r = 9.0;
        for start_deg in [0.0f64, 37.0, 150.0, 264.0] {
            for sweep_deg in [-300.0f64, -190.0, -95.0, -40.0, 40.0, 95.0, 190.0, 300.0] {
                let a0 = start_deg.to_radians();
                let a1 = a0 + sweep_deg.to_radians();
                let on = |a: f64| pt(centre.x + r * a.cos(), centre.y + r * a.sin());
                let from = on(a0);
                let to = on(a1);
                let large = sweep_deg.abs() > 180.0;
                let sweep = sweep_deg > 0.0;

                let cubics = arc_to_cubics(from, r, large, sweep, to);
                assert!(!cubics.is_empty());

                // Every sampled point must sit on the intended circle.
                let mut cur = from;
                let mut samples = Vec::new();
                for (c1, c2, end) in &cubics {
                    for i in 1..=8 {
                        let p = eval_cubic(cur, *c1, *c2, *end, i as f64 / 8.0);
                        assert!(
                            (p.dist(centre) - r).abs() < 0.02,
                            "start {start_deg} sweep {sweep_deg}: {:?} is off the circle",
                            p
                        );
                        samples.push(p);
                    }
                    cur = *end;
                }
                assert!(cur.dist(to) < 1e-9, "endpoint drift");

                // And it must pass through the midpoint of the requested arc,
                // not the midpoint of the complementary one.
                let want_mid = on(a0 + sweep_deg.to_radians() / 2.0);
                let closest = samples
                    .iter()
                    .map(|p| p.dist(want_mid))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    closest < 0.5,
                    "start {start_deg} sweep {sweep_deg}: traced the wrong arc \
                     (nearest sample to the true midpoint was {closest} away)"
                );
            }
        }
    }

    #[test]
    fn turn_angle_signs() {
        let t = turn_angle(pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0));
        assert!(t > 0.0);
        let t2 = turn_angle(pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, -1.0));
        assert!(t2 < 0.0);
    }
}
