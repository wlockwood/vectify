//! Vector format writers.
//!
//! All four writers consume the same [`VectorImage`], which has already been
//! scaled and coordinate-quantised by the pipeline. Rounding earlier rather
//! than per-writer is deliberate: it guarantees that a border shared by two
//! shapes serialises to byte-identical numbers on both sides, whichever format
//! is being written.

pub mod dxf;
pub mod eps;
pub mod pdf;
pub mod svg;

use crate::color::Rgba8;
use crate::config::{OutputConfig, VectorFormat};
use crate::geom::{arc_to_cubics, Point, Seg, SubPath};
use crate::model::VectorImage;

/// A drawing operation reduced to the primitives every page-description
/// language understands: no arcs, and already in the target coordinate system.
#[derive(Clone, Copy, Debug)]
pub enum PathOp {
    Move(Point),
    Line(Point),
    Curve(Point, Point, Point),
    Close,
}

/// Lower a subpath to `PathOp`s, expanding arcs into cubics and optionally
/// flipping the y axis.
///
/// SVG puts the origin at the top left with y running down; PostScript, PDF and
/// DXF all put it at the bottom left with y running up. Passing the page height
/// here performs that flip once, in one place.
pub fn lower(sp: &SubPath, flip_height: Option<f64>) -> Vec<PathOp> {
    let f = |p: Point| match flip_height {
        Some(h) => Point::new(p.x, h - p.y),
        None => p,
    };
    let mut ops = vec![PathOp::Move(f(sp.start))];
    let mut cur = sp.start;
    for seg in &sp.segs {
        match *seg {
            Seg::Line { to } => {
                ops.push(PathOp::Line(f(to)));
                cur = to;
            }
            Seg::Cubic { c1, c2, to } => {
                ops.push(PathOp::Curve(f(c1), f(c2), f(to)));
                cur = to;
            }
            Seg::Arc { r, large, sweep, to } => {
                for (c1, c2, end) in arc_to_cubics(cur, r, large, sweep, to) {
                    ops.push(PathOp::Curve(f(c1), f(c2), f(end)));
                }
                cur = to;
            }
        }
    }
    if sp.closed {
        ops.push(PathOp::Close);
    }
    ops
}

/// Format a number compactly: fixed precision with trailing zeros stripped, and
/// no negative zero.
pub fn num(v: f64, precision: u32) -> String {
    if !v.is_finite() {
        return "0".to_string();
    }
    let mut s = format!("{:.*}", precision as usize, v);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" || s.is_empty() {
        s = "0".to_string();
    }
    s
}

/// Flatten a colour onto white. EPS, PDF and DXF fills in this writer are
/// opaque, so a translucent colour is composited rather than silently losing
/// its alpha.
pub fn over_white(c: Rgba8) -> [f64; 3] {
    let a = c.a as f64 / 255.0;
    [
        (c.r as f64 / 255.0) * a + (1.0 - a),
        (c.g as f64 / 255.0) * a + (1.0 - a),
        (c.b as f64 / 255.0) * a + (1.0 - a),
    ]
}

/// Write `image` in the requested format.
pub fn export(image: &VectorImage, cfg: &OutputConfig) -> Vec<u8> {
    match cfg.format {
        VectorFormat::Svg => svg::write(image, cfg).into_bytes(),
        VectorFormat::Eps => eps::write(image, cfg).into_bytes(),
        VectorFormat::Pdf => pdf::write(image, cfg),
        VectorFormat::Dxf => dxf::write(image, cfg).into_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formatting_is_compact() {
        assert_eq!(num(1.0, 3), "1");
        assert_eq!(num(1.5000, 3), "1.5");
        assert_eq!(num(1.23456, 3), "1.235");
        assert_eq!(num(-0.0001, 3), "0");
        assert_eq!(num(0.0, 3), "0");
        assert_eq!(num(-12.25, 2), "-12.25");
        assert_eq!(num(f64::NAN, 3), "0");
    }

    #[test]
    fn translucent_colours_composite_rather_than_vanish() {
        let half = Rgba8::new(0, 0, 0, 128);
        let c = over_white(half);
        assert!((c[0] - 0.498).abs() < 0.01);
    }
}
