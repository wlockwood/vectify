//! The vector document produced by tracing, and the complexity metrics used to
//! judge it.

use serde::{Deserialize, Serialize};

use crate::color::Rgba8;
use crate::geom::{Point, Rect, Seg, SubPath};

/// One filled shape. Its subpaths are the outer boundary plus any holes, and
/// they never overlap another shape's interior: the whole document is a planar
/// subdivision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shape {
    pub color: Rgba8,
    pub subpaths: Vec<SubPath>,
}

impl Shape {
    pub fn anchors(&self) -> usize {
        self.subpaths.iter().map(|s| s.anchors()).sum()
    }

    pub fn handles(&self) -> usize {
        self.subpaths.iter().map(|s| s.handles()).sum()
    }

    pub fn segments(&self) -> usize {
        self.subpaths.iter().map(|s| s.segs.len()).sum()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VectorImage {
    pub width: f64,
    pub height: f64,
    pub shapes: Vec<Shape>,
    /// Optional backdrop painted behind everything.
    pub background: Option<Rgba8>,
}

/// How much geometry the result costs. Fewer points at equal fidelity is the
/// whole game, so these are first-class outputs, not diagnostics.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Complexity {
    pub shapes: usize,
    pub subpaths: usize,
    pub segments: usize,
    /// On-curve points.
    pub anchors: usize,
    /// Off-curve control handles.
    pub handles: usize,
    pub lines: usize,
    pub cubics: usize,
    pub arcs: usize,
}

impl Complexity {
    /// The headline number: every point an editor would show you.
    pub fn total_points(&self) -> usize {
        self.anchors + self.handles
    }
}

impl VectorImage {
    pub fn complexity(&self) -> Complexity {
        let mut c = Complexity {
            shapes: self.shapes.len(),
            ..Default::default()
        };
        for s in &self.shapes {
            c.subpaths += s.subpaths.len();
            c.anchors += s.anchors();
            c.handles += s.handles();
            for sp in &s.subpaths {
                c.segments += sp.segs.len();
                for seg in &sp.segs {
                    match seg {
                        Seg::Line { .. } => c.lines += 1,
                        Seg::Cubic { .. } => c.cubics += 1,
                        Seg::Arc { .. } => c.arcs += 1,
                    }
                }
            }
        }
        c
    }

    pub fn bounds(&self) -> Rect {
        let mut r = Rect::empty();
        for s in &self.shapes {
            for sp in &s.subpaths {
                for p in sp.flatten(0.25) {
                    r.add(p);
                }
            }
        }
        r
    }

    /// Apply a uniform scale to every coordinate, including the canvas size.
    pub fn scaled(&self, factor: f64) -> VectorImage {
        if (factor - 1.0).abs() < 1e-12 {
            return self.clone();
        }
        let map = |p: Point| Point::new(p.x * factor, p.y * factor);
        let shapes = self
            .shapes
            .iter()
            .map(|s| Shape {
                color: s.color,
                subpaths: s
                    .subpaths
                    .iter()
                    .map(|sp| SubPath {
                        start: map(sp.start),
                        closed: sp.closed,
                        segs: sp
                            .segs
                            .iter()
                            .map(|seg| match *seg {
                                Seg::Line { to } => Seg::Line { to: map(to) },
                                Seg::Cubic { c1, c2, to } => Seg::Cubic {
                                    c1: map(c1),
                                    c2: map(c2),
                                    to: map(to),
                                },
                                Seg::Arc { r, large, sweep, to } => Seg::Arc {
                                    r: r * factor,
                                    large,
                                    sweep,
                                    to: map(to),
                                },
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        VectorImage {
            width: self.width * factor,
            height: self.height * factor,
            shapes,
            background: self.background,
        }
    }

    /// Round every coordinate to `decimals`.
    ///
    /// Done here rather than per-exporter so that a shared border is rounded
    /// identically on both sides. Rounding independently at write time would
    /// reintroduce exactly the sub-pixel gaps the shared-edge topology exists
    /// to prevent.
    pub fn quantized(&self, decimals: u32) -> VectorImage {
        let q = |p: Point| p.round_to(decimals);
        let shapes = self
            .shapes
            .iter()
            .map(|s| Shape {
                color: s.color,
                subpaths: s
                    .subpaths
                    .iter()
                    .map(|sp| SubPath {
                        start: q(sp.start),
                        closed: sp.closed,
                        segs: sp
                            .segs
                            .iter()
                            .map(|seg| match *seg {
                                Seg::Line { to } => Seg::Line { to: q(to) },
                                Seg::Cubic { c1, c2, to } => Seg::Cubic {
                                    c1: q(c1),
                                    c2: q(c2),
                                    to: q(to),
                                },
                                Seg::Arc { r, large, sweep, to } => Seg::Arc {
                                    r,
                                    large,
                                    sweep,
                                    to: q(to),
                                },
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        VectorImage {
            width: self.width,
            height: self.height,
            shapes,
            background: self.background,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::pt;

    fn square() -> Shape {
        let mut sp = SubPath::new(pt(0.0, 0.0));
        sp.segs.push(Seg::Line { to: pt(10.0, 0.0) });
        sp.segs.push(Seg::Cubic {
            c1: pt(12.0, 3.0),
            c2: pt(12.0, 7.0),
            to: pt(10.0, 10.0),
        });
        sp.segs.push(Seg::Line { to: pt(0.0, 10.0) });
        sp.segs.push(Seg::Line { to: pt(0.0, 0.0) });
        sp.closed = true;
        Shape {
            color: Rgba8::opaque(255, 0, 0),
            subpaths: vec![sp],
        }
    }

    #[test]
    fn complexity_counts_points_the_way_an_editor_would() {
        let vi = VectorImage {
            width: 10.0,
            height: 10.0,
            shapes: vec![square()],
            background: None,
        };
        let c = vi.complexity();
        assert_eq!(c.shapes, 1);
        assert_eq!(c.segments, 4);
        assert_eq!(c.lines, 3);
        assert_eq!(c.cubics, 1);
        // Closed: four anchors, not five, because the last coincides with the
        // first.
        assert_eq!(c.anchors, 4);
        assert_eq!(c.handles, 2);
        assert_eq!(c.total_points(), 6);
    }

    #[test]
    fn scaling_moves_every_coordinate() {
        let vi = VectorImage {
            width: 10.0,
            height: 10.0,
            shapes: vec![square()],
            background: None,
        };
        let s = vi.scaled(2.0);
        assert_eq!(s.width, 20.0);
        assert_eq!(s.shapes[0].subpaths[0].segs[0].end(), pt(20.0, 0.0));
    }

    #[test]
    fn quantizing_is_identical_for_identical_inputs() {
        // The property that keeps shared borders welded through export.
        let a = pt(3.14159265, 2.71828182);
        let b = pt(3.14159265, 2.71828182);
        assert_eq!(a.round_to(3), b.round_to(3));
        assert_eq!(a.round_to(3), pt(3.142, 2.718));
    }
}
