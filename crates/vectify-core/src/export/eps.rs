//! Encapsulated PostScript writer.
//!
//! PostScript has no arc-through-two-points operator matching SVG's, so arcs
//! are lowered to cubics. Fills use the nonzero winding rule, which matches how
//! the pipeline orients outer rings against holes.

use super::{lower, num, over_white, PathOp};
use crate::config::OutputConfig;
use crate::model::VectorImage;

pub fn write(image: &VectorImage, cfg: &OutputConfig) -> String {
    let p = cfg.precision;
    let w = image.width;
    let h = image.height;
    let mut s = String::with_capacity(4096);

    s.push_str("%!PS-Adobe-3.0 EPSF-3.0\n");
    s.push_str("%%Creator: Vectify\n");
    s.push_str("%%Title: vectify-output\n");
    s.push_str(&format!(
        "%%BoundingBox: 0 0 {} {}\n",
        w.ceil() as i64,
        h.ceil() as i64
    ));
    s.push_str(&format!(
        "%%HiResBoundingBox: 0 0 {} {}\n",
        num(w, p),
        num(h, p)
    ));
    s.push_str("%%LanguageLevel: 2\n");
    s.push_str("%%EndComments\n");
    // Short aliases keep the file substantially smaller on shape-heavy output.
    s.push_str("/m {moveto} bind def\n/l {lineto} bind def\n");
    s.push_str("/c {curveto} bind def\n/h {closepath} bind def\n");
    s.push_str("/f {fill} bind def\n/rg {setrgbcolor} bind def\n");
    s.push_str("%%BeginProlog\n%%EndProlog\n");

    if let Some(bg) = image.background {
        let c = over_white(bg);
        s.push_str(&format!(
            "{} {} {} rg 0 0 {} {} rectfill\n",
            num(c[0], 4),
            num(c[1], 4),
            num(c[2], 4),
            num(w, p),
            num(h, p)
        ));
    }

    for shape in &image.shapes {
        if shape.subpaths.is_empty() {
            continue;
        }
        let c = over_white(shape.color);
        s.push_str(&format!(
            "{} {} {} rg\n",
            num(c[0], 4),
            num(c[1], 4),
            num(c[2], 4)
        ));
        s.push_str("newpath\n");
        for sp in &shape.subpaths {
            for op in lower(sp, Some(h)) {
                match op {
                    PathOp::Move(a) => {
                        s.push_str(&format!("{} {} m\n", num(a.x, p), num(a.y, p)))
                    }
                    PathOp::Line(a) => {
                        s.push_str(&format!("{} {} l\n", num(a.x, p), num(a.y, p)))
                    }
                    PathOp::Curve(a, b, d) => s.push_str(&format!(
                        "{} {} {} {} {} {} c\n",
                        num(a.x, p),
                        num(a.y, p),
                        num(b.x, p),
                        num(b.y, p),
                        num(d.x, p),
                        num(d.y, p)
                    )),
                    PathOp::Close => s.push_str("h\n"),
                }
            }
        }
        s.push_str("f\n");
    }

    s.push_str("showpage\n%%EOF\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba8;
    use crate::geom::{pt, Seg, SubPath};
    use crate::model::Shape;

    fn doc() -> VectorImage {
        let mut sp = SubPath::new(pt(0.0, 0.0));
        sp.segs.push(Seg::Line { to: pt(10.0, 0.0) });
        sp.segs.push(Seg::Line { to: pt(10.0, 20.0) });
        sp.closed = true;
        VectorImage {
            width: 10.0,
            height: 20.0,
            shapes: vec![Shape {
                color: Rgba8::opaque(255, 0, 0),
                subpaths: vec![sp],
            }],
            background: None,
        }
    }

    #[test]
    fn has_required_eps_structure() {
        let s = write(&doc(), &OutputConfig::default());
        assert!(s.starts_with("%!PS-Adobe-3.0 EPSF-3.0"));
        assert!(s.contains("%%BoundingBox: 0 0 10 20"));
        assert!(s.contains("%%EndComments"));
        assert!(s.trim_end().ends_with("%%EOF"));
        assert!(s.contains("1 0 0 rg"));
    }

    #[test]
    fn y_axis_is_flipped_for_postscript() {
        let s = write(&doc(), &OutputConfig::default());
        // Top-left (0,0) in image space is bottom-left (0,20) on the page.
        assert!(s.contains("0 20 m"), "{s}");
        // And (10,20) in image space maps to y=0.
        assert!(s.contains("10 0 l"), "{s}");
    }
}
