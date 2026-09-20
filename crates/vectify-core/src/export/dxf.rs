//! DXF writer (AutoCAD R12 ASCII).
//!
//! DXF is the format that CNC routers, laser cutters and vinyl plotters expect.
//! R12 is chosen deliberately over newer revisions because it is the dialect
//! essentially every machine controller can read.
//!
//! Two consequences follow from the format, and callers should know about both:
//! R12 has no Bezier primitive, so curves are flattened to polylines; and it
//! has no practical fill for arbitrary nested regions, so shapes are written as
//! closed outlines. That is what a cutting workflow wants anyway -- it is the
//! boundary, not the paint, that drives the tool.

use super::num;
use crate::config::OutputConfig;
use crate::model::VectorImage;

/// A DXF group code / value pair.
fn pair(code: i32, value: &str) -> String {
    format!("{code}\n{value}\n")
}

pub fn write(image: &VectorImage, cfg: &OutputConfig) -> String {
    let p = cfg.precision.max(4);
    // Flatten finely: the polyline is the final geometry here, not an
    // approximation of something the consumer will re-evaluate.
    let flat_tol = 0.05;
    let h = image.height;
    let mut s = String::with_capacity(8192);

    // --- Header: just the extents, which most readers use to fit the view.
    s.push_str(&pair(0, "SECTION"));
    s.push_str(&pair(2, "HEADER"));
    s.push_str(&pair(9, "$EXTMIN"));
    s.push_str(&pair(10, "0.0"));
    s.push_str(&pair(20, "0.0"));
    s.push_str(&pair(9, "$EXTMAX"));
    s.push_str(&pair(10, &num(image.width, p)));
    s.push_str(&pair(20, &num(h, p)));
    s.push_str(&pair(0, "ENDSEC"));

    s.push_str(&pair(0, "SECTION"));
    s.push_str(&pair(2, "ENTITIES"));

    for (i, shape) in image.shapes.iter().enumerate() {
        // One layer per shape keeps colours separable in CAD tools, which is
        // how cutting jobs are usually organised.
        let layer = format!("VECTIFY_{i}");
        let true_color = ((shape.color.r as u32) << 16)
            | ((shape.color.g as u32) << 8)
            | (shape.color.b as u32);

        for sp in &shape.subpaths {
            let pts = sp.flatten(flat_tol);
            if pts.len() < 2 {
                continue;
            }
            s.push_str(&pair(0, "POLYLINE"));
            s.push_str(&pair(8, &layer));
            s.push_str(&pair(420, &true_color.to_string()));
            s.push_str(&pair(66, "1")); // vertices follow
            s.push_str(&pair(10, "0.0"));
            s.push_str(&pair(20, "0.0"));
            s.push_str(&pair(30, "0.0"));
            // Bit 1 marks the polyline closed.
            s.push_str(&pair(70, if sp.closed { "1" } else { "0" }));

            // A closed polyline must not repeat its first point as its last.
            let n = if sp.closed && pts.len() > 1 && pts[0].dist(pts[pts.len() - 1]) < 1e-9 {
                pts.len() - 1
            } else {
                pts.len()
            };
            for q in &pts[..n] {
                s.push_str(&pair(0, "VERTEX"));
                s.push_str(&pair(8, &layer));
                s.push_str(&pair(10, &num(q.x, p)));
                s.push_str(&pair(20, &num(h - q.y, p)));
                s.push_str(&pair(30, "0.0"));
            }
            s.push_str(&pair(0, "SEQEND"));
            s.push_str(&pair(8, &layer));
        }
    }

    s.push_str(&pair(0, "ENDSEC"));
    s.push_str(&pair(0, "EOF"));
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
        sp.segs.push(Seg::Line { to: pt(0.0, 0.0) });
        sp.closed = true;
        VectorImage {
            width: 10.0,
            height: 20.0,
            shapes: vec![Shape {
                color: Rgba8::opaque(0x11, 0x22, 0x33),
                subpaths: vec![sp],
            }],
            background: None,
        }
    }

    #[test]
    fn has_required_sections() {
        let s = write(&doc(), &OutputConfig::default());
        assert!(s.starts_with("0\nSECTION\n"));
        assert!(s.contains("2\nHEADER\n"));
        assert!(s.contains("2\nENTITIES\n"));
        assert!(s.contains("0\nPOLYLINE\n"));
        assert!(s.contains("0\nSEQEND\n"));
        assert!(s.trim_end().ends_with("EOF"));
    }

    #[test]
    fn group_codes_come_in_pairs() {
        // Every DXF line alternates code, value. An odd count means a malformed
        // file that some readers accept and others reject silently.
        let s = write(&doc(), &OutputConfig::default());
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len() % 2, 0, "odd number of DXF lines");
        for c in lines.iter().step_by(2) {
            assert!(c.trim().parse::<i32>().is_ok(), "not a group code: {c:?}");
        }
    }

    #[test]
    fn closed_polyline_does_not_repeat_its_first_vertex() {
        let s = write(&doc(), &OutputConfig::default());
        let verts = s.matches("0\nVERTEX\n").count();
        assert_eq!(verts, 3, "expected three distinct vertices, got {verts}");
    }

    #[test]
    fn true_colour_is_encoded() {
        let s = write(&doc(), &OutputConfig::default());
        let want = ((0x11u32) << 16 | (0x22u32) << 8 | 0x33u32).to_string();
        assert!(s.contains(&format!("420\n{want}\n")), "missing true colour");
    }
}
