//! Minimal PDF 1.7 writer.
//!
//! Produces a single-page document with one uncompressed content stream. The
//! cross-reference table is byte-offset sensitive, so the file is assembled
//! into a byte buffer and the offsets are recorded as each object is written.

use super::{lower, num, over_white, PathOp};
use crate::config::OutputConfig;
use crate::model::VectorImage;

fn content_stream(image: &VectorImage, cfg: &OutputConfig) -> String {
    let p = cfg.precision;
    let h = image.height;
    let mut s = String::with_capacity(4096);

    if let Some(bg) = image.background {
        let c = over_white(bg);
        s.push_str(&format!(
            "{} {} {} rg 0 0 {} {} re f\n",
            num(c[0], 4),
            num(c[1], 4),
            num(c[2], 4),
            num(image.width, p),
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
        // `f` fills with the nonzero winding rule, which is what the pipeline's
        // outer/hole orientation assumes.
        s.push_str("f\n");
    }
    s
}

pub fn write(image: &VectorImage, cfg: &OutputConfig) -> Vec<u8> {
    let content = content_stream(image, cfg);
    let mut out: Vec<u8> = Vec::with_capacity(content.len() + 1024);
    let mut offsets: Vec<usize> = Vec::new();

    out.extend_from_slice(b"%PDF-1.7\n");
    // A binary comment marks the file as containing binary data, which keeps
    // naive tools from mangling it in transit.
    out.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

    let push_obj = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &str| {
        offsets.push(out.len());
        let n = offsets.len();
        out.extend_from_slice(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    push_obj(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>");
    push_obj(
        &mut out,
        &mut offsets,
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    );
    push_obj(
        &mut out,
        &mut offsets,
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] \
             /Contents 4 0 R /Resources << /ProcSet [/PDF] >> >>",
            num(image.width, cfg.precision),
            num(image.height, cfg.precision)
        ),
    );

    // The content stream object is written by hand because its body contains
    // the raw stream between `stream`/`endstream` markers.
    offsets.push(out.len());
    out.extend_from_slice(
        format!(
            "4 0 obj\n<< /Length {} >>\nstream\n",
            content.len()
        )
        .as_bytes(),
    );
    out.extend_from_slice(content.as_bytes());
    out.extend_from_slice(b"endstream\nendobj\n");

    let xref_at = out.len();
    let count = offsets.len() + 1;
    out.extend_from_slice(format!("xref\n0 {count}\n").as_bytes());
    // Every xref entry must be exactly 20 bytes.
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{:010} {:05} n \n", off, 0).as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
        )
        .as_bytes(),
    );
    out
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
                color: Rgba8::opaque(0, 128, 255),
                subpaths: vec![sp],
            }],
            background: None,
        }
    }

    #[test]
    fn has_pdf_header_and_trailer() {
        let b = write(&doc(), &OutputConfig::default());
        let s = String::from_utf8_lossy(&b);
        assert!(s.starts_with("%PDF-1.7"));
        assert!(s.contains("/Type /Catalog"));
        assert!(s.contains("/MediaBox [0 0 10 20]"));
        assert!(s.contains("startxref"));
        assert!(s.trim_end().ends_with("%%EOF"));
    }

    /// Byte offset of `needle` in `hay`, searching from the end.
    fn rfind_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.len() > hay.len() {
            return None;
        }
        (0..=hay.len() - needle.len())
            .rev()
            .find(|&i| &hay[i..i + needle.len()] == needle)
    }

    fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.len() > hay.len() {
            return None;
        }
        (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
    }

    #[test]
    fn xref_offsets_point_at_their_objects() {
        // A wrong offset here yields a file that opens in some readers and not
        // others, so it is worth checking precisely.
        //
        // This works on raw bytes throughout. The header carries a deliberately
        // non-UTF-8 binary marker, so decoding lossily would substitute
        // replacement characters and shift every offset being verified.
        let b = write(&doc(), &OutputConfig::default());
        // Find the table itself, not the `startxref` pointer that follows it
        // and also ends in "xref".
        let xref_at = rfind_bytes(&b, b"\nxref\n").expect("xref table") + 1;
        let table = &b[xref_at..];

        assert!(table.starts_with(b"xref\n0 5\n"));
        let entries = &table[b"xref\n0 5\n".len()..];
        assert_eq!(&entries[..20], b"0000000000 65535 f \n");

        for n in 1..=4usize {
            let e = &entries[n * 20..n * 20 + 20];
            assert_eq!(e[19], b'\n', "entry {n} is not 20 bytes");
            let off: usize = std::str::from_utf8(&e[..10])
                .unwrap()
                .parse()
                .expect("offset");
            let want = format!("{n} 0 obj");
            assert!(
                b[off..].starts_with(want.as_bytes()),
                "object {n}: offset {off} points at {:?}",
                String::from_utf8_lossy(&b[off..(off + 12).min(b.len())])
            );
        }

        // And startxref must point at the table we just validated.
        let sx = rfind_bytes(&b, b"startxref\n").expect("startxref") + "startxref\n".len();
        let end = sx + b[sx..].iter().position(|&c| c == b'\n').unwrap();
        let declared: usize = std::str::from_utf8(&b[sx..end]).unwrap().parse().unwrap();
        assert_eq!(declared, xref_at);
    }

    #[test]
    fn declared_stream_length_matches_reality() {
        let b = write(&doc(), &OutputConfig::default());
        let idx = find_bytes(&b, b"/Length ").expect("length") + b"/Length ".len();
        let end = idx + b[idx..].iter().position(|&c| c == b' ').unwrap();
        let declared: usize = std::str::from_utf8(&b[idx..end]).unwrap().parse().unwrap();
        let start = find_bytes(&b, b"stream\n").expect("stream") + b"stream\n".len();
        let stop = find_bytes(&b, b"endstream").expect("endstream");
        assert_eq!(declared, stop - start);
    }
}
