//! The tracing pipeline: raster in, vector document out.
//!
//! Stage order matters and is not arbitrary:
//!
//! 1. **Segment** into flat, connected regions.
//! 2. **Build topology** -- a planar subdivision whose borders are shared arcs.
//! 3. **Refine sub-pixel** -- move contour vertices onto the edges the
//!    anti-aliasing implies. Done before corner detection, because corner
//!    detection on a raw staircase would be measuring the pixel grid rather
//!    than the artwork.
//! 4. **Detect corners and de-noise** -- decide which bends are real, and smooth
//!    only where the contour is actually noisy.
//! 5. **Fit** each arc exactly once, to the fewest primitives that hold it.
//! 6. **Assemble** regions from the fitted arcs, each arc referenced forwards by
//!    one region and backwards by its neighbour.
//!
//! Step 5 happening once per *arc* rather than once per *region* is what makes
//! shared borders identical rather than merely similar.

use std::collections::HashMap;
use std::time::Instant;

use crate::color::{LabCache, Rgba8};
use crate::config::VectorizeConfig;
use crate::corners;
use crate::fitting;
use crate::geom::{Point, Seg, SubPath};
use crate::model::{Shape, VectorImage};
use crate::raster::Raster;
use crate::segment::{self, Segmentation};
use crate::subpixel;
use crate::topology::{self, ArcRef, Topology};

/// Timing and shape counts from one run, for the GUI and the benchmark report.
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceStats {
    pub regions: usize,
    pub arcs: usize,
    pub contour_vertices: usize,
    pub corners: usize,
    pub segment_ms: f64,
    pub topology_ms: f64,
    pub subpixel_ms: f64,
    pub corners_ms: f64,
    pub fit_ms: f64,
    pub assemble_ms: f64,
}

impl TraceStats {
    pub fn total_ms(&self) -> f64 {
        self.segment_ms
            + self.topology_ms
            + self.subpixel_ms
            + self.corners_ms
            + self.fit_ms
            + self.assemble_ms
    }
}

pub struct TraceResult {
    pub image: VectorImage,
    pub stats: TraceStats,
}

/// Reverse a fitted command list that runs from `start` to its final point.
fn reverse_segs(start: Point, segs: &[Seg]) -> Vec<Seg> {
    let mut out = Vec::with_capacity(segs.len());
    for i in (0..segs.len()).rev() {
        let from = if i == 0 { start } else { segs[i - 1].end() };
        out.push(segs[i].reversed(from));
    }
    out
}

/// Trace an image into a vector document.
pub fn vectorize(img: &Raster, cfg: &VectorizeConfig) -> TraceResult {
    let cache = LabCache::new();
    vectorize_with_cache(img, cfg, &cache)
}

/// Trace, reusing a shared Lab lookup table. The auto-picker runs many configs
/// over one image and would otherwise rebuild the table every time.
pub fn vectorize_with_cache(
    img: &Raster,
    cfg: &VectorizeConfig,
    cache: &LabCache,
) -> TraceResult {
    let mut stats = TraceStats::default();

    let t = Instant::now();
    let seg = segment::segment(img, &cfg.segment, cache);
    stats.segment_ms = t.elapsed().as_secs_f64() * 1000.0;
    stats.regions = seg.region_count();

    let t = Instant::now();
    let mut topo = topology::build(&seg);
    stats.topology_ms = t.elapsed().as_secs_f64() * 1000.0;
    stats.arcs = topo.arcs.len();
    stats.contour_vertices = topo.total_arc_vertices();

    let t = Instant::now();
    subpixel::refine(&mut topo, &seg, img, &cfg.subpixel);
    stats.subpixel_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    corners::analyze(&mut topo, &cfg.corners);
    stats.corners_ms = t.elapsed().as_secs_f64() * 1000.0;
    stats.corners = topo.arcs.iter().map(|a| a.corners.len()).sum();

    let t = Instant::now();
    let fitted = fit_arcs(&topo, cfg);
    stats.fit_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let image = assemble(&topo, &seg, &fitted, cfg);
    stats.assemble_ms = t.elapsed().as_secs_f64() * 1000.0;

    TraceResult { image, stats }
}

/// One fitted arc: the commands, plus the exact start point they begin from.
struct FittedArc {
    start: Point,
    segs: Vec<Seg>,
}

fn fit_arcs(topo: &Topology, cfg: &VectorizeConfig) -> Vec<FittedArc> {
    topo.arcs
        .iter()
        .map(|arc| {
            // De-duplicating can shift indices, so corners are remapped by
            // matching positions rather than carried across blindly.
            let pts = fitting::dedupe(&arc.points);
            if pts.len() < 2 {
                return FittedArc {
                    start: arc.points[0],
                    segs: Vec::new(),
                };
            }
            let closed = arc.loop_arc && pts.len() > 2;
            let mut corner_idx = remap_corners(&arc.points, &pts, &arc.corners);
            if cfg.fit.merge_pass && corner_idx.len() > 2 {
                corner_idx = fitting::prune_corners(&pts, &corner_idx, closed, &cfg.fit);
            }
            let segs = fitting::fit_contour(&pts, &corner_idx, closed, &cfg.fit);
            FittedArc {
                start: pts[0],
                segs,
            }
        })
        .collect()
}

/// Translate corner indices from the raw point list to the de-duplicated one.
fn remap_corners(orig: &[Point], deduped: &[Point], corners: &[usize]) -> Vec<usize> {
    if orig.len() == deduped.len() {
        return corners.to_vec();
    }
    let mut map = Vec::with_capacity(orig.len());
    let mut j = 0usize;
    for p in orig {
        if j + 1 < deduped.len() && deduped[j + 1].dist_sq(*p) < 1e-20 {
            j += 1;
        }
        map.push(j);
    }
    let mut out: Vec<usize> = corners
        .iter()
        .filter_map(|&c| map.get(c).copied())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Concatenate fitted arcs into closed subpaths, one set per region.
fn assemble(
    topo: &Topology,
    seg: &Segmentation,
    fitted: &[FittedArc],
    cfg: &VectorizeConfig,
) -> VectorImage {
    let mut shapes: Vec<Shape> = Vec::new();

    for (region, rings) in topo.region_rings.iter().enumerate() {
        if rings.is_empty() {
            continue;
        }
        let region = region as u32;
        // Transparent regions are simply not painted. Because the document is a
        // planar subdivision, leaving one face empty is exactly right: nothing
        // else was covering it.
        if seg.is_transparent(region) {
            continue;
        }

        let mut subpaths = Vec::new();
        for ring in rings {
            if let Some(sp) = build_subpath(&ring.refs, fitted) {
                subpaths.push(sp);
            }
        }
        if subpaths.is_empty() {
            continue;
        }

        let color = if cfg.output.recolor_regions {
            let c = seg.region_color[region as usize];
            Rgba8::new(c.r, c.g, c.b, 255)
        } else {
            seg.region_fill(region)
        };

        shapes.push(Shape { color, subpaths });
    }

    if cfg.output.group_by_color {
        shapes = group_by_color(shapes);
    }

    let image = VectorImage {
        width: seg.width as f64,
        height: seg.height as f64,
        shapes,
        background: cfg.output.background,
    };
    let image = image.scaled(cfg.output.scale);
    image.quantized(cfg.output.precision)
}

fn build_subpath(refs: &[ArcRef], fitted: &[FittedArc]) -> Option<SubPath> {
    let mut segs: Vec<Seg> = Vec::new();
    let mut start: Option<Point> = None;

    for r in refs {
        let f = &fitted[r.arc as usize];
        if f.segs.is_empty() {
            continue;
        }
        let (s, list) = if r.reversed {
            (f.segs.last().unwrap().end(), reverse_segs(f.start, &f.segs))
        } else {
            (f.start, f.segs.clone())
        };
        if start.is_none() {
            start = Some(s);
        }
        segs.extend(list);
    }

    let start = start?;
    if segs.is_empty() {
        return None;
    }
    // Weld the ring shut. The endpoints already agree to within floating point
    // because both sides came from the same shared junction node; this removes
    // the last ulp of disagreement so the exporter emits an exact close.
    if let Some(last) = segs.last_mut() {
        last.set_end(start);
    }
    Some(SubPath {
        start,
        segs,
        closed: true,
    })
}

/// Merge shapes that share a fill colour into a single path.
///
/// Safe with the nonzero fill rule even when one such region sits inside
/// another's hole: the hole ring and the inner region's outer ring wind
/// oppositely and cancel, leaving the interior filled exactly once.
fn group_by_color(shapes: Vec<Shape>) -> Vec<Shape> {
    let mut order: Vec<Rgba8> = Vec::new();
    let mut buckets: HashMap<Rgba8, Vec<SubPath>> = HashMap::new();
    for s in shapes {
        if !buckets.contains_key(&s.color) {
            order.push(s.color);
        }
        buckets.entry(s.color).or_default().extend(s.subpaths);
    }
    order
        .into_iter()
        .filter_map(|c| {
            buckets
                .remove(&c)
                .map(|subpaths| Shape { color: c, subpaths })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Preset, SegmentMode};

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Raster {
        let mut r = Raster::new(w, h);
        for y in 0..h {
            for x in 0..w {
                r.set(x, y, c);
            }
        }
        r
    }

    #[test]
    fn traces_a_square_into_two_shapes() {
        let mut img = solid(40, 40, [1.0, 1.0, 1.0, 1.0]);
        for y in 10..30 {
            for x in 10..30 {
                img.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        let mut cfg = Preset::BlackAndWhite.config();
        cfg.segment.mode = SegmentMode::Binary;
        let out = vectorize(&img, &cfg);
        let c = out.image.complexity();
        assert_eq!(c.shapes, 2, "background plus square");
        // A square should cost four line segments per ring and no handles at
        // all. Anything more means corners or straight runs were mishandled.
        assert_eq!(c.handles, 0, "a square should need no curve handles");
        assert!(c.segments <= 12, "segments: {}", c.segments);
    }

    #[test]
    fn a_circle_costs_few_points() {
        let mut img = solid(64, 64, [1.0, 1.0, 1.0, 1.0]);
        for y in 0..64 {
            for x in 0..64 {
                let d = ((x as f64 + 0.5 - 32.0).powi(2) + (y as f64 + 0.5 - 32.0).powi(2)).sqrt();
                // Analytic anti-aliasing so the sub-pixel pass has real data.
                let cov = (d - 20.0).clamp(-0.5, 0.5) + 0.5;
                let v = cov as f32;
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        let cfg = Preset::BlackAndWhite.config();
        let out = vectorize(&img, &cfg);
        let c = out.image.complexity();
        assert!(
            c.total_points() < 60,
            "circle cost {} points: {:?}",
            c.total_points(),
            c
        );
    }

    #[test]
    fn shared_borders_are_geometrically_identical() {
        // Two colour fields meeting along a diagonal. The border must appear in
        // both shapes as the very same list of coordinates -- reversed, but
        // point for point identical. That is the property that prevents
        // fringing and gaps.
        let mut img = Raster::new(48, 48);
        for y in 0..48 {
            for x in 0..48 {
                let c = if (x as f64) * 0.6 + (y as f64) * 0.8 < 30.0 {
                    [0.85, 0.15, 0.15, 1.0]
                } else {
                    [0.15, 0.25, 0.85, 1.0]
                };
                img.set(x, y, c);
            }
        }
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 2;
        cfg.output.group_by_color = false;
        let out = vectorize(&img, &cfg);
        assert_eq!(out.image.shapes.len(), 2);

        let flat: Vec<Vec<Point>> = out
            .image
            .shapes
            .iter()
            .map(|s| s.subpaths[0].flatten(0.01))
            .collect();

        // Every vertex of the shared border in shape 0 must coincide exactly
        // with one in shape 1.
        let mut matched = 0usize;
        for p in &flat[0] {
            if flat[1].iter().any(|q| p.dist_sq(*q) < 1e-18) {
                matched += 1;
            }
        }
        assert!(
            matched > flat[0].len() / 4,
            "only {matched} of {} vertices were shared",
            flat[0].len()
        );
    }

    #[test]
    fn transparent_regions_are_left_unpainted() {
        let mut img = Raster::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let inside = x >= 8 && x < 24 && y >= 8 && y < 24;
                img.set(
                    x,
                    y,
                    if inside {
                        [0.9, 0.2, 0.2, 1.0]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    },
                );
            }
        }
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 2;
        let out = vectorize(&img, &cfg);
        assert_eq!(out.image.shapes.len(), 1, "only the opaque square is painted");
        assert_eq!(out.image.shapes[0].color.a, 255);
    }

    const WHITE: Rgba8 = Rgba8 { r: 255, g: 255, b: 255, a: 255 };

    /// White page, a red square and a blue square.
    fn page_with_two_squares() -> Raster {
        let mut img = solid(60, 40, [1.0, 1.0, 1.0, 1.0]);
        for y in 10..30 {
            for x in 6..26 {
                img.set(x, y, [0.85, 0.1, 0.1, 1.0]);
            }
            for x in 34..54 {
                img.set(x, y, [0.1, 0.2, 0.85, 1.0]);
            }
        }
        img
    }

    #[test]
    fn a_keyed_colour_gets_no_shape() {
        let img = page_with_two_squares();
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 3;

        let plain = vectorize(&img, &cfg).image;
        assert_eq!(plain.shapes.len(), 3, "page plus two squares");

        cfg.segment.transparent_colors = vec![WHITE];
        let keyed = vectorize(&img, &cfg).image;
        assert_eq!(keyed.shapes.len(), 2, "only the squares are painted");
        assert!(
            keyed.shapes.iter().all(|s| s.color.luma() < 250.0),
            "no shape may be white: {:?}",
            keyed.shapes.iter().map(|s| s.color).collect::<Vec<_>>()
        );
    }

    #[test]
    fn several_colours_can_be_keyed_at_once() {
        let img = page_with_two_squares();
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 3;
        cfg.segment.transparent_colors = vec![WHITE, Rgba8::opaque(26, 51, 217)];
        let out = vectorize(&img, &cfg).image;
        assert_eq!(out.shapes.len(), 1, "only the red square is left");
        assert!(out.shapes[0].color.r > 150 && out.shapes[0].color.b < 100);
    }

    #[test]
    fn a_keyed_colour_enclosed_by_a_shape_leaves_a_real_hole() {
        // A blue square with a white square inside it, on a white page. Keying
        // white must punch the inner square out of the blue shape as a hole
        // ring, not paint a white shape over it.
        let mut img = solid(50, 50, [1.0, 1.0, 1.0, 1.0]);
        for y in 8..42 {
            for x in 8..42 {
                img.set(x, y, [0.1, 0.2, 0.85, 1.0]);
            }
        }
        for y in 20..30 {
            for x in 20..30 {
                img.set(x, y, [1.0, 1.0, 1.0, 1.0]);
            }
        }
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 2;
        cfg.segment.transparent_colors = vec![WHITE];
        let out = vectorize(&img, &cfg).image;
        assert_eq!(out.shapes.len(), 1);
        assert_eq!(out.shapes[0].subpaths.len(), 2, "outer boundary plus the hole");
    }

    #[test]
    fn keying_works_for_two_tone_tracing() {
        let mut img = solid(40, 40, [1.0, 1.0, 1.0, 1.0]);
        for y in 10..30 {
            for x in 10..30 {
                img.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        let mut cfg = Preset::BlackAndWhite.config();
        cfg.segment.transparent_colors = vec![WHITE];
        let out = vectorize(&img, &cfg).image;
        assert_eq!(out.shapes.len(), 1, "just the ink");
        assert!(out.shapes[0].color.luma() < 10.0);
    }

    #[test]
    fn keying_works_alongside_real_alpha_transparency() {
        // Left third: genuinely transparent. Middle third: white. Right third:
        // red. Both the alpha and the keyed colour must be left unpainted.
        let mut img = Raster::new(60, 20);
        for y in 0..20 {
            for x in 0..60 {
                img.set(
                    x,
                    y,
                    match x / 20 {
                        0 => [0.0, 0.0, 0.0, 0.0],
                        1 => [1.0, 1.0, 1.0, 1.0],
                        _ => [0.85, 0.1, 0.1, 1.0],
                    },
                );
            }
        }
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 2;
        cfg.segment.transparent_colors = vec![WHITE];
        let out = vectorize(&img, &cfg).image;
        assert_eq!(out.shapes.len(), 1, "only the red third is painted");
        assert!(out.shapes[0].color.r > 150 && out.shapes[0].color.g < 100);
    }

    #[test]
    fn a_key_that_is_not_in_the_image_changes_nothing() {
        let img = page_with_two_squares();
        let mut cfg = Preset::Logo.config();
        cfg.segment.colors = 3;
        let plain = vectorize(&img, &cfg).image;
        cfg.segment.transparent_colors = vec![Rgba8::opaque(0, 200, 0)];
        let keyed = vectorize(&img, &cfg).image;
        assert_eq!(plain.shapes.len(), keyed.shapes.len());
        assert_eq!(plain.complexity().segments, keyed.complexity().segments);
    }

    #[test]
    fn keying_a_background_leaves_the_remaining_edges_untouched() {
        // An anti-aliased disc on white. The disc's outline is reconstructed
        // from the blend between the disc and the white around it, so keying the
        // white out must not disturb that: the surviving shape has to be
        // geometrically identical to the one traced without keying, and still
        // sit on the true edge to sub-pixel accuracy.
        let mut img = solid(64, 64, [1.0, 1.0, 1.0, 1.0]);
        for y in 0..64 {
            for x in 0..64 {
                let d = ((x as f64 + 0.5 - 32.0).powi(2) + (y as f64 + 0.5 - 32.0).powi(2)).sqrt();
                let cov = ((20.0 - d).clamp(-0.5, 0.5) + 0.5) as f32;
                let v = 1.0 - cov;
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        let mut cfg = Preset::BlackAndWhite.config();
        let plain = vectorize(&img, &cfg).image;
        cfg.segment.transparent_colors = vec![WHITE];
        let keyed = vectorize(&img, &cfg).image;

        let ink = |v: &VectorImage| -> Vec<Vec<Point>> {
            let s = v.shapes.iter().find(|s| s.color.luma() < 128.0).expect("ink shape");
            s.subpaths.iter().map(|sp| sp.flatten(0.01)).collect()
        };
        assert_eq!(keyed.shapes.len(), 1);
        assert_eq!(ink(&plain), ink(&keyed), "keying must not move the ink's edges");

        let b = keyed.bounds();
        let (w, h) = (b.width(), b.height());
        assert!(
            (w - 40.0).abs() < 0.25 && (h - 40.0).abs() < 0.25,
            "disc traced as {w:.3} x {h:.3}, expected about 40 x 40"
        );
    }

    #[test]
    fn old_configs_without_the_keying_fields_still_load() {
        // Configs are serialised alongside results so they can be reproduced.
        // Ones written before this option existed must keep deserialising.
        let mut value = serde_json::to_value(Preset::Logo.config()).unwrap();
        let segment = value["segment"].as_object_mut().unwrap();
        segment.remove("transparent_colors");
        segment.remove("transparent_tolerance");
        let cfg: VectorizeConfig = serde_json::from_value(value).unwrap();
        assert!(cfg.segment.transparent_colors.is_empty());
        assert_eq!(
            cfg.segment.transparent_tolerance,
            crate::config::DEFAULT_TRANSPARENT_TOLERANCE
        );
    }

    #[test]
    fn every_subpath_is_closed_and_welded() {
        let mut img = Raster::new(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                let c = match (x / 13, y / 13) {
                    (0, _) => [0.9, 0.1, 0.1, 1.0],
                    (1, 0) => [0.1, 0.9, 0.1, 1.0],
                    (1, _) => [0.1, 0.1, 0.9, 1.0],
                    _ => [0.9, 0.9, 0.1, 1.0],
                };
                img.set(x, y, c);
            }
        }
        let mut cfg = Preset::Clipart.config();
        cfg.segment.colors = 4;
        let out = vectorize(&img, &cfg);
        assert!(!out.image.shapes.is_empty());
        for s in &out.image.shapes {
            for sp in &s.subpaths {
                assert!(sp.closed);
                assert!(
                    sp.end().dist(sp.start) < 1e-9,
                    "subpath not welded: {:?} vs {:?}",
                    sp.end(),
                    sp.start
                );
            }
        }
    }

    #[test]
    fn pixel_art_preset_preserves_hard_blocks() {
        // 4x scaled pixel art: the block edges are intentional and must survive
        // exactly, with no sub-pixel guessing and no rounding.
        let mut img = Raster::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let on = ((x / 4) + (y / 4)) % 2 == 0;
                let v = if on { 0.1 } else { 0.9 };
                img.set(x, y, [v, v, v, 1.0]);
            }
        }
        let cfg = Preset::PixelArt.config();
        let out = vectorize(&img, &cfg);
        let c = out.image.complexity();
        assert_eq!(c.handles, 0, "pixel art must stay perfectly rectilinear");
        assert!(c.shapes >= 2);
    }

    #[test]
    fn handles_degenerate_images() {
        for (w, h) in [(1u32, 1u32), (1, 16), (16, 1), (2, 2)] {
            let img = solid(w, h, [0.4, 0.5, 0.6, 1.0]);
            let cfg = Preset::Logo.config();
            let out = vectorize(&img, &cfg);
            assert_eq!(out.image.width, w as f64);
            assert_eq!(out.image.height, h as f64);
        }
    }

    #[test]
    fn grouping_merges_same_coloured_regions() {
        let mut img = solid(40, 20, [1.0, 1.0, 1.0, 1.0]);
        for y in 5..15 {
            for x in 4..12 {
                img.set(x, y, [0.1, 0.1, 0.1, 1.0]);
            }
            for x in 26..34 {
                img.set(x, y, [0.1, 0.1, 0.1, 1.0]);
            }
        }
        let mut cfg = Preset::BlackAndWhite.config();
        cfg.output.group_by_color = true;
        let grouped = vectorize(&img, &cfg).image;
        cfg.output.group_by_color = false;
        let split = vectorize(&img, &cfg).image;

        assert_eq!(grouped.shapes.len(), 2, "one shape per colour");
        assert_eq!(split.shapes.len(), 3, "background plus two squares");
        // Grouping must not change the geometry, only how it is bundled.
        assert_eq!(
            grouped.complexity().segments,
            split.complexity().segments
        );
    }
}
