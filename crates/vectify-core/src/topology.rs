//! Planar topology extraction: the "no pancakes" pass.
//!
//! Most multi-colour tracers vectorise each colour independently and stack the
//! resulting silhouettes. Because each silhouette is traced and smoothed on its
//! own, the border that two neighbouring colours are supposed to *share* ends up
//! as two different curves. Wherever they disagree you get a sliver of the layer
//! underneath showing through (colour fringing) or a visible overlap, and moving
//! one shape in an editor tears a gap next to it.
//!
//! This module instead builds a genuine planar subdivision. It works on the
//! *crack graph*: the lattice of pixel corners, with an edge wherever two
//! adjacent pixels belong to different regions. That graph is cut at junctions
//! (lattice nodes where three or more regions meet) into maximal chains called
//! arcs. Every arc has exactly one region on its left and one on its right.
//!
//! Each arc is then refined and fitted exactly once, and each of the two
//! regions it separates references that same fitted curve -- one forwards, one
//! reversed. Shared borders are therefore identical by construction, not by
//! numerical luck, and junction points are shared vertices.

use std::collections::HashMap;

use crate::geom::{polygon_area, Point};
use crate::segment::{Segmentation, OUTSIDE};

pub const NO_NODE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum EdgeId {
    /// Horizontal crack from node (x, y) to (x+1, y), separating the pixels
    /// above and below it.
    H(u32, u32),
    /// Vertical crack from node (x, y) to (x, y+1), separating the pixels left
    /// and right of it.
    V(u32, u32),
}

/// A maximal chain of crack edges between two junctions, separating exactly two
/// regions.
#[derive(Clone, Debug)]
pub struct Arc {
    /// Lattice polyline including both endpoints. Endpoint positions are
    /// authoritative in `Topology::nodes`; `points[0]` and `points[last]` are
    /// kept in sync with them.
    pub points: Vec<Point>,
    pub start_node: u32,
    pub end_node: u32,
    /// Region on the left when walking `points` forwards.
    pub left: u32,
    /// Region on the right when walking `points` forwards.
    pub right: u32,
    /// True when this arc is a closed loop whose start and end coincide.
    pub loop_arc: bool,
    /// True when the arc runs along the outside of the image. These must not be
    /// moved by sub-pixel refinement: there is no anti-aliasing data beyond the
    /// image edge, and a shape that filled the canvas should still fill it.
    pub on_border: bool,
    /// Indices into `points` that the corner detector marked as hard corners.
    pub corners: Vec<usize>,
}

impl Arc {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArcRef {
    pub arc: u32,
    pub reversed: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Ring {
    pub refs: Vec<ArcRef>,
    /// Signed lattice area. Negative rings are outer boundaries, positive ones
    /// are holes, under the winding our tracer produces in y-down space.
    pub area: f64,
}

impl Ring {
    pub fn is_hole(&self) -> bool {
        self.area > 0.0
    }
}

pub struct Topology {
    /// Shared junction positions. Both arcs meeting at a junction read their
    /// endpoint from here, so a junction can never split into two points.
    pub nodes: Vec<Point>,
    pub arcs: Vec<Arc>,
    /// Boundary rings per region, indexed by region id.
    pub region_rings: Vec<Vec<Ring>>,
}

impl Topology {
    pub fn arc_points(&self, r: ArcRef) -> Vec<Point> {
        let a = &self.arcs[r.arc as usize];
        if r.reversed {
            a.points.iter().rev().copied().collect()
        } else {
            a.points.clone()
        }
    }

    pub fn total_arc_vertices(&self) -> usize {
        self.arcs.iter().map(|a| a.points.len()).sum()
    }
}

struct Builder<'a> {
    seg: &'a Segmentation,
    w: u32,
    h: u32,
    node_id: Vec<u32>,
    nodes: Vec<Point>,
}

impl<'a> Builder<'a> {
    fn new(seg: &'a Segmentation) -> Self {
        let w = seg.width;
        let h = seg.height;
        Builder {
            seg,
            w,
            h,
            node_id: vec![NO_NODE; ((w + 1) as usize) * ((h + 1) as usize)],
            nodes: Vec::new(),
        }
    }

    #[inline]
    fn node_index(&self, i: u32, j: u32) -> usize {
        (j as usize) * ((self.w + 1) as usize) + (i as usize)
    }

    #[inline]
    fn h_exists(&self, x: u32, y: u32) -> bool {
        // Separates pixel (x, y-1) above from pixel (x, y) below.
        self.seg.region_at_signed(x as i64, y as i64 - 1)
            != self.seg.region_at_signed(x as i64, y as i64)
    }

    #[inline]
    fn v_exists(&self, x: u32, y: u32) -> bool {
        // Separates pixel (x-1, y) left from pixel (x, y) right.
        self.seg.region_at_signed(x as i64 - 1, y as i64)
            != self.seg.region_at_signed(x as i64, y as i64)
    }

    /// The up-to-four crack edges meeting at lattice node (i, j).
    fn incident(&self, i: u32, j: u32) -> [Option<EdgeId>; 4] {
        let mut out = [None; 4];
        if i >= 1 && self.h_exists(i - 1, j) {
            out[0] = Some(EdgeId::H(i - 1, j));
        }
        if i < self.w && self.h_exists(i, j) {
            out[1] = Some(EdgeId::H(i, j));
        }
        if j >= 1 && self.v_exists(i, j - 1) {
            out[2] = Some(EdgeId::V(i, j - 1));
        }
        if j < self.h && self.v_exists(i, j) {
            out[3] = Some(EdgeId::V(i, j));
        }
        out
    }

    fn degree(&self, i: u32, j: u32) -> usize {
        self.incident(i, j).iter().filter(|e| e.is_some()).count()
    }

    /// A node is a junction when more or fewer than two cracks meet there, so
    /// that a degree-2 node is always an interior point of exactly one arc. The
    /// four image corners are forced to be junctions as well: they are real
    /// corners of the artwork and must never be smoothed away.
    fn is_junction(&self, i: u32, j: u32) -> bool {
        let d = self.degree(i, j);
        if d == 0 {
            return false;
        }
        if d != 2 {
            return true;
        }
        (i == 0 || i == self.w) && (j == 0 || j == self.h)
    }

    fn other_end(&self, e: EdgeId, from: (u32, u32)) -> (u32, u32) {
        match e {
            EdgeId::H(x, y) => {
                if from == (x, y) {
                    (x + 1, y)
                } else {
                    (x, y)
                }
            }
            EdgeId::V(x, y) => {
                if from == (x, y) {
                    (x, y + 1)
                } else {
                    (x, y)
                }
            }
        }
    }

    fn intern_node(&mut self, i: u32, j: u32) -> u32 {
        let k = self.node_index(i, j);
        if self.node_id[k] == NO_NODE {
            self.node_id[k] = self.nodes.len() as u32;
            self.nodes.push(Point::new(i as f64, j as f64));
        }
        self.node_id[k]
    }
}

/// Regions on either side of a step from `a` to `b` (adjacent lattice nodes).
/// Derived geometrically rather than by case analysis: offset half a pixel
/// along the leftward normal from the step midpoint and see which pixel that
/// lands in.
fn sides(seg: &Segmentation, a: (u32, u32), b: (u32, u32)) -> (u32, u32) {
    let ax = a.0 as f64;
    let ay = a.1 as f64;
    let bx = b.0 as f64;
    let by = b.1 as f64;
    let mx = (ax + bx) * 0.5;
    let my = (ay + by) * 0.5;
    let dx = bx - ax;
    let dy = by - ay;
    // perp() in y-down space: the leftward normal of the direction of travel.
    let (lx, ly) = (dy, -dx);
    let left = seg.region_at_signed(
        (mx + lx * 0.5).floor() as i64,
        (my + ly * 0.5).floor() as i64,
    );
    let right = seg.region_at_signed(
        (mx - lx * 0.5).floor() as i64,
        (my - ly * 0.5).floor() as i64,
    );
    (left, right)
}

pub fn build(seg: &Segmentation) -> Topology {
    let mut b = Builder::new(seg);
    let w = b.w;
    let h = b.h;

    let mut used: HashMap<EdgeId, bool> = HashMap::new();
    let mut arcs: Vec<Arc> = Vec::new();

    // --- Pass 1: arcs that run between junctions -------------------------
    for j in 0..=h {
        for i in 0..=w {
            if !b.is_junction(i, j) {
                continue;
            }
            let start_node = b.intern_node(i, j);
            for e in b.incident(i, j).into_iter().flatten() {
                if used.get(&e).copied().unwrap_or(false) {
                    continue;
                }
                let arc = walk_arc(&mut b, &mut used, (i, j), e, start_node);
                if let Some(arc) = arc {
                    arcs.push(arc);
                }
            }
        }
    }

    // --- Pass 2: closed loops with no junction on them --------------------
    // A lone blob on a plain background produces one of these: every node on
    // its contour has degree 2, so pass 1 never found an entry point.
    for j in 0..=h {
        for i in 0..=w {
            for e in b.incident(i, j).into_iter().flatten() {
                if used.get(&e).copied().unwrap_or(false) {
                    continue;
                }
                let start_node = b.intern_node(i, j);
                if let Some(mut arc) = walk_arc(&mut b, &mut used, (i, j), e, start_node) {
                    arc.loop_arc = true;
                    arcs.push(arc);
                }
            }
        }
    }

    let nodes = b.nodes;
    let region_rings = assemble_rings(seg, &arcs, &nodes);

    Topology {
        nodes,
        arcs,
        region_rings,
    }
}

/// Follow a chain of degree-2 nodes from `start` along `first` until another
/// junction (or the starting node) is reached.
fn walk_arc(
    b: &mut Builder,
    used: &mut HashMap<EdgeId, bool>,
    start: (u32, u32),
    first: EdgeId,
    start_node: u32,
) -> Option<Arc> {
    let mut points = vec![Point::new(start.0 as f64, start.1 as f64)];
    let mut cur = start;
    let mut edge = first;

    let (left, right) = sides(b.seg, start, b.other_end(first, start));
    let mut on_border = left == OUTSIDE || right == OUTSIDE;

    loop {
        used.insert(edge, true);
        let next = b.other_end(edge, cur);
        points.push(Point::new(next.0 as f64, next.1 as f64));
        cur = next;

        if b.is_junction(cur.0, cur.1) || cur == start {
            break;
        }
        // Degree 2: continue through the other incident edge.
        let inc = b.incident(cur.0, cur.1);
        let mut nxt = None;
        for e in inc.into_iter().flatten() {
            if e != edge && !used.get(&e).copied().unwrap_or(false) {
                nxt = Some(e);
                break;
            }
        }
        match nxt {
            Some(e) => {
                let seg_sides = sides(b.seg, cur, b.other_end(e, cur));
                on_border |= seg_sides.0 == OUTSIDE || seg_sides.1 == OUTSIDE;
                edge = e;
            }
            None => break,
        }
    }

    if points.len() < 2 {
        return None;
    }
    let end_node = b.intern_node(cur.0, cur.1);
    let loop_arc = start_node == end_node;

    Some(Arc {
        points,
        start_node,
        end_node,
        left,
        right,
        loop_arc,
        on_border,
        corners: Vec::new(),
    })
}

/// Chain each region's directed arcs into closed rings.
fn assemble_rings(seg: &Segmentation, arcs: &[Arc], nodes: &[Point]) -> Vec<Vec<Ring>> {
    let region_count = seg.region_count();
    let mut out: Vec<Vec<Ring>> = vec![Vec::new(); region_count];

    // Directed references bounding each region: forward where the region lies
    // on the left, reversed where it lies on the right.
    let mut by_region: Vec<Vec<ArcRef>> = vec![Vec::new(); region_count];
    for (i, a) in arcs.iter().enumerate() {
        if a.left != OUTSIDE && (a.left as usize) < region_count {
            by_region[a.left as usize].push(ArcRef { arc: i as u32, reversed: false });
        }
        if a.right != OUTSIDE && (a.right as usize) < region_count {
            by_region[a.right as usize].push(ArcRef { arc: i as u32, reversed: true });
        }
    }

    for (region, refs) in by_region.into_iter().enumerate() {
        if refs.is_empty() {
            continue;
        }
        out[region] = chain_refs(&refs, arcs, nodes);
    }
    out
}

/// Start node of a directed arc reference.
fn ref_start_node(a: &Arc, r: ArcRef) -> u32 {
    if r.reversed {
        a.end_node
    } else {
        a.start_node
    }
}

fn ref_end_node(a: &Arc, r: ArcRef) -> u32 {
    if r.reversed {
        a.start_node
    } else {
        a.end_node
    }
}

/// Direction leaving the start of a directed reference.
fn ref_out_dir(a: &Arc, r: ArcRef) -> Point {
    let p = &a.points;
    if r.reversed {
        (p[p.len() - 2] - p[p.len() - 1]).normalized()
    } else {
        (p[1] - p[0]).normalized()
    }
}

/// Direction arriving at the end of a directed reference.
fn ref_in_dir(a: &Arc, r: ArcRef) -> Point {
    let p = &a.points;
    if r.reversed {
        (p[0] - p[1]).normalized()
    } else {
        (p[p.len() - 1] - p[p.len() - 2]).normalized()
    }
}

fn chain_refs(refs: &[ArcRef], arcs: &[Arc], nodes: &[Point]) -> Vec<Ring> {
    let mut outgoing: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, r) in refs.iter().enumerate() {
        outgoing
            .entry(ref_start_node(&arcs[r.arc as usize], *r))
            .or_default()
            .push(i);
    }

    let mut used = vec![false; refs.len()];
    let mut rings = Vec::new();

    for seed in 0..refs.len() {
        if used[seed] {
            continue;
        }
        let mut ring_refs = Vec::new();
        let mut cur = seed;
        let start_node = ref_start_node(&arcs[refs[seed].arc as usize], refs[seed]);

        loop {
            used[cur] = true;
            ring_refs.push(refs[cur]);
            let arc = &arcs[refs[cur].arc as usize];
            let end_node = ref_end_node(arc, refs[cur]);
            if end_node == start_node && !ring_refs.is_empty() {
                break;
            }
            let in_dir = ref_in_dir(arc, refs[cur]);
            let Some(cands) = outgoing.get(&end_node) else { break };

            // Where several of this region's arcs leave the same junction, take
            // the sharpest left turn. That hugs the region as tightly as
            // possible, which keeps a pinch point (a region touching itself
            // corner-to-corner) from swallowing the wrong lobe.
            let mut best: Option<(f64, usize)> = None;
            for &c in cands {
                if used[c] {
                    continue;
                }
                let out_dir = ref_out_dir(&arcs[refs[c].arc as usize], refs[c]);
                let turn = in_dir.cross(out_dir).atan2(in_dir.dot(out_dir));
                if best.is_none_or(|b| turn < b.0) {
                    best = Some((turn, c));
                }
            }
            match best {
                Some((_, c)) => cur = c,
                None => break,
            }
        }

        if ring_refs.is_empty() {
            continue;
        }
        let area = ring_area(&ring_refs, arcs, nodes);
        rings.push(Ring { refs: ring_refs, area });
    }
    rings
}

fn ring_area(refs: &[ArcRef], arcs: &[Arc], nodes: &[Point]) -> f64 {
    let mut pts: Vec<Point> = Vec::new();
    for r in refs {
        let a = &arcs[r.arc as usize];
        let n = a.points.len();
        if r.reversed {
            for k in (0..n).rev() {
                let p = if k == n - 1 {
                    nodes
                        .get(a.end_node as usize)
                        .copied()
                        .unwrap_or(a.points[k])
                } else if k == 0 {
                    nodes
                        .get(a.start_node as usize)
                        .copied()
                        .unwrap_or(a.points[k])
                } else {
                    a.points[k]
                };
                if pts.last().is_none_or(|l| l.dist_sq(p) > 1e-18) {
                    pts.push(p);
                }
            }
        } else {
            for k in 0..n {
                let p = if k == 0 {
                    nodes
                        .get(a.start_node as usize)
                        .copied()
                        .unwrap_or(a.points[k])
                } else if k == n - 1 {
                    nodes
                        .get(a.end_node as usize)
                        .copied()
                        .unwrap_or(a.points[k])
                } else {
                    a.points[k]
                };
                if pts.last().is_none_or(|l| l.dist_sq(p) > 1e-18) {
                    pts.push(p);
                }
            }
        }
    }
    polygon_area(&pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::LabCache;
    use crate::config::{SegmentConfig, SegmentMode};
    use crate::raster::Raster;
    use crate::segment::segment;

    fn seg_of(img: &Raster, colors: usize) -> Segmentation {
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            mode: if colors == 2 { SegmentMode::Binary } else { SegmentMode::Palette },
            colors,
            despeckle_area: 0,
            ..Default::default()
        };
        segment(img, &cfg, &cache)
    }

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
    fn square_on_background_has_one_interior_loop_and_a_border() {
        let mut img = solid(20, 20, [1.0, 1.0, 1.0, 1.0]);
        for y in 5..15 {
            for x in 5..15 {
                img.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        let seg = seg_of(&img, 2);
        let topo = build(&seg);

        let sq = seg.region_at(10, 10);
        let rings = &topo.region_rings[sq as usize];
        assert_eq!(rings.len(), 1, "square should have exactly one ring");
        assert!(!rings[0].is_hole(), "the square's own ring is an outer ring");
        // 10x10 square: the crack contour encloses exactly 100 lattice units.
        assert!((rings[0].area.abs() - 100.0).abs() < 1e-6, "area {}", rings[0].area);

        let bg = seg.region_at(0, 0);
        let bg_rings = &topo.region_rings[bg as usize];
        assert_eq!(bg_rings.len(), 2, "background has an outer ring plus a hole");
        assert_eq!(bg_rings.iter().filter(|r| r.is_hole()).count(), 1);
    }

    #[test]
    fn the_shared_border_is_one_arc_referenced_twice() {
        // This is the central guarantee of the module: the square and the
        // background do not each own a copy of the boundary between them.
        let mut img = solid(20, 20, [1.0, 1.0, 1.0, 1.0]);
        for y in 5..15 {
            for x in 5..15 {
                img.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        let seg = seg_of(&img, 2);
        let topo = build(&seg);
        let sq = seg.region_at(10, 10);
        let bg = seg.region_at(0, 0);

        let sq_arcs: Vec<u32> = topo.region_rings[sq as usize]
            .iter()
            .flat_map(|r| r.refs.iter().map(|a| a.arc))
            .collect();
        let hole = topo.region_rings[bg as usize]
            .iter()
            .find(|r| r.is_hole())
            .expect("background hole");
        let hole_arcs: Vec<u32> = hole.refs.iter().map(|a| a.arc).collect();

        assert_eq!(sq_arcs.len(), 1);
        assert_eq!(hole_arcs, sq_arcs, "both regions must cite the same arc id");

        // And they cite it in opposite directions.
        let sq_ref = topo.region_rings[sq as usize][0].refs[0];
        let bg_ref = hole.refs[0];
        assert_eq!(sq_ref.arc, bg_ref.arc);
        assert_ne!(sq_ref.reversed, bg_ref.reversed);
    }

    #[test]
    fn three_regions_meet_at_junctions() {
        // Three vertical stripes: the two internal borders are separate arcs,
        // each shared by the pair of stripes it separates.
        let mut img = Raster::new(30, 10);
        for y in 0..10 {
            for x in 0..30 {
                let c = match x / 10 {
                    0 => [0.9, 0.1, 0.1, 1.0],
                    1 => [0.1, 0.85, 0.15, 1.0],
                    _ => [0.1, 0.15, 0.9, 1.0],
                };
                img.set(x, y, c);
            }
        }
        let seg = seg_of(&img, 3);
        assert_eq!(seg.region_count(), 3);
        let topo = build(&seg);

        let mid = seg.region_at(15, 5);
        let mid_rings = &topo.region_rings[mid as usize];
        assert_eq!(mid_rings.len(), 1);
        // Middle stripe is bounded by: left border arc, right border arc, and
        // the two image-edge runs along top and bottom.
        assert!(mid_rings[0].refs.len() >= 3);
        assert!((mid_rings[0].area.abs() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn every_arc_separates_exactly_two_regions_consistently() {
        let mut img = Raster::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                let c = if (x / 6 + y / 6) % 2 == 0 {
                    [0.1, 0.1, 0.1, 1.0]
                } else {
                    [0.9, 0.9, 0.9, 1.0]
                };
                img.set(x, y, c);
            }
        }
        let seg = seg_of(&img, 2);
        let topo = build(&seg);
        for (i, arc) in topo.arcs.iter().enumerate() {
            assert_ne!(arc.left, arc.right, "arc {i} does not separate anything");
            // Re-derive the sides at every step and confirm they never change.
            for k in 0..arc.points.len() - 1 {
                let a = (arc.points[k].x as u32, arc.points[k].y as u32);
                let b = (arc.points[k + 1].x as u32, arc.points[k + 1].y as u32);
                let (l, r) = sides(&seg, a, b);
                assert_eq!((l, r), (arc.left, arc.right), "arc {i} flips sides at step {k}");
            }
        }
    }

    #[test]
    fn region_rings_are_closed() {
        let mut img = Raster::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let d = ((x as f64 - 16.0).powi(2) + (y as f64 - 16.0).powi(2)).sqrt();
                let c = if d < 10.0 {
                    [0.1, 0.2, 0.8, 1.0]
                } else {
                    [0.95, 0.95, 0.9, 1.0]
                };
                img.set(x, y, c);
            }
        }
        let seg = seg_of(&img, 2);
        let topo = build(&seg);
        for rings in &topo.region_rings {
            for ring in rings {
                let mut pts: Vec<Point> = Vec::new();
                for r in &ring.refs {
                    pts.extend(topo.arc_points(*r));
                }
                assert!(pts.len() > 2);
                assert!(
                    pts[0].dist(*pts.last().unwrap()) < 1e-9,
                    "ring not closed: {:?} .. {:?}",
                    pts[0],
                    pts.last().unwrap()
                );
            }
        }
    }

    #[test]
    fn image_corners_are_junctions() {
        let img = solid(8, 8, [0.2, 0.4, 0.6, 1.0]);
        let seg = seg_of(&img, 2);
        let topo = build(&seg);
        // A single uniform region: its boundary is the image border, which must
        // be cut at the four corners rather than traced as one smooth loop.
        let r = seg.region_at(4, 4);
        let refs: usize = topo.region_rings[r as usize]
            .iter()
            .map(|ring| ring.refs.len())
            .sum();
        assert_eq!(refs, 4, "expected four border arcs, got {refs}");
        for arc in &topo.arcs {
            assert!(arc.on_border);
        }
    }
}
