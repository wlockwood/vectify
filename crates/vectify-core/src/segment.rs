//! Turning a quantised image into a map of labelled, connected regions.
//!
//! Two maps come out of this stage. `labels` holds a palette index per pixel.
//! `regions` holds a *connected component* id per pixel, so two separate blobs
//! of the same colour are distinct regions. The topology pass works on regions
//! rather than labels, because the boundary between two same-coloured but
//! disconnected blobs is still a real boundary that has to be traced.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::color::{Lab, LabCache, Rgba8};
use crate::config::SegmentConfig;
use crate::quantize::{build_palette, flatness_weights, Palette};
use crate::raster::Raster;

/// Sentinel region id for "outside the image". The topology pass adds a virtual
/// border of this region so that shapes touching the edge still close.
pub const OUTSIDE: u32 = u32::MAX;

pub struct Segmentation {
    pub width: u32,
    pub height: u32,
    /// Palette index per pixel.
    pub labels: Vec<u32>,
    /// Connected-component id per pixel.
    pub regions: Vec<u32>,
    /// Palette index of each region.
    pub region_label: Vec<u32>,
    /// Pixel count of each region.
    pub region_area: Vec<u32>,
    /// Flatness-weighted mean colour of each region, taken from the *source*
    /// pixels rather than the palette. The sub-pixel solver needs the real
    /// colours either side of an edge to invert the blend correctly, and a
    /// region's true interior colour can sit slightly off its palette entry.
    pub region_color: Vec<Rgba8>,
    pub palette: Palette,
}

impl Segmentation {
    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    #[inline]
    pub fn region_at(&self, x: u32, y: u32) -> u32 {
        self.regions[self.idx(x, y)]
    }

    /// Region id treating out-of-bounds coordinates as `OUTSIDE`.
    #[inline]
    pub fn region_at_signed(&self, x: i64, y: i64) -> u32 {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            OUTSIDE
        } else {
            self.region_at(x as u32, y as u32)
        }
    }

    pub fn region_count(&self) -> usize {
        self.region_area.len()
    }

    /// Fill colour for a region: transparent regions are reported as such so
    /// the exporter can drop them.
    pub fn region_fill(&self, region: u32) -> Rgba8 {
        if self.is_transparent(region) {
            return Rgba8::TRANSPARENT;
        }
        let label = self.region_label[region as usize] as usize;
        self.palette.entries[label].color
    }

    /// Whether a region is left unpainted: outside the image, alpha-transparent,
    /// or one of the colours the config asked to have keyed out.
    pub fn is_transparent(&self, region: u32) -> bool {
        if region == OUTSIDE {
            return true;
        }
        let label = self.region_label[region as usize] as usize;
        self.palette.is_transparent(label)
    }
}

/// Assign every pixel to its nearest palette entry.
fn assign_labels(img: &Raster, palette: &Palette, cfg: &SegmentConfig, cache: &LabCache) -> Vec<u32> {
    let alpha_slot = palette.alpha_slot;
    img.data
        .iter()
        .map(|c| {
            if let Some(t) = alpha_slot {
                if c[3] <= cfg.alpha_threshold {
                    return t as u32;
                }
            }
            palette.nearest(cache.lookup([c[0], c[1], c[2]])) as u32
        })
        .collect()
}

/// The image a trace made with `cfg` should be scored against.
///
/// A colour-keyed region is deliberately absent from the vector output, so
/// scoring the render against the untouched original would count every such
/// pixel as an error -- and the picker would then try to buy the "error" back.
/// Pixels within `transparent_tolerance` of a keyed colour are made transparent
/// here instead, so the scorer sees what the trace was asked to produce.
///
/// Borrows when nothing is keyed, so the common case costs nothing.
pub fn scoring_reference<'a>(
    img: &'a Raster,
    cfg: &SegmentConfig,
    cache: &LabCache,
) -> Cow<'a, Raster> {
    if cfg.transparent_colors.is_empty() {
        return Cow::Borrowed(img);
    }
    let keys: Vec<Lab> = cfg
        .transparent_colors
        .iter()
        .map(|k| cache.lookup([k.r as f32 / 255.0, k.g as f32 / 255.0, k.b as f32 / 255.0]))
        .collect();
    let tolerance_sq = cfg.transparent_tolerance.max(0.0).powi(2);

    let mut out = img.clone();
    for c in out.data.iter_mut() {
        let lab = cache.lookup([c[0], c[1], c[2]]);
        if keys.iter().any(|k| k.dist_sq(lab) <= tolerance_sq) {
            c[3] = 0.0;
        }
    }
    Cow::Owned(out)
}

/// 4-connected component labelling over equal label values.
fn connected_components(
    width: u32,
    height: u32,
    labels: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = labels.len();
    let mut regions = vec![u32::MAX; n];
    let mut region_label: Vec<u32> = Vec::new();
    let mut region_area: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let w = width as i64;
    let h = height as i64;

    for start in 0..n {
        if regions[start] != u32::MAX {
            continue;
        }
        let label = labels[start];
        let id = region_label.len() as u32;
        region_label.push(label);
        let mut area = 0u32;

        regions[start] = id;
        stack.push(start as u32);
        while let Some(p) = stack.pop() {
            area += 1;
            let x = (p as i64) % w;
            let y = (p as i64) / w;
            let mut visit = |nx: i64, ny: i64, stack: &mut Vec<u32>| {
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    return;
                }
                let q = (ny * w + nx) as usize;
                if regions[q] == u32::MAX && labels[q] == label {
                    regions[q] = id;
                    stack.push(q as u32);
                }
            };
            visit(x - 1, y, &mut stack);
            visit(x + 1, y, &mut stack);
            visit(x, y - 1, &mut stack);
            visit(x, y + 1, &mut stack);
        }
        region_area.push(area);
    }
    (regions, region_label, region_area)
}

/// Absorb regions below `min_area` into whichever neighbour they share the most
/// border with. Run repeatedly, smallest first, so that a cluster of specks
/// collapses rather than merely shuffling between each other.
fn despeckle(
    width: u32,
    height: u32,
    labels: &mut [u32],
    min_area: u32,
) -> bool {
    if min_area == 0 {
        return false;
    }
    let (regions, region_label, region_area) = connected_components(width, height, labels);
    let small: Vec<u32> = (0..region_area.len() as u32)
        .filter(|&r| region_area[r as usize] < min_area)
        .collect();
    if small.is_empty() {
        return false;
    }

    // Shared border length from each small region to each neighbouring region.
    let mut contacts: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
    let w = width as i64;
    let h = height as i64;
    let is_small: Vec<bool> = region_area.iter().map(|&a| a < min_area).collect();

    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) as usize;
            let r = regions[p];
            if !is_small[r as usize] {
                continue;
            }
            let entry = contacts.entry(r).or_default();
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let q = (ny * w + nx) as usize;
                let rq = regions[q];
                if rq != r {
                    *entry.entry(rq).or_insert(0) += 1;
                }
            }
        }
    }

    // Smallest first: a 1px speck should be swallowed before a 5px one gets to
    // vote on where it goes.
    let mut order = small.clone();
    order.sort_by_key(|&r| region_area[r as usize]);

    // Current label of each region, updated as merges resolve, so chains of
    // merges settle onto a final surviving label.
    let mut current: Vec<u32> = region_label.clone();
    let mut changed = false;

    for r in order {
        let Some(nb) = contacts.get(&r) else { continue };
        // Prefer a large neighbour; among equals, prefer the longest border.
        let mut best: Option<(u32, u32, u32)> = None; // (border, area, region)
        for (&other, &border) in nb {
            if other == u32::MAX {
                continue;
            }
            let area = region_area[other as usize];
            let key = (border, area, other);
            if best.is_none_or(|b| (b.0, b.1) < (key.0, key.1)) {
                best = Some(key);
            }
        }
        let Some((_, _, target)) = best else { continue };
        if current[target as usize] == current[r as usize] {
            continue;
        }
        current[r as usize] = current[target as usize];
        changed = true;
    }

    if changed {
        for (p, label) in labels.iter_mut().enumerate() {
            *label = current[regions[p] as usize];
        }
    }
    changed
}

/// Flatness-weighted mean source colour per region.
fn region_colors(
    img: &Raster,
    regions: &[u32],
    region_count: usize,
    palette: &Palette,
    region_label: &[u32],
    weights: &[f32],
) -> Vec<Rgba8> {
    let mut acc = vec![[0.0f64; 5]; region_count];
    for (p, &r) in regions.iter().enumerate() {
        let c = img.data[p];
        // Add a floor to the weight so a region made entirely of edge pixels
        // still gets a colour rather than dividing by zero.
        let w = (weights[p] as f64).max(0.02);
        let a = &mut acc[r as usize];
        a[0] += c[0] as f64 * w;
        a[1] += c[1] as f64 * w;
        a[2] += c[2] as f64 * w;
        a[3] += c[3] as f64 * w;
        a[4] += w;
    }
    acc.iter()
        .enumerate()
        .map(|(r, a)| {
            if a[4] < 1e-9 {
                palette.entries[region_label[r] as usize].color
            } else {
                Rgba8::from_f32([
                    (a[0] / a[4]) as f32,
                    (a[1] / a[4]) as f32,
                    (a[2] / a[4]) as f32,
                    (a[3] / a[4]) as f32,
                ])
            }
        })
        .collect()
}

pub fn segment(img: &Raster, cfg: &SegmentConfig, cache: &LabCache) -> Segmentation {
    let palette = build_palette(img, cfg, cache);
    let mut labels = assign_labels(img, &palette, cfg, cache);

    // Two or three despeckle rounds are enough in practice; the third rarely
    // changes anything and the loop exits early when it does not.
    for _ in 0..3 {
        if !despeckle(img.width, img.height, &mut labels, cfg.despeckle_area) {
            break;
        }
    }

    let (regions, region_label, region_area) = connected_components(img.width, img.height, &labels);
    let weights = flatness_weights(img);
    let region_color = region_colors(
        img,
        &regions,
        region_label.len(),
        &palette,
        &region_label,
        &weights,
    );

    Segmentation {
        width: img.width,
        height: img.height,
        labels,
        regions,
        region_label,
        region_area,
        region_color,
        palette,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SegmentConfig, SegmentMode};

    fn cfg_binary() -> SegmentConfig {
        SegmentConfig {
            mode: SegmentMode::Binary,
            despeckle_area: 0,
            ..Default::default()
        }
    }

    /// A white field with two separate black squares.
    fn two_squares() -> Raster {
        let mut r = Raster::new(20, 10);
        for y in 0..10 {
            for x in 0..20 {
                r.set(x, y, [1.0, 1.0, 1.0, 1.0]);
            }
        }
        for y in 2..6 {
            for x in 2..6 {
                r.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
            for x in 12..16 {
                r.set(x, y, [0.0, 0.0, 0.0, 1.0]);
            }
        }
        r
    }

    #[test]
    fn disconnected_blobs_become_separate_regions() {
        let cache = LabCache::new();
        let seg = segment(&two_squares(), &cfg_binary(), &cache);
        // One background plus two squares.
        assert_eq!(seg.region_count(), 3, "regions: {:?}", seg.region_area);
        let a = seg.region_at(3, 3);
        let b = seg.region_at(13, 3);
        assert_ne!(a, b, "same-coloured but disconnected blobs must not share a region");
        assert_eq!(seg.region_area[a as usize], 16);
        assert_eq!(seg.region_area[b as usize], 16);
    }

    #[test]
    fn despeckle_removes_lone_pixels() {
        let mut img = Raster::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.set(x, y, [1.0, 1.0, 1.0, 1.0]);
            }
        }
        // A single stray dark pixel, the shape JPEG noise and scanner dust take.
        img.set(8, 8, [0.0, 0.0, 0.0, 1.0]);

        let cache = LabCache::new();
        let mut cfg = cfg_binary();
        cfg.despeckle_area = 0;
        let kept = segment(&img, &cfg, &cache);
        assert_eq!(kept.region_count(), 2, "speck should survive with despeckle off");

        cfg.despeckle_area = 4;
        let cleaned = segment(&img, &cfg, &cache);
        assert_eq!(cleaned.region_count(), 1, "speck should be absorbed");
    }

    #[test]
    fn transparent_pixels_get_their_own_label() {
        let mut img = Raster::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let a = if x < 4 { 1.0 } else { 0.0 };
                img.set(x, y, [0.2, 0.4, 0.6, a]);
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
        assert!(seg.palette.alpha_slot.is_some());
        assert!(seg.is_transparent(seg.region_at(6, 4)));
        assert!(!seg.is_transparent(seg.region_at(1, 4)));
    }

    /// Left half white, right half red: an opaque image with a colour to key out.
    fn white_and_red() -> Raster {
        let mut r = Raster::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                r.set(x, y, if x < 4 { [1.0, 1.0, 1.0, 1.0] } else { [0.8, 0.1, 0.1, 1.0] });
            }
        }
        r
    }

    fn palette_cfg() -> SegmentConfig {
        SegmentConfig {
            mode: SegmentMode::Palette,
            colors: 2,
            despeckle_area: 0,
            ..Default::default()
        }
    }

    #[test]
    fn a_keyed_colour_becomes_a_transparent_region() {
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            transparent_colors: vec![Rgba8::opaque(255, 255, 255)],
            ..palette_cfg()
        };
        let seg = segment(&white_and_red(), &cfg, &cache);
        assert!(seg.is_transparent(seg.region_at(1, 4)), "white should be keyed out");
        assert!(!seg.is_transparent(seg.region_at(6, 4)), "red should still be painted");
        assert_eq!(seg.region_fill(seg.region_at(1, 4)), Rgba8::TRANSPARENT);
        assert_eq!(seg.palette.alpha_slot, None, "an opaque image needs no alpha slot");
    }

    #[test]
    fn keyed_regions_keep_their_real_colour_for_edge_solving() {
        // The sub-pixel solver reads `region_color` to invert edge blends. A
        // keyed region must still report the colour it really is -- it is the
        // background the neighbouring shape's anti-aliasing was blended with.
        let cache = LabCache::new();
        let cfg = SegmentConfig {
            transparent_colors: vec![Rgba8::opaque(255, 255, 255)],
            ..palette_cfg()
        };
        let seg = segment(&white_and_red(), &cfg, &cache);
        let bg = seg.region_color[seg.region_at(1, 4) as usize];
        assert_eq!((bg.r, bg.g, bg.b, bg.a), (255, 255, 255, 255));
    }

    #[test]
    fn scoring_reference_borrows_when_nothing_is_keyed() {
        let cache = LabCache::new();
        let img = white_and_red();
        let r = scoring_reference(&img, &palette_cfg(), &cache);
        assert!(matches!(r, Cow::Borrowed(_)));
    }

    #[test]
    fn scoring_reference_clears_only_the_keyed_pixels() {
        let cache = LabCache::new();
        let img = white_and_red();
        let cfg = SegmentConfig {
            transparent_colors: vec![Rgba8::opaque(255, 255, 255)],
            ..palette_cfg()
        };
        let r = scoring_reference(&img, &cfg, &cache);
        assert_eq!(r.get(1, 4)[3], 0.0, "white pixel should be transparent");
        assert_eq!(r.get(6, 4)[3], 1.0, "red pixel should be untouched");
        assert_eq!(img.get(1, 4)[3], 1.0, "the original must not be modified");
    }

    #[test]
    fn region_colors_track_the_source_not_the_palette() {
        let mut img = Raster::new(12, 6);
        for y in 0..6 {
            for x in 0..12 {
                let c = if x < 6 {
                    [0.80, 0.20, 0.20, 1.0]
                } else {
                    [0.20, 0.30, 0.85, 1.0]
                };
                img.set(x, y, c);
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
        let left = seg.region_color[seg.region_at(2, 3) as usize];
        assert!((left.r as i32 - 204).abs() < 12, "got {:?}", left);
        assert!((left.g as i32 - 51).abs() < 12, "got {:?}", left);
    }
}
