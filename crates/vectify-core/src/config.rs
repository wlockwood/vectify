//! Tunable parameters for the tracing pipeline, plus named presets.
//!
//! Every knob that materially changes output lives here, and the whole struct
//! is serialisable. That matters for two reasons: the auto-picker works by
//! generating and scoring populations of these, and any result the GUI shows
//! can be reproduced exactly from the config recorded alongside it.

use serde::{Deserialize, Serialize};

use crate::color::Rgba8;

/// How the image is divided into flat colour regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentMode {
    /// Two regions split at a luminance threshold. Best for line art, scans
    /// and silhouettes.
    Binary,
    /// A fixed number of colours found by clustering.
    Palette,
}

/// Algorithm used to choose the palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaletteMethod {
    /// Weighted k-means in CIELAB. Slower, but follows the real colour
    /// distribution and is the better choice for most artwork.
    KMeans,
    /// Recursive median cut in RGB. Fast and deterministic; tends to preserve
    /// rarely-used accent colours that k-means would merge away.
    MedianCut,
}

/// Default CIELAB radius for matching `SegmentConfig::transparent_colors`.
/// About the size of the difference between white and a very pale tint of it:
/// wide enough to absorb a noisy background, narrow enough not to reach a real
/// light colour.
pub const DEFAULT_TRANSPARENT_TOLERANCE: f32 = 8.0;

fn default_transparent_tolerance() -> f32 {
    DEFAULT_TRANSPARENT_TOLERANCE
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentConfig {
    pub mode: SegmentMode,
    pub palette_method: PaletteMethod,
    /// Target colour count for `Palette` mode.
    pub colors: usize,
    /// Luminance split for `Binary` mode, in 0..1. `None` selects Otsu.
    pub threshold: Option<f64>,
    /// Swap which side of the threshold is treated as ink.
    pub invert: bool,
    /// Pixels at or below this alpha become a transparent region that is
    /// omitted from the output entirely.
    pub alpha_threshold: f32,
    /// Colours to leave unpainted, exactly like fully transparent pixels.
    ///
    /// A region in one of these colours gets no shape of its own: the shapes
    /// around it simply end at its border. Because the document is a planar
    /// subdivision that leaves a true hole, not a white shape stacked behind
    /// the artwork, and the borders of the surrounding shapes are still
    /// reconstructed from the real colour contrast. The alpha of each entry is
    /// ignored.
    #[serde(default)]
    pub transparent_colors: Vec<Rgba8>,
    /// How far, in CIELAB distance, a palette colour may sit from one of
    /// `transparent_colors` and still count as it. Real backgrounds are rarely
    /// the exact colour asked for -- JPEG noise and gradients pull the
    /// clustered colour a little off -- so this is a radius, not an equality.
    #[serde(default = "default_transparent_tolerance")]
    pub transparent_tolerance: f32,
    /// Regions smaller than this many pixels are absorbed into their
    /// most-shared neighbour. This is the main defence against JPEG mosquito
    /// noise and stray scanner speckle.
    pub despeckle_area: u32,
    /// When building the palette, down-weight pixels that sit on a strong
    /// gradient. Anti-aliased edge pixels are *mixtures* of two real colours,
    /// not colours in their own right; letting them vote creates phantom
    /// palette entries that show up as halo outlines around every shape.
    pub ignore_edge_pixels: bool,
    /// Iteration cap for k-means.
    pub kmeans_iterations: usize,
    /// Seeds the deterministic PRNG used for k-means++ initialisation, so runs
    /// are reproducible.
    pub seed: u64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        SegmentConfig {
            mode: SegmentMode::Palette,
            palette_method: PaletteMethod::KMeans,
            colors: 8,
            threshold: None,
            invert: false,
            alpha_threshold: 0.5,
            transparent_colors: Vec::new(),
            transparent_tolerance: DEFAULT_TRANSPARENT_TOLERANCE,
            despeckle_area: 4,
            ignore_edge_pixels: true,
            kmeans_iterations: 24,
            seed: 0x5EED,
        }
    }
}

/// Sub-pixel edge reconstruction: reading anti-aliasing to recover where the
/// original mathematical boundary fell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubpixelConfig {
    pub enabled: bool,
    /// How far along the normal, in pixels, to search for the true edge.
    pub search_radius: f64,
    /// Hard cap on how far a vertex may move. Keeps a bad local solve from
    /// tearing the contour.
    pub max_shift: f64,
    /// Minimum colour separation between the two regions before the solve is
    /// trusted. Below this, the anti-aliasing signal is buried in noise.
    pub min_contrast: f32,
    /// Smoothing passes applied to the displacement field along each contour.
    /// Displacements are estimated independently per vertex, so a little
    /// smoothing removes solve jitter without moving the edge.
    pub smooth_passes: usize,
    /// Solve/re-estimate rounds. The coverage inversion needs to know the edge
    /// orientation, but the only orientation available on the first pass comes
    /// from the raw staircase contour, which is badly wrong for shallow slopes.
    /// Each round re-measures orientation from the previous round's much
    /// straighter contour, so this converges quickly -- two rounds already
    /// recover shallow edges to a few hundredths of a pixel.
    pub iterations: usize,
}

impl Default for SubpixelConfig {
    fn default() -> Self {
        SubpixelConfig {
            enabled: true,
            search_radius: 1.5,
            max_shift: 0.85,
            min_contrast: 0.06,
            smooth_passes: 1,
            iterations: 3,
        }
    }
}

/// Separating intentional geometry from compression artefacts and noise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CornerConfig {
    /// Turn angle, in degrees, above which a vertex may be a corner.
    pub threshold_deg: f64,
    /// Neighbourhood radii, in vertices, at which turn is measured. A true
    /// corner bends at every scale; noise and pixel stair-stepping bend only at
    /// the smallest. Requiring agreement across scales is what separates them.
    pub scales: Vec<usize>,
    /// Fraction of scales that must exceed the threshold, in 0..1.
    pub persistence: f64,
    /// Minimum spacing between accepted corners, in vertices.
    pub suppress_radius: usize,
    /// Adapt smoothing to locally measured noise instead of applying one
    /// blanket filter to the whole canvas.
    pub noise_adaptive: bool,
    /// Overall smoothing strength in 0..1, applied between corners only.
    pub smoothing: f64,
}

impl Default for CornerConfig {
    fn default() -> Self {
        CornerConfig {
            threshold_deg: 60.0,
            scales: vec![1, 2, 3, 5, 8],
            persistence: 0.6,
            suppress_radius: 2,
            noise_adaptive: true,
            smoothing: 0.5,
        }
    }
}

/// Curve fitting: turning refined contours into the fewest primitives that
/// still hold the shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitConfig {
    /// Maximum allowed deviation from the traced contour, in pixels.
    pub tolerance: f64,
    /// Try straight lines first. Almost always worth it: a line costs zero
    /// control handles.
    pub allow_lines: bool,
    /// Try circular arcs before falling back to cubics.
    pub allow_arcs: bool,
    /// Emit arcs as SVG `A` commands rather than cubic approximations. More
    /// compact and mathematically exact, but some consumers handle arcs badly.
    pub emit_true_arcs: bool,
    /// Newton-Raphson reparameterisation rounds during cubic fitting.
    pub reparam_iterations: usize,
    /// After fitting, attempt to replace adjacent segment pairs with a single
    /// segment where tolerance still allows.
    pub merge_pass: bool,
    /// Recursion cap for the split-and-refit loop.
    pub max_depth: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        FitConfig {
            tolerance: 0.35,
            allow_lines: true,
            allow_arcs: true,
            emit_true_arcs: false,
            reparam_iterations: 6,
            merge_pass: true,
            max_depth: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorFormat {
    Svg,
    Eps,
    Pdf,
    Dxf,
}

impl VectorFormat {
    pub fn extension(self) -> &'static str {
        match self {
            VectorFormat::Svg => "svg",
            VectorFormat::Eps => "eps",
            VectorFormat::Pdf => "pdf",
            VectorFormat::Dxf => "dxf",
        }
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "svg" => Some(VectorFormat::Svg),
            "eps" | "ps" => Some(VectorFormat::Eps),
            "pdf" => Some(VectorFormat::Pdf),
            "dxf" => Some(VectorFormat::Dxf),
            _ => None,
        }
    }

    pub fn all() -> &'static [VectorFormat] {
        &[
            VectorFormat::Svg,
            VectorFormat::Eps,
            VectorFormat::Pdf,
            VectorFormat::Dxf,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: VectorFormat,
    /// Decimal places in emitted coordinates. Three is plenty at pixel scale
    /// and keeps files small; shared borders are rounded identically on both
    /// sides so they stay welded.
    pub precision: u32,
    /// Merge all regions sharing a palette colour into one path element.
    pub group_by_color: bool,
    /// Fill each region with its own measured mean colour instead of its
    /// palette entry. Raises fidelity, especially on shaded artwork, at the
    /// cost of producing more distinct colours than the palette asked for.
    pub recolor_regions: bool,
    /// Uniform scale applied on output.
    pub scale: f64,
    /// Paint an explicit background rectangle behind everything.
    pub background: Option<Rgba8>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            format: VectorFormat::Svg,
            precision: 3,
            group_by_color: true,
            recolor_regions: false,
            scale: 1.0,
            background: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VectorizeConfig {
    pub segment: SegmentConfig,
    pub subpixel: SubpixelConfig,
    pub corners: CornerConfig,
    pub fit: FitConfig,
    pub output: OutputConfig,
}

/// Named starting points. The auto-picker seeds its search from these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Preset {
    /// Two-tone line art, signatures, stencils.
    BlackAndWhite,
    /// Flat-colour logos with hard edges.
    Logo,
    /// Illustrations and clip art with a moderate palette.
    Clipart,
    /// Detailed artwork with soft shading.
    Artwork,
    /// Photographic input; many colours, smooth everything.
    Photo,
    /// Pixel art and screenshots: preserve hard pixel edges exactly, no
    /// sub-pixel guessing and no smoothing.
    PixelArt,
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Preset::BlackAndWhite => "black-and-white",
            Preset::Logo => "logo",
            Preset::Clipart => "clipart",
            Preset::Artwork => "artwork",
            Preset::Photo => "photo",
            Preset::PixelArt => "pixel-art",
        }
    }

    pub fn all() -> &'static [Preset] {
        &[
            Preset::BlackAndWhite,
            Preset::Logo,
            Preset::Clipart,
            Preset::Artwork,
            Preset::Photo,
            Preset::PixelArt,
        ]
    }

    pub fn from_name(s: &str) -> Option<Preset> {
        Preset::all()
            .iter()
            .copied()
            .find(|p| p.name() == s.to_ascii_lowercase())
    }

    pub fn config(self) -> VectorizeConfig {
        let mut c = VectorizeConfig::default();
        match self {
            Preset::BlackAndWhite => {
                c.segment.mode = SegmentMode::Binary;
                c.segment.colors = 2;
                c.segment.despeckle_area = 6;
                c.fit.tolerance = 0.3;
            }
            Preset::Logo => {
                c.segment.colors = 6;
                c.segment.despeckle_area = 8;
                c.corners.threshold_deg = 50.0;
                c.fit.tolerance = 0.3;
            }
            Preset::Clipart => {
                c.segment.colors = 12;
                c.segment.despeckle_area = 5;
                c.fit.tolerance = 0.4;
            }
            Preset::Artwork => {
                c.segment.colors = 24;
                c.segment.despeckle_area = 4;
                c.corners.threshold_deg = 70.0;
                c.corners.smoothing = 0.6;
                c.fit.tolerance = 0.5;
            }
            Preset::Photo => {
                c.segment.colors = 48;
                c.segment.despeckle_area = 6;
                c.corners.threshold_deg = 80.0;
                c.corners.smoothing = 0.75;
                c.fit.tolerance = 0.7;
                // Photographs have no flat palette to be faithful to, so
                // per-region colour is pure fidelity gain here.
                c.output.recolor_regions = true;
            }
            Preset::PixelArt => {
                // Every pixel edge here is intentional. Turning off sub-pixel
                // reconstruction and smoothing, and demanding a tight
                // tolerance, reproduces the blocks exactly.
                c.segment.colors = 32;
                c.segment.despeckle_area = 0;
                c.segment.ignore_edge_pixels = false;
                c.subpixel.enabled = false;
                c.corners.threshold_deg = 30.0;
                c.corners.smoothing = 0.0;
                c.corners.noise_adaptive = false;
                c.fit.tolerance = 0.05;
                c.fit.allow_arcs = false;
            }
        }
        c
    }
}
