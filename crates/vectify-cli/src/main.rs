//! Vectify command line: trace, auto-select, score and benchmark.
//!
//! NOTE: 100% vibe coded as a one-off tool; not a maintained product.

mod bench;
mod report;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use vectify_core::auto::{auto_select, profile, AutoConfig};
use vectify_core::color::{LabCache, Rgba8};
use vectify_core::config::{
    PaletteMethod, Preset, SegmentMode, VectorFormat, VectorizeConfig,
    DEFAULT_TRANSPARENT_TOLERANCE,
};
use vectify_core::export;
use vectify_core::raster::Raster;
use vectify_core::score::{score, ScoreConfig};
use vectify_core::segment::scoring_reference;
use vectify_core::vectorize::vectorize;

#[derive(Parser)]
#[command(
    name = "vectify",
    version,
    about = "Trace raster images into vector art, and measure how well it went."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Trace an image with explicit settings.
    Trace(TraceArgs),
    /// Try many settings, measure each, and keep the best.
    Auto(AutoArgs),
    /// Score an existing vector file against the raster it came from.
    Score(ScoreArgs),
    /// Run the built-in benchmark suite.
    Bench(BenchArgs),
    /// Write the built-in test images to a directory as PNGs.
    GenTestdata(GenArgs),
    /// Report what the profiler makes of an image.
    Inspect(InspectArgs),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum PresetArg {
    BlackAndWhite,
    Logo,
    Clipart,
    Artwork,
    Photo,
    PixelArt,
}

impl From<PresetArg> for Preset {
    fn from(p: PresetArg) -> Preset {
        match p {
            PresetArg::BlackAndWhite => Preset::BlackAndWhite,
            PresetArg::Logo => Preset::Logo,
            PresetArg::Clipart => Preset::Clipart,
            PresetArg::Artwork => Preset::Artwork,
            PresetArg::Photo => Preset::Photo,
            PresetArg::PixelArt => Preset::PixelArt,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum PaletteArg {
    Kmeans,
    MedianCut,
}

fn parse_color(s: &str) -> Result<Rgba8, String> {
    Rgba8::from_hex(s).ok_or_else(|| format!("{s:?} is not a colour; use #rrggbb or #rgb"))
}

/// Colours to leave unpainted. Shared by `trace` and `auto`.
#[derive(Args, Clone)]
struct TransparencyArgs {
    /// Leave this colour unpainted, as if it were transparent: regions of it
    /// get no shape and the artwork around them is cut out of the page. Takes
    /// #rrggbb or #rgb. Repeat the flag, or separate colours with commas, to
    /// key out several.
    // Field names double as clap argument ids, which must be unique across the
    // flattened command: a bare `colors` here would collide with `--colors`.
    #[arg(long, value_name = "COLOR", value_parser = parse_color, value_delimiter = ',')]
    transparent: Vec<Rgba8>,
    /// How far a colour in the image may be from a --transparent colour and
    /// still count as it, as a CIELAB distance. Raise it for noisy or
    /// off-white backgrounds.
    #[arg(long, default_value_t = DEFAULT_TRANSPARENT_TOLERANCE)]
    transparent_tolerance: f32,
}

/// Settings shared by `trace` and, as starting points, by `auto`.
#[derive(Args, Clone)]
struct TuningArgs {
    /// Starting preset.
    #[arg(long, value_enum, default_value = "logo")]
    preset: PresetArg,
    /// Number of colours to reduce to.
    #[arg(long)]
    colors: Option<usize>,
    /// Two-tone output, split at a luminance threshold.
    #[arg(long)]
    binary: bool,
    /// Luminance threshold in 0..1 for `--binary`. Omit to choose it by Otsu.
    #[arg(long)]
    threshold: Option<f64>,
    /// Maximum curve deviation, in pixels. Lower is more faithful and costs
    /// more control points.
    #[arg(long)]
    tolerance: Option<f64>,
    /// Smoothing strength in 0..1.
    #[arg(long)]
    smoothing: Option<f64>,
    /// Corner detection threshold in degrees.
    #[arg(long)]
    corner_threshold: Option<f64>,
    /// Discard regions smaller than this many pixels.
    #[arg(long)]
    despeckle: Option<u32>,
    /// Palette selection algorithm.
    #[arg(long, value_enum)]
    palette: Option<PaletteArg>,
    /// Turn off sub-pixel edge reconstruction.
    #[arg(long)]
    no_subpixel: bool,
    /// Emit circular arcs as SVG arc commands rather than cubic curves.
    #[arg(long)]
    true_arcs: bool,
    /// Do not fit circular arcs at all.
    #[arg(long)]
    no_arcs: bool,
    /// Give every region its own measured colour instead of a palette entry.
    #[arg(long)]
    recolor: bool,
    /// Emit one path per region instead of one per colour.
    #[arg(long)]
    no_group: bool,
    /// Uniform output scale.
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Decimal places in emitted coordinates.
    #[arg(long, default_value_t = 3)]
    precision: u32,
    #[command(flatten)]
    transparency: TransparencyArgs,
}

impl TuningArgs {
    fn build(&self, format: VectorFormat) -> VectorizeConfig {
        let mut c = Preset::from(self.preset).config();
        c.segment.transparent_colors = self.transparency.transparent.clone();
        c.segment.transparent_tolerance = self.transparency.transparent_tolerance;
        if self.binary {
            c.segment.mode = SegmentMode::Binary;
        }
        if let Some(k) = self.colors {
            c.segment.colors = k.max(2);
            if !self.binary {
                c.segment.mode = SegmentMode::Palette;
            }
        }
        if let Some(t) = self.threshold {
            c.segment.threshold = Some(t);
        }
        if let Some(t) = self.tolerance {
            c.fit.tolerance = t;
        }
        if let Some(s) = self.smoothing {
            c.corners.smoothing = s.clamp(0.0, 1.0);
        }
        if let Some(d) = self.corner_threshold {
            c.corners.threshold_deg = d;
        }
        if let Some(d) = self.despeckle {
            c.segment.despeckle_area = d;
        }
        if let Some(p) = self.palette {
            c.segment.palette_method = match p {
                PaletteArg::Kmeans => PaletteMethod::KMeans,
                PaletteArg::MedianCut => PaletteMethod::MedianCut,
            };
        }
        if self.no_subpixel {
            c.subpixel.enabled = false;
        }
        if self.true_arcs {
            c.fit.allow_arcs = true;
            c.fit.emit_true_arcs = true;
        }
        if self.no_arcs {
            c.fit.allow_arcs = false;
        }
        c.output.recolor_regions = self.recolor;
        c.output.group_by_color = !self.no_group;
        c.output.scale = self.scale;
        c.output.precision = self.precision;
        c.output.format = format;
        c
    }
}

#[derive(Args)]
struct TraceArgs {
    /// Input raster image.
    input: PathBuf,
    /// Output vector file. Format is taken from its extension.
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    tuning: TuningArgs,
    /// Measure the result by rendering it back to pixels and comparing.
    #[arg(long)]
    check: bool,
    /// Write the round-trip rendering to this PNG.
    #[arg(long)]
    dump_render: Option<PathBuf>,
}

#[derive(Args)]
struct AutoArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    /// Match percentage a candidate must reach to be considered acceptable.
    #[arg(long, default_value_t = 99.0)]
    target: f64,
    /// CIEDE2000 difference below which a pixel counts as matching.
    #[arg(long, default_value_t = 2.0)]
    delta_e: f32,
    /// Candidates evaluated in the first stage.
    #[arg(long, default_value_t = 12)]
    seeds: usize,
    /// How many of the best seeds to refine.
    #[arg(long, default_value_t = 2)]
    refine: usize,
    /// Search on a copy no larger than this; 0 disables downscaling.
    #[arg(long, default_value_t = 640)]
    search_size: u32,
    /// Show every candidate, not just the winner.
    #[arg(long)]
    show_all: bool,
    /// Write the full report as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Decimal places in emitted coordinates.
    #[arg(long, default_value_t = 3)]
    precision: u32,
    #[command(flatten)]
    transparency: TransparencyArgs,
}

#[derive(Args)]
struct ScoreArgs {
    /// The original raster.
    raster: PathBuf,
    /// The vector file to judge. Must be SVG.
    vector: PathBuf,
    #[arg(long, default_value_t = 2.0)]
    delta_e: f32,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct BenchArgs {
    /// Extra images to include beyond the built-in suite.
    #[arg(long)]
    images: Vec<PathBuf>,
    /// Use the auto-picker for each image rather than a fixed preset.
    #[arg(long)]
    auto: bool,
    /// Preset to use when not running with `--auto`.
    #[arg(long, value_enum, default_value = "logo")]
    preset: PresetArg,
    #[arg(long, default_value_t = 99.0)]
    target: f64,
    /// Write traced SVGs and round-trip renders here.
    #[arg(long)]
    out_dir: Option<PathBuf>,
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Args)]
struct GenArgs {
    /// Directory to write PNGs into.
    dir: PathBuf,
}

#[derive(Args)]
struct InspectArgs {
    input: PathBuf,
}

fn format_for(path: &Path) -> Result<VectorFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    VectorFormat::from_extension(ext).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot infer a vector format from {:?}; use .svg, .eps, .pdf or .dxf",
            path
        )
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Trace(a) => cmd_trace(a),
        Command::Auto(a) => cmd_auto(a),
        Command::Score(a) => cmd_score(a),
        Command::Bench(a) => bench::run(a_to_bench(a)),
        Command::GenTestdata(a) => cmd_gen(a),
        Command::Inspect(a) => cmd_inspect(a),
    }
}

fn a_to_bench(a: BenchArgs) -> bench::BenchOptions {
    bench::BenchOptions {
        extra_images: a.images,
        auto: a.auto,
        preset: a.preset.into(),
        target: a.target,
        out_dir: a.out_dir,
        json: a.json,
    }
}

fn cmd_trace(a: TraceArgs) -> Result<()> {
    let img = Raster::load(&a.input)?;
    let format = format_for(&a.output)?;
    let cfg = a.tuning.build(format);

    let result = vectorize(&img, &cfg);
    let bytes = export::export(&result.image, &cfg.output);
    std::fs::write(&a.output, &bytes)
        .with_context(|| format!("writing {}", a.output.display()))?;

    let c = result.image.complexity();
    println!(
        "{} -> {}  ({}x{})",
        a.input.display(),
        a.output.display(),
        img.width,
        img.height
    );
    println!(
        "  {} shapes, {} segments, {} anchors + {} handles = {} points",
        c.shapes,
        c.segments,
        c.anchors,
        c.handles,
        c.total_points()
    );
    println!(
        "  primitives: {} lines, {} curves, {} arcs",
        c.lines, c.cubics, c.arcs
    );
    println!(
        "  {:.1} KB, traced in {:.0} ms",
        bytes.len() as f64 / 1024.0,
        result.stats.total_ms()
    );

    if a.check || a.dump_render.is_some() {
        let sc = ScoreConfig::default();
        // Judge a colour-keyed trace against the image it was asked to
        // reproduce, not one whose keyed-out pixels count as missing.
        let reference = scoring_reference(&img, &cfg.segment, &LabCache::new());
        let report = score(&reference, &result.image, &sc)?;
        println!("{}", report::format_report(&report));
        if let Some(p) = a.dump_render {
            let rendered = vectify_core::score::render(&result.image, 1)?;
            rendered.save(&p)?;
            println!("  round-trip render written to {}", p.display());
        }
    }
    Ok(())
}

fn cmd_auto(a: AutoArgs) -> Result<()> {
    let img = Raster::load(&a.input)?;
    let format = format_for(&a.output)?;

    let mut cfg = AutoConfig {
        max_seeds: a.seeds,
        refine_top: a.refine,
        search_max_dimension: if a.search_size == 0 {
            None
        } else {
            Some(a.search_size)
        },
        transparent_colors: a.transparency.transparent.clone(),
        transparent_tolerance: a.transparency.transparent_tolerance,
        ..Default::default()
    };
    cfg.score.target_match = a.target;
    cfg.score.delta_e_threshold = a.delta_e;

    println!(
        "Searching for the best settings for {} ({}x{})...",
        a.input.display(),
        img.width,
        img.height
    );
    let last = std::sync::atomic::AtomicUsize::new(0);
    let result = auto_select(&img, &cfg, &|done, total| {
        // Only redraw when the count actually advances, so parallel workers do
        // not fight over the line.
        let prev = last.swap(done, std::sync::atomic::Ordering::Relaxed);
        if done > prev {
            eprint!("\r  evaluated {done}/{total} candidates");
        }
    });
    eprintln!();

    let Some(best) = result.best() else {
        bail!("no candidate could be evaluated");
    };

    println!("{}", report::format_profile(&result.profile));
    if result.searched_downscaled {
        println!("  (searched on a downscaled copy; winners re-measured at full size)");
    }

    if a.show_all {
        println!("\n{}", report::format_candidate_table(&result.candidates, a.target));
    }

    println!("\nBest: {}", best.label);
    println!("{}", report::format_config(&best.config));
    println!("{}", report::format_report(&best.report));
    println!(
        "  searched {} candidates in {:.1} s",
        result.candidates.len(),
        result.elapsed_ms / 1000.0
    );
    if !best.report.meets(a.target) {
        println!(
            "  note: no candidate reached the {:.1}% target; this was the closest",
            a.target
        );
    }

    let mut out_cfg = best.config.clone();
    out_cfg.output.format = format;
    out_cfg.output.precision = a.precision;
    let traced = vectorize(&img, &out_cfg);
    let bytes = export::export(&traced.image, &out_cfg.output);
    std::fs::write(&a.output, &bytes)
        .with_context(|| format!("writing {}", a.output.display()))?;
    println!("\nWrote {} ({:.1} KB)", a.output.display(), bytes.len() as f64 / 1024.0);

    if let Some(p) = a.json {
        let json = report::auto_result_json(&result, a.target);
        std::fs::write(&p, serde_json::to_string_pretty(&json)?)?;
        println!("Report written to {}", p.display());
    }
    Ok(())
}

fn cmd_score(a: ScoreArgs) -> Result<()> {
    let img = Raster::load(&a.raster)?;
    let svg = std::fs::read_to_string(&a.vector)
        .with_context(|| format!("reading {}", a.vector.display()))?;
    let rendered = vectify_core::score::rasterize_svg(&svg, 1)?;
    if rendered.width != img.width || rendered.height != img.height {
        bail!(
            "size mismatch: raster is {}x{} but the vector renders at {}x{}",
            img.width,
            img.height,
            rendered.width,
            rendered.height
        );
    }
    let mut sc = ScoreConfig::default();
    sc.delta_e_threshold = a.delta_e;
    let mut report = vectify_core::score::compare(&img, &rendered, &sc);
    report.output_bytes = svg.len();

    if a.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{} vs {}", a.raster.display(), a.vector.display());
        println!("{}", report::format_report(&report));
    }
    Ok(())
}

fn cmd_gen(a: GenArgs) -> Result<()> {
    std::fs::create_dir_all(&a.dir)?;
    for t in vectify_core::synth::suite() {
        let path = a.dir.join(format!("{}.png", t.name));
        t.image.save(&path)?;
        println!(
            "{:<14} {}x{}  {}",
            t.name,
            t.image.width,
            t.image.height,
            t.description
        );
    }
    println!("\nWritten to {}", a.dir.display());
    Ok(())
}

fn cmd_inspect(a: InspectArgs) -> Result<()> {
    let img = Raster::load(&a.input)?;
    let p = profile(&img);
    println!("{} ({}x{})", a.input.display(), img.width, img.height);
    println!("{}", report::format_profile(&p));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_definition_is_valid() {
        // Catches things the compiler cannot, such as two flattened argument
        // groups whose fields share a name and so collide as clap ids -- which
        // otherwise only surfaces as a panic the first time the binary runs.
        Cli::command().debug_assert();
    }

    fn trace_args(extra: &[&str]) -> TraceArgs {
        let mut argv = vec!["vectify", "trace", "in.png", "-o", "out.svg"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).expect("arguments should parse").command {
            Command::Trace(a) => a,
            _ => unreachable!("parsed a trace command"),
        }
    }

    #[test]
    fn transparent_colours_reach_the_config() {
        let a = trace_args(&["--transparent", "#fff", "--transparent", "00ff00"]);
        let cfg = a.tuning.build(VectorFormat::Svg);
        assert_eq!(
            cfg.segment.transparent_colors,
            vec![Rgba8::opaque(255, 255, 255), Rgba8::opaque(0, 255, 0)]
        );
    }

    #[test]
    fn transparent_colours_can_be_comma_separated() {
        let a = trace_args(&["--transparent", "#ffffff,#000000"]);
        assert_eq!(a.tuning.transparency.transparent.len(), 2);
    }

    #[test]
    fn transparent_tolerance_reaches_the_config() {
        let a = trace_args(&["--transparent", "#fff", "--transparent-tolerance", "15"]);
        let cfg = a.tuning.build(VectorFormat::Svg);
        assert_eq!(cfg.segment.transparent_tolerance, 15.0);
    }

    #[test]
    fn nothing_is_transparent_unless_asked() {
        let cfg = trace_args(&[]).tuning.build(VectorFormat::Svg);
        assert!(cfg.segment.transparent_colors.is_empty());
    }

    #[test]
    fn a_bad_colour_is_rejected_with_a_useful_message() {
        let err = Cli::try_parse_from(["vectify", "trace", "in.png", "-o", "o.svg", "--transparent", "banana"])
            .err()
            .expect("should not parse");
        assert!(err.to_string().contains("not a colour"), "{err}");
    }
}
