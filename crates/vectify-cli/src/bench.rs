//! The benchmark harness.
//!
//! Runs every test image through the pipeline, scores the round trip, and
//! prints one row per case plus an aggregate. The aggregate is what tells you
//! whether a change to the engine helped or hurt overall, rather than just
//! moving quality between cases.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;

use vectify_core::auto::{auto_select, AutoConfig};
use vectify_core::config::{Preset, VectorizeConfig};
use vectify_core::export;
use vectify_core::raster::Raster;
use vectify_core::score::{score, ScoreConfig, ScoreReport};
use vectify_core::synth;
use vectify_core::vectorize::vectorize;

use crate::report;

pub struct BenchOptions {
    pub extra_images: Vec<PathBuf>,
    pub auto: bool,
    pub preset: Preset,
    pub target: f64,
    pub out_dir: Option<PathBuf>,
    pub json: Option<PathBuf>,
}

struct Row {
    name: String,
    description: String,
    pixels: usize,
    report: ScoreReport,
    config_label: String,
    trace_ms: f64,
    search_ms: f64,
    /// Score achieved by the scene's own exact geometry, where it is known.
    ///
    /// This is the number the tracer is really competing against. A perfect
    /// vectorisation still does not reproduce the input pixel for pixel,
    /// because the renderer used for scoring anti-aliases differently from the
    /// one that produced the image, so on edge-heavy artwork the ceiling sits
    /// below 100% no matter what any tracer does.
    ceiling: Option<f64>,
}

pub fn run(opts: BenchOptions) -> Result<()> {
    let mut cases: Vec<(String, String, Raster, Option<String>)> = synth::suite()
        .into_iter()
        .map(|t| {
            (
                t.name.to_string(),
                t.description.to_string(),
                t.image,
                t.ideal_svg,
            )
        })
        .collect();

    for p in &opts.extra_images {
        let img = Raster::load(p)?;
        let name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();
        cases.push((name, "user-supplied".to_string(), img, None));
    }

    if let Some(d) = &opts.out_dir {
        std::fs::create_dir_all(d)?;
    }

    let score_cfg = ScoreConfig {
        target_match: opts.target,
        ..Default::default()
    };

    println!(
        "Benchmarking {} images, target {:.1}% match{}\n",
        cases.len(),
        opts.target,
        if opts.auto {
            ", auto-selecting per image"
        } else {
            ""
        }
    );

    let mut rows: Vec<Row> = Vec::new();

    for (name, description, img, ideal) in &cases {
        let mut search_ms = 0.0;
        let (cfg, label): (VectorizeConfig, String) = if opts.auto {
            let auto_cfg = AutoConfig {
                score: score_cfg.clone(),
                // Benchmark images are small; search them in full.
                search_max_dimension: None,
                ..Default::default()
            };
            let res = auto_select(img, &auto_cfg, &|_, _| {});
            search_ms = res.elapsed_ms;
            match res.best() {
                Some(b) => (b.config.clone(), b.label.clone()),
                None => (opts.preset.config(), opts.preset.name().to_string()),
            }
        } else {
            (opts.preset.config(), opts.preset.name().to_string())
        };

        let traced = vectorize(img, &cfg);
        let mut report = score(img, &traced.image, &score_cfg)?;
        report.complexity = traced.image.complexity();

        if let Some(d) = &opts.out_dir {
            let svg = export::svg::write(&traced.image, &cfg.output);
            std::fs::write(d.join(format!("{name}.svg")), &svg)?;
            img.save(d.join(format!("{name}.in.png")))?;
            if let Ok(rendered) = vectify_core::score::render(&traced.image, 1) {
                rendered.save(d.join(format!("{name}.out.png")))?;
                img.difference(&rendered, 8.0)
                    .save(d.join(format!("{name}.diff.png")))?;
            }
        }

        // Where the scene's exact geometry is known, score that too: it bounds
        // what any tracer could achieve on this image under this metric.
        let ceiling = ideal.as_ref().and_then(|svg| {
            let rendered = vectify_core::score::rasterize_svg(svg, 1).ok()?;
            Some(vectify_core::score::compare(img, &rendered, &score_cfg).match_pct)
        });

        rows.push(Row {
            name: name.clone(),
            description: description.clone(),
            pixels: img.pixel_count(),
            report,
            config_label: label,
            trace_ms: traced.stats.total_ms(),
            search_ms,
            ceiling,
        });
    }

    print_table(&rows, opts.target);

    if let Some(p) = &opts.json {
        let payload = json!({
            "target_match": opts.target,
            "auto": opts.auto,
            "cases": rows.iter().map(|r| json!({
                "name": r.name,
                "description": r.description,
                "pixels": r.pixels,
                "config": r.config_label,
                "trace_ms": r.trace_ms,
                "search_ms": r.search_ms,
                "score": report::report_json(&r.report),
            })).collect::<Vec<_>>(),
        });
        std::fs::write(p, serde_json::to_string_pretty(&payload)?)?;
        println!("\nJSON report written to {}", p.display());
    }

    Ok(())
}

fn print_table(rows: &[Row], target: f64) {
    println!(
        "{:<14} {:>7} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>8} {:>7}",
        "case", "match%", "ceil%", "gap", "deltaE", "pts", "size KB", "ok", "trace ms", "pick ms"
    );
    println!("{}", "-".repeat(94));

    for r in rows {
        let (ceil, gap) = match r.ceiling {
            Some(c) => (format!("{c:.2}"), format!("{:.2}", c - r.report.match_pct)),
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "{:<14} {:>7.2} {:>7} {:>6} {:>7.2} {:>6} {:>7.1} {:>6} {:>8.0} {:>7}",
            r.name,
            r.report.match_pct,
            ceil,
            gap,
            r.report.mean_delta_e,
            r.report.complexity.total_points(),
            r.report.output_bytes as f64 / 1024.0,
            if r.report.meets(target) { "yes" } else { "NO" },
            r.trace_ms,
            if r.search_ms > 0.0 {
                format!("{:.0}", r.search_ms)
            } else {
                "-".to_string()
            }
        );
    }

    println!("{}", "-".repeat(94));
    let n = rows.len().max(1) as f64;
    let mean_match: f64 = rows.iter().map(|r| r.report.match_pct).sum::<f64>() / n;
    let worst = rows
        .iter()
        .min_by(|a, b| {
            a.report
                .match_pct
                .partial_cmp(&b.report.match_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| (r.name.as_str(), r.report.match_pct))
        .unwrap_or(("-", 0.0));
    let passed = rows.iter().filter(|r| r.report.meets(target)).count();
    let total_points: usize = rows
        .iter()
        .map(|r| r.report.complexity.total_points())
        .sum();
    let total_px: usize = rows.iter().map(|r| r.pixels).sum();
    let total_ms: f64 = rows.iter().map(|r| r.trace_ms).sum();

    println!(
        "{:<14} {:>7.2} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>8.0}",
        "MEAN/TOTAL",
        mean_match,
        "",
        "",
        "",
        total_points,
        "",
        format!("{passed}/{}", rows.len()),
        total_ms
    );
    println!(
        "\n{passed} of {} images reached the {:.1}% target.  Worst: {} at {:.2}%.",
        rows.len(),
        target,
        worst.0,
        worst.1
    );

    // The pass count above is close to meaningless on its own, because the
    // target is not always reachable. This says how much of the shortfall the
    // tracer is actually responsible for.
    let calibrated: Vec<&Row> = rows.iter().filter(|r| r.ceiling.is_some()).collect();
    if !calibrated.is_empty() {
        let gaps: Vec<f64> = calibrated
            .iter()
            .map(|r| r.ceiling.unwrap() - r.report.match_pct)
            .collect();
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let worst_gap = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_ceiling =
            calibrated.iter().map(|r| r.ceiling.unwrap()).sum::<f64>() / calibrated.len() as f64;
        println!(
            "\nOf those, {} have known geometry. Scoring the exact original shapes gives a\n\
             ceiling of {:.2}% on average -- that is what a flawless tracer would score here,\n\
             the rest being the difference between two renderers' anti-aliasing. Tracing\n\
             lands {:.2} points below that ceiling on average, {:.2} at worst.",
            calibrated.len(),
            mean_ceiling,
            mean_gap,
            worst_gap
        );
    }

    if total_ms > 0.0 {
        println!(
            "Throughput: {:.1} megapixels/second across {:.2} MP of input.",
            (total_px as f64 / 1e6) / (total_ms / 1000.0),
            total_px as f64 / 1e6
        );
    }
}
