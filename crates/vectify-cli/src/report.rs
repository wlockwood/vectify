//! Human-readable and JSON formatting of results.

use serde_json::{json, Value};

use vectify_core::auto::{AutoResult, Candidate, ImageProfile};
use vectify_core::config::{SegmentMode, VectorizeConfig};
use vectify_core::score::ScoreReport;

/// A bar that makes "99.1%" and "96.4%" distinguishable at a glance, which
/// plain numbers in a column are not.
fn bar(fraction: f64, width: usize) -> String {
    let filled = ((fraction.clamp(0.0, 1.0)) * width as f64).round() as usize;
    format!(
        "{}{}",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

pub fn format_report(r: &ScoreReport) -> String {
    let c = r.complexity;
    let mut s = String::new();
    s.push_str(&format!(
        "  match   {:>6.2}%  [{}]   (pixels within deltaE {:.1})\n",
        r.match_pct,
        // Stretch the last few percent, where all the interesting differences
        // live; a linear bar is pinned to full for everything usable.
        bar((r.match_pct - 90.0) / 10.0, 24),
        2.0
    ));
    s.push_str(&format!(
        "  deltaE  mean {:.2}  p95 {:.2}  max {:.2}\n",
        r.mean_delta_e, r.p95_delta_e, r.max_delta_e
    ));
    s.push_str(&format!(
        "  rmse    {:.2}   psnr {:.1} dB   ssim {:.4}\n",
        r.rmse, r.psnr, r.ssim
    ));
    s.push_str(&format!(
        "  points  {} ({} anchors + {} handles) in {} segments, {} shapes\n",
        c.total_points(),
        c.anchors,
        c.handles,
        c.segments,
        c.shapes
    ));
    s.push_str(&format!(
        "  size    {:.1} KB   render {:.0} ms   compare {:.0} ms",
        r.output_bytes as f64 / 1024.0,
        r.render_ms,
        r.compare_ms
    ));
    s
}

pub fn format_profile(p: &ImageProfile) -> String {
    let mut tags: Vec<&str> = Vec::new();
    if p.pixel_art {
        tags.push("pixel-art");
    }
    if p.photographic {
        tags.push("photographic");
    }
    if p.bilevel {
        tags.push("bilevel");
    }
    if p.has_alpha {
        tags.push("has-alpha");
    }
    if tags.is_empty() {
        tags.push("general artwork");
    }
    format!(
        "  profile: {} | {} colours | anti-aliasing {:.0}% | flat {:.0}% | edges {:.1}%",
        tags.join(", "),
        p.unique_colors,
        p.antialiasing * 100.0,
        p.flatness * 100.0,
        p.edge_density * 100.0
    )
}

pub fn format_config(c: &VectorizeConfig) -> String {
    let mode = match c.segment.mode {
        SegmentMode::Binary => "binary".to_string(),
        SegmentMode::Palette => format!("{} colours", c.segment.colors),
    };
    let transparent = if c.segment.transparent_colors.is_empty() {
        String::new()
    } else {
        let list: Vec<String> = c.segment.transparent_colors.iter().map(|k| k.to_hex()).collect();
        format!(", transparent {}", list.join(" "))
    };
    format!(
        "  config:  {mode}, tolerance {:.2}px, smoothing {:.2}, corners {:.0} deg, \
         despeckle {}px{}{}{}",
        c.fit.tolerance,
        c.corners.smoothing,
        c.corners.threshold_deg,
        c.segment.despeckle_area,
        if c.subpixel.enabled { "" } else { ", no sub-pixel" },
        if c.fit.emit_true_arcs { ", true arcs" } else { "" },
        transparent
    )
}

pub fn format_candidate_table(cands: &[Candidate], target: f64) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<24} {:>8} {:>8} {:>7} {:>8} {:>6}\n",
        "candidate", "match%", "points", "ok", "size KB", "ms"
    ));
    s.push_str(&"-".repeat(66));
    s.push('\n');
    for c in cands.iter().take(40) {
        s.push_str(&format!(
            "{:<24} {:>8.2} {:>8} {:>7} {:>8.1} {:>6.0}\n",
            truncate(&c.label, 24),
            c.report.match_pct,
            c.points(),
            if c.report.meets(target) { "yes" } else { "no" },
            c.report.output_bytes as f64 / 1024.0,
            c.stats.total_ms()
        ));
    }
    s
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n.saturating_sub(3)])
    }
}

pub fn report_json(r: &ScoreReport) -> Value {
    json!({
        "match_pct": r.match_pct,
        "mean_delta_e": r.mean_delta_e,
        "p95_delta_e": r.p95_delta_e,
        "max_delta_e": r.max_delta_e,
        "rmse": r.rmse,
        "psnr": r.psnr,
        "ssim": r.ssim,
        "total_points": r.complexity.total_points(),
        "anchors": r.complexity.anchors,
        "handles": r.complexity.handles,
        "segments": r.complexity.segments,
        "shapes": r.complexity.shapes,
        "lines": r.complexity.lines,
        "cubics": r.complexity.cubics,
        "arcs": r.complexity.arcs,
        "output_bytes": r.output_bytes,
    })
}

pub fn auto_result_json(res: &AutoResult, target: f64) -> Value {
    json!({
        "profile": res.profile,
        "elapsed_ms": res.elapsed_ms,
        "searched_downscaled": res.searched_downscaled,
        "target_match": target,
        "candidates": res.candidates.iter().map(|c| json!({
            "label": c.label,
            "meets_target": c.report.meets(target),
            "score": report_json(&c.report),
            "config": c.config,
            "trace_ms": c.stats.total_ms(),
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_scales_and_clamps() {
        assert_eq!(bar(0.0, 4), "----");
        assert_eq!(bar(1.0, 4), "####");
        assert_eq!(bar(0.5, 4), "##--");
        assert_eq!(bar(-5.0, 4), "----");
        assert_eq!(bar(9.0, 4), "####");
    }

    #[test]
    fn truncate_keeps_width() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a-very-long-label-here", 10).len(), 10);
    }
}
