//! Calibration: how close does the tracer get to the best possible answer?
//!
//! A round-trip score below 100% does not by itself mean the tracer did
//! anything wrong. Even a *perfect* vectorisation -- the exact geometry the
//! image was drawn from -- will not reproduce the original pixel for pixel,
//! because the renderer scoring it anti-aliases slightly differently from the
//! renderer that made the input. On a high-contrast edge a coverage difference
//! of one or two percent is already a visible-threshold colour difference.
//!
//! So these tests establish the **ceiling** for each scene by scoring its own
//! exact geometry, and then require the tracer to land close to that ceiling.
//! That separates "our contour is in the wrong place" from "the measurement
//! has a floor", which raw percentages cannot distinguish.

use vectify_core::auto::{auto_select, AutoConfig};
use vectify_core::color::Rgba8;
use vectify_core::geom::Point;
use vectify_core::raster::Raster;
use vectify_core::score::{compare, rasterize_svg, score, ScoreConfig};
use vectify_core::synth::{Scene, Shape};

fn rgb(r: u8, g: u8, b: u8) -> Rgba8 {
    Rgba8::opaque(r, g, b)
}

fn circle_scene() -> Scene {
    Scene {
        width: 160,
        height: 160,
        background: rgb(250, 250, 245),
        items: vec![(
            Shape::Circle {
                centre: Point::new(80.3, 79.7),
                radius: 52.4,
            },
            rgb(25, 110, 190),
        )],
    }
}

fn slant_scene() -> Scene {
    Scene {
        width: 180,
        height: 140,
        background: rgb(255, 255, 255),
        items: vec![
            (
                Shape::Polygon(vec![
                    Point::new(10.0, 12.3),
                    Point::new(170.0, 40.7),
                    Point::new(170.0, 58.1),
                    Point::new(10.0, 29.9),
                ]),
                rgb(40, 40, 40),
            ),
            (
                Shape::Polygon(vec![
                    Point::new(90.0, 72.5),
                    Point::new(170.0, 118.5),
                    Point::new(110.0, 132.0),
                ]),
                rgb(20, 140, 90),
            ),
        ],
    }
}

fn star_scene() -> Scene {
    let centre = Point::new(80.0, 80.0);
    let (outer, inner, points) = (66.0, 27.0, 7usize);
    let mut pts = Vec::new();
    for i in 0..points * 2 {
        let r = if i % 2 == 0 { outer } else { inner };
        let t = i as f64 / (points * 2) as f64 * std::f64::consts::TAU
            - std::f64::consts::FRAC_PI_2;
        pts.push(Point::new(centre.x + r * t.cos(), centre.y + r * t.sin()));
    }
    Scene {
        width: 160,
        height: 160,
        background: rgb(255, 255, 255),
        items: vec![(Shape::Polygon(pts), rgb(230, 160, 20))],
    }
}

struct Calibration {
    ceiling: f64,
    achieved: f64,
    points: usize,
}

fn calibrate(name: &str, scene: &Scene, supersample: u32) -> Calibration {
    let img = scene.render(supersample);
    let cfg = ScoreConfig::default();

    // The ceiling: score the scene's own exact geometry.
    let ideal_svg = scene.to_ideal_svg();
    let ideal_render = rasterize_svg(&ideal_svg, 1).expect("render ideal svg");
    let ceiling = compare(&img, &ideal_render, &cfg);

    // What the tracer actually achieves, letting the auto-picker choose.
    let auto_cfg = AutoConfig {
        max_seeds: 6,
        refine_top: 1,
        refine_rounds: 2,
        search_max_dimension: None,
        ..Default::default()
    };
    let res = auto_select(&img, &auto_cfg, &|_, _| {});
    let best = res.best().expect("a candidate");
    let achieved = score(&img, &{
        let t = vectify_core::vectorize::vectorize(&img, &best.config);
        t.image
    }, &cfg)
    .expect("score traced");

    println!(
        "{name:<12} ceiling {:>6.2}%  achieved {:>6.2}%  gap {:>5.2}pp  points {:>4}  \
         (ceiling deltaE mean {:.3}, achieved {:.3})",
        ceiling.match_pct,
        achieved.match_pct,
        ceiling.match_pct - achieved.match_pct,
        achieved.complexity.total_points(),
        ceiling.mean_delta_e,
        achieved.mean_delta_e,
    );

    Calibration {
        ceiling: ceiling.match_pct,
        achieved: achieved.match_pct,
        points: achieved.complexity.total_points(),
    }
}

#[test]
fn tracer_approaches_the_theoretical_ceiling() {
    let cases: Vec<(&str, Scene)> = vec![
        ("circle", circle_scene()),
        ("slants", slant_scene()),
        ("star", star_scene()),
    ];

    for (name, scene) in cases {
        let c = calibrate(name, &scene, 16);
        assert!(
            c.ceiling > 95.0,
            "{name}: ceiling {:.2}% is implausibly low; the calibration itself is broken",
            c.ceiling
        );
        // The tracer must not give away much against a perfect answer.
        assert!(
            c.achieved >= c.ceiling - 1.5,
            "{name}: achieved {:.2}% against a ceiling of {:.2}%",
            c.achieved,
            c.ceiling
        );
        assert!(c.points > 0);
    }
}

#[test]
fn subpixel_reconstruction_beats_pixel_snapping_end_to_end() {
    // The headline claim, measured on a full round trip rather than on
    // contours: turning sub-pixel reconstruction off should visibly cost
    // accuracy on anti-aliased artwork.
    let scene = circle_scene();
    let img = scene.render(16);
    let sc = ScoreConfig::default();

    let mut cfg = vectify_core::config::Preset::Logo.config();
    cfg.segment.colors = 2;

    let with = score(
        &img,
        &vectify_core::vectorize::vectorize(&img, &cfg).image,
        &sc,
    )
    .unwrap();

    cfg.subpixel.enabled = false;
    let without = score(
        &img,
        &vectify_core::vectorize::vectorize(&img, &cfg).image,
        &sc,
    )
    .unwrap();

    println!(
        "circle: sub-pixel on {:.2}% (deltaE {:.3}), off {:.2}% (deltaE {:.3})",
        with.match_pct, with.mean_delta_e, without.match_pct, without.mean_delta_e
    );
    assert!(
        with.match_pct > without.match_pct,
        "sub-pixel reconstruction did not help: {:.2}% vs {:.2}%",
        with.match_pct,
        without.match_pct
    );
    assert!(
        with.mean_delta_e < without.mean_delta_e * 0.8,
        "mean error barely improved: {:.3} vs {:.3}",
        with.mean_delta_e,
        without.mean_delta_e
    );
}

#[test]
fn hard_edged_input_round_trips_exactly() {
    // No anti-aliasing means no ambiguity: the tracer should reproduce this
    // pixel for pixel, and any shortfall is a real defect rather than a
    // rasteriser difference.
    let scene = Scene {
        width: 120,
        height: 90,
        background: rgb(255, 255, 255),
        items: vec![
            (
                Shape::Polygon(vec![
                    Point::new(15.0, 15.0),
                    Point::new(75.0, 15.0),
                    Point::new(75.0, 55.0),
                    Point::new(15.0, 55.0),
                ]),
                rgb(30, 60, 180),
            ),
            (
                Shape::Polygon(vec![
                    Point::new(55.0, 40.0),
                    Point::new(105.0, 40.0),
                    Point::new(105.0, 75.0),
                    Point::new(55.0, 75.0),
                ]),
                rgb(220, 60, 40),
            ),
        ],
    };
    let img = scene.render_hard();
    let mut cfg = vectify_core::config::Preset::Logo.config();
    cfg.segment.colors = 3;
    let traced = vectify_core::vectorize::vectorize(&img, &cfg);
    let r = score(&img, &traced.image, &ScoreConfig::default()).unwrap();
    println!(
        "hard rects: {:.3}% match, {} points, max deltaE {:.3}",
        r.match_pct,
        r.complexity.total_points(),
        r.max_delta_e
    );
    assert!(
        r.match_pct > 99.9,
        "aliased rectangles only reached {:.2}%",
        r.match_pct
    );
    // Two overlapping rectangles make an L and a rectangle: a handful of
    // corners, all of them lines.
    assert!(
        r.complexity.handles == 0,
        "rectilinear art should need no curve handles, got {}",
        r.complexity.handles
    );
}

/// Sanity check that the calibration harness itself is meaningful: scoring the
/// ideal SVG against a *different* scene must fail badly.
#[test]
fn calibration_harness_can_fail() {
    let a = circle_scene().render(16);
    let wrong = rasterize_svg(&star_scene().to_ideal_svg(), 1).unwrap();
    let r = compare(&a, &wrong, &ScoreConfig::default());
    assert!(r.match_pct < 80.0, "mismatched scenes scored {:.2}%", r.match_pct);
}

/// The scene renderer and the ideal-SVG writer must describe the same picture.
#[test]
fn ideal_svg_matches_the_rendered_scene() {
    for (name, scene) in [
        ("circle", circle_scene()),
        ("slants", slant_scene()),
        ("star", star_scene()),
    ] {
        let rendered = scene.render(16);
        let ideal = rasterize_svg(&scene.to_ideal_svg(), 1).unwrap();
        assert_eq!(
            (rendered.width, rendered.height),
            (ideal.width, ideal.height),
            "{name}"
        );
        let r = compare(&rendered, &ideal, &ScoreConfig::default());
        assert!(
            r.match_pct > 95.0,
            "{name}: the ideal SVG does not describe the same scene ({:.2}%)",
            r.match_pct
        );
    }
}

/// Ring scenes exercise holes; confirm the ideal writer handles them too.
#[test]
fn ring_ideal_svg_has_a_hole() {
    let scene = Scene {
        width: 120,
        height: 120,
        background: rgb(255, 255, 255),
        items: vec![(
            Shape::Ring {
                centre: Point::new(60.0, 60.0),
                outer: 50.0,
                inner: 25.0,
            },
            rgb(40, 90, 160),
        )],
    };
    let ideal: Raster = rasterize_svg(&scene.to_ideal_svg(), 1).unwrap();
    // Centre must be background, the band must be the fill.
    let centre = ideal.get(60, 60);
    let band = ideal.get(60, 22);
    assert!(centre[0] > 0.9 && centre[2] > 0.9, "hole not punched: {:?}", centre);
    assert!(band[2] > 0.5 && band[0] < 0.3, "band not filled: {:?}", band);
}
