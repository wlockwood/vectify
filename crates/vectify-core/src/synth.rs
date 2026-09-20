//! Synthetic test images with known ground truth.
//!
//! The benchmark needs inputs whose correct answer is known exactly, and it
//! needs them to exercise the specific things this engine claims to do well:
//! sub-pixel edges at awkward angles, shared borders between three or more
//! colours, real corners sitting next to compression noise, and hard pixel
//! edges that must not be smoothed.
//!
//! Scenes are rendered by supersampling, which is how a real rasteriser
//! behaves. That matters: the point of the exercise is to invert an actual
//! rasterisation, so the test input has to be one.

use crate::color::Rgba8;
use crate::geom::Point;
use crate::quantize::Rng;
use crate::raster::Raster;

#[derive(Clone, Debug)]
pub enum Shape {
    Polygon(Vec<Point>),
    Circle { centre: Point, radius: f64 },
    /// An annulus, which forces a shape with a genuine hole.
    Ring { centre: Point, outer: f64, inner: f64 },
}

impl Shape {
    fn contains(&self, p: Point) -> bool {
        match self {
            Shape::Polygon(pts) => point_in_polygon(p, pts),
            Shape::Circle { centre, radius } => p.dist(*centre) <= *radius,
            Shape::Ring {
                centre,
                outer,
                inner,
            } => {
                let d = p.dist(*centre);
                d <= *outer && d >= *inner
            }
        }
    }
}

fn point_in_polygon(p: Point, pts: &[Point]) -> bool {
    let mut inside = false;
    let n = pts.len();
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
    }
    inside
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub background: Rgba8,
    /// Painted in order; later items cover earlier ones.
    pub items: Vec<(Shape, Rgba8)>,
}

impl Scene {
    /// Render with `ss` x `ss` supersampling, which is what produces the
    /// anti-aliased coverage the tracer is meant to invert.
    pub fn render(&self, ss: u32) -> Raster {
        let ss = ss.max(1);
        let mut out = Raster::new(self.width, self.height);
        let inv = 1.0 / (ss * ss) as f32;
        let bg = self.background.to_f32();

        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = [0.0f32; 4];
                for sy in 0..ss {
                    for sx in 0..ss {
                        let p = Point::new(
                            x as f64 + (sx as f64 + 0.5) / ss as f64,
                            y as f64 + (sy as f64 + 0.5) / ss as f64,
                        );
                        let mut c = bg;
                        for (shape, colour) in &self.items {
                            if shape.contains(p) {
                                let sc = colour.to_f32();
                                // Source-over in gamma space, matching how
                                // ordinary 2D renderers composite.
                                let a = sc[3];
                                c = [
                                    sc[0] * a + c[0] * (1.0 - a),
                                    sc[1] * a + c[1] * (1.0 - a),
                                    sc[2] * a + c[2] * (1.0 - a),
                                    a + c[3] * (1.0 - a),
                                ];
                            }
                        }
                        for i in 0..4 {
                            acc[i] += c[i];
                        }
                    }
                }
                for v in acc.iter_mut() {
                    *v *= inv;
                }
                out.set(x, y, acc);
            }
        }
        out
    }

    /// Render with no anti-aliasing at all: every pixel takes the colour at its
    /// centre. Models pixel art and screenshots.
    pub fn render_hard(&self) -> Raster {
        self.render(1)
    }

    /// The scene's exact geometry as SVG.
    ///
    /// This is the answer a perfect tracer would produce, and scoring it gives
    /// the *ceiling* for the round-trip metric: whatever it falls short of
    /// 100%, the shortfall is the difference between two rasterisers rather
    /// than anything a tracer could fix. Without that calibration it is
    /// impossible to tell a tracing defect from an artefact of the measurement.
    pub fn to_ideal_svg(&self) -> String {
        let mut s = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\" shape-rendering=\"geometricPrecision\">\n",
            w = self.width,
            h = self.height
        );
        if self.background.a > 0 {
            s.push_str(&format!(
                "<rect width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
                self.width,
                self.height,
                self.background.to_hex()
            ));
        }
        for (shape, colour) in &self.items {
            let fill = colour.to_hex();
            match shape {
                Shape::Polygon(pts) => {
                    let d: Vec<String> = pts
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            format!("{}{:.4} {:.4}", if i == 0 { "M" } else { "L" }, p.x, p.y)
                        })
                        .collect();
                    s.push_str(&format!(
                        "<path d=\"{}Z\" fill=\"{fill}\"/>\n",
                        d.join("")
                    ));
                }
                Shape::Circle { centre, radius } => {
                    s.push_str(&format!(
                        "<circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"{:.4}\" fill=\"{fill}\"/>\n",
                        centre.x, centre.y, radius
                    ));
                }
                Shape::Ring {
                    centre,
                    outer,
                    inner,
                } => {
                    // Two concentric circles with the even-odd rule punches the
                    // hole without relying on subpath winding direction.
                    s.push_str(&format!(
                        "<path fill-rule=\"evenodd\" fill=\"{fill}\" d=\"\
                         M{cx} {cy}m-{o},0a{o},{o} 0 1,0 {d2},0a{o},{o} 0 1,0 -{d2},0\
                         M{cx} {cy}m-{i},0a{i},{i} 0 1,0 {d1},0a{i},{i} 0 1,0 -{d1},0\"/>\n",
                        cx = centre.x,
                        cy = centre.y,
                        o = outer,
                        i = inner,
                        d2 = outer * 2.0,
                        d1 = inner * 2.0
                    ));
                }
            }
        }
        s.push_str("</svg>\n");
        s
    }
}

fn regular_polygon(centre: Point, radius: f64, sides: usize, rotation: f64) -> Shape {
    Shape::Polygon(
        (0..sides)
            .map(|i| {
                let t = rotation + i as f64 / sides as f64 * std::f64::consts::TAU;
                Point::new(centre.x + radius * t.cos(), centre.y + radius * t.sin())
            })
            .collect(),
    )
}

fn star(centre: Point, outer: f64, inner: f64, points: usize) -> Shape {
    let mut pts = Vec::new();
    for i in 0..points * 2 {
        let r = if i % 2 == 0 { outer } else { inner };
        let t = i as f64 / (points * 2) as f64 * std::f64::consts::TAU
            - std::f64::consts::FRAC_PI_2;
        pts.push(Point::new(centre.x + r * t.cos(), centre.y + r * t.sin()));
    }
    Shape::Polygon(pts)
}

/// Add uniform per-channel noise, standing in for sensor noise or a low-quality
/// re-encode.
pub fn add_noise(img: &Raster, amount: f32, seed: u64) -> Raster {
    let mut rng = Rng::new(seed);
    let mut out = img.clone();
    for c in out.data.iter_mut() {
        for k in 0..3 {
            let n = (rng.next_f32() - 0.5) * 2.0 * amount;
            c[k] = (c[k] + n).clamp(0.0, 1.0);
        }
    }
    out
}

/// Approximate JPEG ringing: quantise each 8x8 block's detail so that sharp
/// edges gain the overshoot and mosquito noise a lossy codec introduces.
pub fn add_block_artifacts(img: &Raster, strength: f32) -> Raster {
    let mut out = img.clone();
    let w = img.width;
    let h = img.height;
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            // Block mean.
            let mut mean = [0.0f32; 3];
            let mut n = 0.0f32;
            for y in by..(by + 8).min(h) {
                for x in bx..(bx + 8).min(w) {
                    let c = img.get(x, y);
                    for k in 0..3 {
                        mean[k] += c[k];
                    }
                    n += 1.0;
                }
            }
            if n == 0.0 {
                continue;
            }
            for v in mean.iter_mut() {
                *v /= n;
            }
            // Pull each pixel partway toward the block mean, then overshoot
            // where it differs most -- the characteristic ringing signature.
            for y in by..(by + 8).min(h) {
                for x in bx..(bx + 8).min(w) {
                    let c = img.get(x, y);
                    let mut o = c;
                    for k in 0..3 {
                        let d = c[k] - mean[k];
                        o[k] = (mean[k] + d * (1.0 + strength) - d.signum() * strength * 0.12
                            * d.abs().min(0.5))
                        .clamp(0.0, 1.0);
                    }
                    out.set(x, y, o);
                }
            }
        }
    }
    out
}

/// One benchmark case: a name, the input image, and what it is meant to probe.
pub struct TestImage {
    pub name: &'static str,
    pub description: &'static str,
    pub image: Raster,
    /// The scene's exact geometry, for cases built from one.
    ///
    /// Scoring this instead of a traced result gives the *ceiling* for the
    /// round-trip metric, because even a perfect vectorisation is rasterised by
    /// a different renderer than the one that produced the input. Without it a
    /// score of 98.5% is uninterpretable: it could mean the tracer is 1.5%
    /// wrong, or that 1.5% is simply unreachable.
    pub ideal_svg: Option<String>,
}

/// Build a case from a scene, keeping its exact geometry alongside the pixels.
fn from_scene(
    name: &'static str,
    description: &'static str,
    scene: Scene,
    supersample: u32,
) -> TestImage {
    let image = if supersample <= 1 {
        scene.render_hard()
    } else {
        scene.render(supersample)
    };
    TestImage {
        name,
        description,
        image,
        ideal_svg: Some(scene.to_ideal_svg()),
    }
}

/// Build a case from pixels alone, where no closed-form geometry exists.
fn from_raster(name: &'static str, description: &'static str, image: Raster) -> TestImage {
    TestImage {
        name,
        description,
        image,
        ideal_svg: None,
    }
}

fn rgb(r: u8, g: u8, b: u8) -> Rgba8 {
    Rgba8::opaque(r, g, b)
}

/// Five wedges meeting at a single point: the hardest case for shared-border
/// integrity, since a tracer that stacks silhouettes leaves seams radiating
/// from the centre.
fn wedge_scene() -> Scene {
    let c = Point::new(80.0, 80.0);
    let wedge = |a0: f64, a1: f64| {
        let mut pts = vec![c];
        let steps = 24;
        for i in 0..=steps {
            let t = a0 + (a1 - a0) * i as f64 / steps as f64;
            pts.push(Point::new(c.x + 110.0 * t.cos(), c.y + 110.0 * t.sin()));
        }
        Shape::Polygon(pts)
    };
    let tau = std::f64::consts::TAU;
    Scene {
        width: 160,
        height: 160,
        background: rgb(245, 245, 245),
        items: vec![
            (wedge(0.0, tau / 5.0), rgb(210, 70, 60)),
            (wedge(tau / 5.0, 2.0 * tau / 5.0), rgb(60, 150, 210)),
            (wedge(2.0 * tau / 5.0, 3.0 * tau / 5.0), rgb(240, 190, 50)),
            (wedge(3.0 * tau / 5.0, 4.0 * tau / 5.0), rgb(90, 180, 90)),
            (wedge(4.0 * tau / 5.0, tau), rgb(140, 90, 180)),
        ],
    }
}

/// The standard benchmark suite.
pub fn suite() -> Vec<TestImage> {
    let mut out = Vec::new();

    // 1. Hard-edged rectangles: must round-trip exactly, no smoothing, no drift.
    out.push(from_scene("hard-rects", "Aliased rectangles; geometry must survive untouched", Scene {
            width: 160,
            height: 120,
            background: rgb(255, 255, 255),
            items: vec![
                (
                    Shape::Polygon(vec![
                        Point::new(20.0, 20.0),
                        Point::new(100.0, 20.0),
                        Point::new(100.0, 70.0),
                        Point::new(20.0, 70.0),
                    ]),
                    rgb(30, 60, 180),
                ),
                (
                    Shape::Polygon(vec![
                        Point::new(70.0, 50.0),
                        Point::new(140.0, 50.0),
                        Point::new(140.0, 100.0),
                        Point::new(70.0, 100.0),
                    ]),
                    rgb(220, 60, 40),
                ),
            ],
        }, 1));

    // 2. Anti-aliased circle: the classic control-point economy test.
    out.push(from_scene("aa-circle", "Anti-aliased disc; few points, sub-pixel accurate", Scene {
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
        }, 8));

    // 3. Slanted edges at awkward angles: where pixel-snapping tracers show
    //    stair-stepping most clearly.
    out.push(from_scene("slants", "Shallow and steep straight edges; sub-pixel edge recovery", Scene {
            width: 200,
            height: 140,
            background: rgb(255, 255, 255),
            items: vec![
                (
                    Shape::Polygon(vec![
                        Point::new(10.0, 12.3),
                        Point::new(190.0, 40.7),
                        Point::new(190.0, 58.1),
                        Point::new(10.0, 29.9),
                    ]),
                    rgb(40, 40, 40),
                ),
                (
                    Shape::Polygon(vec![
                        Point::new(18.4, 70.0),
                        Point::new(46.9, 132.0),
                        Point::new(66.2, 132.0),
                        Point::new(37.7, 70.0),
                    ]),
                    rgb(200, 90, 20),
                ),
                (
                    Shape::Polygon(vec![
                        Point::new(90.0, 72.5),
                        Point::new(180.0, 118.5),
                        Point::new(120.0, 132.0),
                    ]),
                    rgb(20, 140, 90),
                ),
            ],
        }, 8));

    // 4. Three-colour junctions: the shared-border test. Any tracer that stacks
    //    silhouettes leaves seams along these meetings.
    out.push(from_scene(
        "junctions",
        "Wedges meeting at a point; shared-border integrity",
        wedge_scene(),
        8,
    ));

    // 5. Sharp corners: a star has unmistakable vertices that must stay sharp.
    out.push(from_scene("star", "Sharp convex and reflex corners; corner preservation", Scene {
            width: 160,
            height: 160,
            background: rgb(255, 255, 255),
            items: vec![(
                star(Point::new(80.0, 80.0), 66.0, 27.0, 7),
                rgb(230, 160, 20),
            )],
        }, 8));

    // 6. A shape with a genuine hole, plus nesting.
    out.push(from_scene("rings", "Nested holes; correct even-odd topology", Scene {
            width: 160,
            height: 160,
            background: rgb(255, 255, 255),
            items: vec![
                (
                    Shape::Ring {
                        centre: Point::new(80.0, 80.0),
                        outer: 70.0,
                        inner: 48.0,
                    },
                    rgb(40, 90, 160),
                ),
                (
                    Shape::Ring {
                        centre: Point::new(80.0, 80.0),
                        outer: 36.0,
                        inner: 18.0,
                    },
                    rgb(200, 70, 60),
                ),
            ],
        }, 8));

    // 7. Flat-colour logo: several shapes, hard palette, the bread-and-butter
    //    case.
    out.push(from_scene("logo", "Flat-colour logo; palette fidelity and compactness", Scene {
            width: 200,
            height: 200,
            background: rgb(255, 255, 255),
            items: vec![
                (
                    regular_polygon(Point::new(100.0, 100.0), 86.0, 6, 0.3),
                    rgb(28, 42, 92),
                ),
                (
                    regular_polygon(Point::new(100.0, 100.0), 58.0, 3, -0.4),
                    rgb(240, 200, 40),
                ),
                (
                    Shape::Circle {
                        centre: Point::new(100.0, 118.0),
                        radius: 24.0,
                    },
                    rgb(220, 60, 50),
                ),
            ],
        }, 8));

    // 8. Pixel art: hard blocks that must not be smoothed or shifted.
    out.push(from_raster("pixel-art", "Blocky pixel art; must stay perfectly rectilinear", {
            let mut img = Raster::new(128, 128);
            let palette = [
                [0.10, 0.10, 0.15],
                [0.85, 0.25, 0.25],
                [0.95, 0.85, 0.40],
                [0.20, 0.55, 0.85],
                [0.95, 0.95, 0.95],
            ];
            let mut rng = Rng::new(0xA11CE);
            // 8x8 blocks of flat colour.
            let mut cells = vec![0usize; 16 * 16];
            for c in cells.iter_mut() {
                *c = rng.below(palette.len());
            }
            for y in 0..128u32 {
                for x in 0..128u32 {
                    let c = palette[cells[((y / 8) * 16 + (x / 8)) as usize]];
                    img.set(x, y, [c[0], c[1], c[2], 1.0]);
                }
            }
            img
        }));

    // 9. Noisy artwork: real corners buried in compression-like artefacts. This
    //    is the case that punishes a single blanket smoothing filter.
    out.push(from_raster("noisy-shapes", "Sharp shapes plus block artefacts and noise", {
            let clean = Scene {
                width: 180,
                height: 140,
                background: rgb(250, 250, 250),
                items: vec![
                    (
                        Shape::Polygon(vec![
                            Point::new(20.0, 20.0),
                            Point::new(84.0, 20.0),
                            Point::new(84.0, 120.0),
                            Point::new(20.0, 120.0),
                        ]),
                        rgb(40, 40, 120),
                    ),
                    (
                        star(Point::new(130.0, 70.0), 44.0, 19.0, 5),
                        rgb(220, 120, 30),
                    ),
                ],
            }
            .render(8);
            add_noise(&add_block_artifacts(&clean, 0.35), 0.02, 0x5A17)
        }));

    // 10. Transparency: alpha must survive and not be painted onto a
    //     background.
    out.push(from_scene("alpha", "Shapes on a transparent background", Scene {
            width: 140,
            height: 140,
            background: Rgba8::TRANSPARENT,
            items: vec![
                (
                    Shape::Circle {
                        centre: Point::new(60.0, 70.0),
                        radius: 40.0,
                    },
                    rgb(200, 50, 90),
                ),
                (
                    regular_polygon(Point::new(95.0, 75.0), 38.0, 5, 0.2),
                    rgb(40, 150, 180),
                ),
            ],
        }, 8));

    // 11. Smooth gradient: the hardest case for a flat-fill tracer, included so
    //     the benchmark reports an honest worst case rather than only wins.
    out.push(from_raster("gradient", "Smooth gradient; worst case for flat-region tracing", {
            let mut img = Raster::new(160, 120);
            for y in 0..120u32 {
                for x in 0..160u32 {
                    let fx = x as f32 / 159.0;
                    let fy = y as f32 / 119.0;
                    img.set(x, y, [0.15 + 0.7 * fx, 0.25 + 0.5 * fy, 0.85 - 0.5 * fx, 1.0]);
                }
            }
            img
        }));

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_builds_and_is_well_formed() {
        let s = suite();
        assert!(s.len() >= 10);
        for t in &s {
            assert!(t.image.width > 0 && t.image.height > 0, "{}", t.name);
            assert_eq!(
                t.image.data.len(),
                (t.image.width * t.image.height) as usize,
                "{}",
                t.name
            );
            for c in &t.image.data {
                for v in c {
                    assert!(v.is_finite() && (0.0..=1.0).contains(v), "{}", t.name);
                }
            }
        }
    }

    #[test]
    fn supersampling_produces_real_antialiasing() {
        let scene = Scene {
            width: 32,
            height: 32,
            background: rgb(0, 0, 0),
            items: vec![(
                Shape::Circle {
                    centre: Point::new(16.0, 16.0),
                    radius: 10.0,
                },
                rgb(255, 255, 255),
            )],
        };
        let aa = scene.render(8);
        // Somewhere on the rim there must be partially covered pixels.
        let partial = aa
            .data
            .iter()
            .filter(|c| c[0] > 0.05 && c[0] < 0.95)
            .count();
        assert!(partial > 10, "only {partial} partial pixels");

        let hard = scene.render_hard();
        let partial_hard = hard
            .data
            .iter()
            .filter(|c| c[0] > 0.05 && c[0] < 0.95)
            .count();
        assert_eq!(partial_hard, 0, "centre sampling must not anti-alias");
    }

    #[test]
    fn transparent_background_stays_transparent() {
        let all = suite();
        let img = &all.iter().find(|t| t.name == "alpha").unwrap().image;
        assert!(img.get(2, 2)[3] < 0.01, "corner should be transparent");
        assert!(img.has_alpha());
    }

    #[test]
    fn ring_has_a_hole() {
        let r = Shape::Ring {
            centre: Point::new(0.0, 0.0),
            outer: 10.0,
            inner: 5.0,
        };
        assert!(r.contains(Point::new(7.0, 0.0)));
        assert!(!r.contains(Point::new(2.0, 0.0)));
        assert!(!r.contains(Point::new(12.0, 0.0)));
    }

    #[test]
    fn polygon_containment_handles_concavity() {
        let s = star(Point::new(0.0, 0.0), 10.0, 4.0, 5);
        assert!(s.contains(Point::new(0.0, 0.0)));
        assert!(!s.contains(Point::new(9.0, 9.0)));
    }
}
