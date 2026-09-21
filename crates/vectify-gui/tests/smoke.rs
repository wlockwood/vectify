//! Headless UI tests.
//!
//! These drive the real widget tree without opening a window, so layout
//! panics, duplicate widget ids, borrow mistakes and broken worker plumbing are
//! caught by `cargo test` rather than by someone launching the app and
//! clicking around.
//!
//! Note that `Harness::run` loops until the UI stops asking to be repainted,
//! which this app never does while a trace is in flight (there is a spinner).
//! These tests therefore drive it a fixed number of frames with `run_steps`.

use std::time::{Duration, Instant};

use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use vectify_core::color::Rgba8;
use vectify_core::config::SegmentMode;
use vectify_core::geom::Point;
use vectify_core::raster::Raster;
use vectify_core::synth::{Scene, Shape};
use vectify_gui::app::VectifyApp;
use vectify_gui::viewer::ViewMode;

/// A small anti-aliased scene: enough structure for a real trace, small enough
/// that the worker finishes promptly.
fn test_image() -> Raster {
    Scene {
        width: 64,
        height: 64,
        background: Rgba8::opaque(250, 250, 245),
        items: vec![
            (
                Shape::Circle {
                    centre: Point::new(28.0, 28.0),
                    radius: 17.0,
                },
                Rgba8::opaque(200, 60, 50),
            ),
            (
                Shape::Polygon(vec![
                    Point::new(34.0, 34.0),
                    Point::new(60.0, 34.0),
                    Point::new(60.0, 58.0),
                    Point::new(34.0, 58.0),
                ]),
                Rgba8::opaque(40, 90, 180),
            ),
        ],
    }
    .render(8)
}

fn app_with_image() -> (egui::Context, VectifyApp) {
    let ctx = egui::Context::default();
    let mut app = VectifyApp::with_context(&ctx);
    app.set_source(&ctx, test_image(), None);
    (ctx, app)
}

#[test]
fn renders_with_no_image_loaded() {
    let ctx = egui::Context::default();
    let app = VectifyApp::with_context(&ctx);
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);
    harness.run_steps(3);
    assert!(harness.state().last_error().is_none());
}

#[test]
fn renders_every_view_mode() {
    let (_ctx, app) = app_with_image();
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);
    for mode in ViewMode::ALL {
        harness.state_mut().set_view(mode);
        harness.run_steps(2);
        assert!(
            harness.state().last_error().is_none(),
            "{:?} reported {:?}",
            mode,
            harness.state().last_error()
        );
    }
}

#[test]
fn a_trace_completes_and_reaches_the_ui() {
    // The full round trip through the worker: queue a trace, let the background
    // thread run, and confirm a plausible result arrives in the UI.
    let (_ctx, mut app) = app_with_image();
    app.config_mut().segment.colors = 3;
    app.flush_debounce();

    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);

    let deadline = Instant::now() + Duration::from_secs(30);
    while !harness.state().has_preview() && Instant::now() < deadline {
        harness.run_steps(1);
        std::thread::sleep(Duration::from_millis(10));
    }

    let app = harness.state();
    assert!(app.last_error().is_none(), "error: {:?}", app.last_error());
    assert!(app.has_preview(), "no preview arrived within the timeout");

    let m = app.preview_match().expect("match percentage");
    let p = app.preview_points().expect("point count");
    assert!(m > 95.0, "preview matched only {m:.2}%");
    assert!((1..5000).contains(&p), "preview used {p} control points");
}

#[test]
fn settings_branches_all_render() {
    // Each of these swaps which controls the settings panel shows; a bad widget
    // id or a borrow mistake in one of those branches would only appear here.
    let (_ctx, app) = app_with_image();
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);

    let variants: [&dyn Fn(&mut VectifyApp); 4] = [
        &|a| {
            let c = a.config_mut();
            c.segment.mode = SegmentMode::Binary;
            c.segment.threshold = None;
        },
        &|a| {
            let c = a.config_mut();
            c.segment.mode = SegmentMode::Binary;
            c.segment.threshold = Some(0.42);
        },
        &|a| {
            let c = a.config_mut();
            c.segment.mode = SegmentMode::Palette;
            c.fit.allow_arcs = false;
        },
        &|a| {
            let c = a.config_mut();
            c.segment.mode = SegmentMode::Palette;
            c.fit.allow_arcs = true;
            c.fit.emit_true_arcs = true;
        },
    ];

    for v in variants {
        v(harness.state_mut());
        harness.run_steps(2);
        assert!(harness.state().last_error().is_none());
    }
}

#[test]
fn rapid_setting_changes_stay_consistent() {
    // Every change queues a trace, replacing the one waiting. The last change
    // made must be the one whose result survives, rather than a stale trace
    // landing afterwards and overwriting it.
    let (_ctx, app) = app_with_image();
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);

    for k in [2usize, 5, 3, 8, 4] {
        harness.state_mut().config_mut().segment.colors = k;
        harness.state_mut().flush_debounce();
        harness.run_steps(2);
    }
    harness.state_mut().config_mut().segment.colors = 6;
    harness.state_mut().flush_debounce();

    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        harness.run_steps(1);
        std::thread::sleep(Duration::from_millis(10));
        let app = harness.state();
        if app.has_preview() && !app.is_working() {
            break;
        }
    }

    let app = harness.state();
    assert!(app.last_error().is_none(), "error: {:?}", app.last_error());
    assert!(app.has_preview());
    assert_eq!(
        app.preview_colors(),
        Some(6),
        "the surviving preview is not from the most recent settings"
    );
}

/// Render the harness through wgpu, or `None` if the machine has no GPU adapter.
///
/// `Harness::render` returns a `Result`, but that does not cover this case: it
/// creates the renderer lazily, and the adapter lookup inside it `expect`s
/// instead of returning an error, so with no adapter (a CI runner, a container)
/// it panics. Only that specific panic is turned into a skip. Any other panic,
/// and any real `Err` from rendering, is a genuine failure and is not hidden.
///
/// The panic's message is still printed by the default hook even though it is
/// caught here, so a passing skip can look like a failure in the log; the
/// "SKIPPED" line the caller prints is the authoritative record.
fn render_unless_no_gpu(harness: &mut Harness<'_, VectifyApp>) -> Option<image::RgbaImage> {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

    match catch_unwind(AssertUnwindSafe(|| harness.render())) {
        Ok(Ok(image)) => Some(image),
        Ok(Err(e)) => panic!("rendering the UI failed: {e}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            if message.contains("Failed to create render state") {
                None
            } else {
                resume_unwind(payload)
            }
        }
    }
}

/// Render the real UI offscreen through wgpu and write a PNG.
///
/// This is both a visual check and a guard against the app failing to render
/// at all on a real GPU path, which the pure-layout tests above cannot catch.
/// The GPU step is skipped when no adapter is available (see
/// [`render_unless_no_gpu`]) so it does not turn into a spurious failure on
/// machines without one; the trace and UI checks before it still run.
#[test]
fn renders_to_an_image() {
    // Use a richer scene than the other tests: this snapshot doubles as the
    // screenshot in the README, so it should show the app doing real work.
    let ctx = egui::Context::default();
    let mut app = VectifyApp::with_context(&ctx);
    let suite = vectify_core::synth::suite();
    let source = suite
        .iter()
        .find(|t| t.name == "logo")
        .map(|t| t.image.clone())
        .unwrap_or_else(test_image);
    app.set_source(&ctx, source, None);
    app.config_mut().segment.colors = 4;
    app.flush_debounce();

    let mut harness = Harness::builder()
        .with_size(egui::vec2(1360.0, 880.0))
        .build_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);

    // Give the worker time to produce a preview so the panels have real
    // numbers in them.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !harness.state().has_preview() && Instant::now() < deadline {
        harness.run_steps(1);
        std::thread::sleep(Duration::from_millis(10));
    }
    harness.run_steps(2);

    // Checked before the GPU step, so that machines which cannot render still
    // verify that the UI produced a real result rather than passing vacuously.
    assert!(harness.state().has_preview(), "no preview arrived within the timeout");
    assert!(harness.state().last_error().is_none(), "{:?}", harness.state().last_error());

    let Some(image) = render_unless_no_gpu(&mut harness) else {
        eprintln!("SKIPPED the offscreen render: this machine has no GPU adapter");
        return;
    };
    let path = std::env::var("VECTIFY_SNAPSHOT")
        .unwrap_or_else(|_| "../../target/vectify-ui.png".to_string());
    image.save(&path).expect("write snapshot");
    eprintln!("UI snapshot written to {path}");
}

/// Drive the app until a preview has arrived for the settings it currently holds.
fn wait_for_settled_preview(harness: &mut Harness<'_, VectifyApp>) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        harness.run_steps(1);
        std::thread::sleep(Duration::from_millis(10));
        let app = harness.state();
        if app.has_preview() && !app.is_working() {
            return;
        }
    }
    panic!("no settled preview within the timeout");
}

#[test]
fn a_transparent_colour_removes_its_shape_without_hurting_the_score() {
    // The blue square is keyed out, so it must be absent from the preview. It
    // must also not be scored as an error: the metrics compare against the image
    // as it was asked to come out. If the worker scored against the raw image,
    // the square's ~15% of the canvas would drag the match far below the bar.
    let (_ctx, mut app) = app_with_image();
    app.config_mut().segment.colors = 3;
    app.flush_debounce();
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);
    wait_for_settled_preview(&mut harness);
    assert_eq!(harness.state().preview_shapes(), Some(3), "page, circle and square");

    harness
        .state_mut()
        .config_mut()
        .segment
        .transparent_colors
        .push(Rgba8::opaque(40, 90, 180));
    harness.state_mut().flush_debounce();
    wait_for_settled_preview(&mut harness);

    let app = harness.state();
    assert!(app.last_error().is_none(), "error: {:?}", app.last_error());
    assert_eq!(app.preview_shapes(), Some(2), "the blue square should be gone");
    let m = app.preview_match().expect("match percentage");
    assert!(m > 95.0, "keyed-out colour was scored as an error: {m:.2}%");
}

#[test]
fn the_add_colour_button_keys_out_the_corner_colour() {
    // The image's corner is the page colour (250, 250, 245). Clicking the real
    // button should add exactly that, as the likely thing to want removed.
    let (_ctx, app) = app_with_image();
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);
    harness.run_steps(2);
    assert!(harness.state().transparent_colors().is_empty());

    harness.get_by_label("Add colour").click();
    harness.run_steps(2);

    assert_eq!(
        harness.state().transparent_colors(),
        [Rgba8::opaque(250, 250, 245)],
        "the new entry should start as the top-left pixel"
    );
    assert!(harness.state().last_error().is_none());

    // With an entry present, the list and tolerance controls render and can be
    // used: remove it again through the UI.
    harness.get_by_label("Remove").click();
    harness.run_steps(2);
    assert!(harness.state().transparent_colors().is_empty(), "Remove should drop the entry");
}

#[test]
fn several_transparent_colours_render_without_id_clashes() {
    // Each row holds a colour picker; two identical widgets in a loop is exactly
    // the shape of code that trips duplicate-id problems.
    let (_ctx, mut app) = app_with_image();
    {
        let c = app.config_mut();
        c.segment.transparent_colors = vec![
            Rgba8::opaque(250, 250, 245),
            Rgba8::opaque(250, 250, 245),
            Rgba8::opaque(0, 0, 0),
        ];
    }
    let mut harness = Harness::new_ui_state(|ui, app: &mut VectifyApp| app.draw(ui), app);
    harness.run_steps(3);
    assert!(harness.state().last_error().is_none());
}
