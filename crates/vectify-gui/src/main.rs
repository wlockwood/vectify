//! Vectify: a desktop raster-to-vector tracer.

// Do not pop up a console window alongside the GUI on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use vectify_gui::app::VectifyApp;

fn main() -> eframe::Result<()> {
    // An optional path lets the app be launched straight onto an image, from a
    // shell or a "open with" association.
    let initial = std::env::args().nth(1).map(std::path::PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 880.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Vectify"),
        ..Default::default()
    };
    eframe::run_native(
        "Vectify",
        options,
        Box::new(move |cc| {
            let mut app = VectifyApp::new(cc);
            if let Some(path) = initial {
                app.open(&cc.egui_ctx, path);
            }
            Ok(Box::new(app))
        }),
    )
}
