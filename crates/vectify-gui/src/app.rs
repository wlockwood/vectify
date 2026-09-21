//! The Vectify desktop application.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, RichText, TextureHandle, TextureOptions, Ui};

use vectify_core::auto::{AutoConfig, AutoResult};
use vectify_core::color::Rgba8;
use vectify_core::config::{
    PaletteMethod, Preset, SegmentMode, VectorFormat, VectorizeConfig,
};
use vectify_core::denoise::BilateralConfig;
use vectify_core::export;
use vectify_core::raster::Raster;
use vectify_core::score::ScoreConfig;

use crate::viewer::{to_color_image, Camera, Canvas, ViewMode};
use crate::worker::{Request, Response, TraceOutcome, Worker};

/// Time to wait after the last parameter change before re-tracing. Long enough
/// that dragging a slider does not start a trace per frame, short enough that
/// the preview still feels attached to the control.
const DEBOUNCE: Duration = Duration::from_millis(220);

struct Source {
    path: Option<PathBuf>,
    raster: Arc<Raster>,
    texture: TextureHandle,
}

struct Preview {
    outcome: Box<TraceOutcome>,
    vector_tex: TextureHandle,
    diff_tex: TextureHandle,
    svg_bytes: usize,
}

pub struct VectifyApp {
    source: Option<Source>,
    preview: Option<Preview>,
    config: VectorizeConfig,
    preset: Preset,
    target_match: f64,
    delta_e: f32,
    denoise_reference: bool,
    denoise: BilateralConfig,
    export_format: VectorFormat,

    worker: Worker,
    awaiting: Option<u64>,
    auto_generation: Option<u64>,
    auto_result: Option<Box<AutoResult>>,

    view: ViewMode,
    camera: Camera,
    wipe: f32,
    show_grid: bool,
    diff_gain: f32,

    dirty_since: Option<Instant>,
    diff_dirty: bool,
    status: String,
    error: Option<String>,
}

impl VectifyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::with_context(&cc.egui_ctx)
    }

    /// Build the app against a bare context. `new` goes through here, and so do
    /// headless tests, so the tested object is the real one.
    pub fn with_context(egui_ctx: &egui::Context) -> Self {
        let ctx = egui_ctx.clone();
        let worker = Worker::new(move || ctx.request_repaint());
        VectifyApp {
            source: None,
            preview: None,
            config: Preset::Logo.config(),
            preset: Preset::Logo,
            target_match: 99.0,
            delta_e: 2.0,
            denoise_reference: false,
            denoise: BilateralConfig::default(),
            export_format: VectorFormat::Svg,
            worker,
            awaiting: None,
            auto_generation: None,
            auto_result: None,
            view: ViewMode::SideBySide,
            camera: Camera::default(),
            wipe: 0.5,
            show_grid: true,
            diff_gain: 8.0,
            dirty_since: None,
            diff_dirty: false,
            status: "Open an image to begin.".to_string(),
            error: None,
        }
    }

    fn score_config(&self) -> ScoreConfig {
        ScoreConfig {
            delta_e_threshold: self.delta_e,
            target_match: self.target_match,
            reference_denoise: self.denoise_reference.then_some(self.denoise),
            ..Default::default()
        }
    }

    /// The amplified difference between the original and the trace. When the
    /// score was measured against a smoothed original, this shows that same
    /// image, or the view would light up with noise the score deliberately
    /// ignored.
    fn difference_image(&self, src: &Raster, outcome: &TraceOutcome) -> Raster {
        match &outcome.reference {
            // The smoothed reference is composited over the scoring background,
            // so the rendering must be too, or transparent areas would differ.
            Some(reference) => reference.difference(
                &outcome.rendered.composite_over(self.score_config().background),
                self.diff_gain,
            ),
            None => src.difference(&outcome.rendered, self.diff_gain),
        }
    }

    /// Load an image from disk, reporting failures in the status bar.
    pub fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        match Raster::load(&path) {
            Ok(raster) => {
                let label = format!(
                    "{} ({}x{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    raster.width,
                    raster.height
                );
                self.set_source(ctx, raster, Some(path));
                self.status = label;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Install a raster as the working image and queue a trace.
    pub fn set_source(&mut self, ctx: &egui::Context, raster: Raster, path: Option<PathBuf>) {
        let texture = ctx.load_texture(
            "original",
            to_color_image(&raster),
            TextureOptions::NEAREST,
        );
        self.status = format!("{}x{}", raster.width, raster.height);
        self.source = Some(Source {
            path,
            raster: Arc::new(raster),
            texture,
        });
        self.preview = None;
        self.auto_result = None;
        self.camera.fitted = false;
        self.error = None;
        self.mark_dirty();
    }

    /// Whether a traced preview is currently available.
    pub fn has_preview(&self) -> bool {
        self.preview.is_some()
    }

    /// The match percentage of the current preview, if any.
    pub fn preview_match(&self) -> Option<f64> {
        self.preview.as_ref().map(|p| p.outcome.report.match_pct)
    }

    /// Control points in the current preview, if any.
    pub fn preview_points(&self) -> Option<usize> {
        self.preview
            .as_ref()
            .map(|p| p.outcome.report.complexity.total_points())
    }

    /// Any error surfaced in the status bar.
    pub fn last_error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// True while a trace is queued or running.
    pub fn is_working(&self) -> bool {
        self.dirty_since.is_some() || self.awaiting.is_some() || self.worker.is_busy()
    }

    /// Colour count of the configuration the current preview was traced with.
    /// Lets a test confirm which settings the visible result actually came
    /// from, rather than assuming.
    pub fn preview_colors(&self) -> Option<usize> {
        self.preview.as_ref().map(|p| p.outcome.config.segment.colors)
    }

    /// Number of painted shapes in the current preview.
    pub fn preview_shapes(&self) -> Option<usize> {
        self.preview.as_ref().map(|p| p.outcome.vector.shapes.len())
    }

    /// Colours currently set to be left unpainted.
    pub fn transparent_colors(&self) -> &[Rgba8] {
        &self.config.segment.transparent_colors
    }

    /// Force the pending trace to start now instead of after the debounce.
    pub fn flush_debounce(&mut self) {
        if self.dirty_since.is_some() {
            self.dirty_since = Some(Instant::now() - DEBOUNCE - Duration::from_millis(10));
        }
    }

    pub fn set_view(&mut self, view: ViewMode) {
        self.view = view;
    }

    /// Mutable access to the tracing settings.
    ///
    /// Handing out the reference implies a change is coming, so this queues a
    /// re-trace. Anything else would let a caller change settings and silently
    /// keep looking at the old result.
    pub fn config_mut(&mut self) -> &mut VectorizeConfig {
        self.mark_dirty();
        &mut self.config
    }

    fn mark_dirty(&mut self) {
        self.dirty_since = Some(Instant::now());
    }

    fn maybe_trace(&mut self) {
        let Some(t) = self.dirty_since else { return };
        if t.elapsed() < DEBOUNCE {
            return;
        }
        let Some(src) = &self.source else {
            self.dirty_since = None;
            return;
        };
        self.dirty_since = None;
        let image = src.raster.clone();
        let config = Box::new(self.config.clone());
        let score = self.score_config();
        let generation = self.worker.submit(move |generation| Request::Trace {
            generation,
            image,
            config,
            score,
        });
        self.awaiting = Some(generation);
    }

    fn start_auto(&mut self) {
        let Some(src) = &self.source else { return };
        let mut cfg = AutoConfig::default();
        cfg.score = self.score_config();
        // Which colours to leave out is a requirement of the result, not
        // something for the search to vary.
        cfg.transparent_colors = self.config.segment.transparent_colors.clone();
        cfg.transparent_tolerance = self.config.segment.transparent_tolerance;
        let image = src.raster.clone();
        let generation = self.worker.submit(move |generation| Request::Auto {
            generation,
            image,
            config: Box::new(cfg),
        });
        self.auto_generation = Some(generation);
        self.awaiting = Some(generation);
        self.status = "Searching for the best settings...".into();
    }

    fn pump(&mut self, ctx: &egui::Context) {
        while let Some(msg) = self.worker.try_recv() {
            match msg {
                Response::Traced {
                    generation,
                    outcome,
                } => {
                    // Discard results the user has already moved past. Only the
                    // generation currently awaited is wanted; anything else is
                    // a trace that was already superseded before it finished.
                    if self.awaiting != Some(generation) {
                        continue;
                    }
                    let diff = self
                        .source
                        .as_ref()
                        .map(|s| self.difference_image(&s.raster, &outcome));
                    let vector_tex = ctx.load_texture(
                        "vector",
                        to_color_image(&outcome.rendered),
                        TextureOptions::NEAREST,
                    );
                    let diff_tex = ctx.load_texture(
                        "diff",
                        to_color_image(diff.as_ref().unwrap_or(&outcome.rendered)),
                        TextureOptions::NEAREST,
                    );
                    let svg_bytes = export::svg::write(&outcome.vector, &outcome.config.output).len();
                    self.preview = Some(Preview {
                        outcome,
                        vector_tex,
                        diff_tex,
                        svg_bytes,
                    });
                    self.awaiting = None;
                    self.diff_dirty = false;
                }
                Response::AutoDone { generation, result } => {
                    if self.auto_generation == Some(generation) {
                        if let Some(best) = result.best() {
                            self.config = best.config.clone();
                            self.status = format!(
                                "Auto picked \"{}\": {:.2}% match, {} points",
                                best.label,
                                best.report.match_pct,
                                best.report.complexity.total_points()
                            );
                        }
                        self.auto_result = Some(result);
                        self.auto_generation = None;
                        self.awaiting = None;
                        self.mark_dirty();
                    }
                }
                Response::Failed {
                    generation,
                    message,
                } => {
                    if self.awaiting == Some(generation) {
                        self.awaiting = None;
                    }
                    self.error = Some(message);
                }
            }
        }
    }

    /// Rebuild the difference texture after a gain change.
    fn refresh_difference(&mut self, ctx: &egui::Context) {
        if !self.diff_dirty {
            return;
        }
        self.diff_dirty = false;
        let (Some(src), Some(preview)) = (&self.source, &self.preview) else {
            return;
        };
        let diff = self.difference_image(&src.raster, &preview.outcome);
        let tex = ctx.load_texture("diff", to_color_image(&diff), TextureOptions::NEAREST);
        if let Some(preview) = &mut self.preview {
            preview.diff_tex = tex;
        }
    }

    fn export(&mut self) {
        let Some(preview) = &self.preview else { return };
        let Some(src) = &self.source else { return };
        let stem = src
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "traced".into());
        let ext = self.export_format.extension();

        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{stem}.{ext}"))
            .add_filter(ext.to_uppercase(), &[ext])
            .save_file()
        else {
            return;
        };

        let mut out_cfg = self.config.output.clone();
        out_cfg.format = self.export_format;
        // Re-run rather than reusing the preview: the preview is always SVG,
        // and the other writers need the document built for their own format.
        let traced = vectify_core::vectorize::vectorize(&src.raster, &{
            let mut c = self.config.clone();
            c.output = out_cfg.clone();
            c
        });
        let bytes = export::export(&traced.image, &out_cfg);
        match std::fs::write(&path, &bytes) {
            Ok(()) => {
                self.status = format!(
                    "Wrote {} ({:.1} KB)",
                    path.display(),
                    bytes.len() as f64 / 1024.0
                )
            }
            Err(e) => self.error = Some(format!("writing {}: {e}", path.display())),
        }
        let _ = preview;
    }
}

impl eframe::App for VectifyApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

impl VectifyApp {
    /// The whole UI. Kept separate from the `eframe::App` method so tests can
    /// call it without constructing an `eframe::Frame`.
    pub fn draw(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        self.pump(&ctx);
        self.refresh_difference(&ctx);
        self.maybe_trace();
        if self.dirty_since.is_some() || self.worker.is_busy() {
            ctx.request_repaint_after(Duration::from_millis(60));
        }

        // Drag and drop.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        if let Some(p) = dropped.into_iter().next() {
            self.open(&ctx, p);
        }

        self.top_bar(ui);
        self.side_panel(ui);
        self.status_bar(ui);
        self.central(ui);
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Vectify");
                ui.separator();
                if ui
                    .button("Open image...")
                    .on_hover_text("Load a raster image to trace. You can also drag and drop a file onto the window.")
                    .clicked()
                {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter(
                            "Images",
                            &[
                                "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp",
                                "tga", "ico", "ppm", "pgm", "pnm", "qoi", "dds", "ff", "exr",
                                "hdr",
                            ],
                        )
                        .add_filter("All files", &["*"])
                        .pick_file()
                    {
                        let ctx = ui.ctx().clone();
                        self.open(&ctx, p);
                    }
                }

                let enabled = self.source.is_some();
                ui.add_enabled_ui(enabled, |ui| {
                    if ui
                        .button(RichText::new("Auto-select algorithm").strong())
                        .on_hover_text(
                            "Trace this image with many different settings, score each by \
                             rendering it back to pixels, and keep the one that hits the \
                             fidelity target with the fewest control points.",
                        )
                        .clicked()
                    {
                        self.start_auto();
                    }

                    ui.separator();
                    egui::ComboBox::from_id_salt("preset")
                        .selected_text(self.preset.name())
                        .show_ui(ui, |ui| {
                            for p in Preset::all() {
                                if ui
                                    .selectable_label(self.preset == *p, p.name())
                                    .clicked()
                                {
                                    self.preset = *p;
                                    // A preset is a starting point for how to
                                    // trace, not for what to leave out or where
                                    // to save, so those survive the switch.
                                    let format = self.config.output.format;
                                    let keyed = std::mem::take(&mut self.config.segment.transparent_colors);
                                    let tolerance = self.config.segment.transparent_tolerance;
                                    self.config = p.config();
                                    self.config.output.format = format;
                                    self.config.segment.transparent_colors = keyed;
                                    self.config.segment.transparent_tolerance = tolerance;
                                    self.mark_dirty();
                                }
                            }
                        })
                        .response
                        .on_hover_text(
                            "A starting point for every setting below, tuned for a kind of \
                             artwork. The auto-picker seeds its search from these too.",
                        );

                    ui.separator();
                    egui::ComboBox::from_id_salt("format")
                        .selected_text(self.export_format.extension().to_uppercase())
                        .show_ui(ui, |ui| {
                            for f in VectorFormat::all() {
                                if ui
                                    .selectable_label(
                                        self.export_format == *f,
                                        f.extension().to_uppercase(),
                                    )
                                    .clicked()
                                {
                                    self.export_format = *f;
                                }
                            }
                        })
                        .response
                        .on_hover_text("File format used by \"Export...\".");
                    if ui
                        .add_enabled(self.preview.is_some(), egui::Button::new("Export..."))
                        .on_hover_text(
                            "Save the current trace to disk in the format selected above.",
                        )
                        .clicked()
                    {
                        self.export();
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if self.auto_generation.is_some() {
                    let done = self.worker.progress.0.load(Ordering::Relaxed);
                    let total = self.worker.progress.1.load(Ordering::Relaxed).max(1);
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total as f32)
                            .desired_width(220.0)
                            .text(format!("{done}/{total} candidates")),
                    );
                } else if self.worker.is_busy() || self.dirty_since.is_some() {
                    ui.spinner();
                    ui.label("Tracing...");
                }
                if let Some(e) = &self.error {
                    ui.colored_label(Color32::from_rgb(255, 120, 110), e);
                } else {
                    ui.label(&self.status);
                }
            });
        });
    }

    fn central(&mut self, ui: &mut Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                for m in ViewMode::ALL {
                    let hover = match m {
                        ViewMode::Original => "Show only the source image.",
                        ViewMode::Vector => "Show only the traced result, rendered back to pixels.",
                        ViewMode::SideBySide => "Show the original and the trace next to each other.",
                        ViewMode::Wipe => {
                            "Overlay the original and the trace with a draggable divider, \
                             for spotting sub-pixel edge shifts that side-by-side hides."
                        }
                        ViewMode::Difference => {
                            "Highlight pixels where the trace's colour differs from the \
                             original, amplified by the gain slider."
                        }
                    };
                    if ui
                        .selectable_label(self.view == m, m.label())
                        .on_hover_text(hover)
                        .clicked()
                    {
                        self.view = m;
                    }
                }
                ui.separator();
                if ui
                    .button("Fit")
                    .on_hover_text("Zoom and centre the image to fill the canvas.")
                    .clicked()
                {
                    self.camera.fitted = false;
                }
                if ui
                    .button("1:1")
                    .on_hover_text("Reset zoom to 100%, one screen pixel per image pixel.")
                    .clicked()
                {
                    self.camera.zoom = 1.0;
                    self.camera.pan = egui::Vec2::ZERO;
                    self.camera.fitted = true;
                }
                ui.label(format!("{:.0}%", self.camera.zoom * 100.0));
                ui.checkbox(&mut self.show_grid, "Pixel grid")
                    .on_hover_text("Draw pixel boundaries once zoomed in far enough to see them.");
                if self.view == ViewMode::Wipe {
                    ui.add(egui::Slider::new(&mut self.wipe, 0.0..=1.0).text("wipe"))
                        .on_hover_text(
                            "Position of the divider: 0 shows all original, 1 shows all trace.",
                        );
                }
                if self.view == ViewMode::Difference
                    && ui
                        .add(egui::Slider::new(&mut self.diff_gain, 1.0..=32.0).text("gain"))
                        .on_hover_text(
                            "Amplifies small colour differences so they are visible. Display \
                             only; it does not affect the match score.",
                        )
                        .changed()
                {
                    // Display-only: rebuild the difference texture, but do not
                    // re-trace, which would be a long wait for a view control.
                    self.diff_dirty = true;
                }
            });
            ui.separator();

            let Some(src) = &self.source else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new("Open an image, or drop one here.")
                            .size(18.0)
                            .color(Color32::from_gray(150)),
                    );
                });
                return;
            };

            let canvas = Canvas {
                original: Some(&src.texture),
                vector: self.preview.as_ref().map(|p| &p.vector_tex),
                difference: self.preview.as_ref().map(|p| &p.diff_tex),
                image_size: egui::vec2(src.raster.width as f32, src.raster.height as f32),
                mode: self.view,
                wipe: self.wipe,
                show_grid: self.show_grid,
            };
            let hovered = canvas.show(ui, &mut self.camera);
            if let Some((x, y)) = hovered {
                if x >= 0.0
                    && y >= 0.0
                    && (x as u32) < src.raster.width
                    && (y as u32) < src.raster.height
                {
                    let c = src.raster.pixel_rgba8(x as u32, y as u32);
                    ui.label(format!(
                        "({:.0}, {:.0})  {}  alpha {}",
                        x.floor(),
                        y.floor(),
                        c.to_hex(),
                        c.a
                    ));
                }
            }
        });
    }

    fn side_panel(&mut self, ui: &mut Ui) {
        egui::Panel::right("side").default_size(330.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.metrics_section(ui);
                ui.add_space(6.0);
                self.settings_section(ui);
                ui.add_space(6.0);
                self.candidates_section(ui);
            });
        });
    }

    fn metrics_section(&mut self, ui: &mut Ui) {
        ui.heading("Result");
        let Some(p) = &self.preview else {
            ui.label(RichText::new("No trace yet.").color(Color32::from_gray(140)));
            return;
        };
        let r = &p.outcome.report;
        let c = r.complexity;

        let meets = r.match_pct >= self.target_match;
        let colour = if meets {
            Color32::from_rgb(120, 210, 130)
        } else {
            Color32::from_rgb(230, 190, 90)
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{:.2}%", r.match_pct)).size(26.0).color(colour));
            ui.vertical(|ui| {
                ui.label("of pixels match");
                ui.label(
                    RichText::new(format!("within deltaE {:.1}", self.delta_e))
                        .small()
                        .color(Color32::from_gray(150)),
                );
                if r.reference_denoised {
                    ui.label(
                        RichText::new("vs smoothed original")
                            .small()
                            .color(Color32::from_gray(150)),
                    );
                }
            });
        });

        egui::Grid::new("metrics").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("Control points");
            ui.label(RichText::new(c.total_points().to_string()).strong());
            ui.end_row();
            ui.label("  anchors / handles");
            ui.label(format!("{} / {}", c.anchors, c.handles));
            ui.end_row();
            ui.label("Segments");
            ui.label(format!(
                "{} ({} lines, {} curves, {} arcs)",
                c.segments, c.lines, c.cubics, c.arcs
            ));
            ui.end_row();
            ui.label("Shapes");
            ui.label(c.shapes.to_string());
            ui.end_row();
            ui.label("SVG size");
            ui.label(format!("{:.1} KB", p.svg_bytes as f64 / 1024.0));
            ui.end_row();
            ui.label("deltaE mean / p95 / max");
            ui.label(format!(
                "{:.2} / {:.2} / {:.2}",
                r.mean_delta_e, r.p95_delta_e, r.max_delta_e
            ));
            ui.end_row();
            ui.label("RMSE / PSNR");
            ui.label(format!("{:.2} / {:.1} dB", r.rmse, r.psnr));
            ui.end_row();
            ui.label("SSIM");
            ui.label(format!("{:.4}", r.ssim));
            ui.end_row();
            ui.label("Trace time");
            ui.label(format!("{:.0} ms", p.outcome.stats.total_ms()));
            ui.end_row();
        });

        ui.add_space(4.0);
        ui.label(RichText::new("Palette").small());
        ui.horizontal_wrapped(|ui| {
            let mut seen: Vec<vectify_core::color::Rgba8> = Vec::new();
            for s in &p.outcome.vector.shapes {
                if !seen.contains(&s.color) {
                    seen.push(s.color);
                }
            }
            for c in seen.iter().take(64) {
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                ui.painter().rect_filled(
                    rect,
                    2.0,
                    Color32::from_rgb(c.r, c.g, c.b),
                );
                response.on_hover_text(c.to_hex());
            }
        });
    }

    /// The colour a new "transparent colour" entry starts as: the image's
    /// top-left pixel, since a background is the usual thing to leave out and
    /// the corner is almost always background. White when there is no image, or
    /// when the corner is itself transparent and so carries no real colour.
    fn suggested_transparent_color(&self) -> Rgba8 {
        let corner = self
            .source
            .as_ref()
            .filter(|s| s.raster.width > 0 && s.raster.height > 0)
            .map(|s| s.raster.pixel_rgba8(0, 0))
            .filter(|c| c.a >= 128);
        match corner {
            Some(c) => Rgba8::opaque(c.r, c.g, c.b),
            None => Rgba8::opaque(255, 255, 255),
        }
    }

    /// Controls for colours to leave unpainted. Returns whether anything changed.
    fn transparent_colors_ui(&mut self, ui: &mut Ui) -> bool {
        let mut changed = false;
        ui.add_space(4.0);
        ui.label("Transparent colours").on_hover_text(
            "Regions in these colours get no shape at all: the artwork around them is \
             cut out of the page, leaving a real hole rather than a shape painted in \
             that colour. Useful for removing a background.",
        );

        let mut remove = None;
        for (i, color) in self.config.segment.transparent_colors.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let mut rgb = [color.r, color.g, color.b];
                if ui.color_edit_button_srgb(&mut rgb).changed() {
                    *color = Rgba8::opaque(rgb[0], rgb[1], rgb[2]);
                    changed = true;
                }
                ui.monospace(color.to_hex());
                if ui
                    .small_button("Remove")
                    .on_hover_text("Paint this colour again.")
                    .clicked()
                {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.config.segment.transparent_colors.remove(i);
            changed = true;
        }

        if ui
            .button("Add colour")
            .on_hover_text(
                "Leave another colour unpainted. It starts as the image's top-left pixel, \
                 which is usually the background; click the swatch to change it.",
            )
            .clicked()
        {
            let color = self.suggested_transparent_color();
            self.config.segment.transparent_colors.push(color);
            changed = true;
        }

        if !self.config.segment.transparent_colors.is_empty() {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.config.segment.transparent_tolerance, 0.0..=40.0)
                        .text("tolerance"),
                )
                .on_hover_text(
                    "How far a colour in the image may be from one of these and still count \
                     as it (CIELAB distance). Raise it for a noisy or off-white background; \
                     lower it if it starts swallowing light colours you want kept.",
                )
                .changed();
        }
        changed
    }

    fn settings_section(&mut self, ui: &mut Ui) {
        ui.separator();
        ui.heading("Settings");
        let mut changed = false;

        egui::CollapsingHeader::new("Segmentation")
            .default_open(true)
            .show(ui, |ui| {
                let mut binary = self.config.segment.mode == SegmentMode::Binary;
                if ui
                    .checkbox(&mut binary, "Two-tone (black and white)")
                    .on_hover_text(
                        "Split into just two regions at a luminance threshold, instead of a \
                         colour palette. Best for line art, scans and silhouettes.",
                    )
                    .changed()
                {
                    self.config.segment.mode = if binary {
                        SegmentMode::Binary
                    } else {
                        SegmentMode::Palette
                    };
                    changed = true;
                }
                if binary {
                    let mut auto_t = self.config.segment.threshold.is_none();
                    if ui
                        .checkbox(&mut auto_t, "Automatic threshold (Otsu)")
                        .on_hover_text(
                            "Pick the luminance split automatically from the image histogram, \
                             instead of setting it by hand.",
                        )
                        .changed()
                    {
                        self.config.segment.threshold = if auto_t { None } else { Some(0.5) };
                        changed = true;
                    }
                    if let Some(t) = &mut self.config.segment.threshold {
                        changed |= ui
                            .add(egui::Slider::new(t, 0.0..=1.0).text("threshold"))
                            .on_hover_text(
                                "Luminance split point, in 0..1. Pixels above become one \
                                 region, pixels below the other.",
                            )
                            .changed();
                    }
                    changed |= ui
                        .checkbox(&mut self.config.segment.invert, "Invert")
                        .on_hover_text("Swap which side of the threshold is treated as ink.")
                        .changed();
                } else {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.segment.colors, 2..=128)
                                .text("colours"),
                        )
                        .on_hover_text("Target number of colours in the palette.")
                        .changed();
                    let mut kmeans =
                        self.config.segment.palette_method == PaletteMethod::KMeans;
                    if ui
                        .checkbox(&mut kmeans, "k-means palette (else median cut)")
                        .on_hover_text(
                            "K-means: weighted clustering in CIELAB, slower but follows the \
                             real colour distribution and is the better choice for most \
                             artwork. Median cut: fast and deterministic, and tends to \
                             preserve rarely-used accent colours that k-means would merge away.",
                        )
                        .changed()
                    {
                        self.config.segment.palette_method = if kmeans {
                            PaletteMethod::KMeans
                        } else {
                            PaletteMethod::MedianCut
                        };
                        changed = true;
                    }
                }
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.segment.despeckle_area, 0..=64)
                            .text("despeckle (px)"),
                    )
                    .on_hover_text("Absorb regions smaller than this into their neighbours.")
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut self.config.segment.ignore_edge_pixels,
                        "Ignore edge pixels when choosing colours",
                    )
                    .on_hover_text(
                        "Anti-aliased edge pixels are blends of two real colours. Letting \
                         them vote creates phantom palette entries that show up as halos.",
                    )
                    .changed();
                changed |= self.transparent_colors_ui(ui);
            });

        egui::CollapsingHeader::new("Sub-pixel edges")
            .default_open(true)
            .show(ui, |ui| {
                changed |= ui
                    .checkbox(
                        &mut self.config.subpixel.enabled,
                        "Reconstruct edges from anti-aliasing",
                    )
                    .on_hover_text(
                        "Read the brightness of blended edge pixels to work out where the \
                         original boundary fell between pixels, instead of snapping to the \
                         pixel grid.",
                    )
                    .changed();
                ui.add_enabled_ui(self.config.subpixel.enabled, |ui| {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.subpixel.max_shift, 0.1..=1.5)
                                .text("max shift (px)"),
                        )
                        .on_hover_text(
                            "Hard cap on how far a vertex may move off the pixel grid. Keeps \
                             a bad local solve from tearing the contour.",
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.subpixel.iterations, 1..=6)
                                .text("solve rounds"),
                        )
                        .on_hover_text(
                            "Each round re-measures edge orientation from the previous \
                             round's straighter contour. More rounds recover shallow, \
                             low-angle edges more precisely; two already gets most of the \
                             way there.",
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.config.subpixel.smooth_passes, 0..=4)
                                .text("smoothing passes"),
                        )
                        .on_hover_text(
                            "Smooths the per-vertex displacement estimates along each \
                             contour, removing solve jitter without moving the edge.",
                        )
                        .changed();
                });
            });

        egui::CollapsingHeader::new("Corners and noise")
            .default_open(true)
            .show(ui, |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.corners.threshold_deg, 10.0..=150.0)
                            .text("corner angle"),
                    )
                    .on_hover_text("Bends sharper than this may be treated as hard corners.")
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.corners.persistence, 0.0..=1.0)
                            .text("scale agreement"),
                    )
                    .on_hover_text(
                        "How many measurement scales must agree before a bend counts as a \
                         corner. This is what separates real vertices from pixel \
                         stair-stepping and compression noise.",
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.corners.smoothing, 0.0..=1.0)
                            .text("smoothing"),
                    )
                    .on_hover_text(
                        "Overall smoothing strength, applied between corners only so \
                         intentional vertices stay sharp.",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut self.config.corners.noise_adaptive,
                        "Smooth only where it is noisy",
                    )
                    .on_hover_text(
                        "Measure noise per contour and smooth in proportion, instead of \
                         applying one filter to the whole canvas.",
                    )
                    .changed();
            });

        egui::CollapsingHeader::new("Curve fitting")
            .default_open(true)
            .show(ui, |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.fit.tolerance, 0.02..=3.0)
                            .logarithmic(true)
                            .text("tolerance (px)"),
                    )
                    .on_hover_text("Lower is more faithful and costs more control points.")
                    .changed();
                changed |= ui
                    .checkbox(&mut self.config.fit.allow_arcs, "Fit circular arcs")
                    .on_hover_text(
                        "Try circular arcs before falling back to Bezier curves, wherever a \
                         contour segment is well described by one.",
                    )
                    .changed();
                ui.add_enabled_ui(self.config.fit.allow_arcs, |ui| {
                    changed |= ui
                        .checkbox(
                            &mut self.config.fit.emit_true_arcs,
                            "Emit true arc commands",
                        )
                        .on_hover_text(
                            "More compact and exact, but a few consumers handle SVG arcs \
                             poorly.",
                        )
                        .changed();
                });
                changed |= ui
                    .checkbox(&mut self.config.fit.merge_pass, "Drop unnecessary corners")
                    .on_hover_text(
                        "After fitting, try replacing adjacent segment pairs with a single \
                         segment wherever tolerance still allows, to save control points.",
                    )
                    .changed();
            });

        egui::CollapsingHeader::new("Output")
            .default_open(false)
            .show(ui, |ui| {
                changed |= ui
                    .checkbox(
                        &mut self.config.output.group_by_color,
                        "One path per colour",
                    )
                    .on_hover_text(
                        "Merge all regions sharing a palette colour into a single path \
                         element, instead of emitting one path per region.",
                    )
                    .changed();
                changed |= ui
                    .checkbox(
                        &mut self.config.output.recolor_regions,
                        "Colour each region individually",
                    )
                    .on_hover_text(
                        "Higher fidelity on shaded artwork, at the cost of more distinct \
                         colours than the palette asked for.",
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.config.output.precision, 0..=6)
                            .text("decimal places"),
                    )
                    .on_hover_text(
                        "Decimal places in emitted coordinates. Three is plenty at pixel \
                         scale and keeps files small; shared borders are rounded identically \
                         on both sides so they stay welded.",
                    )
                    .changed();
            });

        egui::CollapsingHeader::new("Scoring")
            .default_open(false)
            .show(ui, |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.target_match, 90.0..=100.0)
                            .text("target match %"),
                    )
                    .on_hover_text(
                        "Match percentage a result must reach to count as acceptable. Drives \
                         the green/amber colouring of the match score, and the \"ok\" column \
                         and pick in Auto-select.",
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.delta_e, 0.5..=10.0)
                            .text("deltaE threshold"),
                    )
                    .on_hover_text(
                        "A pixel counts as matching when its colour difference from the \
                         original is below this. Around 1.0 is the threshold of human \
                         perceptibility.",
                    )
                    .changed();
                changed |= ui
                    .checkbox(&mut self.denoise_reference, "Smooth original before scoring")
                    .on_hover_text(
                        "Score against an edge-preserving (bilateral) smoothing of the \
                         original, so JPEG artefacts and noise that a flat region rightly \
                         ignores do not count as error. Edges are kept, unlike a plain \
                         blur. Only the score and difference view change: the tracer still \
                         sees the original. Scores measured this way run higher and are \
                         not comparable with ones that were not. Costs a few seconds on \
                         a large image.",
                    )
                    .changed();
                if self.denoise_reference {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.denoise.range_sigma, 1.0..=15.0)
                                .text("smoothing strength"),
                        )
                        .on_hover_text(
                            "Colour difference below which neighbouring pixels are averaged \
                             together. Higher removes stronger noise, but softens \
                             low-contrast edges in the reference.",
                        )
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut self.denoise.radius, 1..=6).text("radius"))
                        .on_hover_text("Smoothing window radius in pixels.")
                        .changed();
                }
                ui.label(
                    RichText::new(
                        "Note: even a perfect vectorisation scores around 99.5% here, \
                         because the renderer scoring it anti-aliases slightly differently \
                         from whatever produced the original.",
                    )
                    .small()
                    .color(Color32::from_gray(140)),
                );
            });

        if changed {
            self.mark_dirty();
        }
    }

    fn candidates_section(&mut self, ui: &mut Ui) {
        let Some(result) = &self.auto_result else { return };
        ui.separator();
        ui.heading("Candidates tried");
        ui.label(
            RichText::new(format!(
                "{} evaluated in {:.1} s. Click one to apply it.",
                result.candidates.len(),
                result.elapsed_ms / 1000.0
            ))
            .small()
            .color(Color32::from_gray(150)),
        );

        let mut apply: Option<VectorizeConfig> = None;
        egui::Grid::new("cands")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                ui.label(RichText::new("setting").strong());
                ui.label(RichText::new("match").strong());
                ui.label(RichText::new("points").strong());
                ui.label(RichText::new("ok").strong());
                ui.end_row();
                for c in result.candidates.iter().take(30) {
                    if ui
                        .selectable_label(false, &c.label)
                        .on_hover_text("Click to apply this candidate's settings.")
                        .clicked()
                    {
                        apply = Some(c.config.clone());
                    }
                    ui.label(format!("{:.2}%", c.report.match_pct));
                    ui.label(c.report.complexity.total_points().to_string());
                    let meets = c.report.meets(self.target_match);
                    ui.colored_label(
                        if meets {
                            Color32::from_rgb(120, 210, 130)
                        } else {
                            Color32::from_gray(130)
                        },
                        if meets { "yes" } else { "-" },
                    );
                    ui.end_row();
                }
            });
        if let Some(cfg) = apply {
            self.config = cfg;
            self.mark_dirty();
        }
    }
}
