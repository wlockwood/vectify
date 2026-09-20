//! The image canvas: zoom, pan, and the comparison modes.
//!
//! Judging a tracer by eye means comparing two images pixel for pixel at high
//! zoom, so the viewer is built around that: nearest-neighbour sampling so
//! pixels stay square and countable, a shared transform across both images so
//! they never drift apart, and a wipe mode for spotting sub-pixel edge shifts
//! that a side-by-side view hides.

use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, Ui, Vec2};

use vectify_core::raster::Raster;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    Original,
    Vector,
    SideBySide,
    Wipe,
    Difference,
}

impl ViewMode {
    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Original => "Original",
            ViewMode::Vector => "Vector",
            ViewMode::SideBySide => "Side by side",
            ViewMode::Wipe => "Wipe",
            ViewMode::Difference => "Difference",
        }
    }

    pub const ALL: [ViewMode; 5] = [
        ViewMode::Original,
        ViewMode::Vector,
        ViewMode::SideBySide,
        ViewMode::Wipe,
        ViewMode::Difference,
    ];
}

pub struct Camera {
    /// Screen pixels per image pixel. `None` until the first fit.
    pub zoom: f32,
    pub pan: Vec2,
    pub fitted: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            zoom: 1.0,
            pan: Vec2::ZERO,
            fitted: false,
        }
    }
}

impl Camera {
    pub fn fit(&mut self, viewport: Vec2, image: Vec2) {
        if image.x <= 0.0 || image.y <= 0.0 {
            return;
        }
        let scale = (viewport.x / image.x).min(viewport.y / image.y) * 0.95;
        self.zoom = scale.clamp(0.02, 64.0);
        self.pan = Vec2::ZERO;
        self.fitted = true;
    }
}

/// Convert a raster to a texture-ready image.
pub fn to_color_image(r: &Raster) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied(
        [r.width as usize, r.height as usize],
        &r.to_rgba8_bytes(),
    )
}

/// A checkerboard behind the artwork, so transparent regions read as
/// transparent rather than as white.
fn paint_checker(ui: &Ui, rect: Rect) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(60));
    let size = 8.0f32;
    let mut y = rect.top();
    let mut row = 0;
    while y < rect.bottom() {
        let mut x = rect.left();
        let mut col = row;
        while x < rect.right() {
            if col % 2 == 0 {
                let cell = Rect::from_min_size(
                    Pos2::new(x, y),
                    Vec2::new(size.min(rect.right() - x), size.min(rect.bottom() - y)),
                );
                painter.rect_filled(cell, 0.0, Color32::from_gray(80));
            }
            x += size;
            col += 1;
        }
        y += size;
        row += 1;
    }
}

pub struct Canvas<'a> {
    pub original: Option<&'a TextureHandle>,
    pub vector: Option<&'a TextureHandle>,
    pub difference: Option<&'a TextureHandle>,
    pub image_size: Vec2,
    pub mode: ViewMode,
    pub wipe: f32,
    pub show_grid: bool,
}

impl Canvas<'_> {
    /// Draw the canvas, handling pan, zoom and the wipe handle. Returns the
    /// image-space coordinate under the pointer, if any, for the readout.
    pub fn show(&self, ui: &mut Ui, cam: &mut Camera) -> Option<(f32, f32)> {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, Sense::click_and_drag());

        if !cam.fitted {
            cam.fit(available, self.image_size);
        }

        // Zoom about the pointer so the thing under the cursor stays put, which
        // is the only zoom behaviour that feels right when inspecting detail.
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let old = cam.zoom;
                let factor = (scroll * 0.004).exp();
                cam.zoom = (cam.zoom * factor).clamp(0.02, 64.0);
                if let Some(pointer) = response.hover_pos() {
                    let centre = rect.center() + cam.pan;
                    let to_pointer = pointer - centre;
                    cam.pan += to_pointer * (1.0 - cam.zoom / old);
                }
            }
        }
        if response.dragged() {
            cam.pan += response.drag_delta();
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(32));

        let draw_size = self.image_size * cam.zoom;
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

        match self.mode {
            ViewMode::SideBySide => {
                // Two panes sharing one transform: the same pan and zoom apply
                // to both, so corresponding pixels stay aligned.
                let half = Rect::from_min_max(rect.min, Pos2::new(rect.center().x - 1.0, rect.max.y));
                let other = Rect::from_min_max(Pos2::new(rect.center().x + 1.0, rect.min.y), rect.max);
                for (pane, tex) in [(half, self.original), (other, self.vector)] {
                    let target = Rect::from_center_size(pane.center() + cam.pan, draw_size);
                    let clipped = ui.painter_at(pane);
                    clipped.rect_filled(pane, 0.0, Color32::from_gray(32));
                    paint_checker(ui, pane.intersect(target));
                    if let Some(t) = tex {
                        clipped.image(t.id(), target, uv, Color32::WHITE);
                    }
                }
                painter.line_segment(
                    [
                        Pos2::new(rect.center().x, rect.top()),
                        Pos2::new(rect.center().x, rect.bottom()),
                    ],
                    Stroke::new(1.0, Color32::from_gray(110)),
                );
            }
            ViewMode::Wipe => {
                let target = Rect::from_center_size(rect.center() + cam.pan, draw_size);
                paint_checker(ui, rect.intersect(target));
                let split = rect.left() + rect.width() * self.wipe.clamp(0.0, 1.0);
                if let Some(t) = self.original {
                    let left = Rect::from_min_max(rect.min, Pos2::new(split, rect.max.y));
                    ui.painter_at(left).image(t.id(), target, uv, Color32::WHITE);
                }
                if let Some(t) = self.vector {
                    let right = Rect::from_min_max(Pos2::new(split, rect.min.y), rect.max);
                    ui.painter_at(right).image(t.id(), target, uv, Color32::WHITE);
                }
                painter.line_segment(
                    [Pos2::new(split, rect.top()), Pos2::new(split, rect.bottom())],
                    Stroke::new(2.0, Color32::from_rgb(255, 190, 60)),
                );
            }
            mode => {
                let tex = match mode {
                    ViewMode::Original => self.original,
                    ViewMode::Vector => self.vector,
                    ViewMode::Difference => self.difference,
                    _ => None,
                };
                let target = Rect::from_center_size(rect.center() + cam.pan, draw_size);
                if mode != ViewMode::Difference {
                    paint_checker(ui, rect.intersect(target));
                }
                if let Some(t) = tex {
                    painter.image(t.id(), target, uv, Color32::WHITE);
                }
            }
        }

        // A pixel grid, but only once pixels are big enough for it to clarify
        // rather than clutter.
        if self.show_grid && cam.zoom >= 8.0 {
            let target = Rect::from_center_size(rect.center() + cam.pan, draw_size);
            // Faint enough to read as a grid rather than as texture on top of
            // flat colour, and fading in with zoom so it never dominates.
            let alpha = (((cam.zoom - 8.0) / 8.0).clamp(0.0, 1.0) * 22.0 + 10.0) as u8;
            let stroke = Stroke::new(0.5, Color32::from_white_alpha(alpha));
            let mut x = target.left();
            while x <= target.right() {
                if x >= rect.left() && x <= rect.right() {
                    painter.line_segment(
                        [
                            Pos2::new(x, target.top().max(rect.top())),
                            Pos2::new(x, target.bottom().min(rect.bottom())),
                        ],
                        stroke,
                    );
                }
                x += cam.zoom;
            }
            let mut y = target.top();
            while y <= target.bottom() {
                if y >= rect.top() && y <= rect.bottom() {
                    painter.line_segment(
                        [
                            Pos2::new(target.left().max(rect.left()), y),
                            Pos2::new(target.right().min(rect.right()), y),
                        ],
                        stroke,
                    );
                }
                y += cam.zoom;
            }
        }

        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::from_gray(70)),
            StrokeKind::Inside,
        );

        response.hover_pos().map(|p| {
            let target = Rect::from_center_size(rect.center() + cam.pan, draw_size);
            (
                (p.x - target.left()) / cam.zoom,
                (p.y - target.top()) / cam.zoom,
            )
        })
    }
}
