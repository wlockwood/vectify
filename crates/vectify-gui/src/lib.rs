//! Vectify desktop application.
//!
//! Exposed as a library as well as a binary so that the UI can be driven
//! headlessly by tests.
//!
//! NOTE: 100% vibe coded as a one-off tool; not a maintained product.

pub mod app;
pub mod viewer;
pub mod worker;

/// The 256x256 application icon, shared with the repository's other brand assets
/// in `assets/` so there is one copy of the artwork rather than one per crate.
const ICON_PNG: &[u8] = include_bytes!("../../../assets/icon.png");

/// The window and taskbar icon.
///
/// The PNG is embedded at compile time, so a failure to decode it is a broken
/// build artefact rather than a runtime condition; it is checked by a test, and
/// this panics rather than quietly launching with no icon.
pub fn app_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(ICON_PNG).expect("assets/icon.png is a valid PNG")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_icon_decodes_to_a_square_with_a_transparent_corner() {
        let icon = app_icon();
        assert_eq!((icon.width, icon.height), (256, 256), "window icons must be square");
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        let alpha = |x: usize, y: usize| icon.rgba[(y * 256 + x) * 4 + 3];
        assert_eq!(alpha(0, 0), 0, "the corner should be transparent, not a white box");
        assert!(alpha(128, 128) > 250, "the centre of the mark should be opaque");
    }
}
