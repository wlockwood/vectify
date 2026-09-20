//! Raster image loading and sampling.
//!
//! Pixels are stored as gamma-encoded sRGB with straight (un-premultiplied)
//! alpha, in 0..1 floats. See `color.rs` for why gamma space is the right
//! working space for this engine.

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::color::Rgba8;
use crate::geom::Point;

#[derive(Clone)]
pub struct Raster {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width * height` entries of `[r, g, b, a]`.
    pub data: Vec<[f32; 4]>,
}

impl Raster {
    pub fn new(width: u32, height: u32) -> Self {
        Raster {
            width,
            height,
            data: vec![[0.0; 4]; (width as usize) * (height as usize)],
        }
    }

    /// Decode any raster format the `image` crate supports. Format is detected
    /// from content rather than extension, so a mislabelled file still loads.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let reader = image::ImageReader::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .with_guessed_format()
            .with_context(|| format!("detecting format of {}", path.display()))?;
        let img = reader
            .decode()
            .with_context(|| format!("decoding {}", path.display()))?;
        Ok(Self::from_image(&img))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes).context("decoding image from memory")?;
        Ok(Self::from_image(&img))
    }

    pub fn from_image(img: &image::DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data = rgba
            .pixels()
            .map(|p| {
                [
                    p.0[0] as f32 / 255.0,
                    p.0[1] as f32 / 255.0,
                    p.0[2] as f32 / 255.0,
                    p.0[3] as f32 / 255.0,
                ]
            })
            .collect();
        Raster { width, height, data }
    }

    pub fn from_rgba8(width: u32, height: u32, bytes: &[u8]) -> Result<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if bytes.len() < expected {
            bail!(
                "buffer too small: {} bytes for {}x{} image (need {})",
                bytes.len(),
                width,
                height,
                expected
            );
        }
        let data = bytes[..expected]
            .chunks_exact(4)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            })
            .collect();
        Ok(Raster { width, height, data })
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        self.data[self.idx(x, y)]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, c: [f32; 4]) {
        let i = self.idx(x, y);
        self.data[i] = c;
    }

    /// Fetch with edge clamping, taking signed coordinates.
    #[inline]
    pub fn get_clamped(&self, x: i64, y: i64) -> [f32; 4] {
        let x = x.clamp(0, self.width as i64 - 1) as u32;
        let y = y.clamp(0, self.height as i64 - 1) as u32;
        self.get(x, y)
    }

    pub fn pixel_count(&self) -> usize {
        self.data.len()
    }

    /// Bilinear sample in **lattice coordinates**, where pixel centres are at
    /// half-integers. This is the sampler the sub-pixel edge solver walks along
    /// its normals with, so getting the half-pixel offset right here is what
    /// keeps reconstructed edges from being biased by half a pixel.
    pub fn sample(&self, p: Point) -> [f32; 4] {
        let fx = p.x - 0.5;
        let fy = p.y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = (fx - x0) as f32;
        let ty = (fy - y0) as f32;
        let x0 = x0 as i64;
        let y0 = y0 as i64;

        let c00 = self.get_clamped(x0, y0);
        let c10 = self.get_clamped(x0 + 1, y0);
        let c01 = self.get_clamped(x0, y0 + 1);
        let c11 = self.get_clamped(x0 + 1, y0 + 1);

        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * tx;
            let bot = c01[i] + (c11[i] - c01[i]) * tx;
            out[i] = top + (bot - top) * ty;
        }
        out
    }

    pub fn has_alpha(&self) -> bool {
        self.data.iter().any(|c| c[3] < 0.999)
    }

    /// Composite over an opaque background colour, yielding a fully opaque
    /// image. Blending happens in gamma space to match how the alpha was
    /// almost certainly produced.
    pub fn composite_over(&self, bg: [f32; 3]) -> Raster {
        let data = self
            .data
            .iter()
            .map(|c| {
                let a = c[3];
                [
                    c[0] * a + bg[0] * (1.0 - a),
                    c[1] * a + bg[1] * (1.0 - a),
                    c[2] * a + bg[2] * (1.0 - a),
                    1.0,
                ]
            })
            .collect();
        Raster {
            width: self.width,
            height: self.height,
            data,
        }
    }

    pub fn to_rgba8_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for c in &self.data {
            for i in 0..4 {
                out.push((c[i].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        out
    }

    pub fn pixel_rgba8(&self, x: u32, y: u32) -> Rgba8 {
        Rgba8::from_f32(self.get(x, y))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let buf = image::RgbaImage::from_raw(self.width, self.height, self.to_rgba8_bytes())
            .context("building output buffer")?;
        buf.save(path)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Box-downsample by an integer factor. Used to turn a supersampled render
    /// into a comparison image without introducing resampling artefacts of its
    /// own.
    pub fn downsample(&self, factor: u32) -> Raster {
        if factor <= 1 {
            return self.clone();
        }
        let w = self.width / factor;
        let h = self.height / factor;
        let mut out = Raster::new(w.max(1), h.max(1));
        let inv = 1.0 / (factor * factor) as f32;
        for y in 0..out.height {
            for x in 0..out.width {
                let mut acc = [0.0f32; 4];
                for dy in 0..factor {
                    for dx in 0..factor {
                        let c = self.get(x * factor + dx, y * factor + dy);
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

    /// Area-average resample to an arbitrary size.
    ///
    /// Box filtering rather than bilinear: when shrinking, bilinear samples
    /// only four source pixels no matter how far it is shrinking, which aliases
    /// badly and would misrepresent fine detail to the auto-picker's search.
    pub fn resized(&self, w: u32, h: u32) -> Raster {
        let w = w.max(1);
        let h = h.max(1);
        if w == self.width && h == self.height {
            return self.clone();
        }
        let mut out = Raster::new(w, h);
        let sx = self.width as f64 / w as f64;
        let sy = self.height as f64 / h as f64;
        for y in 0..h {
            let y0 = (y as f64 * sy).floor() as u32;
            let y1 = (((y + 1) as f64 * sy).ceil() as u32).min(self.height).max(y0 + 1);
            for x in 0..w {
                let x0 = (x as f64 * sx).floor() as u32;
                let x1 = (((x + 1) as f64 * sx).ceil() as u32).min(self.width).max(x0 + 1);
                let mut acc = [0.0f32; 4];
                let mut n = 0.0f32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let c = self.get(xx, yy);
                        for i in 0..4 {
                            acc[i] += c[i];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    for v in acc.iter_mut() {
                        *v /= n;
                    }
                }
                out.set(x, y, acc);
            }
        }
        out
    }

    /// Shrink so neither side exceeds `max_dim`, preserving aspect ratio.
    /// Returns a clone untouched if it already fits.
    pub fn fit_within(&self, max_dim: u32) -> Raster {
        let longest = self.width.max(self.height);
        if longest <= max_dim || max_dim == 0 {
            return self.clone();
        }
        let scale = max_dim as f64 / longest as f64;
        self.resized(
            ((self.width as f64 * scale).round() as u32).max(1),
            ((self.height as f64 * scale).round() as u32).max(1),
        )
    }

    /// Per-pixel absolute difference, amplified, as an opaque visualisation.
    pub fn difference(&self, other: &Raster, gain: f32) -> Raster {
        let mut out = Raster::new(self.width, self.height);
        for y in 0..self.height.min(other.height) {
            for x in 0..self.width.min(other.width) {
                let a = self.get(x, y);
                let b = other.get(x, y);
                let d = [
                    ((a[0] - b[0]).abs() * gain).min(1.0),
                    ((a[1] - b[1]).abs() * gain).min(1.0),
                    ((a[2] - b[2]).abs() * gain).min(1.0),
                    1.0,
                ];
                out.set(x, y, d);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::pt;

    fn ramp() -> Raster {
        let mut r = Raster::new(4, 1);
        for x in 0..4 {
            let v = x as f32 / 3.0;
            r.set(x, 0, [v, v, v, 1.0]);
        }
        r
    }

    #[test]
    fn sampling_at_pixel_centre_is_exact() {
        let r = ramp();
        for x in 0..4 {
            let s = r.sample(pt(x as f64 + 0.5, 0.5));
            assert!((s[0] - x as f32 / 3.0).abs() < 1e-6, "x={x} got {:?}", s);
        }
    }

    #[test]
    fn sampling_midway_interpolates() {
        let r = ramp();
        // Lattice x=1.0 is the boundary between pixels 0 and 1.
        let s = r.sample(pt(1.0, 0.5));
        let want = (0.0 + 1.0 / 3.0) / 2.0;
        assert!((s[0] - want).abs() < 1e-6, "got {:?} want {want}", s);
    }

    #[test]
    fn composite_over_white() {
        let mut r = Raster::new(1, 1);
        r.set(0, 0, [0.0, 0.0, 0.0, 0.5]);
        let c = r.composite_over([1.0, 1.0, 1.0]);
        assert!((c.get(0, 0)[0] - 0.5).abs() < 1e-6);
        assert!((c.get(0, 0)[3] - 1.0).abs() < 1e-6);
    }
}
