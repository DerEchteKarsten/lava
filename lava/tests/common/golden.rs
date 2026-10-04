//! Golden-image comparison: rendered RGBA8 pixels against PNGs in `lava/tests/golden`
//!
//! Run with `LAVA_BLESS=1` to (re)write the expected images, then review them by eye.
use std::{fs::File, io::BufWriter, path::PathBuf};

/// A channel may differ by this much (of 255) before a pixel counts as different.
const CHANNEL_TOLERANCE: u8 = 2;
/// Fraction of pixels that may differ; absorbs rasterisation differences between GPU vendors.
const MAX_DIFFERENT_PIXELS: f64 = 0.001;

fn golden_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden"))
}

fn failure_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-failures")
}

fn write_png(path: &PathBuf, pixels: &[u8], [width, height]: [u32; 2]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(pixels)
        .unwrap();
}

fn read_png(path: &PathBuf) -> Option<(Vec<u8>, [u32; 2])> {
    let decoder = png::Decoder::new(std::io::BufReader::new(File::open(path).ok()?));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(
        (info.color_type, info.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight),
        "{} is not an 8-bit RGBA image",
        path.display()
    );
    pixels.truncate(info.buffer_size());
    Some((pixels, [info.width, info.height]))
}

/// Result of comparing two RGBA8 images of the same size.
pub struct Difference {
    /// Number of pixels with a channel off by more than the tolerance.
    pub different_pixels: usize,
    /// Largest difference of any channel.
    pub max_delta: u8,
    /// Differing pixels in red on black, for looking at.
    pub image: Vec<u8>,
}

pub fn compare(actual: &[u8], expected: &[u8]) -> Difference {
    assert_eq!(actual.len(), expected.len());
    let mut difference = Difference {
        different_pixels: 0,
        max_delta: 0,
        image: Vec::with_capacity(actual.len()),
    };
    for (a, e) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let delta = a.iter().zip(e).map(|(a, e)| a.abs_diff(*e)).max().unwrap();
        difference.max_delta = difference.max_delta.max(delta);
        let differs = delta > CHANNEL_TOLERANCE;
        difference.different_pixels += differs as usize;
        difference.image.extend_from_slice(if differs {
            &[255, 0, 0, 255]
        } else {
            &[0, 0, 0, 255]
        });
    }
    difference
}

/// Whether a comparison counts as a match.
pub fn matches(difference: &Difference, pixel_count: usize) -> bool {
    difference.different_pixels as f64 <= pixel_count as f64 * MAX_DIFFERENT_PIXELS
}

/// Fails the test unless `pixels` (tightly packed RGBA8, `size` = [width, height]) match the
/// golden image `name`. On a mismatch the actual image and a diff are written next to the
/// test binaries for inspection.
#[track_caller]
pub fn assert_golden(name: &str, pixels: &[u8], size: [u32; 2]) {
    assert_eq!(
        pixels.len(),
        (size[0] * size[1] * 4) as usize,
        "pixel data does not match {size:?}"
    );
    let path = golden_dir().join(format!("{name}.png"));

    if std::env::var_os("LAVA_BLESS").is_some() {
        write_png(&path, pixels, size);
        eprintln!("blessed {}", path.display());
        return;
    }

    let Some((expected, expected_size)) = read_png(&path) else {
        panic!(
            "golden image {} does not exist; run the tests with LAVA_BLESS=1 to create it, then review it",
            path.display()
        );
    };

    let actual_path = failure_dir().join(format!("{name}.actual.png"));
    if expected_size != size {
        write_png(&actual_path, pixels, size);
        panic!(
            "golden `{name}` is {expected_size:?} but the test rendered {size:?}; actual image: {}",
            actual_path.display()
        );
    }

    let difference = compare(pixels, &expected);
    if !matches(&difference, pixels.len() / 4) {
        let diff_path = failure_dir().join(format!("{name}.diff.png"));
        write_png(&actual_path, pixels, size);
        write_png(&diff_path, &difference.image, size);
        panic!(
            "golden `{name}` does not match: {} of {} pixels differ (largest channel delta {})\n  expected: {}\n  actual:   {}\n  diff:     {}",
            difference.different_pixels,
            pixels.len() / 4,
            difference.max_delta,
            path.display(),
            actual_path.display(),
            diff_path.display()
        );
    }
}
