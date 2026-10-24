//! Golden-image comparison: rendered RGBA8 pixels against PNGs in `lava-tests/tests/golden`
use std::{fs::File, io::BufWriter, path::PathBuf};

/// A channel may differ by this much (of 255) before a pixel counts as different.
const CHANNEL_TOLERANCE: u8 = 2;
/// Fraction of pixels that may differ; absorbs rasterisation differences between GPU vendors.
const MAX_DIFFERENT_PIXELS: f64 = 0.001;

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
    let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden"))
        .join(format!("{name}.png"));

    if std::env::var_os("LAVA_BLESS").is_some() {
        write_png(&path, pixels, size);
        eprintln!("blessed {}", path.display());
        return;
    }

    let Ok(file) = File::open(&path) else {
        panic!(
            "golden image {} does not exist; run the tests with LAVA_BLESS=1 to create it, then review it",
            path.display()
        );
    };
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .unwrap();
    let mut expected = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut expected).unwrap();
    assert_eq!(
        (info.color_type, info.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight),
        "{} is not an 8-bit RGBA image",
        path.display()
    );
    expected.truncate(info.buffer_size());
    let expected_size = [info.width, info.height];

    let actual_path = failure_dir().join(format!("{name}.actual.png"));
    if expected_size != size {
        write_png(&actual_path, pixels, size);
        panic!(
            "golden `{name}` is {expected_size:?} but the test rendered {size:?}; actual image: {}",
            actual_path.display()
        );
    }

    let (mut different_pixels, mut max_delta) = (0, 0);
    let mut diff_image = Vec::with_capacity(pixels.len());
    for (a, e) in pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
    {
        let delta = a.iter().zip(e).map(|(a, e)| a.abs_diff(*e)).max().unwrap();
        max_delta = max_delta.max(delta);
        let differs = delta > CHANNEL_TOLERANCE;
        different_pixels += differs as usize;
        diff_image.extend_from_slice(if differs {
            &[255, 0, 0, 255]
        } else {
            &[0, 0, 0, 255]
        });
    }
    if different_pixels as f64 > (pixels.len() / 4) as f64 * MAX_DIFFERENT_PIXELS {
        let diff_path = failure_dir().join(format!("{name}.diff.png"));
        write_png(&actual_path, pixels, size);
        write_png(&diff_path, &diff_image, size);
        panic!(
            "golden `{name}` does not match: {different_pixels} of {} pixels differ (largest channel delta {max_delta})\n  expected: {}\n  actual:   {}\n  diff:     {}",
            pixels.len() / 4,
            path.display(),
            actual_path.display(),
            diff_path.display()
        );
    }
}
