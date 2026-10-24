//! Golden-image comparison: rendered RGBA8 pixels against PNGs in `lava-tests/tests/golden`, blessed with `LAVA_BLESS=1` only by the one test that owns a golden (`assert_golden`; others compare, `assert_matches_golden`), written atomically
use std::{
    collections::BTreeMap, fs::File, io::BufWriter, panic::Location, path::PathBuf, sync::Mutex,
};

/// A channel may differ by this much (of 255) before a pixel counts as different.
const CHANNEL_TOLERANCE: u8 = 2;
/// Fraction of pixels that may differ; absorbs rasterisation differences between GPU vendors.
const MAX_DIFFERENT_PIXELS: f64 = 0.001;

fn failure_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-failures")
}

/// Writes through a temporary file and a rename, so a test binary comparing at the same time
/// never reads half a PNG.
fn write_png(path: &PathBuf, pixels: &[u8], [width, height]: [u32; 2]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let temporary = path.with_extension(format!("png.{}.tmp", std::process::id()));
    write_png_to(&temporary, pixels, [width, height]);
    std::fs::rename(&temporary, path).unwrap();
}

fn write_png_to(path: &PathBuf, pixels: &[u8], [width, height]: [u32; 2]) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(pixels)
        .unwrap();
}

/// Whether this run rewrites the goldens: `LAVA_BLESS=1`, no other value.
fn blessing() -> bool {
    std::env::var_os("LAVA_BLESS").is_some_and(|value| value == "1")
}

/// Fails the test unless `pixels` (tightly packed RGBA8, `size` = [width, height]) match the
/// golden image `name`, which this test owns: with `LAVA_BLESS=1` it writes the golden instead.
/// Every golden has one owner; other tests that render the same image compare with
/// [`assert_matches_golden`], so two writers never race. On a mismatch the actual image and a
/// diff are written next to the test binaries for inspection.
#[track_caller]
pub fn assert_golden(name: &str, pixels: &[u8], size: [u32; 2]) {
    static OWNERS: Mutex<BTreeMap<String, &'static Location<'static>>> =
        Mutex::new(BTreeMap::new());
    let caller = Location::caller();
    let owner = *OWNERS
        .lock()
        .unwrap()
        .entry(name.to_owned())
        .or_insert(caller);
    assert_eq!(
        owner, caller,
        "golden `{name}` has two owners; compare in one with `assert_matches_golden`"
    );
    if blessing() {
        check_size(name, pixels, size);
        write_png(&golden_path(name), pixels, size);
        eprintln!("blessed {}", golden_path(name).display());
        return;
    }
    assert_matches_golden(name, pixels, size);
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden")).join(format!("{name}.png"))
}

#[track_caller]
fn check_size(name: &str, pixels: &[u8], size: [u32; 2]) {
    assert_eq!(
        pixels.len(),
        (size[0] * size[1] * 4) as usize,
        "pixel data of `{name}` does not match {size:?}"
    );
}

/// Like [`assert_golden`] for a golden another test owns: compares, also while blessing (the
/// owner may be in another test binary; rerun after blessing if it changed).
#[track_caller]
pub fn assert_matches_golden(name: &str, pixels: &[u8], size: [u32; 2]) {
    check_size(name, pixels, size);
    let path = golden_path(name);

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
