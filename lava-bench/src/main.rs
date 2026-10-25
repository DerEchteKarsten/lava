//! lava against hand-written Vulkan: realistic workloads, each recorded through idiomatic lava and by hand with raw `ash` on the same device, pipelines and shaders, timed frame by frame (CPU recording, CPU submission, GPU), or checked under validation with matching checksums (`--check`)

mod check;
mod pipelines;
mod raw;
mod scenario;
mod scenarios;

pub mod bindings {
    lava::include_passes!();
}

use std::io::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lava::queue::{Graphics, Kind};
use lava::{Ctx, Features, Timeline, Wait};
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

use crate::raw::RawQueue;
use crate::scenario::{LavaCtx, RawCtx, Scenario, Stamped, gpu_ns};

/// Frame slots of both modes.
pub const IN_FLIGHT: u32 = 2;
/// The graphics timeline's staging ring.
const STAGING: u64 = 64 << 20;

struct Args {
    scenarios: Vec<String>,
    frames: usize,
    warmup: usize,
    check: bool,
    csv: Option<String>,
}

fn args() -> Args {
    let mut args = Args {
        scenarios: Vec::new(),
        frames: 500,
        warmup: 50,
        check: false,
        csv: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--frames" => args.frames = value("--frames").parse().expect("a frame count"),
            "--warmup" => args.warmup = value("--warmup").parse().expect("a frame count"),
            "--check" => args.check = true,
            "--csv" => args.csv = Some(value("--csv")),
            "--help" | "-h" => {
                println!(
                    "lava-bench [scenario..] [--frames N] [--warmup N] [--check] [--csv path]\n\nScenarios: {}",
                    scenarios::NAMES.join(", ")
                );
                std::process::exit(0);
            }
            name if !name.starts_with('-') => args.scenarios.push(name.to_string()),
            other => panic!("unknown option {other}"),
        }
    }
    if args.scenarios.is_empty() {
        args.scenarios = scenarios::NAMES.iter().map(|s| s.to_string()).collect();
    }
    for name in &args.scenarios {
        assert!(scenarios::NAMES.contains(&name.as_str()), "no scenario {name}");
    }
    args
}

/// Validation messages (warnings and errors) seen so far, under `--check`.
static MESSAGES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static WARNINGS: AtomicUsize = AtomicUsize::new(0);

struct Capture;

struct Message(String);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let meta = event.metadata();
        let validation = meta.target() == "vulkan-validation";
        if (validation || meta.target().starts_with("lava")) && *meta.level() <= tracing::Level::WARN {
            let mut message = Message(String::new());
            event.record(&mut message);
            eprintln!("[{} {}] {}", meta.level(), meta.target(), message.0);
            if validation {
                MESSAGES.lock().unwrap().push(message.0);
            } else {
                WARNINGS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Median, 10th and 90th percentile.
fn percentiles(times: &mut [u64]) -> [u64; 3] {
    times.sort_unstable();
    let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
    [at(0.5), at(0.1), at(0.9)]
}

/// One frame's measurements of one mode, in nanoseconds.
#[derive(Default)]
struct Samples {
    record: Vec<u64>,
    total: Vec<u64>,
    gpu: Vec<u64>,
}

fn lava_frame(scenario: &mut dyn Scenario, graphics: &mut Stamped<Graphics>, check: bool) -> [u64; 3] {
    let mut ctx = LavaCtx {
        graphics,
        check,
        record_ns: 0,
        waits: Vec::new(),
    };
    let start = Instant::now();
    scenario.lava(&mut ctx);
    let total = start.elapsed().as_nanos() as u64;
    let record = ctx.record_ns;
    Wait::all(&ctx.waits).unwrap();
    scenario.block_extra(true);
    let mut spans = scenario.extra_spans(true);
    if !ctx.waits.is_empty() {
        spans.push(ctx.graphics.span());
    }
    [record, total, gpu_ns(&spans)]
}

fn raw_frame(scenario: &mut dyn Scenario, queue: &mut RawQueue, check: bool) -> [u64; 3] {
    let before = queue.value;
    let mut ctx = RawCtx {
        queue,
        check,
        record_ns: 0,
    };
    let start = Instant::now();
    scenario.raw(&mut ctx);
    let total = start.elapsed().as_nanos() as u64;
    let record = ctx.record_ns;
    ctx.queue.block();
    scenario.block_extra(false);
    let mut spans = scenario.extra_spans(false);
    if ctx.queue.value != before {
        spans.push(ctx.queue.stamps((ctx.queue.value as usize - 1) % ctx.queue.slots()));
    }
    [record, total, gpu_ns(&spans)]
}

fn main() {
    let args = args();
    if args.check {
        // SAFETY: no other thread runs yet.
        unsafe {
std::env::set_var("VK_LAYER_VALIDATE_SYNC", "1")
        };
    }
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(Capture)).unwrap();
    let mut config = lava::Config::new(&bindings::PASS_MAP);
    config.validation = args.check;
    // Without `rebar` every `Gpu` buffer is device-local and unmapped and the timeline gets a
    // staging ring, as on a discrete GPU: what `streaming` uploads through.
    config.disabled = Features {
        rebar: true,
        ..Features::NONE
    };
    lava::init(&config).expect("lava init");
    let properties = Ctx::device_properties();
    let device = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) };
    println!("Device: {}", device.to_string_lossy());
    println!(
        "VK_KHR_unified_image_layouts: {}",
        match scenarios::has_unified_layouts() {
            true => "yes (optimal and GENERAL layouts perform alike there)",
            false => "no",
        }
    );

    let mut graphics = Stamped::new(Timeline::<Graphics>::new(IN_FLIGHT, STAGING).unwrap());
    let mut queue = RawQueue::new(Kind::Graphics, IN_FLIGHT as usize);
    let frames = if args.check { 4 } else { args.frames };
    let warmup = if args.check { 0 } else { args.warmup };

    let mut rows = Vec::new();
    let mut failed = false;
    for name in &args.scenarios {
        let Some(mut boxed) = scenarios::make(name, &mut graphics, &mut queue) else {
            println!("{name:<17} skipped: {}", scenarios::skip_reason(name));
            continue;
        };
        let scenario = boxed.as_mut();
        MESSAGES.lock().unwrap().clear();
        let mut lava = Samples::default();
        let mut raw = Samples::default();
        for frame in 0..warmup + frames {
            // The last frame of a check records the checksums.
            let check = args.check && frame + 1 == warmup + frames;
            let [record, total, gpu] = lava_frame(scenario, &mut graphics, check);
            if frame >= warmup {
                lava.record.push(record);
                lava.total.push(total);
                lava.gpu.push(gpu);
            }
            let [record, total, gpu] = raw_frame(scenario, &mut queue, check);
            if frame >= warmup {
                raw.record.push(record);
                raw.total.push(total);
                raw.gpu.push(gpu);
            }
        }
        if args.check {
            let messages = std::mem::take(&mut *MESSAGES.lock().unwrap());
            let (lava, raw) = scenario.checksums();
            let mismatches: Vec<_> = (0..lava.len()).filter(|i| lava[*i] != raw[*i]).collect();
            let ok = messages.is_empty() && mismatches.is_empty() && !lava.contains(&0);
            failed |= !ok;
            println!(
                "{name:<17} {}: {} validation messages, checksums lava {lava:08x?} raw {raw:08x?}{}",
                if ok { "ok" } else { "FAILED" },
                messages.len(),
                match mismatches.is_empty() {
                    true => String::new(),
                    false => format!(", mismatched at {mismatches:?}"),
                }
            );
        } else {
            rows.push((name.clone(), lava, raw));
        }
        drop(boxed);
        lava::wait_idle().unwrap();
    }

    if !args.check {
        report(&mut rows, frames, args.csv.as_deref());
    }
    drop(graphics);
    drop(queue);
    lava::wait_idle().unwrap();
    pipelines::destroy_all();
    lava::shutdown().unwrap();
    if failed {
        std::process::exit(1);
    }
}

fn report(rows: &mut [(String, Samples, Samples)], frames: usize, csv: Option<&str>) {
    println!(
        "\nPer frame in µs: median, then the 10th–90th percentile, of {frames} frames, lava and raw interleaved."
    );
    println!("  record: building and recording the commands (lava's submit closure, raw's recording)");
    println!("  submit: begin until submitted (lava: begin..submit; raw: pool reset, begin, record, end, vkQueueSubmit2)");
    println!("  gpu:    timestamps around the frame's commands");
    println!("  overhead: (lava − raw) / raw of the medians\n");
    println!(
        "{:<17} {:>8} {:>8} {:>7}  {:>8} {:>8} {:>7}  {:>8} {:>8} {:>7}",
        "scenario", "record", "raw", "+%", "submit", "raw", "+%", "gpu", "raw", "+%"
    );
    let mut lines = vec![
        "scenario,lava_record_ns,raw_record_ns,lava_submit_ns,raw_submit_ns,lava_gpu_ns,raw_gpu_ns,record_p10,record_p90,raw_record_p10,raw_record_p90,submit_p10,submit_p90,raw_submit_p10,raw_submit_p90,gpu_p10,gpu_p90,raw_gpu_p10,raw_gpu_p90".to_string(),
    ];
    let us = |ns: u64| ns as f64 / 1000.0;
    let pct = |a: u64, b: u64| (a as f64 - b as f64) / b as f64 * 100.0;
    for (name, lava, raw) in rows.iter_mut() {
        let l = [&mut lava.record, &mut lava.total, &mut lava.gpu].map(|t| percentiles(t));
        let r = [&mut raw.record, &mut raw.total, &mut raw.gpu].map(|t| percentiles(t));
        println!(
            "{name:<17} {:>8.1} {:>8.1} {:>+6.0}%  {:>8.1} {:>8.1} {:>+6.0}%  {:>8.1} {:>8.1} {:>+6.1}%",
            us(l[0][0]),
            us(r[0][0]),
            pct(l[0][0], r[0][0]),
            us(l[1][0]),
            us(r[1][0]),
            pct(l[1][0], r[1][0]),
            us(l[2][0]),
            us(r[2][0]),
            pct(l[2][0], r[2][0]),
        );
        let range = |p: [u64; 3]| format!("{:.0}–{:.0}", us(p[1]), us(p[2]));
        println!(
            "{:<17} {:>8} {:>8} {:>7}  {:>8} {:>8} {:>7}  {:>8} {:>8}",
            "",
            range(l[0]),
            range(r[0]),
            "",
            range(l[1]),
            range(r[1]),
            "",
            range(l[2]),
            range(r[2]),
        );
        lines.push(format!(
            "{name},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            l[0][0], r[0][0], l[1][0], r[1][0], l[2][0], r[2][0], l[0][1], l[0][2], r[0][1],
            r[0][2], l[1][1], l[1][2], r[1][1], r[1][2], l[2][1], l[2][2], r[2][1], r[2][2]
        ));
    }
    if let Some(path) = csv {
        let mut file = std::fs::File::create(path).expect("the CSV file");
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        println!("\nWrote {path}");
    }
}
