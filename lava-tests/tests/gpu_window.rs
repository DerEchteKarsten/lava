//! Windowed GPU tests: init with a display, surfaces, swapchains (recreated by size where the surface allows it, else by present mode or composite alpha), presenting and its misuse, the acquire waited for at the first access's stage, surfaces dropped with a current image, present modes, composite alpha, HDR metadata, surface formats and colour spaces, several windows presented at once; each step checks for validation errors
mod common;

use std::process::ExitCode;

use common::panic_message;
use lava::{
    Ctx,
    bindless::{self, ImageHandle, Storage},
    image::{
        format::{self, A2B10G10R10UnormPack32, B8G8R8A8Unorm, ClearValue, R8Unorm},
        usage::usage,
    },
    present::{
        ColorSpace, CompositeAlpha, PresentMode, PresentSurface, SurfaceFormat, present_all,
    },
    queue::{Graphics, Timeline},
    vk,
};
use lava_tests::bindings::*;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const SIZE: glam::UVec2 = glam::UVec2::new(256, 256);

type Surface = PresentSurface<usage![ColorAttachment, Storage, TransferDst]>;

/// The swapchain format the tests ask for (the images are storage images too).
const FORMATS: [SurfaceFormat; 1] = [SurfaceFormat {
    format: format::Dynamic(vk::Format::B8G8R8A8_UNORM),
    color_space: ColorSpace::SrgbNonlinear,
}];

fn create_window(event_loop: &ActiveEventLoop, title: &str) -> Window {
    event_loop
        .create_window(
            Window::default_attributes()
                .with_title(title)
                .with_inner_size(PhysicalSize::new(SIZE.x, SIZE.y)),
        )
        .expect("failed to create a window")
}

fn surface_for(window: &Window) -> lava::Result<Surface> {
    surface_preferring(window, &FORMATS)
}

fn surface_preferring(window: &Window, formats: &[SurfaceFormat]) -> lava::Result<Surface> {
    // SAFETY: the windows stay open until the teardown waited for the device, which destroys
    // the surfaces.
    unsafe {
        PresentSurface::new(
            &window.display_handle().unwrap().as_raw(),
            &window.window_handle().unwrap().as_raw(),
            formats,
        )
    }
}

/// Acquires, clears to `color` and presents an image; returns its handle.
fn present_frame(
    timeline: &mut Timeline<Graphics>,
    surface: &mut Surface,
    size: glam::UVec2,
    color: [f32; 4],
) -> ImageHandle<Storage> {
    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(size).unwrap(), "the window has an extent");
    let handle = surface.current_image().handle();
    submission
        .submit(|cmd| {
            cmd.clear_image(surface.current_image(), ClearValue::Float(color));
            cmd.present(surface);
        })
        .unwrap();
    surface.present().unwrap();
    handle
}

/// Acquires an image and clears and presents it in a submission, without `surface.present()`.
fn submit_without_presenting(timeline: &mut Timeline<Graphics>, surface: &mut Surface) {
    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    submission
        .submit(|cmd| {
            cmd.clear_image(surface.current_image(), ClearValue::Float([0.5; 4]));
            cmd.present(surface);
        })
        .unwrap();
}

fn run(event_loop: &ActiveEventLoop) {
    let window = create_window(event_loop, "lava test");
    // Each step fails on the validation errors it raised, before it reports ok.
    let step = |name: &str| {
        let errors = common::take_validation_errors();
        assert!(
            errors.is_empty(),
            "{name}: Vulkan validation errors:\n{}",
            errors.join("\n\n")
        );
        println!("test {name} ... ok");
    };

    common::capture_validation_errors();
    let config = lava::Config {
        display: Some(window.display_handle().unwrap().as_raw()),
        validation: true,
        ..lava::Config::new(&PASS_MAP)
    };
    lava::init(&config).expect("lava init with a display failed");
    common::assert_validation_active();
    assert!(
        lava::init(&lava::Config::new(&PASS_MAP)).is_err(),
        "second init must fail"
    );
    step("init_with_display");

    let mut surface = surface_for(&window).unwrap();
    assert_eq!(
        surface.size(),
        glam::UVec2::ZERO,
        "no swapchain before the first acquire"
    );
    let modes = surface.present_modes();
    assert_eq!(modes[0], PresentMode::Fifo);
    assert!(surface.supports(PresentMode::Fifo));
    assert_eq!(surface.present_mode(), PresentMode::Fifo);
    assert_eq!(surface.format().0, vk::Format::B8G8R8A8_UNORM);
    assert_eq!(surface.surface_format(), FORMATS[0]);
    assert!(surface.is_preferred_format());
    step("surface_reports_its_format_and_modes");

    let mut timeline = Timeline::<Graphics>::new(2, 0).unwrap();
    let mut handles = Vec::new();
    for color in [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ] {
        handles.push(present_frame(&mut timeline, &mut surface, SIZE, color));
    }
    // The requested extent where the surface lets the application choose it (Wayland), else
    // the window's (X11, where a window manager may have resized the window).
    let window_extent = || glam::UVec2::new(window.inner_size().width, window.inner_size().height);
    assert!(
        surface.size() == SIZE || surface.size() == window_extent(),
        "the swapchain is the requested or the window's extent: {}",
        surface.size()
    );
    // Bound as storage, the only descriptor its usage allows.
    for handle in &handles {
        assert!(handle.storage_slot().is_some());
        assert_eq!(handle.sampled_slot(), None);
    }
    step("acquire_clear_present");

    // The current swapchain's slots: frames until the last twice its image count showed only
    // that many images (a window manager may still be resizing the window, and X11 recreates
    // suboptimal swapchains at the same extent).
    let (mut frames, mut current) = (Vec::new(), Vec::new());
    let mut extent = surface.size();
    for _ in 0..64 {
        let handle = present_frame(&mut timeline, &mut surface, SIZE, [1.0, 1.0, 1.0, 1.0]);
        frames.push((handle, surface.size()));
        let n = surface.num_images();
        if frames.len() >= 2 * n {
            let last = &frames[frames.len() - 2 * n..];
            extent = surface.size();
            let mut handles: Vec<_> = last.iter().map(|&(h, _)| h).collect();
            handles.sort_by_key(|h| h.storage_slot());
            handles.dedup();
            if handles.len() == n && last.iter().all(|&(_, size)| size == extent) {
                current = handles;
                break;
            }
        }
    }
    assert!(
        !current.is_empty(),
        "the swapchain settles within 64 frames"
    );
    let old_images = surface.num_images();
    // A smaller window's swapchain is smaller where the surface lets the application choose
    // its extent (Wayland); where the extent is the window's (X11), it stays, and another
    // present mode or composite alpha recreates the swapchain instead.
    let mut new = present_frame(&mut timeline, &mut surface, SIZE / 2, [1.0, 1.0, 0.0, 1.0]);
    let resized = surface.size();
    assert!(
        resized == SIZE / 2 || resized == window_extent(),
        "the swapchain is the requested or the window's extent: {resized}"
    );
    let size = SIZE / 2;
    // Recreated by the new size, or (X11) by a window manager still resizing the window.
    let by_size = resized == SIZE / 2 || resized != extent;
    let recreated = if by_size {
        println!("the swapchain follows the requested size or the window's");
        true
    } else if let Some(&mode) = (surface.present_modes().iter()).find(|&&m| m != PresentMode::Fifo)
    {
        println!("the swapchain has the window's extent: recreated by {mode:?}");
        surface.set_present_mode(mode)
    } else if let Some(alpha) = (CompositeAlpha::ALL.into_iter())
        .find(|&a| a != surface.composite_alpha() && surface.supports_composite_alpha(a))
    {
        println!("the swapchain has the window's extent: recreated by {alpha:?}");
        surface.set_composite_alpha(alpha)
    } else {
        false
    };
    if recreated {
        if !by_size {
            new = present_frame(&mut timeline, &mut surface, size, [1.0, 0.0, 1.0, 1.0]);
        }
        assert!(
            !current.contains(&new),
            "the old swapchain's slots stay taken until it is destroyed"
        );
        // An acquire destroys the old swapchain once nothing uses it, freeing its slots.
        for _ in 0..2 {
            lava::wait_idle().unwrap();
            present_frame(&mut timeline, &mut surface, size, [0.0, 0.0, 0.0, 1.0]);
        }
        let lowest = (current.iter())
            .map(|h| h.storage_slot().unwrap())
            .min()
            .unwrap();
        let reserved = bindless::reserve::<Storage>(old_images as u32).unwrap();
        assert!(
            reserved.slots().start <= lowest,
            "{:?} lies above the freed slot {lowest}",
            reserved.slots()
        );
        drop(reserved);
    } else {
        eprintln!(
            "SKIPPED a_new_size_recreates_the_swapchain: the extent is fixed and there is no other present mode or composite alpha"
        );
    }
    assert!(surface.set_present_mode(PresentMode::Fifo));
    step("a_new_size_recreates_the_swapchain_and_frees_the_old_slots");

    assert!(!surface.acquire(glam::UVec2::ZERO).unwrap());
    assert!(!surface.acquire(glam::UVec2::new(0, 64)).unwrap());
    step("no_image_without_an_extent");

    // An image no submission used stays current, through `present` too.
    assert!(surface.acquire(SIZE).unwrap());
    let unused = surface.current_image().handle::<Storage>();
    surface.present().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    assert_eq!(surface.current_image().handle::<Storage>(), unused);
    let drawn = present_frame(&mut timeline, &mut surface, SIZE, [0.0, 1.0, 1.0, 1.0]);
    assert_eq!(drawn, unused, "the next frame draws into it");
    // One a submission presented is presented by the next acquire.
    submit_without_presenting(&mut timeline, &mut surface);
    present_frame(&mut timeline, &mut surface, SIZE, [1.0, 1.0, 1.0, 1.0]);
    step("unused_images_stay_current");

    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    let message = panic_message(|| {
        submission
            .submit(|cmd| cmd.clear_image(surface.current_image(), ClearValue::Float([0.5; 4])))
    });
    assert!(
        message.contains("must present it"),
        "the submission that uses the current image presents it: {message:?}"
    );
    submit_without_presenting(&mut timeline, &mut surface);
    let submission = timeline.begin().unwrap();
    let message = panic_message(|| submission.submit(|cmd| cmd.present(&mut surface)));
    assert!(
        message.contains("a submission presented the current image already"),
        "an image is presented by one submission: {message}"
    );
    surface.present().unwrap();
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 0.0, 0.0, 1.0]);
    step("misuse_panics");

    // Presenting or acquiring inside a submission would use a semaphore it signals.
    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    let message = panic_message(|| {
        let _ = submission.submit(|cmd| {
            cmd.present(&mut surface);
            let _ = surface.present();
        });
    });
    assert!(message.contains("call it after `submit`"), "{message}");
    // The failed recording was replaced by its barriers and still presents the image.
    surface.present().unwrap();
    let submission = timeline.begin().unwrap();
    let message = panic_message(|| {
        let _ = submission.submit(|_| {
            let _ = surface.acquire(SIZE);
        });
    });
    assert!(message.contains("before or after `submit`"), "{message}");
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 0.0, 0.0, 1.0]);
    step("present_and_acquire_inside_a_submission_panic");

    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    let message = panic_message(|| {
        let _ = submission.submit(|cmd| {
            cmd.present(&mut surface);
            cmd.clear_image(surface.current_image(), ClearValue::Float([0.5; 4]));
        });
    });
    assert!(
        message.contains("present must be the image's last command"),
        "{message}"
    );
    surface.present().unwrap();
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 0.0, 0.0, 1.0]);
    step("an_access_after_present_panics");

    let modes = surface.present_modes().to_vec();
    for &mode in &modes {
        assert!(surface.set_present_mode(mode));
        present_frame(&mut timeline, &mut surface, SIZE, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(surface.present_mode(), mode);
    }
    for &mode in PresentMode::ALL {
        if !modes.contains(&mode) {
            assert!(!surface.set_present_mode(mode), "{mode:?} isn't supported");
            assert_ne!(surface.present_mode(), mode);
        }
    }
    #[allow(unreachable_patterns)]
    let latest_ready = PresentMode::ALL.iter().find(|m| {
        !matches!(
            m,
            PresentMode::Fifo
                | PresentMode::FifoRelaxed
                | PresentMode::Mailbox
                | PresentMode::Immediate
        )
    });
    match latest_ready {
        Some(mode) if surface.supports(*mode) => println!("presented with {mode:?}"),
        _ => eprintln!("SKIPPED FifoLatestReady: the surface doesn't support it"),
    }
    assert!(surface.set_present_mode(PresentMode::Fifo));
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 1.0, 0.0, 1.0]);
    step("present_modes_change");

    let alpha = surface.composite_alpha();
    assert!(surface.supports_composite_alpha(alpha));
    for other in CompositeAlpha::ALL {
        if surface.set_composite_alpha(other) {
            present_frame(&mut timeline, &mut surface, SIZE, [0.0, 0.0, 1.0, 0.5]);
            assert_eq!(surface.composite_alpha(), other);
        }
    }
    assert!(surface.set_composite_alpha(alpha));
    step("composite_alpha_changes");

    if Ctx::features().hdr_metadata {
        let metadata = lava::present::HdrMetadata {
            red: glam::Vec2::new(0.708, 0.292),
            green: glam::Vec2::new(0.170, 0.797),
            blue: glam::Vec2::new(0.131, 0.046),
            white_point: glam::Vec2::new(0.3127, 0.3290),
            max_luminance: 1000.0,
            min_luminance: 0.001,
            max_content_light_level: 1000.0,
            max_frame_average_light_level: 400.0,
        };
        surface.set_hdr_metadata(&metadata).unwrap();
        present_frame(&mut timeline, &mut surface, SIZE, [1.0, 1.0, 1.0, 1.0]);
        step("hdr_metadata");
    } else {
        eprintln!("SKIPPED hdr_metadata: the device has no VK_EXT_hdr_metadata");
    }

    let second_window = create_window(event_loop, "lava test 2");
    let mut second = surface_for(&second_window).unwrap();
    assert_ne!(second.handle(), surface.handle());
    let other = present_frame(&mut timeline, &mut second, SIZE, [1.0, 1.0, 0.0, 1.0]);
    let first = present_frame(&mut timeline, &mut surface, SIZE, [1.0, 0.0, 1.0, 1.0]);
    assert_ne!(other, first);
    step("second_window_has_its_own_surface");

    // One submission draws into both, one call presents both.
    for color in [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]] {
        let submission = timeline.begin().unwrap();
        assert!(surface.acquire(SIZE).unwrap() && second.acquire(SIZE).unwrap());
        submission
            .submit(|cmd| {
                cmd.clear_image(surface.current_image(), ClearValue::Float(color));
                cmd.clear_image(second.current_image(), ClearValue::Float(color));
                cmd.present(&mut surface);
                cmd.present(&mut second);
            })
            .unwrap();
        present_all(&mut [&mut surface, &mut second]).unwrap();
    }
    // A surface whose image no submission presented keeps it.
    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap() && second.acquire(SIZE).unwrap());
    let kept = second.current_image().handle::<Storage>();
    submission
        .submit(|cmd| {
            cmd.clear_image(surface.current_image(), ClearValue::Float([0.0; 4]));
            cmd.present(&mut surface);
        })
        .unwrap();
    present_all(&mut [&mut surface, &mut second]).unwrap();
    assert!(second.acquire(SIZE).unwrap());
    assert_eq!(second.current_image().handle::<Storage>(), kept);
    present_frame(&mut timeline, &mut second, SIZE, [0.0; 4]);
    step("present_all_presents_several_surfaces_at_once");

    // HDR first where the surface has it, else sRGB.
    let hdr = [
        SurfaceFormat::of::<A2B10G10R10UnormPack32>(ColorSpace::Hdr10St2084),
        SurfaceFormat::of::<B8G8R8A8Unorm>(ColorSpace::SrgbNonlinear),
    ];
    let third_window = create_window(event_loop, "lava test 3");
    let mut third = surface_preferring(&third_window, &hdr).unwrap();
    assert!(third.is_preferred_format());
    assert!(
        hdr.contains(&third.surface_format()),
        "{:?}",
        third.surface_format()
    );
    println!("the HDR test surface presents {:?}", third.surface_format());
    present_frame(&mut timeline, &mut third, SIZE, [0.5, 0.5, 0.5, 1.0]);
    drop(third);
    // A format no surface shows falls back, reported.
    let unshown = [SurfaceFormat::of::<R8Unorm>(ColorSpace::SrgbNonlinear)];
    let mut fallback = surface_preferring(&third_window, &unshown).unwrap();
    assert!(!fallback.is_preferred_format());
    assert_ne!(fallback.surface_format(), unshown[0]);
    present_frame(&mut timeline, &mut fallback, SIZE, [0.5, 0.5, 0.5, 1.0]);
    drop(fallback);
    step("surface_formats_and_colour_spaces");

    // The presenting submission's first access of the image waits for its acquire at that
    // access's stage: a compute shader's storage write, or a rendering's attachment output.
    for first in ["compute", "rendering"] {
        let submission = timeline.begin().unwrap();
        assert!(surface.acquire(SIZE).unwrap());
        submission
            .submit(|cmd| {
                let image = surface.current_image();
                match first {
                    "compute" => cmd.compute(
                        &TestComputeImage::push_bindings(
                            TestComputeImageArgs {
                                target: image,
                                tint: glam::Vec4::ONE,
                                offset: glam::UVec2::ZERO,
                                size: SIZE,
                            },
                            None,
                        ),
                        [SIZE.x / 8, SIZE.y / 8, 1],
                    ),
                    _ => cmd
                        .raster(
                            SIZE,
                            [lava::command_buffer::Attachment::color(
                                image,
                                Some(ClearValue::Float([0.0, 0.5, 0.0, 1.0])),
                            )],
                        )
                        .record("clear only"),
                }
                cmd.present(&mut surface);
            })
            .unwrap();
        surface.present().unwrap();
    }
    step("the_first_access_waits_for_the_acquire_at_its_stage");

    // A surface dropped with a current image: acquired only, or presented by a submission but
    // not by `present()`.
    for presented in [false, true] {
        let window = create_window(event_loop, "lava test 4");
        let mut dropped = surface_for(&window).unwrap();
        present_frame(&mut timeline, &mut dropped, SIZE, [0.0, 0.0, 1.0, 1.0]);
        if presented {
            submit_without_presenting(&mut timeline, &mut dropped);
        } else {
            assert!(dropped.acquire(SIZE).unwrap());
        }
        drop(dropped);
        lava::wait_idle().unwrap();
    }
    step("dropping_a_surface_with_a_current_image");

    // Destroys the surfaces while the windows are still open.
    drop(second);
    drop(surface);
    drop(timeline);
    lava::wait_idle().unwrap();
    step("teardown");

    step("no_validation_errors");
}

#[derive(Default)]
struct App {
    outcome: Option<std::thread::Result<()>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.outcome.is_none() {
            self.outcome = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || run(event_loop),
            )));
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

fn main() -> ExitCode {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() && std::env::var_os("DISPLAY").is_none() {
        eprintln!("SKIPPED gpu_window: no Wayland or X11 display available");
        return ExitCode::SUCCESS;
    }

    let event_loop = EventLoop::new().expect("failed to create an event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop failed");

    match app.outcome {
        Some(Ok(())) => {
            println!("\ntest result: ok.");
            ExitCode::SUCCESS
        }
        Some(Err(_)) => {
            println!("\ntest result: FAILED.");
            ExitCode::FAILURE
        }
        None => {
            println!("\ntest result: FAILED. The event loop never resumed.");
            ExitCode::FAILURE
        }
    }
}
