//! Windowed GPU tests: init with a display, surfaces, swapchains, presenting, multi-window
mod common;

use std::process::ExitCode;

use lava::{
    bindless::{BindlessHandle, NULL_HANDLE},
    image::{format::ClearValue, usage::ColorAttachmentStorageTransferDst},
    present::PresentSurface,
    queue::{Graphics, Timeline},
    state::raw_vulkan as vk,
};
use lava_tests::bindings::PASS_MAP;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const SIZE: glam::UVec2 = glam::UVec2::new(256, 256);

type Surface = PresentSurface<ColorAttachmentStorageTransferDst>;

/// The swapchain format the tests ask for (the images are storage images too).
const FORMATS: [vk::SurfaceFormatKHR; 1] = [vk::SurfaceFormatKHR {
    format: vk::Format::B8G8R8A8_UNORM,
    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
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
    // SAFETY: the windows stay open until the teardown waited for the device, which destroys
    // the surfaces.
    unsafe {
        PresentSurface::new(
            &window.display_handle().unwrap().as_raw(),
            &window.window_handle().unwrap().as_raw(),
            &FORMATS,
        )
    }
}

/// Acquires, clears to `color` and presents an image; returns its handle.
fn present_frame(
    timeline: &mut Timeline<Graphics>,
    surface: &mut Surface,
    size: glam::UVec2,
    color: [f32; 4],
) -> BindlessHandle {
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
    let step = |name: &str| println!("test {name} ... ok");

    common::capture_validation_errors();
    let config = lava::Config {
        display: Some(window.display_handle().unwrap().as_raw()),
        validation: true,
        ..lava::Config::new(&PASS_MAP)
    };
    lava::init(&config).expect("lava init with a display failed");
    assert!(
        lava::init(&lava::Config::new(&PASS_MAP)).is_err(),
        "second init must fail"
    );
    step("init_with_display");

    let mut surface = surface_for(&window).unwrap();
    assert_eq!(surface.size(), glam::UVec2::ZERO, "no swapchain before the first acquire");
    let modes = surface.present_modes();
    assert_eq!(modes[0].0, vk::PresentModeKHR::FIFO);
    assert_eq!(surface.format().0, vk::Format::B8G8R8A8_UNORM);
    step("surface_reports_its_format_and_modes");

    let mut timeline = Timeline::<Graphics>::new(2).unwrap();
    let mut handles = Vec::new();
    for color in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]] {
        handles.push(present_frame(&mut timeline, &mut surface, SIZE, color));
    }
    assert_eq!(surface.size(), SIZE);
    // Bound in the storage set, the only one its usage allows.
    for handle in &handles {
        assert_ne!(handle.descriptor_index_set1, NULL_HANDLE);
        assert_eq!(handle.descriptor_index_set0, NULL_HANDLE);
    }
    step("acquire_clear_present");

    let old_images = surface.num_images();
    present_frame(&mut timeline, &mut surface, SIZE / 2, [1.0, 1.0, 0.0, 1.0]);
    assert_eq!(surface.size(), SIZE / 2);
    let new = present_frame(&mut timeline, &mut surface, SIZE / 2, [1.0, 0.0, 1.0, 1.0]);
    assert!(!handles.contains(&new), "the old swapchain's slots stay taken until it is destroyed");
    // An acquire destroys the old swapchain once nothing uses it, freeing the lowest slots.
    for _ in 0..2 {
        lava::wait_idle().unwrap();
        present_frame(&mut timeline, &mut surface, SIZE / 2, [0.0, 0.0, 0.0, 1.0]);
    }
    let lowest = handles.iter().map(|h| h.descriptor_index_set1).min().unwrap();
    let reserved = lava::bindless::reserve_storage(old_images as u32).unwrap();
    assert_eq!(reserved.start(), lowest);
    drop(reserved);
    step("a_new_size_recreates_the_swapchain_and_frees_the_old_slots");

    assert!(!surface.acquire(glam::UVec2::ZERO).unwrap());
    assert!(!surface.acquire(glam::UVec2::new(0, 64)).unwrap());
    step("no_image_without_an_extent");

    // An image no submission used stays current, through `present` too.
    assert!(surface.acquire(SIZE).unwrap());
    let unused = surface.current_image().handle();
    surface.present().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    assert_eq!(surface.current_image().handle(), unused);
    let drawn = present_frame(&mut timeline, &mut surface, SIZE, [0.0, 1.0, 1.0, 1.0]);
    assert_eq!(drawn, unused, "the next frame draws into it");
    // One a submission presented is presented by the next acquire.
    submit_without_presenting(&mut timeline, &mut surface);
    present_frame(&mut timeline, &mut surface, SIZE, [1.0, 1.0, 1.0, 1.0]);
    step("unused_images_stay_current");

    let submission = timeline.begin().unwrap();
    assert!(surface.acquire(SIZE).unwrap());
    let unpresented = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        submission.submit(|cmd| cmd.clear_image(surface.current_image(), ClearValue::Float([0.5; 4])))
    }));
    assert!(unpresented.is_err(), "the submission that uses the current image presents it");
    submit_without_presenting(&mut timeline, &mut surface);
    let submission = timeline.begin().unwrap();
    let twice = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        submission.submit(|cmd| cmd.present(&mut surface))
    }));
    assert!(twice.is_err(), "an image is presented by one submission");
    surface.present().unwrap();
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 0.0, 0.0, 1.0]);
    step("misuse_panics");

    let modes = surface.present_modes().len();
    surface.set_present_mode(modes - 1);
    present_frame(&mut timeline, &mut surface, SIZE, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(surface.present_mode(), modes - 1);
    surface.set_present_mode(0);
    present_frame(&mut timeline, &mut surface, SIZE, [0.0, 1.0, 0.0, 1.0]);
    step("present_modes_change");

    let second_window = create_window(event_loop, "lava test 2");
    let mut second = surface_for(&second_window).unwrap();
    assert_ne!(second.handle(), surface.handle());
    let other = present_frame(&mut timeline, &mut second, SIZE, [1.0, 1.0, 0.0, 1.0]);
    let first = present_frame(&mut timeline, &mut surface, SIZE, [1.0, 0.0, 1.0, 1.0]);
    assert_ne!(other, first);
    step("second_window_has_its_own_surface");

    // Destroys the surfaces while the windows are still open.
    drop(second);
    drop(surface);
    drop(timeline);
    lava::wait_idle().unwrap();
    step("teardown");

    let errors = common::take_validation_errors();
    assert!(
        errors.is_empty(),
        "Vulkan validation errors:\n{}",
        errors.join("\n\n")
    );
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
        println!("SKIPPED gpu_window: no Wayland or X11 display available");
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
