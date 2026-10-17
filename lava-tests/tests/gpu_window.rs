//! Windowed GPU tests: init with a display, surfaces, swapchains, presenting, multi-window
//!
//! Runs without the libtest harness because the window needs the main thread's event loop.
//! The tests share one lava context and run in order; a window shows up briefly while they
//! present. Without a Wayland or X11 display the binary reports that it skipped.
mod common;

use std::process::ExitCode;

use lava::{
    bindless::BindlessHandle,
    epoch,
    image::{
        format::{ClearValue, Format},
        usage::ColorAttachmentStorageTransferDst,
    },
    state::raw_vulkan as vk,
    vkobjects::{
        queue::{Binary, FrameSlot, Gfx, Queue, Semaphore},
        surface::Surface,
        swapchain::Swapchain,
    },
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
        Surface::new(
            &window.display_handle().unwrap().as_raw(),
            &window.window_handle().unwrap().as_raw(),
        )
    }
}

/// Acquires an image, clears it, and presents it. The frame's semaphores are parked in
/// `semaphores`: the presentation engine may still use them after this returns.
fn present_frame(
    queue: &Queue<Gfx>,
    swapchain: &Swapchain<ColorAttachmentStorageTransferDst>,
    color: [f32; 4],
    semaphores: &mut Vec<Semaphore<Binary>>,
) {
    let image_available = Semaphore::<Binary>::new().unwrap();
    let render_finished = Semaphore::<Binary>::new().unwrap();
    let index = swapchain.aquire_image(&image_available, None).unwrap();
    assert!((index as usize) < swapchain.num_images());

    let mut slot = FrameSlot::new(queue).unwrap();
    slot.begin(epoch::reserve())
        .unwrap()
        .execute(
            queue, &[image_available.info()],
            &[render_finished.info()],
            |cmd| {
                cmd.clear_image(swapchain.image(index), ClearValue::Float(color));
                cmd.present(swapchain.image(index));
            },
        )
        .unwrap();
    queue
        .present(swapchain, index, &[&render_finished])
        .unwrap();
    drop(slot);
    semaphores.extend([image_available, render_finished]);
}

/// Everything that needs a window, in order. Panics on the first failure.
/// The swapchain format the tests ask for (the images are storage images too).
const FORMATS: [vk::SurfaceFormatKHR; 1] = [vk::SurfaceFormatKHR {
    format: vk::Format::B8G8R8A8_UNORM,
    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
}];

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
    assert!(!surface.formats().is_empty());
    assert!(!surface.present_modes().is_empty());
    assert!(surface.capabilities().min_image_count >= 1);
    step("surface_reports_formats_modes_and_capabilities");

    let mut swapchain = Swapchain::new(&mut surface, None, Some(SIZE), &FORMATS).unwrap();
    assert_eq!(swapchain.size(), SIZE);
    assert!(swapchain.num_images() >= surface.capabilities().min_image_count as usize);
    assert!(
        surface
            .formats()
            .iter()
            .any(|f| f.format == swapchain.format().vk_format())
    );
    let handles: Vec<_> = (0..swapchain.num_images() as u32)
        .map(|i| swapchain.image(i).handle())
        .collect();
    // Bound in the storage set, the only one its usage allows, each image at its own slot.
    for (i, handle) in handles.iter().enumerate() {
        assert_ne!(handle.descriptor_index_set1, lava::bindless::NULL_HANDLE);
        assert_eq!(handle.descriptor_index_set0, lava::bindless::NULL_HANDLE);
        assert!(!handles[..i].contains(handle));
    }
    step("swapchain_creation");

    let queue = Queue::<Gfx>::new();
    let mut semaphores = Vec::new();
    present_frame(&queue, &swapchain, [1.0, 0.0, 0.0, 1.0], &mut semaphores);
    present_frame(&queue, &swapchain, [0.0, 1.0, 0.0, 1.0], &mut semaphores);
    step("acquire_clear_present");

    let new_size = SIZE / 2;
    // The old swapchain is handed back and can be dropped: it is destroyed once the frames
    // and presents that may use it are done.
    let old = swapchain.recreate(&mut surface, new_size).unwrap();
    assert_eq!(old.size(), SIZE);
    assert_eq!(swapchain.size(), new_size);
    // The old swapchain's slots stay taken until it is destroyed: the new images are bound
    // at others.
    let new_handles: Vec<BindlessHandle> = (0..swapchain.num_images() as u32)
        .map(|i| swapchain.image(i).handle())
        .collect();
    assert!(new_handles.iter().all(|handle| !handles.contains(handle)));
    present_frame(&queue, &swapchain, [0.0, 0.0, 1.0, 1.0], &mut semaphores);
    drop(old);
    step("swapchain_recreate_binds_other_slots");

    // Destroyed once the frames and presents that may use it are done, which an idle device
    // is: then its slots are free again, and the first free ones.
    lava::wait_idle().unwrap();
    let lowest = handles.iter().map(|h| h.descriptor_index_set1).min().unwrap();
    let reserved = lava::bindless::reserve_storage(handles.len() as u32).unwrap();
    assert_eq!(reserved.start(), lowest);
    drop(reserved);
    step("dropped_swapchains_free_their_slots");

    let second_window = create_window(event_loop, "lava test 2");
    let mut second_surface = surface_for(&second_window).unwrap();
    let second_swapchain = Swapchain::new(&mut second_surface, None, Some(SIZE), &FORMATS).unwrap();
    assert!(!new_handles.contains(&second_swapchain.image(0).handle()));
    assert_ne!(second_surface.handle(), surface.handle());
    present_frame(
        &queue,
        &second_swapchain,
        [1.0, 1.0, 0.0, 1.0],
        &mut semaphores,
    );
    present_frame(&queue, &swapchain, [1.0, 0.0, 1.0, 1.0], &mut semaphores);
    step("second_window_has_its_own_surface_and_swapchain");

    // A swapchain keeps its surface, so it may go after it (destroying the surface first would
    // be a validation error). Waiting for the device destroys everything dropped before, while
    // the windows are still open.
    drop(second_surface);
    drop(second_swapchain);
    drop(swapchain);
    drop(surface);
    drop(semaphores);
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
