//! Windowed GPU tests: init with a display, surfaces, swapchains, presenting, multi-window
//!
//! Runs without the libtest harness because the window needs the main thread's event loop.
//! The tests share one lava context and run in order; a window shows up briefly while they
//! present. Without a Wayland or X11 display the binary reports that it skipped.
mod common;

use std::process::ExitCode;

use lava::{
    bindless::{BindlessHandle, BindlessWrites},
    image::format::{self, Format},
    state::Ctx,
    vkobjects::{
        queue::{Binary, FrameSlot, Gfx, PendingAccesses, Queue, Semaphore},
        surface::Surface,
        swapchain::Swapchain,
    },
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const SIZE: [u32; 2] = [256, 256];

fn create_window(event_loop: &ActiveEventLoop, title: &str) -> Window {
    event_loop
        .create_window(
            Window::default_attributes()
                .with_title(title)
                .with_inner_size(PhysicalSize::new(SIZE[0], SIZE[1])),
        )
        .expect("failed to create a window")
}

fn surface_for(window: &Window) -> lava::Result<Surface> {
    Surface::new(
        &window.display_handle().unwrap().as_raw(),
        &window.window_handle().unwrap().as_raw(),
    )
}

/// Acquires an image, clears it, and presents it. The frame's semaphores are parked in
/// `semaphores`: the presentation engine may still use them after this returns.
fn present_frame(
    queue: &Queue<Gfx>,
    swapchain: &Swapchain,
    color: [f32; 4],
    semaphores: &mut Vec<Semaphore<Binary>>,
) {
    let image_available = Semaphore::<Binary>::new().unwrap();
    let render_finished = Semaphore::<Binary>::new().unwrap();
    let index = swapchain.aquire_image(&image_available, None).unwrap();
    assert!((index as usize) < swapchain.num_images());

    let mut slot = FrameSlot::new(queue).unwrap();
    slot.begin()
        .unwrap()
        .execute(
            queue,
            PendingAccesses::default(),
            &[image_available.info()],
            &[render_finished.info()],
            |cmd| {
                cmd.clear_image(swapchain.image(index), color);
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
fn run(event_loop: &ActiveEventLoop) {
    let window = create_window(event_loop, "lava test");
    let step = |name: &str| println!("test {name} ... ok");

    common::capture_validation_errors();
    lava::init(
        Some(&window.display_handle().unwrap().as_raw()),
        true,
        false,
    )
    .expect("lava init with a display failed");
    assert!(Ctx::features().present);
    assert!(
        lava::init(None, false, false).is_err(),
        "second init must fail"
    );
    step("init_with_display");

    let mut surface = surface_for(&window).unwrap();
    assert!(!surface.formats.is_empty());
    assert!(!surface.present_modes.is_empty());
    assert!(surface.capabilities.min_image_count >= 1);
    step("surface_reports_formats_modes_and_capabilities");

    let mut swapchain = Swapchain::new(&surface, None, Some(SIZE)).unwrap();
    assert_eq!(swapchain.image(0).handle, BindlessHandle::none());
    let mut writes = BindlessWrites::default();
    swapchain.bind_storage(0, &mut writes);
    writes.submit();
    assert_eq!(swapchain.size, SIZE);
    assert!(swapchain.num_images() >= surface.capabilities.min_image_count as usize);
    // Creating a swapchain fixes the format behind `format::Swapchain`.
    assert!(
        surface
            .formats
            .iter()
            .any(|f| f.format == format::Swapchain::format())
    );
    let handles: Vec<_> = (0..swapchain.num_images() as u32)
        .map(|i| swapchain.image(i).handle)
        .collect();
    for (i, handle) in handles.iter().enumerate() {
        assert_eq!(handle.descriptor_index_set1, i as u32);
        assert_eq!(handle.descriptor_index_set0, lava::bindless::NULL_HANDLE);
    }
    step("swapchain_creation");

    let queue = Queue::<Gfx>::new().unwrap();
    let mut semaphores = Vec::new();
    present_frame(&queue, &swapchain, [1.0, 0.0, 0.0, 1.0], &mut semaphores);
    present_frame(&queue, &swapchain, [0.0, 1.0, 0.0, 1.0], &mut semaphores);
    step("acquire_clear_present");

    let new_size = [SIZE[0] / 2, SIZE[1] / 2];
    surface.refresh_capabilities().unwrap();
    swapchain.recreate(&surface, new_size).unwrap();
    assert_eq!(swapchain.size, new_size);
    // Shaders keep addressing the swapchain images through the same bindless slots.
    for (i, handle) in handles.iter().enumerate().take(swapchain.num_images()) {
        assert_eq!(swapchain.image(i as u32).handle, *handle);
    }
    present_frame(&queue, &swapchain, [0.0, 0.0, 1.0, 1.0], &mut semaphores);
    step("swapchain_recreate_keeps_bindless_handles");

    let second_window = create_window(event_loop, "lava test 2");
    let second_surface = surface_for(&second_window).unwrap();
    let mut second_swapchain = Swapchain::new(&second_surface, None, Some(SIZE)).unwrap();
    let mut writes = BindlessWrites::default();
    second_swapchain.bind_storage(8, &mut writes);
    writes.submit();
    assert_eq!(second_swapchain.image(0).handle.descriptor_index_set1, 8);
    assert_ne!(second_surface.handle, surface.handle);
    present_frame(
        &queue,
        &second_swapchain,
        [1.0, 1.0, 0.0, 1.0],
        &mut semaphores,
    );
    present_frame(&queue, &swapchain, [1.0, 0.0, 1.0, 1.0], &mut semaphores);
    step("second_window_has_its_own_surface_and_swapchain");

    // Swapchains go before their surfaces, surfaces before their windows.
    drop(second_swapchain);
    drop(second_surface);
    drop(swapchain);
    drop(surface);
    // Destroying the swapchains waited for the device, so the semaphores are idle now.
    drop(semaphores);
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
