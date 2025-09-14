#![feature(int_roundings)]
#![feature(f16)]
#![feature(lock_value_accessors)]
use std::{time::Instant};

use anyhow::{Ok, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use winit::{application::ApplicationHandler, dpi::{PhysicalSize, Size}, event::{self, Event, WindowEvent}, event_loop::{EventLoop, EventLoopBuilder}, platform::wayland::EventLoopBuilderExtWayland, raw_window_handle::{HasDisplayHandle, HasWindowHandle}, window::{Window, WindowAttributes, WindowId}};
use crate::{bindless::BindlessDescriptorHeap, pipelines::{RasterPipelineHandle, ShaderPath}, state::Ctx, vkobjects::{buffer::{Buffer, DynamicBuffer}, queue::CommandBuffer}};

pub mod bindless;
pub mod pipelines;
pub mod vkobjects;
pub mod state;


pub const FRAMES_IN_FLIGHT: usize = 3;

pub fn init<T: HasDisplayHandle+HasWindowHandle>(window: Option<&T>, enable_validation: bool) -> Result<()> {
    Ctx::init(window, enable_validation)?;
    BindlessDescriptorHeap::init()?;
    Ok(())
}



struct App<F: Fn(&vk::CommandBuffer, usize) -> Result<()>, Y: Fn() -> Result<()>> {
    window: Option<Window>,
    start_time: Instant,
    test_func: F,
    after_init: Y,
}

impl<F: Fn(&vk::CommandBuffer, usize) -> Result<()>, Y: Fn() -> Result<()>> ApplicationHandler for App<F, Y> {
    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _: WindowId, event: WindowEvent) {
        println!("{event:?}");
        match event {
            WindowEvent::CloseRequested => {
                println!("Close was requested; stopping");
                event_loop.exit();
            },
            WindowEvent::Resized(_) => {
                self.window.as_ref().expect("resize event without a window").request_redraw();
            },
            WindowEvent::RedrawRequested => {
                if self.start_time.elapsed().as_secs_f32() > 2.0 {
                    event_loop.exit();

                    return;
                }
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                let window = self.window.as_ref().expect("redraw request without a window");

                // Notify that you're about to draw.
                window.pre_present_notify();

                // Draw.
                Ctx::next_frame(&mut self.test_func).expect("failed to draw frame");

                // For contiguous redraw loop you can request a redraw from here.
                window.request_redraw();
            },
            _ => (),
        }   
    }
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop.create_window(Window::default_attributes()).expect("failed to create window");
            crate::init(Some(&window), false).unwrap();
            (self.after_init)();
            self.window = Some(window);
        }
        self.window.as_ref().expect("resumed event without a window").request_redraw();
    }
}


#[test]
fn test_init() {
    let event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .unwrap();
    
    let mut app = App {
        window: None,
        start_time: Instant::now(),
        test_func: |cmd: &vk::CommandBuffer, _| {
            Ok(())
        },
        after_init: || {
            Ok(())
        },
    };
    event_loop.run_app(&mut app).unwrap();
}


#[test]
fn pipelines() {
    let event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .unwrap();

    // let pipeline = RasterPipelineHandle {
    //     backface_culling: true,
    //     model: pipelines::PipelineModel::Vertex { vertex: ShaderPath {
    //         path: "shaders/test.vert.spv"
    //     } }
    // }
    
    let mut app = App {
        window: None,
        start_time: Instant::now(),
        test_func: |cmd: &vk::CommandBuffer, _| {
            Ok(())
        },
        after_init: || {
            Ok(())
        },
    };
    event_loop.run_app(&mut app).unwrap();
}


#[test]
fn buffers() {

    crate::init::<winit::window::Window>(None, false).unwrap();

    let buffer = Buffer::new(
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        1024,
    ).unwrap();

    let staging_buffer = Buffer::new(
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        1024,
    ).unwrap();

    let data = [67u8; 1024];
    staging_buffer.copy_data_to_buffer(&data).unwrap();

    Ctx::queue().execute_command_wait(|cmd_buf| {
        let copy_region = vk::BufferCopy::default()
            .size(1024);
        unsafe {
            Ctx::device().cmd_copy_buffer(
                *cmd_buf,
                staging_buffer.buffer,
                buffer.buffer,
                &[copy_region],
            );
        }
    }).unwrap();

    Ctx::queue().execute_command_wait(|cmd_buf| {
        let copy_region = vk::BufferCopy::default()
            .size(1024);
        unsafe {
            Ctx::device().cmd_copy_buffer(
                *cmd_buf,
                buffer.buffer,
                staging_buffer.buffer,
                &[copy_region],
            );
        }
    }).unwrap();

    staging_buffer.read(1024).unwrap().into_iter().for_each(|b: u8| {
        assert_eq!(b, 67u8);
    });


    let mut dynamic_buffer = DynamicBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, 256, None).unwrap();
    
    Ctx::queue().execute_command_wait(|cmd_buf| {
        dynamic_buffer.copy_from(&staging_buffer, 0, 1024);
    }).unwrap();

    Ctx::queue().execute_command_wait(|cmd_buf| {
        let copy_region = vk::BufferCopy::default()
            .size(1024);
        unsafe {
            Ctx::device().cmd_copy_buffer(
                *cmd_buf,
                dynamic_buffer.buffer.buffer,
                staging_buffer.buffer,
                &[copy_region],
            );
        }
    }).unwrap();

    staging_buffer.read(1024).unwrap().into_iter().for_each(|b: u8| {
        assert_eq!(b, 67u8);
    });

}