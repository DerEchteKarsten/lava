use std::{fs, io::Write, path::PathBuf};

use bytemuck::Pod;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{Attribute, DeriveInput, Expr, PathArguments, Type, TypePtr, TypeTuple, parse_macro_input};

#[proc_macro_attribute]
pub fn shader(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as syn::ItemFn);
    let f = func.clone();
    let func_name = func.sig.ident;

    let stage = func.attrs
        .iter()
        .find(|e| e.path().is_ident("spirv"))
        .expect("Function does not have a spirv attribute")
        .parse_args::<proc_macro2::TokenStream>()
        .unwrap()
        .to_string();

    let stage = match stage {
        s if s.contains("compute") => quote!(ash::vk::PipelineStageFlags2::COMPUTE_SHADER),
        s if s.contains("vertex") => quote!(ash::vk::PipelineStageFlags2::VERTEX_SHADER),
        s if s.contains("fragment") => quote!(ash::vk::PipelineStageFlags2::FRAGMENT_SHADER),
        s if s.contains("ray_generation") => quote!(ash::vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR),
        _ => panic!("Shader Stage not suported (yet)")
    };

    let binding_type = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| if let syn::FnArg::Typed(pat_type) = arg { Some(pat_type) } else {None})
        .find(|arg| {
            let Some(attrib) = arg.attrs.iter().find(|e| e.path().is_ident("spirv")) else { return false;  };
            attrib.parse_args::<proc_macro2::TokenStream>().unwrap().to_string().contains("push_constant")
        }).map(|e| {
            let ty = e.ty.as_ref().clone();
            if let Type::Reference(ty) = ty
                && let Type::Path(ty) = ty.elem.as_ref().clone() {
                let path = format_ident!("C{}",ty.path.segments.last().unwrap().ident);
                quote!(#path)
            }else {
                quote!(compile_error!("Push Constant Type needs to be a struct"))
            }
        })
        .unwrap_or(quote!(compile_error!("")));

    let func_name_string = func_name.to_string();
    let s = func_name.span().unwrap().file();
    let path = s
        .split("/")
        .skip(1)
        .map(|e| {
            e.trim_end_matches(".rs")
        })
        .chain(std::iter::once(func_name_string.as_str()))
        .fold("".to_string(), |acc, s| {
            if acc == "" {
                s.to_string()
            }else {
                format!("{}::{}", acc, s)
            }
        });
        
    let expandet = quote! {
        pub struct #func_name;

        impl lava::command_buffer::Shader for #func_name {
            const STAGE: ash::vk::PipelineStageFlags2 = #stage;
            type GpuBinding = #binding_type;
            const ENTRY: &'static str = #path;
        }
    };

    let out_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let mut cpu_file = PathBuf::from(out_dir);
    cpu_file.push("target");
    cpu_file.push("bindings");
    if !fs::exists(&cpu_file).unwrap() {
        fs::create_dir(&cpu_file).unwrap();
    }
    cpu_file.push(format!("{}.cpu.rs", func_name_string));

    fs::write(&cpu_file, expandet.to_string()).expect("Could not write CPU bindings");

    quote!(#f).into()
}


#[proc_macro_derive(Binding)]
pub fn bindings(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let gpu_name = format_ident!("C{}", struct_name.to_string());

    let fields = match &input.data {
        syn::Data::Struct(str) => &str.fields,
        _ => return quote!(compile_error!("Texture can not be written to but still hast write_only tag")).into(), 
    };

    let mut cpu_fields = Vec::new();
    let mut gpu_fields = Vec::new();
    let mut field_constructors = Vec::new(); 
    let mut states = Vec::new();
    let mut checks = Vec::new();
    for field in fields {
        let name = field.ident.as_ref().unwrap();
        let mut pc = false;
        let (cpu_type, gpu_type) = if let Type::Path(ty) =  &field.ty {
            let name = ty.path.segments.last().unwrap().ident.to_string();
            let generic = if let PathArguments::AngleBracketed(args) = &ty.path.segments.last().unwrap().arguments {
                Some(args.args.first().unwrap())
            }else {
                None
            };
            match name.as_str() {
                "Image" | "MutImage" => (
                    quote!(&'a lava::vkobjects::image::Image),
                    quote!(u64),
                ),
                "Ptr" | "MutPtr" => {
                    let generic = generic.unwrap();
                (
                    quote!(&'a lava::vkobjects::buffer::Buffer<#generic>),
                    quote!(u64),
                )},
                _ => { 
                    pc = true;
                    (
                        quote!(#ty),
                        quote!(#ty),
                    )
                }
            }
        } else {
            (quote!(compile_error!("Unsuported Type")),quote!())
        };

        cpu_fields.push(quote! {
            pub #name: #cpu_type
        });
        gpu_fields.push(quote! {
            pub #name: #gpu_type
        });
        if pc {
            field_constructors.push(quote! {
                #name: bindings.#name
            });
            continue;
        }

        states.push(if let Type::Path(ty) = &field.ty {
            let type_name = ty.path.segments.last().unwrap().ident.to_string();
            let (layout, access, usage, aspect, field_constructor, handle) = match type_name.as_str() {
                "Image" => {
                    (
                        quote!(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        quote!(ash::vk::AccessFlags2::SHADER_SAMPLED_READ),
                        Some(quote!(ash::vk::ImageUsageFlags::SAMPLED)),
                        quote!(lava::vkobjects::image::get_aspects(bindings.#name.format)),
                        quote!(#name: bindings.#name.bindless_handle.unwrap() as u64),
                        quote!(lava::command_buffer::ResourceHandle::Image((bindings.#name.view, bindings.#name.handle))),
                    )
                },
                "MutImage" => {
                    (
                        quote!(ash::vk::ImageLayout::GENERAL),
                        quote!(ash::vk::AccessFlags2::SHADER_STORAGE_READ | ash::vk::AccessFlags2::SHADER_STORAGE_WRITE),
                        Some(quote!(ash::vk::ImageUsageFlags::STORAGE)),
                        quote!(lava::vkobjects::image::get_aspects(bindings.#name.format)),
                        quote!(#name: bindings.#name.bindless_handle.unwrap() as u64),
                        quote!(lava::command_buffer::ResourceHandle::Image((bindings.#name.view, bindings.#name.handle)))
                    )
                },
                "Ptr" => {
                    (
                        quote!(ash::vk::ImageLayout::UNDEFINED),
                        quote!(ash::vk::AccessFlags2::SHADER_SAMPLED_READ),
                        None,
                        quote!(ash::vk::ImageAspectFlags::empty()),
                        quote!(#name: bindings.#name.address),
                        quote!(lava::command_buffer::ResourceHandle::Buffer(bindings.#name.handle))
                    )
                },
                "MutPtr" => {
                    (
                        quote!(ash::vk::ImageLayout::UNDEFINED),
                        quote!(ash::vk::AccessFlags2::SHADER_STORAGE_READ | ash::vk::AccessFlags2::SHADER_STORAGE_WRITE),
                        None,
                        quote!(ash::vk::ImageAspectFlags::empty()),
                        quote!(#name: bindings.#name.address ),
                        quote!(lava::command_buffer::ResourceHandle::Buffer(bindings.#name.handle))
                    )
                },
                _ => unreachable!()
            };
            field_constructors.push(field_constructor);
            let name_str = name.to_string();
            if let Some(usage) = usage {
                checks.push(quote! {
                    assert!(bindings.#name.usage.contains(#usage), "Field {} needs {:#?} usage flag", #name_str, #usage);
                });
            }
            quote!(
                (
                    #handle,
                    lava::command_buffer::ResourceState {
                        access: #access,
                        stages,
                        layout: #layout,
                        aspect: #aspect,
                    }
                )
            )
        }else {
            quote!(compile_error!("Unsuported Type"))
        });
    }


    let expandet = quote! { 
        #[derive(Clone, Copy)]   
        pub struct #gpu_name {
            #(#gpu_fields,)*
        }
        
        pub struct #struct_name <'a> {
            #(#cpu_fields,)*
        }

        unsafe impl bytemuck::Pod for #gpu_name {}
        unsafe impl bytemuck::Zeroable for #gpu_name {}

        impl lava::command_buffer::Binding for #gpu_name {
            type CpuBinding<'a> = #struct_name<'a>;
            fn from_cpu_binding<'a>(bindings: &Self::CpuBinding<'a>) -> Self {
                #(#checks)*
                
                Self {
                    #(#field_constructors,)*
                }
            }
            fn resources<'a>(bindings: &Self::CpuBinding<'a>, stages: ash::vk::PipelineStageFlags2) -> Vec<(lava::command_buffer::ResourceHandle, lava::command_buffer::ResourceState)> {
                vec![
                    #(#states,)*
                ]
            }
        }
    };

    let out_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let mut cpu_file = PathBuf::from(out_dir);
    cpu_file.push("target");
    cpu_file.push("bindings");
    cpu_file.push(format!("{}.cpu.rs", struct_name));

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(false)
        .write(true)
        .truncate(true)
        .open(cpu_file)
        .unwrap();

    file.write(expandet.to_string().as_bytes()).unwrap();

    TokenStream::new()
}



fn main() {
    
}