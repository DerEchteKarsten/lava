//! Proc macros for lava: records caller locations for Vulkan validation messages
use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, parse_macro_input};

/// Records the caller's location in `crate::state::CALLSITE` for the duration of the function.
///
/// The location is held by a `crate::state::CallsiteGuard`, so it is released on every exit
/// path (early `return`, `?`, panic) and only by the outermost traced call.
#[proc_macro_attribute]
pub fn validation_trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    #[cfg(not(debug_assertions))]
    return item;

    let item: Item = parse_macro_input!(item as Item);

    let Item::Fn(mut item) = item else {
        return quote!(compile_error!("only functions are supported");).into();
    };

    item.attrs.push(syn::parse_quote!(#[track_caller]));

    item.block.stmts.insert(
        0,
        syn::parse_quote!(
            let _callsite_guard =
                crate::state::CallsiteGuard::enter(::std::panic::Location::caller());
        ),
    );

    quote!(#item).into()
}
