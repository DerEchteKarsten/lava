//! Proc macros for lava: records caller locations for Vulkan validation messages
use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, parse_macro_input};

/// Records the caller's location in `crate::state::CALLSITE` while the function runs, released
/// on every exit path by a `crate::state::CallsiteGuard`. Only in debug builds of the crate
/// that uses it.
#[proc_macro_attribute]
pub fn validation_trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: Item = parse_macro_input!(item as Item);

    let Item::Fn(mut item) = item else {
        return quote!(compile_error!("only functions are supported");).into();
    };

    item
        .attrs
        .push(syn::parse_quote!(#[cfg_attr(debug_assertions, track_caller)]));

    item.block.stmts.insert(
        0,
        syn::parse_quote!(
            #[cfg(debug_assertions)]
            let _callsite_guard =
                crate::state::CallsiteGuard::enter(::std::panic::Location::caller());
        ),
    );

    quote!(#item).into()
}
