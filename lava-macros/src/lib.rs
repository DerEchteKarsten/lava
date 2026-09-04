use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, Stmt, parse_macro_input};

#[proc_macro_attribute]
pub fn validation_trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    #[cfg(not(debug_assertions))]
    return item;

    let item: Item = parse_macro_input!(item as Item);

    let Item::Fn(mut item) = item else {
        return quote!(compiler_error!("only functions are supported")).into();
    };

    item.attrs.push(syn::parse_quote!(#[track_caller]));

    item.block.stmts.insert(
        0,
        syn::parse_quote!(if crate::state::CALLSITE.get().is_none() {
            crate::state::CALLSITE.set(Some(std::panic::Location::caller().clone()));
        }),
    );

    let release_stmt = syn::parse_quote!(crate::state::CALLSITE.set(None););

    if let Some(Stmt::Expr(_, t)) = item.block.stmts.last()
        && t.is_none()
    {
        let last_statement = item.block.stmts.pop();
        item.block
            .stmts
            .push(syn::parse_quote!(let result_PROC_MACRO = #last_statement;));
        item.block.stmts.push(release_stmt);
        item.block
            .stmts
            .push(syn::parse_quote!(return result_PROC_MACRO;));
    } else {
        item.block.stmts.push(release_stmt);
    }

    quote!(#item).into()
}
