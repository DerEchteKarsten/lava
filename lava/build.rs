//! Derives the cfgs of the draw state a raster pipeline bakes in (`bake_*`) from lava's cargo features: a state is baked where its capability is compiled and its `dynamic-*` feature isn't asserted
use std::env;

/// `(cfg, capabilities (any of them), the dynamic state's cargo feature)`.
const BAKED: &[(&str, &[&str], &str)] = &[
    ("bake_blend", &[], "dynamic-blend"),
    (
        "bake_polygon_mode",
        &["fill-mode-non-solid"],
        "dynamic-polygon-mode",
    ),
    ("bake_depth_clamp", &["depth-clamp"], "dynamic-depth-clamp"),
    (
        "bake_depth_clip",
        &["depth-clip-enable"],
        "dynamic-depth-clip",
    ),
    (
        "bake_depth_clip_negative_one_to_one",
        &["depth-clip-control"],
        "dynamic-depth-clip-negative-one-to-one",
    ),
    ("bake_logic_op", &["logic-op"], "dynamic-logic-op"),
    (
        "bake_conservative_rasterization",
        &["conservative-rasterization"],
        "dynamic-conservative-rasterization",
    ),
    (
        "bake_line_mode",
        &["rectangular-lines", "bresenham-lines", "smooth-lines"],
        "dynamic-line-mode",
    ),
    (
        "bake_line_stipple",
        &["line-stipple"],
        "dynamic-line-stipple",
    ),
    (
        "bake_provoking_vertex",
        &["provoking-vertex"],
        "dynamic-provoking-vertex",
    ),
];

fn main() {
    let on = |feature: &str| {
        let name = format!("CARGO_FEATURE_{}", feature.to_uppercase().replace('-', "_"));
        env::var_os(name).is_some()
    };
    for (cfg, capabilities, dynamic) in BAKED {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
        let capability = capabilities.is_empty() || capabilities.iter().any(|c| on(c));
        if capability && !on(&format!("assert-{dynamic}")) {
            println!("cargo::rustc-cfg={cfg}");
        }
    }
}
