//! lava-build-slang's hot reload (the `hot-reload` feature, a debugging aid): recompiles the pass files that change (every file if an include does) and hands their SPIR-V to lava, which replaces their pipelines right away; a panic fails only its file
use std::{
    iter,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{OnceLock, mpsc},
    time::Duration,
};

use lava::pipeline::{self, Reload, pass_map};
use notify::EventHandler;

use crate::{
    Optimization, SLANG_DIR,
    compile::{self, slang},
};

/// How the build script compiled the passes, generated as `$OUT_DIR/lava/hot_reload.rs`.
pub struct Build {
    pub search_paths: &'static [&'static str],
    pub defines: &'static [(&'static str, &'static str)],
    /// The absolute pass and include directories.
    pub watch_dirs: &'static [&'static str],
    pub optimization: Optimization,
    pub debug_info: bool,
    /// One per pass of `PASS_MAP`, in its order.
    pub passes: &'static [BuiltPass],
}

pub struct BuiltPass {
    /// Its name in `PASS_MAP`.
    pub name: &'static str,
    /// The absolute path of its `.slang` file.
    pub source: &'static str,
    /// The entry point it was linked with alone, in a file of several compute entry points.
    pub entry: Option<&'static str>,
    /// The `PassDesc::layout_hash` the bindings were generated from; a reload that changes it
    /// is refused.
    pub layout: u64,
}

/// Starts the hot reload of the passes compiled by the calling crate's build script, after
/// `lava::init` got their `PASS_MAP`.
#[macro_export]
macro_rules! watch {
    () => {{
        use $crate::{
            Optimization,
            hot_reload::{Build, BuiltPass},
        };
        static BUILD: Build = include!(concat!(env!("OUT_DIR"), "/lava/hot_reload.rs"));
        $crate::hot_reload::watch(&BUILD)
    }};
}

/// Whether `build` is the build of lava's pass map: the same passes by name, in its order.
fn matches_pass_map(build: &Build) -> Result<(), String> {
    let map = pass_map().passes;
    if build.passes.len() != map.len() {
        return Err(format!(
            "its build has {} passes, lava's `PASS_MAP` {}",
            build.passes.len(),
            map.len()
        ));
    }
    match (build.passes.iter().zip(map)).find(|(built, entry)| built.name != entry.name) {
        Some((built, entry)) => Err(format!(
            "its build's pass `{}` is `{}` in lava's `PASS_MAP`",
            built.name, entry.name
        )),
        None => Ok(()),
    }
}

/// Recompiles the passes of `changed` files (every pass if one of them is no pass's source, an
/// include) and replaces their pipelines, logging each result; a panic fails only its file.
fn reload(
    build: &Build,
    global: &slang::GlobalSession,
    options: &compile::Options,
    changed: &[PathBuf],
) {
    let is_source_of = |path: &PathBuf, pass: &BuiltPass| path == Path::new(pass.source);
    let all =
        (changed.iter()).any(|path| !build.passes.iter().any(|pass| is_source_of(path, pass)));
    let mut sources: Vec<&str> = (build.passes.iter())
        .filter(|pass| all || changed.iter().any(|path| is_source_of(path, pass)))
        .map(|pass| pass.source)
        .collect();
    sources.dedup();
    for source in sources {
        let reloaded = catch_unwind(AssertUnwindSafe(|| {
            reload_file(build, global, options, source)
        }));
        if let Err(panic) = reloaded {
            let message = (panic.downcast_ref::<String>().map(String::as_str))
                .or(panic.downcast_ref::<&str>().copied())
                .unwrap_or("a panic");
            tracing::error!(target: "shader_file_watcher", "{source}: the reload panicked: {message}");
        }
    }
}

fn reload_file(
    build: &Build,
    global: &slang::GlobalSession,
    options: &compile::Options,
    source: &str,
) {
    // A fresh session: a session keeps the modules it loaded.
    let session = compile::session(global, options);
    let compiled = compile::compile_file(&session, source);
    for warning in &compiled.warnings {
        tracing::warn!(target: "shader_file_watcher", "{warning}");
    }
    let passes = match compiled.passes {
        Ok(passes) => passes,
        Err(errors) => {
            tracing::error!(target: "shader_file_watcher", "{source}:\n{}", errors.join("\n"));
            return;
        }
    };
    for (index, built) in build
        .passes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.source == source)
    {
        let Some(pass) = passes.iter().find(|p| p.name == built.name) else {
            tracing::error!(
                target: "shader_file_watcher",
                "{source}: the pass `{}` is gone, which the generated bindings can't follow: run `cargo build`",
                built.name
            );
            continue;
        };
        if pass.desc.layout_hash() != built.layout {
            tracing::error!(
                target: "shader_file_watcher",
                "{source}: `{}`'s push constants, specialization constants, entry points or exports changed, \
                 which the generated bindings can't follow: run `cargo build`",
                built.name
            );
            continue;
        }
        match pipeline::reload_pass(index, &pass.spirv) {
            Ok(Reload::Replaced) => {
                tracing::info!(target: "shader_file_watcher", "Reloaded pass `{}`", built.name)
            }
            Ok(Reload::Skipped) => tracing::info!(
                target: "shader_file_watcher",
                "Skipped pass `{}`: the device can't run it",
                built.name
            ),
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "{source}: `{}`: {e}", built.name)
            }
        }
    }
}

/// Watches the build's directories and [`SLANG_DIR`] and compiles changed passes on a thread of
/// its own. Later calls do nothing.
pub fn watch(build: &'static Build) {
    use notify::Watcher;

    static WATCHER: OnceLock<Option<notify::RecommendedWatcher>> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let watch = || -> std::result::Result<notify::RecommendedWatcher, String> {
            matches_pass_map(build)?;
            let dirs: Vec<&Path> = (build.watch_dirs.iter().copied())
                .chain([SLANG_DIR])
                .map(Path::new)
                .collect();
            // A binary moved off the build machine has no shaders to watch.
            if let Some(missing) = dirs.iter().find(|dir| !dir.is_dir()) {
                return Err(format!("{} is not a directory", missing.display()));
            }
            let (changed, rx) = mpsc::channel();
            let mut watcher =
                notify::recommended_watcher(FileWatcher { changed }).map_err(|e| e.to_string())?;
            for dir in &dirs {
                watcher
                    .watch(dir, notify::RecursiveMode::Recursive)
                    .map_err(|e| format!("{}: {e}", dir.display()))?;
            }
            std::thread::Builder::new()
                .name("shader_hot_reload".into())
                .spawn(move || {
                    let Some(global) = slang::GlobalSession::new() else {
                        tracing::error!(target: "shader_file_watcher", "failed to create slang session, hot reloading is disabled");
                        return;
                    };
                    let options = compile::Options {
                        search_paths: build.search_paths,
                        defines: build.defines,
                        optimization: build.optimization,
                        debug_info: build.debug_info,
                    };
                    while let Ok(first) = rx.recv() {
                        std::thread::sleep(Duration::from_millis(50));
                        let changed: Vec<PathBuf> = iter::once(first).chain(rx.try_iter()).collect();
                        reload(build, &global, &options, &changed);
                    }
                })
                .map_err(|e| e.to_string())?;
            Ok(watcher)
        };
        match watch() {
            Ok(watcher) => Some(watcher),
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "shader hot reload is off: {e}");
                None
            }
        }
    });
}

/// Whether a changed file can change a pass: any file but hidden ones and editors' backups and
/// swap files.
fn may_change_passes(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    !(name.starts_with('.')
        || name.ends_with('~')
        || [".swp", ".swx", ".tmp", ".bak"]
            .iter()
            .any(|ext| name.ends_with(ext))
        || name.chars().all(|c| c.is_ascii_digit()))
}

struct FileWatcher {
    changed: mpsc::Sender<PathBuf>,
}

impl EventHandler for FileWatcher {
    fn handle_event(&mut self, e: notify::Result<notify::Event>) {
        let event = match e {
            Ok(event) => event,
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "{e}");
                return;
            }
        };

        if !(event.kind.is_modify() || event.kind.is_create()) {
            return;
        }

        for path in event.paths {
            if may_change_passes(&path) {
                let _ = self.changed.send(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_forwards_modified_and_created_files_but_backups() {
        use notify::{
            Event, EventKind,
            event::{AccessKind, CreateKind, ModifyKind, RemoveKind},
        };
        let (changed, received) = mpsc::channel();
        let mut watcher = FileWatcher { changed };
        for (kind, paths) in [
            (
                EventKind::Modify(ModifyKind::Any),
                &[
                    "/s/passes/a.slang",
                    "/s/include/b.slangh",
                    "/s/.a.slang.swp",
                    "/s/a.slang~",
                ][..],
            ),
            (
                EventKind::Create(CreateKind::File),
                &["/s/include/new.h", "/s/4913"],
            ),
            (EventKind::Remove(RemoveKind::File), &["/s/passes/a.slang"]),
            (EventKind::Access(AccessKind::Any), &["/s/passes/a.slang"]),
        ] {
            let event = paths
                .iter()
                .fold(Event::new(kind), |e, p| e.add_path(p.into()));
            watcher.handle_event(Ok(event));
        }
        watcher.handle_event(Err(notify::Error::generic("watch failed")));
        drop(watcher);
        assert_eq!(
            received.iter().collect::<Vec<_>>(),
            [
                PathBuf::from("/s/passes/a.slang"),
                PathBuf::from("/s/include/b.slangh"),
                PathBuf::from("/s/include/new.h")
            ]
        );
    }
}
