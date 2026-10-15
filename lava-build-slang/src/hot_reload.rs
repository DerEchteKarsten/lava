//! lava-build-slang's hot reload (the `hot-reload` feature, a debugging aid): recompiles the passes whose sources change and hands their SPIR-V to lava, which replaces their pipelines right away
use std::{
    iter,
    path::{Path, PathBuf},
    sync::{OnceLock, mpsc},
    time::Duration,
};

use lava::pipeline::{self, pass_map};
use notify::EventHandler;

use crate::{
    Optimization, SLANG_DIR,
    compile::{self, slang},
};

/// How the build script compiled the passes, which the hot reload compiles them with again:
/// generated next to the bindings (`$OUT_DIR/lava/hot_reload.rs`) and included by [`watch!`].
pub struct Build {
    /// The absolute include directories.
    pub search_paths: &'static [&'static str],
    /// The preprocessor macros (name, value).
    pub defines: &'static [(&'static str, &'static str)],
    /// The absolute pass and include directories.
    pub watch_dirs: &'static [&'static str],
    pub optimization: Optimization,
    /// One per pass of `PASS_MAP`, in its order.
    pub passes: &'static [BuiltPass],
}

pub struct BuiltPass {
    /// The absolute path of its `.slang` file.
    pub source: &'static str,
    /// `compile::layout_hash` of the push constants, specialization constants and entry points
    /// the bindings were generated from; a reload that changes it is refused, since the Rust
    /// side can't follow.
    pub layout: u64,
}

/// Starts the hot reload of the passes compiled by the calling crate's build script: call it
/// in the crate that includes the bindings (it reads that crate's `OUT_DIR`), after
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

/// Watches the build's directories and [`SLANG_DIR`], and compiles changed passes on a thread
/// of its own; [`watch!`] calls it with the build of the calling crate. Later calls do nothing.
pub fn watch(build: &'static Build) {
    use notify::Watcher;

    static WATCHER: OnceLock<Option<notify::RecommendedWatcher>> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let watch = || -> std::result::Result<notify::RecommendedWatcher, String> {
            // The source directories of the build machine: a moved binary has no shaders to
            // watch, and keeps the pipelines it started with.
            if build.passes.len() != pass_map().passes.len() {
                return Err("its build isn't the one of lava's `PASS_MAP`".to_string());
            }
            let dirs: Vec<&Path> = (build.watch_dirs.iter().copied())
                .chain([SLANG_DIR])
                .map(Path::new)
                .collect();
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
                .spawn(move || compile_changed_passes(build, rx))
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
            if path.extension().is_some_and(|ext| ext == "slang") {
                let _ = self.changed.send(path);
            }
        }
    }
}

fn compile_changed_passes(build: &Build, changed: mpsc::Receiver<PathBuf>) {
    let Some(global) = slang::GlobalSession::new() else {
        tracing::error!(target: "shader_file_watcher", "failed to create slang session, hot reloading is disabled");
        return;
    };

    while let Ok(first) = changed.recv() {
        std::thread::sleep(Duration::from_millis(50));
        let paths: Vec<PathBuf> = iter::once(first).chain(changed.try_iter()).collect();

        for pass_index in passes_to_reload(&paths, build.passes) {
            let pass = &build.passes[pass_index];

            let reloaded = compile_pass(&global, build, pass_index).and_then(|spirv| {
                tracing::info!(target: "shader_file_watcher", "Reloading pass `{}`", pass.source);
                pipeline::reload_pass(pass_index, spirv.as_slice()).map_err(|e| e.to_string())
            });
            if let Err(e) = reloaded {
                tracing::error!(target: "shader_file_watcher", "{}:\n{e}", pass.source);
            }
        }
    }
}

/// Indices of the passes that have to be recompiled after `changed` files were modified:
/// the passes whose own source changed, or every pass as soon as a non-pass file (an include)
/// is among them.
fn passes_to_reload(changed: &[PathBuf], passes: &[BuiltPass]) -> Vec<usize> {
    let is_source_of = |path: &PathBuf, pass: &BuiltPass| path == Path::new(pass.source);
    let all = changed
        .iter()
        .any(|path| !passes.iter().any(|pass| is_source_of(path, pass)));
    passes
        .iter()
        .enumerate()
        .filter(|(_, pass)| all || changed.iter().any(|path| is_source_of(path, pass)))
        .map(|(index, _)| index)
        .collect()
}

/// The pass's new SPIR-V, unless its push constants, specialization constants or entry
/// points changed, which only `cargo build` can follow.
fn compile_pass(
    global: &slang::GlobalSession,
    build: &Build,
    pass_index: usize,
) -> std::result::Result<slang::Blob, String> {
    let pass = &build.passes[pass_index];
    let options = compile::Options {
        search_paths: build.search_paths,
        defines: build.defines,
        optimization: build.optimization,
    };
    let session = compile::session(global, &options);
    let program = compile::link(&session, pass.source).map_err(|e| e.to_string())?;
    let layout = program.layout(0).map_err(|e| e.to_string())?;
    if compile::layout_hash(layout) != pass.layout {
        return Err(
            "its push constants, specialization constants or entry points changed, \
                    which the generated bindings can't follow: run `cargo build`"
                .to_string(),
        );
    }
    program.target_code(0).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(source: &'static str) -> BuiltPass {
        BuiltPass { source, layout: 0 }
    }

    #[test]
    fn changed_pass_sources_reload_only_those_passes() {
        let passes = [
            pass("/repo/shaders/passes/a.slang"),
            pass("/repo/shaders/passes/b.slang"),
            pass("/repo/shaders/passes/c.slang"),
        ];
        let changed = [
            PathBuf::from("/repo/shaders/passes/c.slang"),
            PathBuf::from("/repo/shaders/passes/a.slang"),
        ];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 2]);
        assert!(passes_to_reload(&[], &passes).is_empty());
    }

    #[test]
    fn a_changed_include_reloads_every_pass() {
        let passes = [pass("/repo/shaders/passes/a.slang"), pass("/repo/shaders/passes/b.slang")];
        let changed = [
            PathBuf::from("/repo/shaders/passes/a.slang"),
            PathBuf::from("/repo/shaders/include/math.slang"),
        ];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 1]);
    }

    #[test]
    fn pass_sources_match_on_whole_path_components() {
        // `xa.slang` is not pass `a.slang`; it counts as an unknown file, i.e. an include.
        let passes = [pass("/repo/shaders/passes/a.slang"), pass("/repo/shaders/passes/b.slang")];
        let changed = [PathBuf::from("/repo/shaders/passes/xa.slang")];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 1]);
    }

    fn watch(events: Vec<notify::Result<notify::Event>>) -> Vec<PathBuf> {
        let (changed, received) = mpsc::channel();
        let mut watcher = FileWatcher { changed };
        for event in events {
            watcher.handle_event(event);
        }
        drop(watcher);
        received.iter().collect()
    }

    fn event(kind: notify::EventKind, paths: &[&str]) -> notify::Result<notify::Event> {
        let mut event = notify::Event::new(kind);
        for path in paths {
            event = event.add_path(PathBuf::from(path));
        }
        Ok(event)
    }

    #[test]
    fn watcher_forwards_modified_and_created_slang_files() {
        use notify::{
            EventKind,
            event::{CreateKind, ModifyKind},
        };
        let paths = watch(vec![
            event(
                EventKind::Modify(ModifyKind::Any),
                &["/s/passes/a.slang", "/s/notes.txt"],
            ),
            event(
                EventKind::Create(CreateKind::File),
                &["/s/include/new.slang"],
            ),
        ]);
        assert_eq!(
            paths,
            [
                PathBuf::from("/s/passes/a.slang"),
                PathBuf::from("/s/include/new.slang")
            ]
        );
    }

    #[test]
    fn watcher_ignores_other_events_and_errors() {
        use notify::{
            EventKind,
            event::{AccessKind, RemoveKind},
        };
        let paths = watch(vec![
            event(EventKind::Remove(RemoveKind::File), &["/s/passes/a.slang"]),
            event(EventKind::Access(AccessKind::Any), &["/s/passes/a.slang"]),
            Err(notify::Error::generic("watch failed")),
        ]);
        assert!(paths.is_empty());
    }
}
