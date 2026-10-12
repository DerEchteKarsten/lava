//! Shader hot reload (the `hot-reload` feature, a debugging aid): recompiles the passes whose sources change and replaces their pipelines right away
use std::{
    iter,
    path::{Path, PathBuf},
    sync::{OnceLock, mpsc},
    time::Duration,
};

use notify::EventHandler;
use shader_slang as slang;

use crate::{
    bindings::PASS_MAP,
    pipeline::{PassEntry, PipelineManager},
    slang_compile::{self, SHADER_DIR},
};

/// Watches the shader directory and compiles changed passes on a thread of its own.
pub(crate) fn watch(manager: &'static PipelineManager) {
    use notify::Watcher;

    static WATCHER: OnceLock<Option<notify::RecommendedWatcher>> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let watch = || -> std::result::Result<notify::RecommendedWatcher, String> {
            // The source directory of the build machine: a moved binary has no shaders to
            // watch, and keeps the pipelines it started with.
            let dirs = ["passes", "include"].map(|dir| Path::new(SHADER_DIR).join(dir));
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
                .spawn(move || compile_changed_passes(manager, rx))
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

fn compile_changed_passes(manager: &'static PipelineManager, changed: mpsc::Receiver<PathBuf>) {
    let Some(global) = slang::GlobalSession::new() else {
        tracing::error!(target: "shader_file_watcher", "failed to create slang session, hot reloading is disabled");
        return;
    };

    while let Ok(first) = changed.recv() {
        std::thread::sleep(Duration::from_millis(50));
        let paths: Vec<PathBuf> = iter::once(first).chain(changed.try_iter()).collect();

        for pass_index in passes_to_reload(&paths, &PASS_MAP) {
            let pass = &PASS_MAP[pass_index];

            let reloaded = compile_pass(&global, pass_index).and_then(|spirv| {
                tracing::info!(target: "shader_file_watcher", "Reloading pass `{}`", pass.source);
                manager
                    .reload_pass(pass_index, spirv.as_slice())
                    .map_err(|e| e.to_string())
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
fn passes_to_reload(changed: &[PathBuf], passes: &[PassEntry]) -> Vec<usize> {
    let is_source_of = |path: &PathBuf, pass: &PassEntry| path.ends_with(pass.source);
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
    pass_index: usize,
) -> std::result::Result<slang::Blob, String> {
    let pass = &PASS_MAP[pass_index];
    let options = slang_compile::Options {
        test_passes: cfg!(feature = "test-passes"),
        profiling: cfg!(feature = "profiling"),
        pass_index,
    };
    let session = slang_compile::session(global, &options);
    let program = slang_compile::link(&session, pass.source).map_err(|e| e.to_string())?;
    let layout = program.layout(0).map_err(|e| e.to_string())?;
    if slang_compile::layout_hash(layout) != pass.layout {
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
    use crate::pipeline::PassKind;

    fn pass(source: &'static str) -> PassEntry {
        PassEntry {
            name: "",
            path: "",
            source,
            kind: PassKind::Compute { entry: "main\0" },
            index: 0,
            constants: &[],
            constants_size: 0,
            regions: &[],
            layout: 0,
        }
    }

    #[test]
    fn changed_pass_sources_reload_only_those_passes() {
        let passes = [
            pass("passes/a.slang"),
            pass("passes/b.slang"),
            pass("passes/c.slang"),
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
        let passes = [pass("passes/a.slang"), pass("passes/b.slang")];
        let changed = [
            PathBuf::from("/repo/shaders/passes/a.slang"),
            PathBuf::from("/repo/shaders/include/math.slang"),
        ];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 1]);
    }

    #[test]
    fn pass_sources_match_on_whole_path_components() {
        // `xa.slang` is not pass `a.slang`; it counts as an unknown file, i.e. an include.
        let passes = [pass("passes/a.slang"), pass("passes/b.slang")];
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
