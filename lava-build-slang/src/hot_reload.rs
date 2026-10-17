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

/// How the build script compiled the passes, generated as `$OUT_DIR/lava/hot_reload.rs`.
pub struct Build {
    pub search_paths: &'static [&'static str],
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
    /// The `compile::layout_hash` the bindings were generated from; a reload that changes it is
    /// refused.
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

/// Watches the build's directories and [`SLANG_DIR`] and compiles changed passes on a thread of
/// its own. Later calls do nothing.
pub fn watch(build: &'static Build) {
    use notify::Watcher;

    static WATCHER: OnceLock<Option<notify::RecommendedWatcher>> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let watch = || -> std::result::Result<notify::RecommendedWatcher, String> {
            if build.passes.len() != pass_map().passes.len() {
                return Err("its build isn't the one of lava's `PASS_MAP`".to_string());
            }
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
                    };

                    while let Ok(first) = rx.recv() {
                        std::thread::sleep(Duration::from_millis(50));
                        let changed: Vec<PathBuf> = iter::once(first).chain(rx.try_iter()).collect();

                        // A changed file that is no pass's source is an include: reload every pass.
                        let is_source_of = |path: &PathBuf, pass: &BuiltPass| path == Path::new(pass.source);
                        let all = changed
                            .iter()
                            .any(|path| !build.passes.iter().any(|pass| is_source_of(path, pass)));
                        for (pass_index, pass) in build.passes.iter().enumerate() {
                            if !(all || changed.iter().any(|path| is_source_of(path, pass))) {
                                continue;
                            }
                            let reloaded = (|| {
                                let session = compile::session(&global, &options);
                                let program =
                                    compile::link(&session, pass.source).map_err(|e| e.to_string())?;
                                let layout = program.layout(0).map_err(|e| e.to_string())?;
                                if compile::layout_hash(layout) != pass.layout {
                                    return Err(
                                        "its push constants, specialization constants or entry points changed, \
                                                which the generated bindings can't follow: run `cargo build`"
                                            .to_string(),
                                    );
                                }
                                let spirv = program.target_code(0).map_err(|e| e.to_string())?;
                                tracing::info!(target: "shader_file_watcher", "Reloading pass `{}`", pass.source);
                                pipeline::reload_pass(pass_index, spirv.as_slice()).map_err(|e| e.to_string())
                            })();
                            if let Err(e) = reloaded {
                                tracing::error!(target: "shader_file_watcher", "{}:\n{e}", pass.source);
                            }
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_forwards_only_modified_and_created_slang_files() {
        use notify::{
            Event, EventKind,
            event::{AccessKind, CreateKind, ModifyKind, RemoveKind},
        };
        let (changed, received) = mpsc::channel();
        let mut watcher = FileWatcher { changed };
        for (kind, paths) in [
            (EventKind::Modify(ModifyKind::Any), &["/s/passes/a.slang", "/s/notes.txt"][..]),
            (EventKind::Create(CreateKind::File), &["/s/include/new.slang"]),
            (EventKind::Remove(RemoveKind::File), &["/s/passes/a.slang"]),
            (EventKind::Access(AccessKind::Any), &["/s/passes/a.slang"]),
        ] {
            let event = paths.iter().fold(Event::new(kind), |e, p| e.add_path(p.into()));
            watcher.handle_event(Ok(event));
        }
        watcher.handle_event(Err(notify::Error::generic("watch failed")));
        drop(watcher);
        assert_eq!(
            received.iter().collect::<Vec<_>>(),
            [
                PathBuf::from("/s/passes/a.slang"),
                PathBuf::from("/s/include/new.slang")
            ]
        );
    }
}
