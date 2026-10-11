//! Shader hot reload (the `hot-reload` feature, a debugging aid): recompiles the passes whose sources change and swaps their pipelines at the next frame
use std::{
    iter,
    path::{Path, PathBuf},
    sync::{OnceLock, atomic::Ordering, mpsc},
    time::Duration,
};

use notify::EventHandler;
use shader_slang as slang;

use crate::{
    bindings::PASS_MAP,
    command_buffer::{PIPELINES, PassEntry, PipelineManager},
    slang_compile::{self, SHADER_DIR},
};

/// Frames begun after a reload before the pipelines it replaced are destroyed. Frames begun
/// before the reload may still use them, so this has to exceed the frames any loop keeps in
/// flight, counted over every `FrameSlot`.
const RETIRED_FRAMES: u64 = 8;

/// Called by every `FrameSlot::begin`: swaps in the passes compiled since, and destroys the
/// pipelines swapped out `RETIRED_FRAMES` frames ago.
pub(crate) fn begin_frame() {
    let Some(manager) = PIPELINES.get() else {
        return;
    };
    let frame = manager.frame.fetch_add(1, Ordering::Relaxed);
    let pending = std::mem::take(&mut *manager.pending_reloads.lock().unwrap());
    let mut retired = manager.retired.lock().unwrap();
    retired.retain(|(at, _)| at + RETIRED_FRAMES > frame);
    for (pass_index, spirv) in pending {
        let replaced = manager.reload_pass(pass_index, &spirv);
        retired.extend(replaced.into_iter().map(|pipeline| (frame, pipeline)));
    }
}

/// Watches the shader directory and compiles changed passes on a thread of its own.
pub(crate) fn watch(manager: &'static PipelineManager) {
    use notify::Watcher;

    static WATCHER: OnceLock<notify::RecommendedWatcher> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let (changed, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("shader_hot_reload".into())
            .spawn(move || compile_changed_passes(manager, rx))
            .unwrap();

        let mut watcher = notify::recommended_watcher(FileWatcher { changed }).unwrap();
        for dir in ["passes", "include"] {
            watcher
                .watch(
                    &Path::new(SHADER_DIR).join(dir),
                    notify::RecursiveMode::Recursive,
                )
                .unwrap();
        }
        watcher
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

            match compile_pass(&global, pass_index) {
                Ok(spirv) => {
                    tracing::info!(target: "shader_file_watcher", "Reloading pass `{}`", pass.source);
                    match manager.pending_reloads.lock() {
                        Ok(mut pending) => pending.push((pass_index, spirv.as_slice().to_vec())),
                        Err(_) => {
                            tracing::error!(target: "shader_file_watcher", "failed to acquire lock on pending shader reloads")
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(target: "shader_file_watcher", "{}:\n{e}", pass.source);
                }
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

fn compile_pass(global: &slang::GlobalSession, pass_index: usize) -> slang::Result<slang::Blob> {
    let options = slang_compile::Options {
        test_passes: cfg!(feature = "test-passes"),
        profiling: cfg!(feature = "profiling"),
        pass_index,
    };
    let session = slang_compile::session(global, &options);
    slang_compile::link(&session, PASS_MAP[pass_index].source)?.target_code(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_buffer::PassKind;

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
