//! Scope observers: what `CommandBuffer::measure` tells when a scope, pass or rendering of a recording is entered and exited, as a tracing subscriber is told about spans, with a recorder that can only record the queries the recording's family can
use std::marker::PhantomData;

use crate::{
    PipelineStageFlags2,
    command_buffer::Recording,
    query::{QueryPool, RecordedOn, Scoped, Timestamp},
    queue::{Graphics, QueueKind},
};

/// A scope as an observer sees it, borrowed for the hook's call: an observer that keeps the
/// name copies it.
#[derive(Clone, Copy, Debug)]
pub struct ScopeInfo<'a> {
    pub name: &'a str,
    /// 0 for the root `measure` names.
    pub depth: u16,
    /// A pass or rendering, named by its generated struct or `record` name; else a
    /// `CommandBuffer::scope` or the root.
    pub pass: bool,
}

/// Watches the root, every `CommandBuffer::scope` and every pass and rendering of a recording
/// on a family of kind `K` while `CommandBuffer::measure` attaches it, each entered and exited
/// once in nesting order. What `on_enter` records comes before the scope's commands, but after
/// a pass's or rendering's barriers (which its scope, debug label and timestamps leave out),
/// what `on_exit` records after them; neither is inside a rendering.
pub trait ScopeObserver<K: QueueKind = Graphics> {
    fn on_enter(&mut self, queries: &mut QueryRecorder<'_, '_, K>, scope: &ScopeInfo<'_>);
    fn on_exit(&mut self, queries: &mut QueryRecorder<'_, '_, K>, scope: &ScopeInfo<'_>);
}

/// What a [`ScopeObserver`] can record at the edges of a scope: the queries a family of kind `K`
/// can record, nothing else.
pub struct QueryRecorder<'a, 'p, K: QueueKind = Graphics> {
    pub(crate) cmd: &'a mut Recording<'p>,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> QueryRecorder<'_, '_, K> {
    pub fn write_timestamp(
        &mut self,
        pool: &QueryPool<Timestamp>,
        index: u32,
        stage: PipelineStageFlags2,
    ) {
        self.cmd.write_timestamp(pool, index, stage);
    }

    /// Begins query `index` of `pool`, which the observer ends at the scope's exit or an outer
    /// scope's. One query of a kind can be active at a time (see [`Self::is_active`]).
    pub fn begin_query<Q: Scoped + RecordedOn<K>>(&mut self, pool: &QueryPool<Q>, index: u32) {
        self.cmd.begin_query(pool, index);
    }

    /// Ends query `index` of `pool`; panics unless it is the active query of its kind.
    pub fn end_query<Q: Scoped + RecordedOn<K>>(&mut self, pool: &QueryPool<Q>, index: u32) {
        self.cmd.end_query(pool, index);
    }

    /// Whether a query of kind `Q` is active, such as a `CommandBuffer::query` around the
    /// scope; another one can't begin then.
    pub fn is_active<Q: Scoped>(&self) -> bool {
        self.cmd.open_query::<Q>().is_some()
    }
}

/// The observer `measure` attached, with its family kind erased so the untyped `Recording` can
/// hold it.
pub(crate) trait Observer {
    fn on_enter(&mut self, cmd: &mut Recording, scope: &ScopeInfo<'_>);
    fn on_exit(&mut self, cmd: &mut Recording, scope: &ScopeInfo<'_>);
}

/// An observer of recordings on a family of kind `K`.
pub(crate) struct Typed<'o, K: QueueKind>(pub(crate) &'o mut dyn ScopeObserver<K>);

impl<K: QueueKind> Observer for Typed<'_, K> {
    fn on_enter(&mut self, cmd: &mut Recording, scope: &ScopeInfo<'_>) {
        let mut queries = QueryRecorder {
            cmd,
            _kind: PhantomData,
        };
        self.0.on_enter(&mut queries, scope);
    }

    fn on_exit(&mut self, cmd: &mut Recording, scope: &ScopeInfo<'_>) {
        let mut queries = QueryRecorder {
            cmd,
            _kind: PhantomData,
        };
        self.0.on_exit(&mut queries, scope);
    }
}
