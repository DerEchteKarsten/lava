//! Scope observers: what `CommandBuffer::measure` tells when a scope, pass or rendering of a recording is entered and exited, as a tracing subscriber is told about spans, with a recorder that can only record queries there
use crate::{
    PipelineStageFlags2,
    command_buffer::{Recording, query_bit},
    query::{QueryPool, Scoped, Timestamp},
};

/// A scope as an observer sees it.
#[derive(Clone, Copy, Debug)]
pub struct ScopeInfo {
    pub name: &'static str,
    /// 0 for the root `measure` names.
    pub depth: u16,
    /// A compute or ray tracing pass, or a rendering (`RasterBuilder::record`), named by its
    /// generated struct or its `record` name; else a `CommandBuffer::scope` or the root.
    pub pass: bool,
}

/// Watches the scopes of a recording while `CommandBuffer::measure` attaches it: the root,
/// every `CommandBuffer::scope`, and every pass and rendering, which are scopes of their own.
/// Each is entered and exited once, in nesting order. Its hooks can record queries that
/// bracket the scope's commands: what `on_enter` records comes before them (a pass's barriers
/// included), what `on_exit` records after them. Passes don't nest, and neither renderings nor
/// passes contain other scopes.
pub trait ScopeObserver {
    fn on_enter(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo);
    fn on_exit(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo);
}

/// What a [`ScopeObserver`] can record at the edges of a scope: queries, nothing else.
pub struct QueryRecorder<'a> {
    pub(crate) cmd: &'a mut Recording,
}

impl QueryRecorder<'_> {
    /// See `CommandBuffer::write_timestamp`.
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
    pub fn begin_query<K: Scoped>(&mut self, pool: &QueryPool<K>, index: u32) {
        assert!(index < pool.count(), "query {index} of a pool of {}", pool.count());
        self.cmd.begin_query(pool, index);
    }

    pub fn end_query<K: Scoped>(&mut self, pool: &QueryPool<K>, index: u32) {
        self.cmd.end_query(pool, index);
    }

    /// Whether a query of kind `K` is active, such as a `CommandBuffer::query` around the
    /// scope; another one can't begin then.
    pub fn is_active<K: Scoped>(&self) -> bool {
        self.cmd.active_queries & query_bit(K::TYPE) != 0
    }
}
