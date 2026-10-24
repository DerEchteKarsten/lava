//! The compaction of acceleration structures as a step per recording: a `Compactor` owns the structures it is given, queries their compacted sizes, compacts them once the GPU wrote the sizes, and hands the compacted structures back once their compaction completed
use std::sync::Arc;

use crate::{
    command_buffer::CommandBuffer,
    error::Result,
    query::{CompactedSize, QueryPool},
    queue::{HasCompute, timeline::poll},
    vkobjects::acceleration_structure::{AccelerationStructure, Level},
};

/// Compacts acceleration structures over a few frames, the six steps of compaction (query the
/// sizes, wait for them, create the compacted structures, compact, wait, swap) driven by one
/// [`Self::record`] per frame:
/// 1. [`Self::add`] a structure with `allow_compaction` whose build was recorded (it may still
///    be in flight), with a key of the caller's (an index, a mesh id).
/// 2. Every [`Self::record`] queries the sizes of the structures added since the last one,
///    compacts those whose sizes the GPU wrote, and moves those whose compaction completed to
///    [`Self::take_finished`], with their keys. It never waits.
/// 3. Use a finished structure instead of its source from then on (rebuild the TLASes that name
///    it: its address is new); the compactor drops the source, which is destroyed once no frame
///    reaches it.
///
/// Until then the compactor keeps the source alive and traceable ([`Self::iter`]). A recording
/// that fails retries its steps in the next one.
pub struct Compactor<L: Level, K = ()> {
    entries: Vec<Entry<L, K>>,
    finished: Vec<(K, AccelerationStructure<L>)>,
}

struct Entry<L: Level, K> {
    key: K,
    source: AccelerationStructure<L>,
    step: Step<L>,
}

enum Step<L: Level> {
    /// Its size isn't queried yet.
    Added,
    /// Its size is query `index` of `pool`, recorded at `at` (timeline, value).
    Queried {
        pool: Arc<QueryPool<CompactedSize>>,
        index: u32,
        at: (usize, u64),
    },
    /// Its compaction into `compacted` was recorded at `at`.
    Compacting {
        compacted: AccelerationStructure<L>,
        at: (usize, u64),
    },
}

/// Whether the submission at `(timeline, value)` completed (polled, never waited for).
fn completed((timeline, value): (usize, u64)) -> bool {
    poll(timeline).is_ok_and(|done| done >= value)
}

impl<L: Level, K> Default for Compactor<L, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: Level, K> Compactor<L, K> {
    pub fn new() -> Self {
        Compactor {
            entries: Vec::new(),
            finished: Vec::new(),
        }
    }

    /// Takes `structure` (with `allow_compaction`) to compact; panics otherwise.
    #[track_caller]
    pub fn add(&mut self, key: K, structure: AccelerationStructure<L>) {
        assert!(
            structure.flags().allow_compaction,
            "a compactor compacts structures built with `allow_compaction`"
        );
        self.entries.push(Entry {
            key,
            source: structure,
            step: Step::Added,
        });
    }

    /// The structures it holds that aren't finished, by key: the sources, until their
    /// compacted structures are taken.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &AccelerationStructure<L>)> {
        self.entries.iter().map(|e| (&e.key, &e.source))
    }

    /// How many structures it holds that aren't finished.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Records the next step of every structure into `cmd`: size queries of the added ones
    /// that were built, compactions of those whose sizes are written; and finishes those whose
    /// compaction completed. A creation that fails (out of memory) leaves the structure where
    /// it was and returns the error after the other steps.
    pub fn record<Q: HasCompute>(&mut self, cmd: &mut CommandBuffer<'_, Q>) -> Result<()> {
        let at = (cmd.rec.timeline, cmd.rec.value);
        let mut error = None;
        let mut entries = Vec::with_capacity(self.entries.len());
        for mut entry in std::mem::take(&mut self.entries) {
            entry.step = match entry.step {
                Step::Compacting { compacted, at } if completed(at) => {
                    if compacted.is_built() {
                        self.finished.push((entry.key, compacted));
                        continue;
                    }
                    // Its recording failed: compact again.
                    Step::Added
                }
                Step::Queried { pool, index, at: queried } => {
                    let mut size = [0];
                    match pool.read(index, &mut size) {
                        Ok(true) => match entry.source.compacted(size[0]) {
                            Ok(compacted) => {
                                cmd.compact(&entry.source, &compacted);
                                Step::Compacting { compacted, at }
                            }
                            Err(err) => {
                                error.get_or_insert(err);
                                Step::Queried { pool, index, at: queried }
                            }
                        },
                        // Done without a size: its recording failed.
                        Ok(false) if completed(queried) => Step::Added,
                        Ok(false) => Step::Queried { pool, index, at: queried },
                        Err(err) => {
                            error.get_or_insert(err);
                            Step::Queried { pool, index, at: queried }
                        }
                    }
                }
                step => step,
            };
            entries.push(entry);
        }
        self.entries = entries;

        let added = |e: &Entry<L, K>| matches!(e.step, Step::Added) && e.source.is_built();
        let count = self.entries.iter().filter(|e| added(e)).count() as u32;
        if count > 0 {
            match QueryPool::compacted_sizes(count) {
                Ok(pool) => {
                    let pool = Arc::new(pool);
                    let sources: Vec<_> = (self.entries.iter()).filter(|e| added(e)).map(|e| &e.source).collect();
                    cmd.query_compacted_sizes(&sources, &pool, 0);
                    let mut index = 0;
                    for entry in self.entries.iter_mut().filter(|e| added(e)) {
                        entry.step = Step::Queried {
                            pool: pool.clone(),
                            index,
                            at,
                        };
                        index += 1;
                    }
                }
                Err(err) => {
                    error.get_or_insert(err);
                }
            }
        }
        error.map_or(Ok(()), Err)
    }

    /// The compacted structures whose compaction completed, with their keys, taken out.
    pub fn take_finished(&mut self) -> Vec<(K, AccelerationStructure<L>)> {
        std::mem::take(&mut self.finished)
    }
}
