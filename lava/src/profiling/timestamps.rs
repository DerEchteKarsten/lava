//! GPU timestamps: the timestamp pool of a frame slot, which times its root scope, every `begin_scope` group and every pass
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::{error::Result, state::Ctx};

const TIMESTAMPS: u32 = 256;

/// A timed region of one submission, in ns since its root scope ("frame") started.
#[derive(Clone, Debug)]
pub struct GpuScope {
    pub name: &'static str,
    pub depth: u16,
    pub start_ns: u64,
    pub end_ns: u64,
}

struct Scope {
    name: &'static str,
    depth: u16,
    /// Its start timestamp; the end is the next one.
    timestamp: u32,
}

/// The timestamp pool of one frame slot and the scopes recorded into it. A scope takes two
/// timestamps; a scope that doesn't fit is skipped.
pub struct TimestampQueries {
    pool: vk::QueryPool,
    valid_bits: u64,
    scopes: Vec<Scope>,
    /// Indices into `scopes` of the open scopes, `None` for a skipped one.
    open: Vec<Option<usize>>,
    next: u32,
    ticks: Vec<u64>,
    results: Vec<GpuScope>,
}

impl TimestampQueries {
    pub(crate) fn new(queue_family: u32) -> Result<Self> {
        let pool = unsafe {
            Ctx::device().create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(TIMESTAMPS),
                None,
            )?
        };
        let bits = Ctx::physical_device().queue_families[queue_family as usize]
            .handel
            .timestamp_valid_bits;
        Ok(Self {
            pool,
            valid_bits: if bits >= 64 { !0 } else { (1 << bits) - 1 },
            scopes: Vec::new(),
            open: Vec::new(),
            next: 0,
            ticks: Vec::new(),
            results: Vec::new(),
        })
    }

    pub(crate) fn reset(&mut self, cmd: vk::CommandBuffer) {
        unsafe { Ctx::device().cmd_reset_query_pool(cmd, self.pool, 0, TIMESTAMPS) };
        self.scopes.clear();
        self.open.clear();
        self.next = 0;
    }

    pub(crate) fn begin(&mut self, cmd: vk::CommandBuffer, name: &'static str) {
        if self.next + 2 > TIMESTAMPS {
            tracing::warn!("more than {} GPU scopes in a frame", TIMESTAMPS / 2);
            self.open.push(None);
            return;
        }
        let timestamp = self.next;
        self.next += 2;
        unsafe {
            Ctx::device().cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::NONE,
                self.pool,
                timestamp,
            )
        };
        self.open.push(Some(self.scopes.len()));
        self.scopes.push(Scope {
            name,
            depth: self.open.len() as u16 - 1,
            timestamp,
        });
    }

    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer) {
        let Some(Some(index)) = self.open.pop() else {
            return;
        };
        unsafe {
            Ctx::device().cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                self.scopes[index].timestamp + 1,
            )
        };
    }

    /// Reads the scopes of the last submission, which must have finished.
    pub(crate) fn read(&mut self) -> Result<()> {
        self.results.clear();
        if self.scopes.is_empty() {
            return Ok(());
        }
        self.ticks.resize(self.next as usize, 0);
        unsafe {
            Ctx::device().get_query_pool_results(
                self.pool,
                0,
                &mut self.ticks,
                vk::QueryResultFlags::TYPE_64,
            )?
        };
        let period = Ctx::device_properties().limits.timestamp_period as f64;
        let (ticks, valid_bits) = (&self.ticks, self.valid_bits);
        let base = ticks[self.scopes[0].timestamp as usize] & valid_bits;
        let ns = |tick: u64| ((tick & valid_bits).saturating_sub(base) as f64 * period) as u64;
        self.results
            .extend(self.scopes.iter().map(|scope| GpuScope {
                name: scope.name,
                depth: scope.depth,
                start_ns: ns(ticks[scope.timestamp as usize]),
                end_ns: ns(ticks[scope.timestamp as usize + 1]),
            }));
        Ok(())
    }

    pub(crate) fn results(&self) -> Option<&[GpuScope]> {
        (!self.results.is_empty()).then_some(&self.results)
    }
}

impl Drop for TimestampQueries {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_query_pool(self.pool, None) };
    }
}
