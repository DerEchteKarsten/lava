//! GPU objects as lava owns them: each lives in an `Arc` of its internal (buffers, images, acceleration structures, samplers), whose `Header` counts it as a lava user until its drop finished and keeps what references between objects need (the buffers that reference it, and dirty bits a write marks them with), and the typed strong references (`Strong`) a referencing buffer holds its children by
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
};

use smallvec::SmallVec;

use crate::vk::{AccessFlags2 as A, PipelineStageFlags2 as S2};
#[cfg(feature = "acceleration-structure")]
use crate::vkobjects::acceleration_structure::StructureInternal;
use crate::{
    bindless::SamplerInternal,
    buffer::BufferInternal,
    command_buffer::{Barriers, access_state::Tracker},
    global::Live,
    image::ImageInternal,
    queue::lock,
    reference::KeepAlive,
};

/// The dirty bit of a layout: an image of the object or below it may be outside `GENERAL`.
pub(crate) const DIRTY_LAYOUT: u32 = 1 << 8;

/// What every object's internal carries.
#[derive(Debug)]
pub(crate) struct Header {
    /// Bit `t` (below 8): this object or something it references has a write recorded on
    /// timeline `t` since a walk last made it visible; [`DIRTY_LAYOUT`]: an image of it or
    /// below it may be outside `GENERAL`. Set by [`mark_dirty`], cleared by walks.
    pub(crate) dirty: AtomicU32,
    /// The buffers that reference it (`reference::KeepAlive`), once each.
    pub(crate) parents: Mutex<SmallVec<[Weak<dyn Object>; 2]>>,
    /// `parents`' length, so writes of objects nothing references skip the lock.
    pub(crate) parent_count: AtomicU32,
    /// Somewhere below this object is a write reference: a command that reaches it may write
    /// through it, so it walks to the writers even when nothing is dirty. Never cleared while
    /// the object lives.
    pub(crate) writes_below: AtomicBool,
    /// Counts the object as a lava user, so `shutdown` refuses while it lives; dropped last.
    _live: Live,
}

impl Header {
    pub(crate) fn new(live: Live) -> Self {
        Self {
            dirty: AtomicU32::new(0),
            parents: Mutex::default(),
            parent_count: AtomicU32::new(0),
            writes_below: AtomicBool::new(false),
            _live: live,
        }
    }

    pub(crate) fn has_parents(&self) -> bool {
        self.parent_count.load(Relaxed) > 0
    }
}

/// What lava's object internals share, for the references between them.
pub(crate) trait Object: Send + Sync + 'static {
    fn header(&self) -> &Header;
    /// Its references to other objects: a buffer's whose elements hold some.
    fn keep_alive(&self) -> Option<&Mutex<KeepAlive>> {
        None
    }
    /// What a write (`stage`, `access`) of an object it references waits for of its own
    /// tracked accesses: the commands that read the object through it recorded only reads of
    /// it (a walk of a clean buffer reaches nothing). None for objects that reference nothing.
    fn reference_hazards(&self, t: &Tracker, log: &[Barriers], stage: S2, access: A) -> Hazard {
        let _ = (t, log, stage, access);
        None
    }
}

/// The stages and accesses a barrier waits for, if any.
pub(crate) type Hazard = Option<(S2, A)>;

/// What a write (`stage`, `access`) of `object` waits for of the objects above it, which
/// commands read it through (`Object::reference_hazards` of every ancestor).
pub(crate) fn ancestor_hazards(
    object: &dyn Object,
    t: &Tracker,
    log: &[Barriers],
    stage: S2,
    access: A,
) -> Hazard {
    if !object.header().has_parents() {
        return None;
    }
    let mut found: Hazard = None;
    let mut add = |hazard: Hazard| {
        if let Some((s, a)) = hazard {
            let (stages, accesses) = found.get_or_insert((S2::NONE, A::NONE));
            *stages |= s;
            *accesses |= a;
        }
    };
    for parent in parents(object) {
        add(parent.reference_hazards(t, log, stage, access));
        add(ancestor_hazards(&*parent, t, log, stage, access));
    }
    found
}

/// A strong reference to an object a referencing buffer holds, typed, since a walk turns it
/// into a command's access of it.
#[derive(Clone, Debug)]
pub(crate) enum Strong {
    Buffer(Arc<BufferInternal>),
    Image(Arc<ImageInternal>),
    /// No barriers: the GPU never writes a sampler.
    Sampler(Arc<SamplerInternal>),
    #[cfg(feature = "acceleration-structure")]
    Structure(Arc<StructureInternal>),
}

impl Strong {
    pub(crate) fn object(&self) -> &dyn Object {
        match self {
            Strong::Buffer(buffer) => &**buffer,
            Strong::Image(image) => &**image,
            Strong::Sampler(sampler) => &**sampler,
            #[cfg(feature = "acceleration-structure")]
            Strong::Structure(structure) => &**structure,
        }
    }

    /// Its address, which tells two references of the same object apart from others.
    pub(crate) fn key(&self) -> usize {
        std::ptr::from_ref(self.object()) as *const () as usize
    }
}

/// Marks `object` with `bits`, and its parents, their parents and so on: stops at a parent that
/// has every bit already, whose ancestors have them too.
pub(crate) fn mark_dirty(object: &dyn Object, bits: u32) {
    let header = object.header();
    header.dirty.fetch_or(bits, Relaxed);
    if !header.has_parents() {
        return;
    }
    for parent in parents(object) {
        if parent.header().dirty.fetch_or(bits, Relaxed) & bits != bits {
            mark_dirty(&*parent, bits);
        }
    }
}

/// Sets `writes_below` on `object` and its ancestors, stopping where it is set already.
pub(crate) fn mark_writes_below(object: &dyn Object) {
    if object.header().writes_below.swap(true, Relaxed) {
        return;
    }
    for parent in parents(object) {
        mark_writes_below(&*parent);
    }
}

/// The live parents of `object`, collected so no lock is held while they are visited.
fn parents(object: &dyn Object) -> SmallVec<[Arc<dyn Object>; 2]> {
    (lock(&object.header().parents).iter())
        .filter_map(Weak::upgrade)
        .collect()
}
