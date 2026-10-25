//! GPU objects as lava owns them: each lives in an `Arc` of its internal (buffers, images, acceleration structures), whose `Header` counts it as a lava user until its drop finished and keeps what references between objects need (the buffers that reference it, and dirty bits a write marks them with), and the typed strong references (`Strong`) a referencing buffer holds its children by
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
};

use smallvec::SmallVec;

use crate::{
    buffer::BufferInternal, global::Live, image::ImageInternal, queue::lock, reference::KeepAlive,
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
}

/// A strong reference to an object a referencing buffer holds, typed, since a walk turns it
/// into a command's access of it.
#[derive(Clone, Debug)]
pub(crate) enum Strong {
    Buffer(Arc<BufferInternal>),
    Image(Arc<ImageInternal>),
}

impl Strong {
    pub(crate) fn object(&self) -> &dyn Object {
        match self {
            Strong::Buffer(buffer) => &**buffer,
            Strong::Image(image) => &**image,
        }
    }

    /// Its address, which tells two references of the same object apart from others.
    pub(crate) fn key(&self) -> usize {
        match self {
            Strong::Buffer(buffer) => Arc::as_ptr(buffer) as *const () as usize,
            Strong::Image(image) => Arc::as_ptr(image) as *const () as usize,
        }
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
