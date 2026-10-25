//! GPU objects as lava owns them: each lives in an `Arc` of its internal (buffers, images, acceleration structures, samplers), whose `Header` counts it as a lava user until its drop finished and keeps what references between objects need (the buffers that reference it, and dirty bits a write marks them with), the typed strong references (`Strong`) a referencing buffer holds its children by, and the keep list (`Keep`) of what a recording names, which its submission holds until it completed
use std::range::Range;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed},
};

use smallvec::SmallVec;

use crate::vk::{AccessFlags2 as A, PipelineStageFlags2 as S2};
#[cfg(feature = "acceleration-structure")]
use crate::vkobjects::acceleration_structure::StructureInternal;
use crate::{
    bindless::SamplerInternal,
    buffer::BufferInternal,
    command_buffer::{
        access_state::{AccessState, Visibility},
        locals::Locals,
    },
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
    /// The recording that kept it last (`Keep::tag`), so a recording keeps it once.
    pub(crate) last_kept: AtomicU64,
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
            last_kept: AtomicU64::new(0),
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
    /// Its tracked accesses (a structure's are its buffer's); none for a sampler.
    fn access_state(&self) -> Option<&Mutex<AccessState>> {
        None
    }
    /// The range of its state the commands that reach things through it access: the whole
    /// buffer, or the structure's memory. None for objects that reference nothing.
    fn referencing_range(&self) -> Option<Range<u64>> {
        None
    }
}

/// The stages and accesses a barrier waits for, if any.
pub(crate) type Hazard = Option<(S2, A)>;

/// What a write (`stage`, `access`) of `object` waits for of the objects above it, which
/// commands read it through (`Object::reference_hazards` of every ancestor).
/// The commands that read the object through an ancestor recorded only reads of the ancestor
/// (a walk of a clean buffer reaches nothing), so those are what the write waits for.
pub(crate) fn ancestor_hazards(
    object: &dyn Object,
    locals: &mut Locals,
    vis: &Visibility,
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
        let t = *locals.tracker();
        if let Some(range) = parent.referencing_range()
            && let Some(state) = locals.of_dyn(&parent)
        {
            add(state.hazards(&t, vis, range, stage, access));
        }
        add(ancestor_hazards(&*parent, locals, vis, stage, access));
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

/// The objects a recording names, which its submission keeps alive until it completed
/// (`timeline::keep_until`): each once, by the recording's tag in the object's header. The
/// tag is the timeline (bits 60 and up), the index of the recording's copy of the object's
/// access state (`Locals`, bits 40 to 59; `NO_INDEX` for none, or one that doesn't fit) and
/// the low 40 bits of the recording's value.
pub(crate) struct Keep {
    /// The recording's timeline and value bits; an object's tag matches with any index.
    tag: u64,
    pub(crate) kept: Vec<crate::queue::timeline::Kept>,
}

const INDEX_SHIFT: u32 = 40;
const INDEX_MASK: u64 = 0xF_FFFF << INDEX_SHIFT;
const NO_INDEX: u64 = INDEX_MASK;

impl Keep {
    /// The keep list of the recording of `value` on timeline `timeline`.
    pub(crate) fn new(timeline: usize, value: u64) -> Self {
        Keep {
            tag: (timeline as u64 + 1) << 60 | (value & ((1 << INDEX_SHIFT) - 1)),
            kept: Vec::new(),
        }
    }

    /// `header`'s tag if it is this recording's.
    fn tag_of(&self, header: &Header) -> Option<u64> {
        let tag = header.last_kept.load(Relaxed);
        (tag & !INDEX_MASK == self.tag).then_some(tag)
    }

    /// The index of the recording's copy of the object's access state, if its header carries
    /// it.
    pub(crate) fn local_index(&self, header: &Header) -> Option<usize> {
        let index = self.tag_of(header)? & INDEX_MASK;
        (index != NO_INDEX).then_some((index >> INDEX_SHIFT) as usize)
    }

    pub(crate) fn object<T: Object>(&mut self, object: &Arc<T>) {
        // A load, so a kept object's repeated accesses never write its header.
        if self.tag_of(object.header()).is_none() {
            object.header().last_kept.store(self.tag | NO_INDEX, Relaxed);
            let object: Arc<dyn Object> = object.clone();
            self.kept.push(crate::queue::timeline::Kept::Object(object));
        }
    }

    /// Drops the index from `header`'s tag: the copy was published.
    pub(crate) fn forget_index(&self, header: &Header) {
        if self.tag_of(header).is_some() {
            header.last_kept.store(self.tag | NO_INDEX, Relaxed);
        }
    }

    /// Keeps `object`, whose access state the recording copied at `index`.
    pub(crate) fn object_with_index(
        &mut self,
        header: &Header,
        object: &Arc<dyn Object>,
        index: usize,
    ) {
        let index = match index < (INDEX_MASK >> INDEX_SHIFT) as usize {
            true => (index as u64) << INDEX_SHIFT,
            false => NO_INDEX,
        };
        header.last_kept.store(self.tag | index, Relaxed);
        (self.kept).push(crate::queue::timeline::Kept::Object(object.clone()));
    }

    /// Something else the submission needs alive, which its caller keeps once.
    pub(crate) fn other(&mut self, other: Arc<dyn std::any::Any + Send + Sync>) {
        self.kept.push(crate::queue::timeline::Kept::Other(other));
    }
}
