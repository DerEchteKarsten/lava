//! Reference counts and tracked image layouts of lava's objects: one cache line each in a global slab, so views, slices and frames keep an object alive without borrowing it
//!
//! An object (image, buffer, swapchain, surface, semaphore) keeps its Vulkan handles inline and
//! a [`KeepAlive`] of its own [`Slot`] here: the count of the handles that keep it alive, where
//! its handles wait once its owner is gone, and the tracked layouts of an image (one for all
//! mips, one per mip of a `MipChain`, one per image of a swapchain), 2 bits each. Views,
//! slices and frame slots hold [`KeepAlive`]s too, and pending accesses a [`Weak`], which
//! tells whether the object still lives by its slot's generation. When the owner drops while others still
//! count, its handles are parked in `ORPHANS` and the last [`KeepAlive`] to drop destroys
//! them, on whatever thread that happens.
//!
//! A slot fills one cache line, so threads counting different objects don't contend. Slots live
//! in chunks that are never moved or freed, so a pointer to one stays valid for the process'
//! life; a slot is only reused once nothing counts it any more.
use std::{
    fmt::Debug,
    mem::ManuallyDrop,
    ops::Range,
    ptr::NonNull,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering, fence},
    },
};

use ash::vk;

/// Layout words of a slot: what is left of the cache line after the count, orphan index and
/// generation.
const LAYOUT_WORDS: usize = 13;
/// Tracked layouts one object can have, 2 bits each: an image's mips, a swapchain's images.
pub(crate) const MAX_LAYOUTS: u32 = LAYOUT_WORDS as u32 * 16;

/// One object's cache line.
#[repr(C, align(64))]
pub(crate) struct Slot {
    /// Handles alive; bit 31: the owner is gone and its handles wait in `ORPHANS`.
    count: AtomicU32,
    /// Where in `ORPHANS` the owner's handles wait, once bit 31 of `count` is set.
    orphan: AtomicU32,
    /// How often the slot was freed, so a [`Weak`] of an earlier object sees it is gone.
    generation: AtomicU32,
    /// 2 bits per layout (`encode`), 16 per word.
    layouts: [AtomicU32; LAYOUT_WORDS],
}

const _: () = assert!(size_of::<Slot>() == 64);

impl Slot {
    const fn new() -> Self {
        Slot {
            count: AtomicU32::new(0),
            orphan: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            layouts: [const { AtomicU32::new(0) }; LAYOUT_WORDS],
        }
    }
}

const CHUNK: usize = 1024;
const ORPHANED: u32 = 1 << 31;
/// What the last handle of an orphaned object sees before its decrement.
const LAST_ORPHANED: u32 = ORPHANED | 1;
/// Like `Arc`: far more handles than any program makes means a leak of `mem::forget`s. Only
/// checked in debug builds.
const MAX_COUNT: u32 = 1 << 30;

static FIRST: [Slot; CHUNK] = [const { Slot::new() }; CHUNK];

struct Slab {
    /// The chunk new slots are taken from, and how many of its slots are taken.
    current: &'static [Slot; CHUNK],
    used: usize,
    free: Vec<NonNull<Slot>>,
}
// The pointers are into `'static` chunks of atomics.
unsafe impl Send for Slab {}

static SLAB: Mutex<Slab> = Mutex::new(Slab {
    current: &FIRST,
    used: 0,
    free: Vec::new(),
});

/// The handles of owners that dropped while still counted, at the index in their slot.
struct Orphans {
    parts: Vec<Option<Box<dyn Send>>>,
    free: Vec<u32>,
}

static ORPHANS: Mutex<Orphans> = Mutex::new(Orphans {
    parts: Vec::new(),
    free: Vec::new(),
});

/// A slot counting 1 (the owner), with every layout `UNDEFINED`.
fn alloc() -> NonNull<Slot> {
    let mut slab = SLAB.lock().unwrap();
    let slot = match slab.free.pop() {
        Some(slot) => slot,
        None => {
            if slab.used == CHUNK {
                // Chunks are never freed.
                slab.current = Box::leak(Box::new([const { Slot::new() }; CHUNK]));
                slab.used = 0;
            }
            let slot = NonNull::from(&slab.current[slab.used]);
            slab.used += 1;
            slot
        }
    };
    let slot_ref = unsafe { slot.as_ref() };
    for word in &slot_ref.layouts {
        word.store(0, Ordering::Relaxed);
    }
    slot_ref.count.store(1, Ordering::Relaxed);
    slot
}

fn free(slot: NonNull<Slot>) {
    let slot_ref = unsafe { slot.as_ref() };
    slot_ref.count.store(0, Ordering::Relaxed);
    slot_ref.generation.fetch_add(1, Ordering::Relaxed);
    SLAB.lock().unwrap().free.push(slot);
}

/// Destroys what the owner of `slot` parked, once the last handle is gone.
#[cold]
fn destroy_orphan(slot: NonNull<Slot>) {
    let index = unsafe { slot.as_ref() }.orphan.load(Ordering::Relaxed);
    let parts = {
        let mut orphans = ORPHANS.lock().unwrap();
        orphans.free.push(index);
        orphans.parts[index as usize].take()
    };
    drop(parts);
    free(slot);
}

/// The 2-bit code of a layout lava tracks. Every access asks for one of these.
fn encode(layout: vk::ImageLayout) -> u32 {
    match layout {
        vk::ImageLayout::UNDEFINED => 0,
        vk::ImageLayout::GENERAL => 1,
        vk::ImageLayout::ATTACHMENT_OPTIMAL => 2,
        vk::ImageLayout::PRESENT_SRC_KHR => 3,
        other => panic!("lava doesn't track the image layout {other:?}"),
    }
}

fn decode(code: u32) -> vk::ImageLayout {
    [
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::ATTACHMENT_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    ][code as usize]
}

/// One count of a lava object: while it lives, the object's Vulkan handles do too, even if
/// its owner (the `Image`, `Buffer`, `Swapchain`, `Surface` or `Semaphore`) was dropped. The
/// owner holds the first one ([`KeepAlive::new`]) and hands it to [`KeepAlive::release`] when it
/// drops; views and slices hold clones, and a frame slot holds those of everything its last
/// submission used until the GPU finished it. Cloning and dropping cost one atomic each.
pub struct KeepAlive(NonNull<Slot>);
unsafe impl Send for KeepAlive {}
unsafe impl Sync for KeepAlive {}

impl KeepAlive {
    /// A new object's count, the owner's. Its layouts start `UNDEFINED`.
    pub(crate) fn new() -> Self {
        KeepAlive(alloc())
    }

    fn slot(&self) -> &Slot {
        unsafe { self.0.as_ref() }
    }

    /// The owner's end: destroys `parts` (the object's Vulkan handles; dropping them destroys
    /// them) now if nothing else counts the object, else parks them for the last handle.
    pub(crate) fn release<T: Send + 'static>(self, parts: T) {
        let this = ManuallyDrop::new(self);
        let slot = this.slot();
        // Only the owner counts: nobody can make a new handle, since that takes one.
        if slot.count.load(Ordering::Acquire) == 1 {
            drop(parts);
            free(this.0);
            return;
        }
        let index = {
            let mut orphans = ORPHANS.lock().unwrap();
            let parts = Some(Box::new(parts) as Box<dyn Send>);
            match orphans.free.pop() {
                Some(index) => {
                    orphans.parts[index as usize] = parts;
                    index
                }
                None => {
                    orphans.parts.push(parts);
                    orphans.parts.len() as u32 - 1
                }
            }
        };
        slot.orphan.store(index, Ordering::Relaxed);
        // Marks the slot orphaned and gives up the owner's count in one step, so a handle
        // dropping meanwhile either sees the mark and destroys, or leaves the last count to us.
        if slot.count.fetch_add(ORPHANED - 1, Ordering::AcqRel) == 1 {
            destroy_orphan(this.0);
        }
    }

    /// Sets the tracked layouts `fields` to `new` and calls `old` with each one's previous
    /// layout, in order.
    pub(crate) fn swap_layouts(
        &self,
        fields: Range<u32>,
        new: vk::ImageLayout,
        mut old: impl FnMut(vk::ImageLayout),
    ) {
        assert!(fields.end <= MAX_LAYOUTS);
        let code = encode(new) * 0x5555_5555;
        let mut field = fields.start;
        while field < fields.end {
            let first = field % 16;
            let count = (fields.end - field).min(16 - first);
            let mask = (u32::MAX >> (32 - 2 * count)) << (2 * first);
            let word = &self.slot().layouts[field as usize / 16];
            let previous = word
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |w| {
                    Some(w & !mask | code & mask)
                })
                .unwrap();
            for i in first..first + count {
                old(decode(previous >> (2 * i) & 3));
            }
            field += count;
        }
    }

    /// A reference to the object that doesn't keep it alive.
    pub(crate) fn downgrade(&self) -> Weak {
        Weak {
            slot: self.0,
            generation: self.slot().generation.load(Ordering::Relaxed),
        }
    }

    /// Whether both count the same object.
    #[cfg(test)]
    fn ptr_eq(&self, other: &KeepAlive) -> bool {
        self.0 == other.0
    }
}

impl Clone for KeepAlive {
    fn clone(&self) -> Self {
        let old = self.slot().count.fetch_add(1, Ordering::Relaxed);
        if cfg!(debug_assertions) && old & !ORPHANED > MAX_COUNT {
            std::process::abort();
        }
        KeepAlive(self.0)
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        match self.slot().count.fetch_sub(1, Ordering::Release) {
            LAST_ORPHANED => {
                fence(Ordering::Acquire);
                destroy_orphan(self.0);
            }
            // An owner dropped without `release`: there is nothing to destroy.
            1 => {
                fence(Ordering::Acquire);
                free(self.0);
            }
            _ => {}
        }
    }
}

/// An object that a [`KeepAlive`] counted, without counting it: [`Weak::is_alive`] until its
/// handles are destroyed. Pending accesses hold one, so they can be forgotten with their object.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Weak {
    slot: NonNull<Slot>,
    generation: u32,
}
// The pointer is into a `'static` chunk of atomics.
unsafe impl Send for Weak {}
unsafe impl Sync for Weak {}

impl Weak {
    pub(crate) fn is_alive(&self) -> bool {
        let slot = unsafe { self.slot.as_ref() };
        slot.generation.load(Ordering::Relaxed) == self.generation
    }
}

impl Debug for KeepAlive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("KeepAlive").field(&self.0).finish()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicUsize};

    /// Parts that count their drops.
    struct Parts(Arc<AtomicUsize>);
    impl Drop for Parts {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn count(keep: &KeepAlive) -> u32 {
        keep.slot().count.load(Ordering::SeqCst)
    }

    /// Layout `field`, after setting it to `set` if given.
    pub(crate) fn layout(
        keep: &KeepAlive,
        field: u32,
        set: Option<vk::ImageLayout>,
    ) -> vk::ImageLayout {
        let mut old = vk::ImageLayout::UNDEFINED;
        keep.swap_layouts(
            field..field + 1,
            set.unwrap_or(vk::ImageLayout::UNDEFINED),
            |l| old = l,
        );
        match set {
            Some(set) => set,
            None => {
                keep.swap_layouts(field..field + 1, old, |_| {});
                old
            }
        }
    }

    #[test]
    fn an_owner_alone_destroys_at_once() {
        let destroyed = Arc::new(AtomicUsize::new(0));
        let owner = KeepAlive::new();
        assert_eq!(count(&owner), 1);
        owner.release(Parts(destroyed.clone()));
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_last_handle_destroys_what_the_owner_left() {
        let destroyed = Arc::new(AtomicUsize::new(0));
        let owner = KeepAlive::new();
        let a = owner.clone();
        let b = a.clone();
        assert_eq!(count(&owner), 3);
        owner.release(Parts(destroyed.clone()));
        drop(a);
        assert_eq!(destroyed.load(Ordering::SeqCst), 0);
        let c = b.clone();
        drop(b);
        assert_eq!(destroyed.load(Ordering::SeqCst), 0);
        drop(c);
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn handles_dropped_before_the_owner_leave_it_alone() {
        let destroyed = Arc::new(AtomicUsize::new(0));
        let owner = KeepAlive::new();
        drop(owner.clone());
        assert_eq!(count(&owner), 1);
        owner.release(Parts(destroyed.clone()));
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn layouts_are_tracked_apart_and_start_undefined() {
        let keep = KeepAlive::new();
        let other = keep.clone();
        layout(&keep, 1, Some(vk::ImageLayout::GENERAL));
        layout(&keep, 20, Some(vk::ImageLayout::PRESENT_SRC_KHR));
        assert_eq!(layout(&other, 0, None), vk::ImageLayout::UNDEFINED);
        assert_eq!(layout(&other, 1, None), vk::ImageLayout::GENERAL);
        assert_eq!(layout(&other, 2, None), vk::ImageLayout::UNDEFINED);
        assert_eq!(layout(&other, 20, None), vk::ImageLayout::PRESENT_SRC_KHR);
        assert!(keep.ptr_eq(&other));
        assert!(!keep.ptr_eq(&KeepAlive::new()));
    }

    /// A swap across words reports every field's old layout in order and sets them all.
    #[test]
    fn swapping_a_range_crosses_words() {
        let keep = KeepAlive::new();
        let attachments = [3, 15, 16, 30, MAX_LAYOUTS - 1];
        for field in attachments {
            layout(&keep, field, Some(vk::ImageLayout::ATTACHMENT_OPTIMAL));
        }
        let mut old = Vec::new();
        keep.swap_layouts(0..MAX_LAYOUTS, vk::ImageLayout::GENERAL, |l| old.push(l));
        assert_eq!(old.len(), MAX_LAYOUTS as usize);
        for (field, layout) in (0..).zip(old) {
            let expected = match attachments.contains(&field) {
                true => vk::ImageLayout::ATTACHMENT_OPTIMAL,
                false => vk::ImageLayout::UNDEFINED,
            };
            assert_eq!(layout, expected, "field {field}");
        }
        assert_eq!(layout(&keep, 17, None), vk::ImageLayout::GENERAL);
    }

    #[test]
    #[should_panic(expected = "doesn't track")]
    fn untracked_layouts_panic() {
        KeepAlive::new().swap_layouts(0..1, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, |_| {});
    }

    #[test]
    fn freed_slots_are_reused_with_fresh_layouts() {
        let keep = KeepAlive::new();
        let slot = keep.0;
        layout(&keep, 4, Some(vk::ImageLayout::GENERAL));
        drop(keep);
        // Another test may take it first; then a fresh slot is just as good.
        let again = KeepAlive::new();
        if again.0 == slot {
            assert_eq!(layout(&again, 4, None), vk::ImageLayout::UNDEFINED);
            assert_eq!(count(&again), 1);
        }
    }

    #[test]
    fn weak_references_die_with_their_object_and_stay_dead_when_the_slot_is_reused() {
        let owner = KeepAlive::new();
        let weak = owner.downgrade();
        let handle = owner.clone();
        owner.release(());
        assert!(weak.is_alive());
        drop(handle);
        assert!(!weak.is_alive());
        // Another test may take the slot first; then any new object will do.
        let again = KeepAlive::new();
        assert!(!weak.is_alive());
        assert!(again.downgrade().is_alive());
    }

    #[test]
    fn racing_handles_destroy_exactly_once() {
        for _ in 0..200 {
            let destroyed = Arc::new(AtomicUsize::new(0));
            let owner = KeepAlive::new();
            let handles: Vec<_> = (0..4).map(|_| owner.clone()).collect();
            let threads: Vec<_> = handles
                .into_iter()
                .map(|handle| std::thread::spawn(move || drop(handle.clone())))
                .collect();
            owner.release(Parts(destroyed.clone()));
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(destroyed.load(Ordering::SeqCst), 1);
        }
    }
}
