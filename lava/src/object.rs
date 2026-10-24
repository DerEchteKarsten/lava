//! GPU objects as lava owns them: each lives in an `Arc` of its internal (buffers, images, acceleration structures, query pools), whose `Header` counts it as a lava user until its drop finished
use crate::global::Live;

/// What every object's internal carries, as its last field.
#[derive(Debug)]
pub(crate) struct Header {
    /// Counts the object as a lava user, so `shutdown` refuses while it lives; dropped last.
    _live: Live,
}

impl Header {
    pub(crate) fn new(live: Live) -> Self {
        Self { _live: live }
    }
}
