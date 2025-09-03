use std::{mem::MaybeUninit, sync::Arc};


struct DeviceProperties {

}

static DEVICE_PROPERTIES: Arc<MaybeUninit<DeviceProperties>> = Arc::new_uninit(); 