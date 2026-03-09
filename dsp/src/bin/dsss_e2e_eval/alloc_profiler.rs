use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

const PHASE_COUNT: usize = 4;

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum Phase {
    Other = 0,
    Tx = 1,
    Channel = 2,
    Rx = 3,
}

impl Phase {
    fn as_index(self) -> usize {
        self as usize
    }

    fn from_index(v: usize) -> Self {
        match v {
            1 => Self::Tx,
            2 => Self::Channel,
            3 => Self::Rx,
            _ => Self::Other,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::Tx => "tx",
            Self::Channel => "channel",
            Self::Rx => "rx",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PhaseSnapshot {
    alloc_count: u64,
    alloc_bytes: u64,
    dealloc_count: u64,
    dealloc_bytes: u64,
    realloc_count: u64,
    realloc_old_bytes: u64,
    realloc_new_bytes: u64,
}

impl PhaseSnapshot {
    fn net_bytes(self) -> i64 {
        self.alloc_bytes as i64 + self.realloc_new_bytes as i64
            - self.dealloc_bytes as i64
            - self.realloc_old_bytes as i64
    }
}

const fn zero_u64_array() -> [AtomicU64; PHASE_COUNT] {
    [const { AtomicU64::new(0) }; PHASE_COUNT]
}

struct PhaseCounters {
    alloc_count: [AtomicU64; PHASE_COUNT],
    alloc_bytes: [AtomicU64; PHASE_COUNT],
    dealloc_count: [AtomicU64; PHASE_COUNT],
    dealloc_bytes: [AtomicU64; PHASE_COUNT],
    realloc_count: [AtomicU64; PHASE_COUNT],
    realloc_old_bytes: [AtomicU64; PHASE_COUNT],
    realloc_new_bytes: [AtomicU64; PHASE_COUNT],
}

impl PhaseCounters {
    const fn new() -> Self {
        Self {
            alloc_count: zero_u64_array(),
            alloc_bytes: zero_u64_array(),
            dealloc_count: zero_u64_array(),
            dealloc_bytes: zero_u64_array(),
            realloc_count: zero_u64_array(),
            realloc_old_bytes: zero_u64_array(),
            realloc_new_bytes: zero_u64_array(),
        }
    }

    fn reset(&self) {
        for i in 0..PHASE_COUNT {
            self.alloc_count[i].store(0, Ordering::Relaxed);
            self.alloc_bytes[i].store(0, Ordering::Relaxed);
            self.dealloc_count[i].store(0, Ordering::Relaxed);
            self.dealloc_bytes[i].store(0, Ordering::Relaxed);
            self.realloc_count[i].store(0, Ordering::Relaxed);
            self.realloc_old_bytes[i].store(0, Ordering::Relaxed);
            self.realloc_new_bytes[i].store(0, Ordering::Relaxed);
        }
    }

    fn snapshot(&self, phase: Phase) -> PhaseSnapshot {
        let i = phase.as_index();
        PhaseSnapshot {
            alloc_count: self.alloc_count[i].load(Ordering::Relaxed),
            alloc_bytes: self.alloc_bytes[i].load(Ordering::Relaxed),
            dealloc_count: self.dealloc_count[i].load(Ordering::Relaxed),
            dealloc_bytes: self.dealloc_bytes[i].load(Ordering::Relaxed),
            realloc_count: self.realloc_count[i].load(Ordering::Relaxed),
            realloc_old_bytes: self.realloc_old_bytes[i].load(Ordering::Relaxed),
            realloc_new_bytes: self.realloc_new_bytes[i].load(Ordering::Relaxed),
        }
    }
}

static CURRENT_PHASE: AtomicU8 = AtomicU8::new(Phase::Other as u8);
static COUNTERS: PhaseCounters = PhaseCounters::new();

fn phase_index_now() -> usize {
    let idx = CURRENT_PHASE.load(Ordering::Relaxed) as usize;
    if idx < PHASE_COUNT {
        idx
    } else {
        Phase::Other.as_index()
    }
}

fn record_alloc(bytes: usize) {
    let idx = phase_index_now();
    COUNTERS.alloc_count[idx].fetch_add(1, Ordering::Relaxed);
    COUNTERS.alloc_bytes[idx].fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_dealloc(bytes: usize) {
    let idx = phase_index_now();
    COUNTERS.dealloc_count[idx].fetch_add(1, Ordering::Relaxed);
    COUNTERS.dealloc_bytes[idx].fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_realloc(old_bytes: usize, new_bytes: usize) {
    let idx = phase_index_now();
    COUNTERS.realloc_count[idx].fetch_add(1, Ordering::Relaxed);
    COUNTERS.realloc_old_bytes[idx].fetch_add(old_bytes as u64, Ordering::Relaxed);
    COUNTERS.realloc_new_bytes[idx].fetch_add(new_bytes as u64, Ordering::Relaxed);
}

pub struct PhaseGuard {
    prev: u8,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        CURRENT_PHASE.store(self.prev, Ordering::Relaxed);
    }
}

pub fn enter(phase: Phase) -> PhaseGuard {
    let prev = CURRENT_PHASE.swap(phase as u8, Ordering::Relaxed);
    PhaseGuard { prev }
}

pub fn reset() {
    COUNTERS.reset();
    CURRENT_PHASE.store(Phase::Other as u8, Ordering::Relaxed);
}

pub fn report_to_stderr(label: &str) {
    let mut total = PhaseSnapshot::default();
    for i in 0..PHASE_COUNT {
        let s = COUNTERS.snapshot(Phase::from_index(i));
        total.alloc_count += s.alloc_count;
        total.alloc_bytes += s.alloc_bytes;
        total.dealloc_count += s.dealloc_count;
        total.dealloc_bytes += s.dealloc_bytes;
        total.realloc_count += s.realloc_count;
        total.realloc_old_bytes += s.realloc_old_bytes;
        total.realloc_new_bytes += s.realloc_new_bytes;
    }

    eprintln!(
        "[alloc-profile] label={} total alloc_count={} alloc_bytes={} dealloc_count={} dealloc_bytes={} realloc_count={} realloc_old_bytes={} realloc_new_bytes={} net_bytes={}",
        label,
        total.alloc_count,
        total.alloc_bytes,
        total.dealloc_count,
        total.dealloc_bytes,
        total.realloc_count,
        total.realloc_old_bytes,
        total.realloc_new_bytes,
        total.net_bytes()
    );

    for i in 0..PHASE_COUNT {
        let phase = Phase::from_index(i);
        let s = COUNTERS.snapshot(phase);
        eprintln!(
            "[alloc-profile] label={} phase={} alloc_count={} alloc_bytes={} dealloc_count={} dealloc_bytes={} realloc_count={} realloc_old_bytes={} realloc_new_bytes={} net_bytes={}",
            label,
            phase.name(),
            s.alloc_count,
            s.alloc_bytes,
            s.dealloc_count,
            s.dealloc_bytes,
            s.realloc_count,
            s.realloc_old_bytes,
            s.realloc_new_bytes,
            s.net_bytes()
        );
    }
}

struct PhaseTrackingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: PhaseTrackingAllocator = PhaseTrackingAllocator;

// SAFETY: Delegates all operations to System allocator and only updates atomic counters.
unsafe impl GlobalAlloc for PhaseTrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record_alloc(layout.size());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record_alloc(layout.size());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_dealloc(layout.size());
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        record_realloc(layout.size(), new_size);
        new_ptr
    }
}
