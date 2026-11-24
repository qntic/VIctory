//! Measures, on this machine's GPU, the memory host copies of title memory
//! live in (`Host::shared_buffer`: a custom heap, write-back, pool L0, i.e.
//! system memory the GPU reads over PCIe) against device-local memory, for
//! the decisions of #407 (title memory as one GPU address space) and the
//! copy costs of #400. Run with
//! `cargo test --release -p victory-gpu -- --ignored --nocapture memory_bench`.

use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const SIZE: usize = 256 << 20;
const ROUNDS: usize = 4;

fn faults() -> u64 {
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// MB/s and page faults per round of `f` over SIZE bytes.
fn time(name: &str, mut f: impl FnMut()) {
    f(); // first touch
    let (started, before) = (Instant::now(), faults());
    for _ in 0..ROUNDS {
        f();
    }
    let s = started.elapsed().as_secs_f64();
    let mb = (SIZE * ROUNDS) as f64 / 1e6;
    eprintln!(
        "BENCH {name}: {:.0} MB/s, {:.0} page faults per {} MB",
        mb / s,
        (faults() - before) as f64 / ROUNDS as f64,
        SIZE >> 20
    );
}

fn buffer(host: &crate::host::Host, heap: &D3D12_HEAP_PROPERTIES, uav: bool) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SIZE as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer.
    unsafe {
        host.device
            .CreateCommittedResource(
                heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut r,
            )
            .unwrap();
    }
    r.unwrap()
}

/// GPU copy of SIZE bytes from `src` to `dst`, MB/s (wall clock around one
/// submission each, after a warm-up).
fn gpu_copy(host: &crate::host::Host, name: &str, dst: &ID3D12Resource, src: &ID3D12Resource) {
    let run = || {
        let list = host.new_list().unwrap();
        // SAFETY: an open list and live buffers in the common state
        // (buffers promote to copy states implicitly).
        unsafe { list.list.CopyBufferRegion(dst, 0, src, 0, SIZE as u64) };
        list.close();
        let t = Instant::now();
        host.execute(&[&list]);
        t.elapsed().as_secs_f64()
    };
    run();
    let s: f64 = (0..ROUNDS).map(|_| run()).sum();
    eprintln!(
        "BENCH GPU {name}: {:.0} MB/s",
        (SIZE * ROUNDS) as f64 / 1e6 / s
    );
}

#[test]
#[ignore = "benchmark"]
fn memory_bench() {
    let Some(host) = crate::host::host() else {
        eprintln!("no host GPU; skipped");
        return;
    };
    let title = vec![0x5au8; SIZE];
    let mut plain = vec![0u8; SIZE];
    let (shared, mapped) = host.shared_buffer(SIZE as u64).expect("shared buffer");
    // SAFETY: the mapping spans SIZE bytes.
    let shared_bytes = unsafe { std::slice::from_raw_parts_mut(mapped, SIZE) };

    time("CPU memcpy into process memory", || {
        plain.copy_from_slice(&title)
    });
    time("CPU memcpy into the L0 write-back heap", || {
        shared_bytes.copy_from_slice(&title)
    });
    let rpm = |dst: *mut u8| {
        let mut done = 0usize;
        // SAFETY: both ranges span SIZE bytes of our own process.
        let ok = unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                title.as_ptr().cast(),
                dst.cast(),
                SIZE,
                &mut done,
            )
        };
        assert!(ok != 0 && done == SIZE);
    };
    time("ReadProcessMemory into process memory", || {
        rpm(plain.as_mut_ptr())
    });
    time("ReadProcessMemory into the L0 write-back heap", || {
        rpm(mapped)
    });
    time("CPU read of the L0 write-back heap", || {
        std::hint::black_box(shared_bytes.iter().step_by(64).fold(0u8, |a, &b| a ^ b));
    });

    let vram = buffer(
        host,
        &D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        true,
    );
    let vram2 = buffer(
        host,
        &D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        true,
    );
    gpu_copy(host, "L0 write-back heap -> VRAM", &vram, &shared);
    gpu_copy(host, "VRAM -> L0 write-back heap", &shared, &vram);
    gpu_copy(host, "VRAM -> VRAM", &vram2, &vram);
}
