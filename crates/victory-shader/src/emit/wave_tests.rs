//! Wave operations of wave32 pixel programs, translated and drawn on the
//! host GPU.
//!
//! A console pixel program runs in 32-lane waves. A host may run the pixel
//! stage 64 lanes wide (AMD RDNA does), and then one host wave holds two
//! console waves (`MaskData::Host` in `lower`). Lane masks read as numbers,
//! lane numbers and lane-indexed reads must still give each console wave
//! its own lanes.
//!
//! Which pixels share a host wave, and in which lanes, is the driver's
//! choice. A native pixel shader (`layout_ps`) reports it first: per pixel
//! the host wave's first lane and the host lane index (`WaveReadLaneFirst`,
//! `WaveGetLaneIndex`), drawn twice to see it does not change. Host lanes
//! 0-31 and 32-63 are then the two console waves, as `MaskData::Host` has
//! them, and every program's export is compared pixel by pixel with what the
//! console gives in that wave: the id of its first lane
//! (`v_readfirstlane_b32`), the lane number (`v_mbcnt_lo_u32_b32 -1, 0`) and
//! the value the program computes. Programs are drawn over the whole target
//! and over a triangle that leaves waves partly covered.
//!
//! The tests skip without a hardware SM 6.6 device and where the host's
//! pixel waves are not 64 lanes wide (on 32-lane hosts a console wave is a
//! host wave, which these programs do not test).

use std::collections::BTreeMap;

use super::bitcode::{Attr, Bin, Cast, Term};
use super::container::{SigElement, signature};
use super::host::{Draw, Host, Root};
use super::shader::{Shader, flags};
use super::{HostShader, emit, validator};
use crate::dxbc::{Container, FourCC, build};
use crate::native::{HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

// ---- GFX10 encodings ----

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;
/// The inline constant `n` (0..=64).
const fn k(n: u8) -> u16 {
    128 + n as u16
}
const MINUS_ONE: u16 = 193;
const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn vop1(op: u32, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | op << 9 | src0 as u32
}
const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
/// VOPC: the result goes to VCC.
const fn vopc(op: u32, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
const fn vop3(op: u32, vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [
        0xD400_0000 | op << 16 | vdst as u32,
        src0 as u32 | (src1 as u32) << 9,
    ]
}
const fn sop1(op: u32, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | op << 8 | src0 as u32
}
const fn sop2(op: u32, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | op << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
/// `exp mrt0, v[a], v[a+1], v[a+2], v[a+3] done vm`.
const fn exp_mrt0(a: u8) -> [u32; 2] {
    let a = a as u32;
    [
        0xF800_180F,
        a | (a + 1) << 8 | (a + 2) << 16 | (a + 3) << 24,
    ]
}

const V_MOV_B32: u32 = 0x01;
const V_READFIRSTLANE_B32: u32 = 0x02;
const V_CVT_F32_U32: u32 = 0x06;
const V_CVT_U32_F32: u32 = 0x07;
const V_LSHRREV_B32: u32 = 0x16;
const V_LSHLREV_B32: u32 = 0x1A;
const V_ADD_NC_U32: u32 = 0x25;
const V_CMP_GT_U32: u32 = 0xC4;
const V_MUL_LO_U32: u32 = 0x169;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const V_MBCNT_LO_U32_B32: u32 = 0x365;
const V_MBCNT_HI_U32_B32: u32 = 0x366;
const S_MOV_B32: u32 = 0x03;
const S_BCNT1_I32_B32: u32 = 0x0F;
const S_FF1_I32_B32: u32 = 0x13;
const S_AND_SAVEEXEC_B32: u32 = 0x3C;
const S_CSELECT_B32: u32 = 0x0A;
const S_AND_B32: u32 = 0x0E;
const S_OR_B32: u32 = 0x10;
const S_CMP_EQ_U32: u32 = 0x06;
const S_CMP_LG_U32: u32 = 0x07;
const S_CBRANCH_EXECNZ: u32 = 0x09;

/// SOPP `op simm16` (branches: dwords after the next instruction).
const fn sopp(op: u32, simm16: i16) -> u32 {
    0xBF80_0000 | op << 16 | simm16 as u16 as u32
}

/// SOPC `op src0, src1`: the result goes to SCC.
const fn sopc(op: u32, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | op << 16 | (src1 as u32) << 8 | src0 as u32
}

/// The multiplier of the pixel hash the condition tests.
const HASH: u32 = 0x9E37_79B1;

/// What every program starts with: v0, v1 = pixel position (x, y);
/// v2 = id (x + 64 y); v4 = 1 + the id of the console wave's first lane;
/// v5 = the lane number.
fn prologue() -> Vec<u32> {
    let mut w = vec![
        vop1(V_CVT_U32_F32, 2, v(0)),
        vop1(V_CVT_U32_F32, 3, v(1)),
        vop2(V_LSHLREV_B32, 3, k(6), 3),
        vop2(V_ADD_NC_U32, 2, v(2), 3),
        vop1(V_READFIRSTLANE_B32, 4, v(2)), // s4
        vop1(V_MOV_B32, 4, 4),              // v4 = s4
        vop2(V_ADD_NC_U32, 4, k(1), 4),
    ];
    w.extend(vop3(V_MBCNT_LO_U32_B32, 5, MINUS_ONE, k(0)));
    w
}

/// VCC = the pixel's condition (see [`cond`]).
fn condition() -> Vec<u32> {
    condition_of(29, 3)
}

/// VCC = (the pixel's hash >> shift) < below: a condition a console wave
/// meets in none, some or all of its lanes, as `below` is small or large.
fn condition_of(shift: u8, below: u8) -> Vec<u32> {
    let mut w = vec![sop1(S_MOV_B32, 10, 0xFF), HASH]; // s10 = HASH
    w.extend(vop3(V_MUL_LO_U32, 8, 10, v(2)));
    w.push(vop2(V_LSHRREV_B32, 8, k(shift), 8));
    w.push(vopc(V_CMP_GT_U32, k(below), 8));
    w
}

fn cond_of(id: u32, shift: u8, below: u8) -> bool {
    id.wrapping_mul(HASH) >> shift < u32::from(below)
}

/// The pixel condition [`condition`] computes.
fn cond(id: u32) -> bool {
    cond_of(id, 29, 3)
}

/// Exports v4-v7 as floats.
fn epilogue() -> Vec<u32> {
    let mut w: Vec<u32> = (4..8).map(|r| vop1(V_CVT_F32_U32, r, v(r))).collect();
    w.extend(exp_mrt0(4));
    w.push(S_ENDPGM);
    w
}

#[test]
fn encodings_are_the_instructions_named() {
    let words: Vec<u32> = [
        prologue(),
        condition(),
        vop3(V_READLANE_B32, 6, v(2), k(5)).to_vec(),
        vop3(V_WRITELANE_B32, 6, k(7), k(5)).to_vec(),
        vop3(V_MBCNT_HI_U32_B32, 6, MINUS_ONE, v(5)).to_vec(),
        vop3(V_READLANE_B32, 6, v(2), 5).to_vec(),
        vec![
            sopc(S_CMP_LG_U32, VCC_LO, 128),
            sopc(S_CMP_EQ_U32, VCC_LO, EXEC_LO),
            sop2(S_CSELECT_B32, 5, 129, 130),
            sop2(S_AND_B32, EXEC_LO, EXEC_LO, VCC_LO),
            sopp(S_CBRANCH_EXECNZ, -3),
            sop1(S_BCNT1_I32_B32, 5, VCC_LO),
            sop1(S_FF1_I32_B32, 5, VCC_LO),
            sop1(S_AND_SAVEEXEC_B32, 6, VCC_LO),
            sop2(S_OR_B32, EXEC_LO, EXEC_LO, 6),
        ],
        epilogue(),
    ]
    .concat();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(false))
        .collect();
    let expected = [
        "v_cvt_u32_f32 v2, v0",
        "v_cvt_u32_f32 v3, v1",
        "v_lshlrev_b32 v3, 6, v3",
        "v_add_nc_u32 v2, v2, v3",
        "v_readfirstlane_b32 s4, v2",
        "v_mov_b32 v4, s4",
        "v_add_nc_u32 v4, 1, v4",
        "v_mbcnt_lo_u32_b32 v5, -1, 0",
        "s_mov_b32 s10, lit(0x9e3779b1)",
        "v_mul_lo_u32 v8, s10, v2",
        "v_lshrrev_b32 v8, 29, v8",
        "v_cmp_gt_u32 vcc_lo, 3, v8",
        "v_readlane_b32 s6, v2, 5",
        "v_writelane_b32 v6, 7, 5",
        "v_mbcnt_hi_u32_b32 v6, -1, v5",
        "v_readlane_b32 s6, v2, s5",
        "s_cmp_lg_u32 vcc_lo, 0",
        "s_cmp_eq_u32 vcc_lo, exec_lo",
        "s_cselect_b32 s5, 1, 2",
        "s_and_b32 exec_lo, exec_lo, vcc_lo",
        "s_cbranch_execnz label_0064",
        "s_bcnt1_i32_b32 s5, vcc_lo",
        "s_ff1_i32_b32 s5, vcc_lo",
        "s_and_saveexec_b32 s6, vcc_lo",
        "s_or_b32 exec_lo, exec_lo, s6",
        "v_cvt_f32_u32 v4, v4",
        "v_cvt_f32_u32 v5, v5",
        "v_cvt_f32_u32 v6, v6",
        "v_cvt_f32_u32 v7, v7",
        "exp mrt0, v4, v5, v6, v7 done vm",
        "s_endpgm",
    ];
    assert_eq!(text, expected);
}

// ---- shaders ----

fn element(name: &str, kind: u8, interp: u8, usage: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage,
        half: false,
    }
}

/// Translates a wave32 pixel program reading the position (x, y in v0, v1)
/// and exporting four floats to target 0.
fn pixel_program(words: &[u32]) -> Result<HostShader, String> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            // POS_X_FLOAT, POS_Y_FLOAT.
            input_addr: 1 << 8 | 1 << 9,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg1 = signature(&[element("SV_Position", 3, 4, 0b11)], true);
    let osg1 = signature(&[element("SV_Target", 16, 0, 0xF)], false);
    let bytes = build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).map_err(|e| e.to_string())?;
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).map_err(e)?;
    let res = resource::track(&p, &native.layout, native.stage).map_err(e)?;
    let io = io::raise(&p, &native).map_err(e)?;
    let st = structure::structure(&p).map_err(e)?;
    let shader = emit(&c, &native, &p, &st, &io, &res).map_err(e)?;
    validate(&shader.dxil);
    Ok(shader)
}

fn validate(dxil: &[u8]) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut dxil.to_vec())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
}

fn finish(s: Shader, pixel: bool) -> HostShader {
    let dxil = s.finish();
    validate(&dxil);
    HostShader {
        stage: if pixel { HwStage::Ps } else { HwStage::Vs },
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
    }
}

/// A vertex shader passing `POSITION` through as `SV_Position`.
fn passthrough_vs() -> HostShader {
    let mut s = Shader::new(false);
    let b = s.entry;
    for col in 0..4 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
        let x = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), x];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.inputs = vec![element("POSITION", 0, 0, 0xF)];
    s.outputs = vec![element("SV_Position", 3, 4, 0xF)];
    s.m.terminate(b, Term::Ret);
    finish(s, false)
}

/// A pixel shader exporting the host's wave layout: 1 + the id of the host
/// wave's first lane, the host lane index, the host wave's lane count, the
/// pixel's id (ids as [`prologue`] computes them).
fn layout_ps() -> HostShader {
    let mut s = Shader::new(true);
    let b = s.entry;
    let (f32t, i32t) = (s.t.f32, s.t.i32);
    let mut xy = Vec::new();
    for col in 0..2 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(i32t)];
        let c = s.op(b, "loadInput.f32", Attr::ReadNone, f32t, 4, &args);
        xy.push(s.m.cast(b, Cast::FpToUi, c, i32t));
    }
    let size = s.m.i32(SIZE);
    let row = s.m.bin(b, Bin::Mul, xy[1], size);
    let id = s.m.bin(b, Bin::Add, xy[0], row);
    let first = s.op(b, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[id]);
    let one = s.m.i32(1);
    let first = s.m.bin(b, Bin::Add, first, one);
    let lane = s.op(b, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
    let count = s.op(b, "waveGetLaneCount", Attr::ReadNone, i32t, 112, &[]);
    s.flags |= flags::WAVE_OPS;
    for (col, x) in [first, lane, count, id].into_iter().enumerate() {
        let f = s.m.cast(b, Cast::UiToFp, x, f32t);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), f];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.inputs = vec![element("SV_Position", 3, 4, 0b11)];
    s.outputs = vec![element("SV_Target", 16, 0, 0xF)];
    s.m.terminate(b, Term::Ret);
    finish(s, true)
}

// ---- drawing ----

const SIZE: u32 = 64;
/// Covers the target.
const FULL: [f32; 12] = [
    -1.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0,
];
/// Leaves waves on its edges partly covered.
const PARTIAL: [f32; 12] = [
    -0.9, -0.75, 0.5, 1.0, 0.3, 0.95, 0.5, 1.0, 0.85, -0.4, 0.5, 1.0,
];

fn draw(host: &Host, ps: &HostShader, vertices: &[f32]) -> Vec<[f32; 4]> {
    let vs = passthrough_vs();
    let (root, pso) = host.pipeline(&vs, ps).expect("pipeline");
    let roots = vec![
        (
            0,
            Root::Constants(vec![0; vs.user_data_dwords().max(1) as usize]),
        ),
        (
            1,
            Root::Constants(vec![0; ps.user_data_dwords().max(1) as usize]),
        ),
    ];
    host.draw(
        &root,
        &pso,
        &Draw {
            roots,
            textures: &[],
            samplers: &[],
            vertices,
            stride: 16,
            size: (SIZE, SIZE),
            mesh_groups: 0,
        },
    )
    .expect("draw")
}

/// A console wave as the host runs it: (console lane, pixel id), by lane.
type Wave = Vec<(u32, u32)>;

/// Each covered pixel's console wave, from the host's layout: a 64-lane
/// host wave holds two console waves, host lanes 0-31 and 32-63 (the split
/// `MaskData::Host` assumes). `None` where the host's pixel waves are not
/// 64 lanes wide or the layout changes between draws.
fn console_waves(host: &Host, vertices: &[f32]) -> Option<BTreeMap<u32, Wave>> {
    let ps = layout_ps();
    let a = draw(host, &ps, vertices);
    if a != draw(host, &ps, vertices) {
        eprintln!("the host's wave layout changes between draws; skipped");
        return None;
    }
    let mut waves: BTreeMap<(u32, u32), Wave> = BTreeMap::new();
    for t in a.iter().filter(|t| t[0] != 0.0) {
        if t[2] != 64.0 {
            eprintln!("pixel waves of {} lanes; skipped", t[2]);
            return None;
        }
        let (first, lane, id) = (t[0] as u32 - 1, t[1] as u32, t[3] as u32);
        waves
            .entry((first, lane >> 5))
            .or_default()
            .push((lane & 31, id));
    }
    let mut by_id = BTreeMap::new();
    for mut w in waves.into_values() {
        w.sort_unstable();
        for &(_, id) in &w {
            by_id.insert(id, w.clone());
        }
    }
    Some(by_id)
}

/// A pixel whose export differs: its id, [first lane, lane, value] got and
/// wanted.
type Mismatch = (u32, [u32; 3], [u32; 3]);

/// Draws `words` over both triangles and compares every pixel's export
/// with the console's: the id of its console wave's first lane, its lane
/// number, and the value `expect` gives for the pixel (by id) in its wave
/// (`None`: not checked).
fn run(
    name: &str,
    words: &[u32],
    expect: impl Fn(&Wave, u32) -> Option<u32>,
) -> Result<(), String> {
    let Ok(host) = Host::new(false) else {
        eprintln!("{name}: no SM 6.6 hardware device; skipped");
        return Ok(());
    };
    let ps = pixel_program(words).map_err(|e| format!("{name}: not translated: {e}"))?;
    // The wave size the driver chose for this translated program (it may
    // differ from the layout probe's): the same program with its colour's
    // w replaced by WaveGetLaneCount.
    super::lower::LANE_COUNT_IN_W.with(|c| c.set(true));
    let sized = pixel_program(words);
    super::lower::LANE_COUNT_IN_W.with(|c| c.set(false));
    let sized = sized.map_err(|e| format!("{name}: not translated with the lane count: {e}"))?;
    let mut counts = std::collections::BTreeMap::new();
    for vertices in [&FULL, &PARTIAL] {
        for t in draw(&host, &sized, vertices).iter().filter(|t| t[0] != 0.0) {
            *counts.entry(t[3] as u32).or_insert(0u32) += 1;
        }
    }
    eprintln!("{name}: translated program's host wave sizes (lanes: pixels) {counts:?}");
    if counts.keys().any(|&n| n != 64) {
        eprintln!(
            "{name}: NOT EXERCISED: the driver ran the translated program in waves of              {:?} lanes, so no host wave held two console waves; skipped",
            counts.keys().collect::<Vec<_>>()
        );
        return Ok(());
    }
    let mut wrong = Vec::new();
    for (shape, vertices) in [("full", &FULL), ("partial", &PARTIAL)] {
        let Some(waves) = console_waves(&host, vertices) else {
            return Ok(());
        };
        let out = draw(&host, &ps, vertices);
        // First lane, lane number, value, coverage.
        let mut counts = [0usize; 4];
        let mut example: [Option<Mismatch>; 4] = [None; 4];
        for (n, t) in out.iter().enumerate() {
            let id = n as u32;
            let Some(wave) = waves.get(&id) else {
                if t[0] != 0.0 {
                    counts[3] += 1;
                }
                continue;
            };
            if t[0] == 0.0 || t[3] as u32 != id {
                counts[3] += 1;
                continue;
            }
            let got = [t[0] as u32 - 1, t[1] as u32, t[2] as u32];
            let value = expect(wave, id);
            let want = [wave[0].1, lane_of(wave, id), value.unwrap_or(got[2])];
            for i in 0..3 {
                if got[i] != want[i] {
                    counts[i] += 1;
                    example[i].get_or_insert((id, got, want));
                }
            }
        }
        let covered = waves.len();
        for (i, what) in ["first lane", "lane number", "value", "coverage"]
            .iter()
            .enumerate()
        {
            if counts[i] > 0 {
                wrong.push(format!(
                    "{shape}: {what} wrong in {} of {covered} pixels, e.g. \
                     (pixel, [first, lane, value] got, wanted) {:?}",
                    counts[i], example[i]
                ));
            }
        }
        eprintln!("{name} {shape}: {covered} pixels; wrong first/lane/value/coverage {counts:?}");
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(format!("{name}:\n  {}", wrong.join("\n  ")))
    }
}

/// The lane number of pixel `id` in `wave`.
fn lane_of(wave: &Wave, id: u32) -> u32 {
    wave.iter().find(|w| w.1 == id).unwrap().0
}

/// Lanes below pixel `id` in `wave` that meet the condition.
fn cond_below(wave: &Wave, id: u32) -> u32 {
    let lane = lane_of(wave, id);
    wave.iter().filter(|w| w.0 < lane && cond(w.1)).count() as u32
}

fn cond_count(wave: &Wave) -> u32 {
    wave.iter().filter(|w| cond(w.1)).count() as u32
}

/// `count << 6 | rank`, as the programs pack them.
fn packed(count: u32, rank: u32) -> u32 {
    count << 6 | rank
}

/// v6 += SGPR `s` << 6 (through v7).
fn pack_scalar(s: u8) -> [u32; 3] {
    [
        vop1(V_MOV_B32, 7, u16::from(s)),
        vop2(V_LSHLREV_B32, 7, k(6), 7),
        vop2(V_ADD_NC_U32, 6, v(6), 7),
    ]
}

// ---- programs ----

/// EXEC read as a number: `s_bcnt1` of it is the console wave's lane count,
/// `v_mbcnt_lo` of it the lane's rank among them.
#[test]
fn exec_counts_and_ranks_are_the_console_waves() {
    let mut w = prologue();
    w.extend(vop3(V_MBCNT_LO_U32_B32, 6, u16::from(EXEC_LO), k(0)));
    w.push(sop1(S_BCNT1_I32_B32, 5, EXEC_LO));
    w.extend(pack_scalar(5));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("exec", &w, |wave, id| {
        let rank = wave.iter().position(|w| w.1 == id).unwrap() as u32;
        Some(packed(wave.len() as u32, rank))
    })
    .unwrap();
}

/// A compare's mask read as a number: `s_bcnt1` counts the console wave's
/// lanes that meet the condition, `v_mbcnt_lo` those below the lane.
#[test]
fn compare_masks_count_the_console_waves_lanes() {
    let mut w = prologue();
    w.extend(condition());
    w.push(sop1(S_BCNT1_I32_B32, 5, VCC_LO));
    w.extend(vop3(V_MBCNT_LO_U32_B32, 6, u16::from(VCC_LO), k(0)));
    w.extend(pack_scalar(5));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("compare", &w, |wave, id| {
        Some(packed(cond_count(wave), cond_below(wave, id)))
    })
    .unwrap();
}

/// `s_ff1` of a compare's mask: the console wave's lowest lane that meets
/// the condition (exported plus one; 0 for none).
#[test]
fn first_set_lane_is_the_console_waves() {
    let mut w = prologue();
    w.extend(condition());
    w.push(sop1(S_FF1_I32_B32, 5, VCC_LO));
    w.push(vop1(V_MOV_B32, 6, 5));
    w.push(vop2(V_ADD_NC_U32, 6, k(1), 6));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("ff1", &w, |wave, _| {
        Some(wave.iter().find(|w| cond(w.1)).map_or(0, |w| w.0 + 1))
    })
    .unwrap();
}

/// `v_readlane_b32` at a constant lane reads that lane of the reader's own
/// console wave. Only waves whose lane 5 is covered are drawn into the
/// check; elsewhere the lane's value is undefined.
#[test]
fn readlane_reads_the_console_waves_lane() {
    let mut w = prologue();
    w.extend(vop3(V_READLANE_B32, 6, v(2), k(5))); // s6
    w.push(vop1(V_MOV_B32, 6, 6));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("readlane", &w, |wave, _| {
        wave.iter().find(|w| w.0 == 5).map(|w| w.1)
    })
    .unwrap();
}

/// Inside `s_and_saveexec` on a compare: `s_bcnt1` and `v_mbcnt_lo` of the
/// narrowed EXEC count the console wave's lanes that took the branch
/// (exported plus one; 0 for lanes outside it).
#[test]
fn a_narrowed_exec_counts_the_console_waves_lanes() {
    let mut w = prologue();
    w.extend(condition());
    w.push(vop1(V_MOV_B32, 6, k(0)));
    w.push(sop1(S_AND_SAVEEXEC_B32, 6, VCC_LO));
    w.push(sop1(S_BCNT1_I32_B32, 7, EXEC_LO));
    w.extend(vop3(V_MBCNT_LO_U32_B32, 6, u16::from(EXEC_LO), k(0)));
    w.extend(pack_scalar(7));
    w.push(vop2(V_ADD_NC_U32, 6, k(1), 6));
    w.push(sop2(S_OR_B32, EXEC_LO, EXEC_LO, 6));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("saveexec", &w, |wave, id| {
        Some(if cond(id) {
            packed(cond_count(wave), cond_below(wave, id)) + 1
        } else {
            0
        })
    })
    .unwrap();
}

/// `v_readlane_b32` at a lane the program computes: the console wave's
/// first lane that meets the condition (`s_ff1`, a console lane number;
/// none: -1, of which the read takes the low five bits).
#[test]
fn readlane_at_a_computed_lane_reads_the_console_waves_lane() {
    let mut w = prologue();
    w.extend(condition());
    w.push(sop1(S_FF1_I32_B32, 5, VCC_LO));
    w.extend(vop3(V_READLANE_B32, 6, v(2), 5)); // s6
    w.push(vop1(V_MOV_B32, 6, 6));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("readlane at s_ff1", &w, |wave, _| {
        let lane = wave.iter().find(|w| cond(w.1)).map_or(31, |w| w.0);
        wave.iter().find(|w| w.0 == lane).map(|w| w.1)
    })
    .unwrap();
}

/// `v_writelane_b32` writes the given lane of the console wave only: lane
/// 5 at a constant lane, and the first lane that meets the condition at a
/// computed one (`s_ff1`; values 7 and 9, the others 0 and 0).
#[test]
fn writelane_writes_the_console_waves_lane() {
    let mut w = prologue();
    w.extend(condition());
    w.push(sop1(S_FF1_I32_B32, 5, VCC_LO));
    w.push(vop1(V_MOV_B32, 6, k(0)));
    w.extend(vop3(V_WRITELANE_B32, 6, k(7), k(5)));
    w.push(vop1(V_MOV_B32, 7, k(0)));
    w.extend(vop3(V_WRITELANE_B32, 7, k(9), 5));
    w.push(vop2(V_ADD_NC_U32, 6, v(6), 7));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("writelane", &w, |wave, id| {
        let lane = lane_of(wave, id);
        let first = wave.iter().find(|w| cond(w.1)).map_or(31, |w| w.0);
        Some(u32::from(lane == 5) * 7 + u32::from(lane == first) * 9)
    })
    .unwrap();
}

/// Wave-wide tests of a compare's mask (`WaveActiveAnyTrue`,
/// `WaveActiveAllTrue`): `s_cmp_lg_u32 vcc_lo, 0` and `s_cmp_eq_u32 vcc_lo,
/// exec_lo` with `s_cselect`, on conditions rare and common enough that
/// console waves meet them in none, some and all of their lanes.
#[test]
fn any_and_all_lanes_are_the_console_waves() {
    for (shift, below) in [(25, 1), (29, 7)] {
        let mut w = prologue();
        w.extend(condition_of(shift, below));
        w.push(sopc(S_CMP_LG_U32, VCC_LO, 128));
        w.push(sop2(S_CSELECT_B32, 5, 129, 130)); // any: 1, none: 2
        w.push(sopc(S_CMP_EQ_U32, VCC_LO, EXEC_LO));
        w.push(sop2(S_CSELECT_B32, 6, 132, 136)); // all: 4, not all: 8
        w.push(vop1(V_MOV_B32, 6, 5));
        w.push(vop1(V_MOV_B32, 7, 6));
        w.push(vop2(V_ADD_NC_U32, 6, v(6), 7));
        w.push(vop1(V_MOV_B32, 7, v(2)));
        w.extend(epilogue());
        let name = format!("any/all, hash >> {shift} < {below}");
        run(&name, &w, |wave, _| {
            let n = wave.iter().filter(|w| cond_of(w.1, shift, below)).count();
            let any = if n > 0 { 1 } else { 2 };
            let all = if n == wave.len() { 4 } else { 8 };
            Some(any + all)
        })
        .unwrap();
    }
}

/// `v_mbcnt_hi_u32_b32` after `v_mbcnt_lo_u32_b32`, as compilers write a
/// lane index for either wave size: a wave32 lane has no lanes of the
/// mask's high half below it, so the pair is the lane number.
#[test]
fn mbcnt_lo_and_hi_are_the_console_lane() {
    let mut w = prologue();
    w.extend(vop3(V_MBCNT_HI_U32_B32, 6, MINUS_ONE, v(5)));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    run("mbcnt hi", &w, |wave, id| Some(lane_of(wave, id))).unwrap();
}

/// A loop the lanes leave at different iterations (1 to 4, by hash), with
/// EXEC read as a number in every iteration: each lane sums the console
/// wave's lane counts of the iterations it ran.
#[test]
fn exec_counts_follow_lanes_leaving_a_loop() {
    let mut w = prologue();
    w.extend([sop1(S_MOV_B32, 10, 0xFF), HASH]);
    w.extend(vop3(V_MUL_LO_U32, 8, 10, v(2)));
    w.push(vop2(V_LSHRREV_B32, 8, k(30), 8));
    w.push(vop2(V_ADD_NC_U32, 8, k(1), 8)); // v8 = iterations
    w.push(vop1(V_MOV_B32, 6, k(0)));
    w.push(vop1(V_MOV_B32, 9, k(0)));
    w.push(sop1(S_MOV_B32, 12, EXEC_LO));
    // loop:
    w.push(sop1(S_BCNT1_I32_B32, 7, EXEC_LO));
    w.push(vop2(V_ADD_NC_U32, 6, 7, 6));
    w.push(vop2(V_ADD_NC_U32, 9, k(1), 9));
    w.push(vopc(V_CMP_GT_U32, v(8), 9)); // more to run
    w.push(sop2(S_AND_B32, EXEC_LO, EXEC_LO, VCC_LO));
    w.push(sopp(S_CBRANCH_EXECNZ, -6));
    w.push(sop1(S_MOV_B32, EXEC_LO, 12));
    w.push(vop1(V_MOV_B32, 7, v(2)));
    w.extend(epilogue());
    let iterations = |id: u32| (id.wrapping_mul(HASH) >> 30) + 1;
    run("loop", &w, |wave, id| {
        let sum = (0..iterations(id))
            .map(|i| wave.iter().filter(|w| iterations(w.1) > i).count() as u32)
            .sum();
        Some(sum)
    })
    .unwrap();
}
