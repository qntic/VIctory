//! Lane masks and lane reads of compute programs, translated and run on the
//! host GPU: one group of 64 threads (thread id in v0), each storing v0 at
//! byte 4 * lane. A wave64 program is one console wave, a wave32 program
//! two.
//!
//! Each test states what the console computes. A program the translator
//! refuses is reported and passes: refusing is safe; computing other
//! values without an error is what the tests catch. They need a hardware
//! SM 6.6 device (and 64-lane waves for wave64 programs) and skip without.
//!
//! The fixture follows `build_lanes_cs` of #269's `valu_tests`.

use super::HostShader;
use super::host::{Dispatch, Host, Root, View};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

// ---- GFX10 encodings ----

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const VCC_HI: u8 = 107;
const EXEC_LO: u8 = 126;
const EXEC_HI: u8 = 127;
const ZERO: u8 = 128;
const MINUS_ONE: u8 = 193;

const fn k(n: u8) -> u16 {
    128 + n as u16
}
const fn v(n: u8) -> u16 {
    256 + n as u16
}
const fn vop1(op: u32, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | op << 9 | src0 as u32
}
const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
const fn vopc(op: u32, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
/// VOP3; for a compare, `vdst` is the SGPR (pair) it writes.
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

const V_CNDMASK_B32: u32 = 0x01;
const V_MOV_B32: u32 = 0x01;
const V_AND_B32: u32 = 0x1B;
const V_CMP_EQ_U32: u32 = 0xC2;
const V_CMP_GT_U32: u32 = 0xC4;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const V_MBCNT_LO_U32_B32: u32 = 0x365;
const V_MBCNT_HI_U32_B32: u32 = 0x366;
const S_MOV_B32: u32 = 0x03;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_I32_B32: u32 = 0x0F;
const S_BCNT1_I32_B64: u32 = 0x10;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_OR_B64: u32 = 0x11;
const S_XOR_B64: u32 = 0x13;

/// `v_mov_b32 v0, s<n>`.
const fn v0_from(s: u8) -> u32 {
    vop1(V_MOV_B32, 0, s as u16)
}

// ---- fixture ----

/// The program: `body` between the descriptor load and the store of v0 at
/// byte 4 * lane; v10 keeps the thread id.
fn lanes_cs(body: &[u32], wave64: bool) -> Result<HostShader, String> {
    let mut words = vec![
        sop1(S_MOV_B32, 5, ZERO),
        // s_load_dwordx4 s[8:11], s[4:5], 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 4 >> 1,
        125 << 25,
        vop1(V_MOV_B32, 10, v(0)),
    ];
    words.extend_from_slice(body);
    // v_lshlrev_b32 v9, 2, v10; buffer_store_dword v0, v9, s[8:11], 0 offen
    words.push(0x3412_1482);
    words.extend_from_slice(&[0x38 << 26 | 0x1C << 18 | 1 << 12, 0x80 << 24 | 2 << 16 | 9]);
    words.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build(&native.code, native.wave64, native.vgprs).map_err(e)?;
    let st = structure::structure(&p).map_err(e)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(e)?;
    let io = io::raise(&p, &native).map_err(e)?;
    let lowered = super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .map_err(e)?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        address_srvs: lowered.address_srvs,
        stride_table: lowered.stride_table,
    };
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut shader.dxil.clone())
            .map_err(|e| format!("DXIL validation: {e}"))?;
    }
    Ok(shader)
}

/// Runs `body` and checks every lane's v0 against `want` (`None`: not
/// checked). Returns without checking where the program is refused or no
/// device can run it.
fn run(name: &str, body: &[u32], wave64: bool, want: impl Fn(u32) -> Option<u32>) {
    let shader = match lanes_cs(body, wave64) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{name}: refused: {e}");
            return;
        }
    };
    let Ok(host) = Host::new(false) else {
        eprintln!("{name}: no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{name}: no pipeline on this host ({e}); skipped");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let got = &out[0][..64];
    let wrong: Vec<(u32, u32, u32)> = (0..64u32)
        .filter_map(|lane| {
            let w = want(lane)?;
            (got[lane as usize] != w).then_some((lane, got[lane as usize], w))
        })
        .collect();
    eprintln!("{name}: translated, {} of 64 lanes wrong", wrong.len());
    assert!(
        wrong.is_empty(),
        "{name}: (lane, got, wanted) {:?}",
        &wrong[..wrong.len().min(8)]
    );
}

// ---- console semantics, wave64 ----

/// `s_bcnt1_i32_b64` of a compare's mask; `v_mbcnt_lo`/`_hi` of all lanes;
/// `v_readlane_b32` of lane 40.
#[test]
fn wave64_mask_counts_ranks_and_lane_reads() {
    // Lanes whose id is below 40.
    let mut cmp = vec![vopc(V_CMP_GT_U32, k(40), 0)];
    cmp.push(sop1(S_BCNT1_I32_B64, 4, VCC_LO));
    cmp.push(v0_from(4));
    run("wave64 s_bcnt1_i32_b64", &cmp, true, |_| Some(40));

    let mut rank = Vec::from(vop3(V_MBCNT_LO_U32_B32, 1, MINUS_ONE as u16, k(0)));
    rank.extend(vop3(V_MBCNT_HI_U32_B32, 0, MINUS_ONE as u16, v(1)));
    run("wave64 v_mbcnt", &rank, true, Some);

    let mut read = Vec::from(vop3(V_READLANE_B32, 4, v(10), k(40)));
    read.push(v0_from(4));
    run("wave64 v_readlane 40", &read, true, |_| Some(40));
}

/// #660: a select on VCC after only its high dword was rewritten. The
/// hardware's VCC_HI is 0, so lanes 32-63 pick src0.
#[test]
fn wave64_select_after_vcc_hi_rewritten() {
    let body = [
        vop2(V_AND_B32, 3, k(1), 0),     // v3 = id & 1
        vopc(V_CMP_EQ_U32, k(0), 3),     // vcc = even lanes
        sop1(S_MOV_B32, VCC_HI, ZERO),   // vcc_hi = 0
        vop1(V_MOV_B32, 4, k(2)),        // v4 = 2
        vop2(V_CNDMASK_B32, 0, k(1), 4), // v0 = vcc ? 2 : 1
    ];
    run("wave64 select after vcc_hi = 0", &body, true, |lane| {
        Some(if lane < 32 && lane % 2 == 0 { 2 } else { 1 })
    });
}

/// #660: EXEC's high dword cleared alone; the store that follows runs in
/// lanes 0-31 only (the others keep the buffer's 0).
#[test]
fn wave64_store_after_exec_hi_cleared() {
    let body = [sop1(S_MOV_B32, EXEC_HI, ZERO), vop1(V_MOV_B32, 0, k(7))];
    run("wave64 store after exec_hi = 0", &body, true, |lane| {
        Some(if lane < 32 { 7 } else { 0 })
    });
}

/// #298: a compare's mask spilled to two lanes of a VGPR, the SGPRs reused,
/// read back and used with `s_and_saveexec`; also its complement.
#[test]
fn wave64_mask_spilled_to_lanes_and_read_back() {
    for complement in [false, true] {
        let mut body = vec![vop1(V_MOV_B32, 0, k(0))];
        body.extend(vop3(V_CMP_GT_U32, 6, k(40), v(10))); // s[6:7] = id < 40
        if complement {
            body.push(sop2(S_XOR_B64, 6, 6, MINUS_ONE));
        }
        body.extend(vop3(V_WRITELANE_B32, 20, 6, k(1)));
        body.extend(vop3(V_WRITELANE_B32, 20, 7, k(2)));
        body.push(sop1(S_MOV_B64, 6, ZERO));
        // s[8:11] hold the store's descriptor.
        body.extend(vop3(V_READLANE_B32, 12, v(20), k(1)));
        body.extend(vop3(V_READLANE_B32, 13, v(20), k(2)));
        body.push(sop1(S_AND_SAVEEXEC_B64, 14, 12));
        body.push(vop1(V_MOV_B32, 0, k(5)));
        body.push(sop2(S_OR_B64, EXEC_LO, EXEC_LO, 14));
        let name = format!("wave64 spilled mask (complement {complement})");
        run(&name, &body, true, |lane| {
            Some(if (lane < 40) != complement { 5 } else { 0 })
        });
    }
}

// ---- console semantics, wave32 (two console waves) ----

/// `s_bcnt1_i32_b32` of a compare's mask and `v_readlane_b32` of lane 5,
/// each in its own console wave.
#[test]
fn wave32_mask_counts_and_lane_reads_per_wave() {
    // Lanes whose id is below 40: wave 0 all 32, wave 1 lanes 32-39.
    let body = [
        vopc(V_CMP_GT_U32, k(40), 0),
        sop1(S_BCNT1_I32_B32, 4, VCC_LO),
        v0_from(4),
    ];
    run("wave32 s_bcnt1_i32_b32", &body, false, |lane| {
        Some(if lane < 32 { 32 } else { 8 })
    });
    let mut read = Vec::from(vop3(V_READLANE_B32, 4, v(10), k(5)));
    read.push(v0_from(4));
    run("wave32 v_readlane 5", &read, false, |lane| {
        Some(if lane < 32 { 5 } else { 37 })
    });
}

#[test]
fn encodings_are_the_instructions_named() {
    let words: Vec<u32> = [
        vec![vop2(V_AND_B32, 3, k(1), 0), vopc(V_CMP_EQ_U32, k(0), 3)],
        vec![
            sop1(S_MOV_B32, VCC_HI, ZERO),
            vop2(V_CNDMASK_B32, 0, k(1), 4),
        ],
        vop3(V_CMP_GT_U32, 6, k(40), v(10)).to_vec(),
        vop3(V_WRITELANE_B32, 20, 6, k(1)).to_vec(),
        vop3(V_READLANE_B32, 8, v(20), k(1)).to_vec(),
        vop3(V_MBCNT_HI_U32_B32, 0, MINUS_ONE as u16, v(1)).to_vec(),
        vec![
            sop1(S_MOV_B64, 6, ZERO),
            sop1(S_AND_SAVEEXEC_B64, 14, 12),
            sop2(S_OR_B64, EXEC_LO, EXEC_LO, 14),
            sop2(S_XOR_B64, 6, 6, MINUS_ONE),
            sop1(S_BCNT1_I32_B64, 4, VCC_LO),
            sop1(S_BCNT1_I32_B32, 4, VCC_LO),
            sop1(S_MOV_B32, EXEC_HI, ZERO),
            S_ENDPGM,
        ],
    ]
    .concat();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect();
    let expected = [
        "v_and_b32 v3, 1, v0",
        "v_cmp_eq_u32 vcc, 0, v3",
        "s_mov_b32 vcc_hi, 0",
        "v_cndmask_b32 v0, 1, v4, vcc",
        "v_cmp_gt_u32 s[6:7], 40, v10",
        "v_writelane_b32 v20, s6, 1",
        "v_readlane_b32 s8, v20, 1",
        "v_mbcnt_hi_u32_b32 v0, -1, v1",
        "s_mov_b64 s[6:7], 0",
        "s_and_saveexec_b64 s[14:15], s[12:13]",
        "s_or_b64 exec, exec, s[14:15]",
        "s_xor_b64 s[6:7], s[6:7], -1",
        "s_bcnt1_i32_b64 s4, vcc",
        "s_bcnt1_i32_b32 s4, vcc_lo",
        "s_mov_b32 exec_hi, 0",
        "s_endpgm",
    ];
    assert_eq!(text, expected);
}
