//! A compute program whose root arguments sit past the 16 user SGPRs reads
//! them through the spill table `s[0:1]` points to (`ir::build_with`,
//! `spilled`). Translated and run on the host GPU: the values it reads must
//! be the root arguments the dispatch binds.

use super::HostShader;
use super::host::{Dispatch, Host, Root, View};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// Four constants at user data 16..19 (table dwords 0..3), the UAV table at
/// user data 20 (table dword 4). The program finds both through the spill
/// table, adds the constants and stores the sum per lane.
#[test]
fn spill_table_arguments_reach_the_program() {
    let words = [
        0xBE82_0300u32, // s_mov_b32 s2, s0
        0x8703_FF01,    // s_and_b32 s3, s1, lit
        0x0000_00FF,    //   0xff
        0xF400_0101,    // s_load_dword s4, s[2:3], 0x10 (table dword 4: the UAV table)
        0xFA00_0010,
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0301, // s_load_dwordx4 s[12:15], s[2:3], 0 (table dwords 0..3)
        0xFA00_0000,
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0 (the UAV's descriptor)
        0xFA00_0000,
        0x8006_0D0C, // s_add_u32 s6, s12, s13
        0x8006_0E06, // s_add_u32 s6, s6, s14
        0x8006_0F06, // s_add_u32 s6, s6, s15
        0x7E14_0300, // v_mov_b32 v10, v0 (thread id)
        0x7E00_0206, // v_mov_b32 v0, s6
        0x3412_1482, // v_lshlrev_b32 v9, 2, v10
        0xE070_1000, // buffer_store_dword v0, v9, s[8:11], 0 offen
        0x8002_0009,
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect();
    assert_eq!(
        text[2..10],
        [
            "s_load_dword s4, s[2:3], 0x000010",
            "s_mov_b32 s5, 0",
            "s_load_dwordx4 s[12:15], s[2:3], null",
            "s_load_dwordx4 s[8:11], s[4:5], null",
            "s_add_u32 s6, s12, s13",
            "s_add_u32 s6, s6, s14",
            "s_add_u32 s6, s6, s15",
            "v_mov_b32 v10, v0",
        ],
        "{text:#?}"
    );
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 16,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 20,
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
            user_data: 20,
            slot: 1,
        }],
        // Two user SGPRs: the spill table's address.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [64, 1, 1],
        }),
        code,
    };
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build_with(&native.code, native.wave64, native.vgprs, true)
        .map_err(e)
        .unwrap();
    // The table loads became copies; the descriptor's load stays.
    let loads = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| i.name().starts_with("s_load"))
        .count();
    assert_eq!(loads, 1, "the table loads were not rewritten");
    let st = structure::structure(&p).map_err(e).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))
        .map_err(e)
        .unwrap();
    let io = io::raise(&p, &native).map_err(e).unwrap();
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
    .map_err(e)
    .unwrap();
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("no pipeline on this host ({e}); skipped");
            return;
        }
    };
    const ARGS: [u32; 4] = [1, 10, 100, 1000];
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => ARGS[u.dword as usize],
            (RootType::Table, 1) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
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
    assert!(got.iter().all(|&x| x == 1111), "sums: {got:?}");
}
