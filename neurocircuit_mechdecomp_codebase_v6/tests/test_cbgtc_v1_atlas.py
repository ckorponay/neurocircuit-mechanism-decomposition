import numpy as np

from neurocircuit.atlas.cbgtc_v1 import (
    CBGTC_V1_N_REGIONS,
    canonical_rois,
    canonical_roi_ids,
    build_graph_mask,
    build_routing_mask,
    build_dynamics_mask,
)


def _idx(roi_id):
    return canonical_roi_ids().index(roi_id)


def test_cbgtc_v1_counts_and_finest_tian_groups():
    rois = canonical_rois()
    assert len(rois) == CBGTC_V1_N_REGIONS == 164
    counts = {}
    for r in rois:
        counts[r.group] = counts.get(r.group, 0) + 1
    assert counts == {
        "C": 100,
        "S": 20,
        "Th": 16,
        "A": 4,
        "H": 10,
        "GPe": 2,
        "GPi": 2,
        "VeP": 2,
        "STN": 2,
        "SNc": 2,
        "SNr": 2,
        "VTA": 2,
    }
    assert all(r.source_atlas == "Tian2020_3T_S3" for r in rois if r.group == "S")
    assert all(r.source_atlas == "Tian2020_3T_S4" for r in rois if r.group in {"Th", "A", "H"})


def test_all_major_input_classes_can_reach_all_striatum():
    m = build_graph_mask("core")
    for src in ["C_L_001", "A_L_Lat", "H_L_HeadM1", "Th_L_VAip", "SNc_L", "VTA_L"]:
        assert m[_idx(src), _idx("S_L_NAcCore")]
        assert m[_idx(src), _idx("S_L_CauTail")]
        assert m[_idx(src), _idx("S_R_PutDP")]


def test_revised_recurrent_limbic_thalamic_and_bg_routes():
    m = build_graph_mask("core")
    # Cortex-limbic/thalamic recurrence.
    assert m[_idx("C_L_001"), _idx("A_L_Lat")]
    assert m[_idx("A_L_Lat"), _idx("C_L_001")]
    assert m[_idx("C_L_001"), _idx("H_L_HeadM1")]
    assert m[_idx("H_L_HeadM1"), _idx("C_L_001")]
    assert m[_idx("C_L_001"), _idx("Th_L_VAip")]
    assert m[_idx("Th_L_VAip"), _idx("C_L_001")]
    assert m[_idx("A_L_Lat"), _idx("H_L_HeadM1")]
    assert m[_idx("H_L_HeadM1"), _idx("A_L_Lat")]

    # Thalamostriatal and canonical basal-ganglia routes.
    assert m[_idx("Th_L_VAip"), _idx("S_L_CauVA")]
    assert m[_idx("S_L_PutVA"), _idx("GPe_L")]
    assert m[_idx("S_L_PutVA"), _idx("GPi_L")]
    assert m[_idx("S_L_PutVA"), _idx("SNr_L")]
    assert m[_idx("S_L_NAcShell"), _idx("VeP_L")]
    assert m[_idx("GPe_L"), _idx("S_L_CauVA")]
    assert m[_idx("GPe_L"), _idx("STN_L")]
    assert m[_idx("STN_L"), _idx("GPe_L")]
    assert m[_idx("STN_L"), _idx("GPi_L")]
    assert m[_idx("GPi_L"), _idx("Th_L_VAip")]
    assert m[_idx("SNr_L"), _idx("Th_L_VAip")]
    assert m[_idx("C_L_001"), _idx("STN_L")]


def test_no_generic_cortico_cortical_routing_in_core_but_extended_ablation_has_it():
    core = build_routing_mask("core")
    ext = build_routing_mask("extended")
    assert not core[_idx("C_L_001"), _idx("C_L_002")]
    assert not core[_idx("C_L_001"), _idx("C_L_010")]
    assert ext[_idx("C_L_001"), _idx("C_L_002")]
    assert ext[_idx("C_L_001"), _idx("C_L_010")]


def test_no_transformer_self_routing():
    m = build_graph_mask("core")
    assert m.shape == (164, 164)
    assert not np.diag(m).any()


def test_routing_is_smaller_than_dynamics_and_core_dynamics_has_within_network_cortex():
    routing = build_routing_mask("core")
    dynamics = build_dynamics_mask("core")
    assert routing.mean() < 0.35
    assert dynamics.mean() < 0.45
    assert dynamics.sum() > routing.sum()
    # Within-network cortical background dynamics are explicit, but not routed.
    assert not routing[_idx("C_L_001"), _idx("C_L_002")]
    assert dynamics[_idx("C_L_001"), _idx("C_L_002")]
    # Cross-network explicit cortical coupling is absent from the core sparse A;
    # the primary model handles distributed cross-network C->C via low rank.
    assert not dynamics[_idx("C_L_001"), _idx("C_L_010")]
    assert build_dynamics_mask("extended")[_idx("C_L_001"), _idx("C_L_010")]
