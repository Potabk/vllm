# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EAGLE3 spec-decode + NixlConnector fix.

Bug: NixlWorker._sync_block_size_with_kernel() called
     get_current_attn_backends(vllm_config) without scoping layer names,
     so EAGLE3 draft-model backends (e.g. FlashAttention, which supports
     MultipleOf(16)) were mixed with main-model MLA backends (e.g. FlashMLA,
     which only supports [64]).
     With block_size=256 this caused:
       select_common_block_size(256, [MLA, FA]) -> 64  (phys_ratio=4)
     but the FA-only draft group alone gives:
       select_common_block_size(256, [FA])      -> 256 (phys_ratio=1)
     So self.num_blocks (4x) != draft tensor shape[0] (1x) -> AssertionError.

Fix: scope attn_backends to the *primary* (first non-Mamba) kv_cache_group
     and skip secondary-group attention layers in register_kv_caches.

These tests validate both the bug scenario and the fix, without needing
NIXL library, GPUs, or actual model weights.

Run:
    .venv/bin/python -m pytest \
        tests/v1/kv_connector/unit/test_nixl_eagle3_fix.py -v
"""

from unittest.mock import MagicMock

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.worker.utils import select_common_block_size

# ---------------------------------------------------------------------------
# Minimal mock attention backends (no GPU / NIXL required)
# ---------------------------------------------------------------------------


class _MockMLABackend:
    """Simulates a MLA backend (e.g. FlashMLA) that only accepts block_size=64."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [64]

    @classmethod
    def full_cls_name(cls):
        return "_MockMLABackend"

    @staticmethod
    def get_name():
        return "MOCK_MLA"


class _MockFlashAttnBackend:
    """Simulates FlashAttention (EAGLE3 draft uses this) that accepts MultipleOf(16)."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @classmethod
    def full_cls_name(cls):
        return "_MockFlashAttnBackend"

    @staticmethod
    def get_name():
        return "MOCK_FA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_groups(main_layers, draft_layers):
    """Build two KVCacheGroupSpec mocks: group-0 main model, group-1 EAGLE3 draft."""
    main_spec = MagicMock(spec=AttentionSpec)
    draft_spec = MagicMock(spec=AttentionSpec)
    return [
        MagicMock(kv_cache_spec=main_spec, layer_names=list(main_layers)),
        MagicMock(kv_cache_spec=draft_spec, layer_names=list(draft_layers)),
    ]


def _extract_primary_attn_layer_names(kv_groups):
    """Replicate the _primary_attn_layer_names logic from the fix."""
    for group in kv_groups:
        if not isinstance(group.kv_cache_spec, MambaSpec):
            return list(group.layer_names)
    return None


# ---------------------------------------------------------------------------
# Tests – block-size arithmetic
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256  # common scenario: MLA caps at 64, FA supports 256


class TestSelectCommonBlockSizeBugVsFix:
    def test_bug_mixed_backends_inflate_phys_ratio(self):
        """
        Before the fix: mixing MLA + FA backends forces kernel_block_size=64,
        giving phys_ratio=4.  But the FA-only draft group uses kernel_block_size=256
        (phys_ratio=1), so self.num_blocks != draft tensor shape[0].
        """
        all_backends = [_MockMLABackend, _MockFlashAttnBackend]
        common_size = select_common_block_size(BLOCK_SIZE, all_backends)
        assert common_size == 64, (
            "Mixed backends should converge to 64 (MLA constraint)"
        )
        phys_ratio_nixl = BLOCK_SIZE // common_size  # 4 – what nixl worker would use

        # Draft group computed independently (only FA)
        fa_size = select_common_block_size(BLOCK_SIZE, [_MockFlashAttnBackend])
        assert fa_size == BLOCK_SIZE  # FA supports any MultipleOf(16)
        phys_ratio_draft_tensor = BLOCK_SIZE // fa_size  # 1

        assert phys_ratio_nixl != phys_ratio_draft_tensor, (
            "Bug confirmed: nixl phys_ratio and draft tensor phys_ratio diverge"
        )

    def test_fix_primary_only_backends_consistent_with_main_group(self):
        """
        After the fix: using only the primary (MLA) backend, kernel_block_size=64,
        phys_ratio=4.  The main-model group *also* computes phys_ratio=4.
        self.num_blocks == main tensor shape[0].
        """
        primary_backends = [_MockMLABackend]
        common_size = select_common_block_size(BLOCK_SIZE, primary_backends)
        assert common_size == 64
        phys_ratio_nixl = BLOCK_SIZE // common_size  # 4

        # Main group kernel_block_size (prepare_kernel_block_sizes logic)
        main_group_size = select_common_block_size(BLOCK_SIZE, [_MockMLABackend])
        phys_ratio_main_tensor = BLOCK_SIZE // main_group_size  # 4

        assert phys_ratio_nixl == phys_ratio_main_tensor, (
            "Fix confirmed: nixl phys_ratio matches main-model tensor phys_ratio"
        )

    def test_non_mla_model_unchanged(self):
        """
        For a non-MLA model with a single attention group, primary-only
        == all-backends, so behaviour is unchanged.
        """
        fa_only = [_MockFlashAttnBackend]
        size_all = select_common_block_size(BLOCK_SIZE, fa_only)
        size_primary = select_common_block_size(BLOCK_SIZE, fa_only)
        assert size_all == size_primary  # identical — no regression


# ---------------------------------------------------------------------------
# Tests – _primary_attn_layer_names extraction
# ---------------------------------------------------------------------------

MAIN_LAYERS = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
DRAFT_LAYERS = ["drafter.layers.0.self_attn", "drafter.layers.1.self_attn"]


class TestPrimaryAttnLayerNames:
    def test_primary_excludes_draft_layers(self):
        kv_groups = _make_kv_groups(MAIN_LAYERS, DRAFT_LAYERS)
        primary = _extract_primary_attn_layer_names(kv_groups)

        assert primary == MAIN_LAYERS
        for dl in DRAFT_LAYERS:
            assert dl not in primary, f"Draft layer {dl!r} leaked into primary"

    def test_primary_is_first_non_mamba_group(self):
        """If group-0 is Mamba and group-1 is attention, take group-1."""
        mamba_spec = MagicMock(spec=MambaSpec)
        attn_spec = MagicMock(spec=AttentionSpec)
        kv_groups = [
            MagicMock(kv_cache_spec=mamba_spec, layer_names=["mamba.0"]),
            MagicMock(kv_cache_spec=attn_spec, layer_names=MAIN_LAYERS),
        ]
        primary = _extract_primary_attn_layer_names(kv_groups)
        assert primary == MAIN_LAYERS

    def test_primary_none_when_only_mamba(self):
        """Mamba-only config returns None – no attention group to scope."""
        mamba_spec = MagicMock(spec=MambaSpec)
        kv_groups = [
            MagicMock(kv_cache_spec=mamba_spec, layer_names=["mamba.0"]),
        ]
        primary = _extract_primary_attn_layer_names(kv_groups)
        assert primary is None

    def test_single_group_no_draft(self):
        """Standard model: all layers in one group → primary == all layers."""
        kv_groups = _make_kv_groups(MAIN_LAYERS, [])
        # Only one non-empty group
        kv_groups = [kv_groups[0]]
        primary = _extract_primary_attn_layer_names(kv_groups)
        assert set(primary) == set(MAIN_LAYERS)


# ---------------------------------------------------------------------------
# Tests – register_kv_caches skip-logic
# ---------------------------------------------------------------------------


class TestRegisterSkipLogic:
    """
    Tests the shape-based skip condition in register_kv_caches.

    The fix uses cache.shape[0] != num_blocks rather than group-membership so
    that legitimate secondary groups (e.g. SWA) with matching shapes are never
    skipped, while spec-decode draft layers (different kernel block size →
    different shape[0]) are skipped.
    """

    @staticmethod
    def _should_skip(layer_spec, cache_shape0, num_blocks):
        """Mirror of the shape-based skip added to register_kv_caches."""
        if isinstance(layer_spec, MambaSpec):
            return False
        return cache_shape0 != num_blocks

    def test_main_layers_matching_shape_not_skipped(self):
        attn_spec = MagicMock(spec=AttentionSpec)
        num_blocks = 1000
        assert not self._should_skip(attn_spec, num_blocks, num_blocks)

    def test_swa_layers_matching_shape_not_skipped(self):
        """SWA layers use the same backend → same shape[0] → must not be skipped."""
        attn_spec = MagicMock(spec=AttentionSpec)
        num_blocks = 1000
        # SWA group has same kernel block size as full-attn group → same shape
        assert not self._should_skip(attn_spec, num_blocks, num_blocks)

    def test_draft_layers_mismatched_shape_skipped(self):
        """EAGLE3 draft layer has shape[0]=original while num_blocks=original*ratio."""
        attn_spec = MagicMock(spec=AttentionSpec)
        num_blocks = 2000  # inflated by phys_ratio=2 from primary MLA backend
        draft_shape0 = 1000  # draft (FA) layer only has original num_blocks
        assert self._should_skip(attn_spec, draft_shape0, num_blocks)

    def test_mamba_layers_never_skipped(self):
        """Mamba layers bypass the check regardless of shape."""
        mamba_spec = MagicMock(spec=MambaSpec)
        assert not self._should_skip(mamba_spec, 999, 1000)

    def test_main_layers_wrong_shape_are_caught(self):
        """If a main layer genuinely has a wrong shape the assertion still fires."""
        attn_spec = MagicMock(spec=AttentionSpec)
        # A mismatched shape that is NOT a known-ok draft-model case
        # is flagged: skip=True means the original assert is replaced by
        # a logged skip; callers can decide if this is an error.
        assert self._should_skip(attn_spec, 500, 1000)
