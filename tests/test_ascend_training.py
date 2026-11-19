"""Run the offline training CLI on explicitly selected Ascend devices."""

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from training.device import bind_device, training_attention


needs_npu = pytest.mark.skipif(
    not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    reason="select available NPUs with ASCEND_RT_VISIBLE_DEVICES",
)


def test_attention_backend_selection():
    assert training_attention("auto", "npu") == "sdpa"
    assert training_attention("auto", "cuda") == "flash_attention_2"
    assert training_attention("sdpa", "cuda") == "sdpa"
    with pytest.raises(ValueError, match="sdpa"):
        training_attention("flash_attention_2", "npu")


def tiny_training_inputs(directory):
    checkpoint = directory / "initial"
    tokens = ["<pad>", "<unk>", "<MASK>", "user", "assistant"] + [
        f"w{i}" for i in range(123)
    ]
    backend = Tokenizer(
        WordLevel({s: i for i, s in enumerate(tokens)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<MASK>",
        chat_template=(
            "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    torch.manual_seed(127)
    config = Qwen3Config(
        vocab_size=len(tokens),
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        mask_token_id=2,
    )
    Qwen3ForCausalLM(config).save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    data = directory / "train.jsonl"
    with data.open("w") as rows:
        for i in range(8):
            rows.write(
                json.dumps(
                    {
                        "id": str(i),
                        "messages": [
                            {"role": "user", "content": f"w{i} w0"},
                            {
                                "role": "assistant",
                                "content": " ".join(f"w{j + i}" for j in range(40 + i)),
                            },
                        ],
                    }
                )
                + "\n"
            )
    return checkpoint, data


@needs_npu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "packed"])
def test_npu_attention_gradients(causal):
    from training.model.configuration_d3d import D3DConfig
    from training.model.modeling_d3d import D3DForCausalLM

    device = bind_device("npu:0")
    torch.manual_seed(103)
    config = D3DConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        pad_token_id=0,
        fuse_cross_entropy=False,
    )
    config._attn_implementation = "sdpa"
    cpu = D3DForCausalLM(config).to(torch.bfloat16).float()
    npu = copy.deepcopy(cpu).to(device, dtype=torch.bfloat16)
    ids = torch.tensor([[8, 12, 4, 2, 7, 2]])
    # A causal two-token prefix and two independent bidirectional draft blocks.
    mask = torch.eye(6, dtype=torch.bool)
    mask[1:, 0] = True
    mask[2:4, :4] = True
    mask[4:, :2] = True
    mask[4:, 4:] = True

    def forward_backward(model, dev):
        logits = model(
            ids.to(dev),
            attention_mask=None if causal else mask.to(dev),
            draft_causal=causal,
            use_cache=False,
        ).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], ids[0].flip(0).to(dev))
        loss.backward()
        gradient = model.model.layers[0].self_attn.q_proj.weight.grad
        return logits.detach().cpu(), gradient.detach().float().cpu()

    reference, reference_grad = forward_backward(cpu, "cpu")
    result, grad = forward_backward(npu, device)
    torch.testing.assert_close(result, reference, rtol=0.04, atol=0.01)
    assert (grad - reference_grad).norm() < 0.05 * reference_grad.norm()


@needs_npu
def test_npu_training_checkpoint(tmp_path):
    count = len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","))
    checkpoint, data = tiny_training_inputs(tmp_path)
    output = tmp_path / "training"
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        port = listener.getsockname()[1]
    env = dict(
        os.environ,
        PY=sys.executable,
        DRAFT_MODEL=str(checkpoint),
        TARGET_MODEL=str(checkpoint),
        DATA_PATH=str(data),
        OUTPUT_DIR=str(output),
        MASTER_PORT=str(port),
        NPROC_PER_NODE=str(count),
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
    )
    subprocess.run(
        [
            "bash",
            "scripts/train.sh",
            "--device",
            "npu",
            "--disable-wandb",
            "--block-size",
            "16",
            "--parallel-anchors-per-example",
            "2",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--max-micro-steps",
            "1",
            "--save-every",
            "1",
            "--workers",
            "1",
            "--lr",
            "0.01",
            "--warmup-steps",
            "0",
            "--min-lr-ratio",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        timeout=360,
    )
    args = json.loads((output / "args.json").read_text())
    assert (args["device"], args["attn_implementation"], args["world_size"]) == (
        "npu",
        "sdpa",
        count,
    )
    assert args["backend"] == "hccl"
    if count > 1:
        assert (args["distributed_type"], args["fsdp_version"]) == ("FSDP", 2)
    saved = output / "step-1"
    trained = load_file(saved / "model.safetensors")
    initial = load_file(checkpoint / "model.safetensors")
    assert trained.keys() == initial.keys()
    assert all(torch.isfinite(weight).all() for weight in trained.values())
    key = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(trained[key], initial[key].bfloat16().to(trained[key].dtype))
    model, info = AutoModelForCausalLM.from_pretrained(
        saved,
        trust_remote_code=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    loaded = model.model.layers[0].self_attn.q_proj.weight.detach()
    assert torch.equal(loaded, trained[key].to(loaded.dtype))
    with torch.no_grad():
        output = model(torch.tensor([[3, 5, 6]]), use_cache=False, draft_causal=True)
    assert torch.isfinite(output.logits).all()
