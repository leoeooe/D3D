# 在昇腾 910B 上从 Qwen3-8B 训练 D3D

本文复现 D3D 论文的主实验流程。默认目标模型为 **Qwen3-8B**。先用目标模型生成的回答把原始 Qwen3-8B 训练成 36 层 D3D 草稿，再逐层剪枝、恢复和选模，交付 31 层的 Hugging Face 检查点。将目标切换为 Qwen3-32B 时，训练数据和验证用的目标模型随之切换，草稿仍从 Qwen3-8B 初始化。[论文 §3.2–3.3、§4.1](https://arxiv.org/html/2611.06425v1#S4.SS1)

## 1. 原文用的训练数据与参数

作者取 **Open-PerfectBlend** 的约 142 万条指令，用各自的目标模型在 **non-thinking** 模式下重新采样回答。可以直接下载作者发布的回答，离线训练无需再生成 142 万次。训练采样设置为 `temperature=0.7`、`top_p=0.8`、`top_k=20`、`min_p=0`、`presence_penalty=0`、`repetition_penalty=1`。论文报告推理评测时使用贪心解码。[原始提示集](https://huggingface.co/datasets/mlabonne/open-perfectblend) · [论文 §4.1](https://arxiv.org/html/2611.06425v1#S4.SS1) · [作者发布的数据表](https://github.com/fudan-generative-vision/D3D#training-data)

| 目标模型 | 对应的训练发布集 | 发布记录数 | JSONL 文件大小 |
| --- | --- | ---: | ---: |
| Qwen3-8B | [PerfectBlend-Qwen3-8B-NoThink-Sampling](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-8B-NoThink-Sampling) | 1,420,909 | 6.23 GB |
| Qwen3-32B | [PerfectBlend-Qwen3-32B-NoThink-Sampling](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-32B-NoThink-Sampling) | 1,420,500 | 6.87 GB |

| 项目 | 论文设置及本指南的对应方式 |
| --- | --- |
| 离线训练 | 3 个 epoch，块长 16，每条最多 128 个随机锚点，分块交叉熵加自回归损失 |
| 数值与批次 | BF16，每个进程批大小 1。原文 2 张 H800、梯度累积 8，全局批大小 16。下文按可用芯片数调整累积次数 |
| 学习率 | 峰值 `1e-5`、余弦衰减至 `1e-6`、1% warmup、权重衰减 0、梯度范数裁剪 1 |
| 剪枝 | 从 36 层逐层减到 31 层，每轮恢复 10,000 **次优化器更新**。选层和选检查点各用独立的 512 个开发样本 |
| 910B 适配 | 本地实现使用 PyTorch AdamW、SDPA 和 HCCL/FSDP2。论文使用 H800 和 8-bit AdamW。原文报告的 360 GPU 小时对应离线对齐阶段 |

论文给出了来源规模和训练轮数，未给出逐条训练清单、两个开发集的样本 ID 或锚点清单。下面使用完整发布文件，按当前训练器的长度条件过滤、按提示去重，再构造固定开发集。每次运行都会记录实际训练行数。这样的划分可在同一数据和软件版本上重建，硬件、优化器和开发集的差异会影响损失、接受长度和最终选中的层。[论文实验设置](https://arxiv.org/html/2611.06425v1#S4.SS1) · [训练器](../training/train/trainer.py) · [恢复训练实现](../training/pruning/train_recovery.py)

## 2. 准备代码、910B 环境和工作目录

本指南对应本地 `feat/ascend` 的训练实现 `f79a148a3c97bc833c747b2241a5199ddc24cc5a`。可在当前开发机 D3D 仓库根目录打包，再把 bundle 复制到 910B。已有该提交的 910B 工作树可直接使用。

```bash
# 开发机
git bundle create /tmp/D3D-ascend.bundle feat/ascend

# 910B，在放置 bundle 的目录执行
git clone -b feat/ascend ./D3D-ascend.bundle D3D
cd D3D
git merge-base --is-ancestor f79a148a3c97bc833c747b2241a5199ddc24cc5a HEAD
git diff --exit-code f79a148a3c97bc833c747b2241a5199ddc24cc5a HEAD -- training common scripts/train.sh scripts/prune.sh requirements-ascend-train.txt
```

以下各段在 910B 的**同一 Bash 会话、D3D 仓库根目录**执行。环境以已装好 CANN 9.2.1、驱动工具 26.1.1、Python 3.12.16、PyTorch 2.10.0+cpu 和 torch_npu 2.10.0.post7 为起点。PyTorch 的 CPU wheel 与 torch_npu 一起提供 NPU 后端。如果这些包已装在专用虚拟环境，直接激活该环境即可。下例适用于框架装在基础 Python 的机器。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按实际 CANN 安装路径调整
set -eo pipefail
python3.12 -m venv --system-site-packages .venv-910b
source .venv-910b/bin/activate
python -m pip install -r requirements-ascend-train.txt
export PY="$(command -v python)"
npu-smi info
npu-smi info -m
```

把下面的卡号换成 910B 上分配给你的 **Chip Phy-ID**。示例假设单机有 4 个空闲芯片。若分配 2 个，填两个 ID，累积次数会自动变成 8。恢复训练还需同时容纳教师、学生及优化器状态，具体可用芯片数和 HBM 容量请以机器为准。将 `D3D_WORK` 设为有足够空间的持久盘绝对路径。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TARGET_SIZE=8B                       # 另一套主实验填 32B
export D3D_WORK="${D3D_WORK:-$HOME/d3d-910b}"
export RUN="$D3D_WORK/run-qwen3-$TARGET_SIZE"
export DRAFT_MODEL="$D3D_WORK/models/Qwen3-8B"
export TARGET_MODEL="$D3D_WORK/models/Qwen3-$TARGET_SIZE"
export RAW="$D3D_WORK/raw/$TARGET_SIZE/perfectblend_qwen3-${TARGET_SIZE,,}-nothink-sampling_regen.jsonl"
export DATA_PATH="$RUN/data/train.jsonl"
export OUTPUT_DIR="$RUN/alignment"
export D3D_TRAIN_DEVICE=npu OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
IFS=, read -ra chips <<< "$ASCEND_RT_VISIBLE_DEVICES"
export NPROC_PER_NODE=${#chips[@]}
(( 16 % NPROC_PER_NODE == 0 )) || { echo '请选择能整除全局批大小 16 的芯片数'; exit 1; }
ACC=$((16 / NPROC_PER_NODE))
devices=()
for ((i=0; i<NPROC_PER_NODE; i++)); do devices+=("npu:$i"); done
mkdir -p "$RUN"
"$PY" - <<'PY'
import os, sys, torch, torch_npu, transformers, accelerate
print(sys.version, torch.__version__, torch_npu.__version__, transformers.__version__, accelerate.__version__)
assert torch.npu.is_available() and torch.npu.device_count() == int(os.environ["NPROC_PER_NODE"])
for i in range(torch.npu.device_count()):
    x = torch.ones(2, device=f"npu:{i}", dtype=torch.bfloat16)
    print(i, torch.npu.get_device_name(i), (x + x).cpu().tolist())
PY
git rev-parse HEAD > "$RUN/code-revision.txt"
"$PY" -m pip freeze > "$RUN/python-packages.txt"
```

预期每个逻辑设备输出 `[2.0, 2.0]`。本地适配验证使用 Transformers 4.57.3 和 Accelerate 1.12.0。可先用仓库自带的小模型检查分块注意力梯度、参数更新及检查点重载，再投入全量数据。

```bash
"$PY" -m pip install pytest
"$PY" -m pytest -q tests/test_ascend_training.py
```

## 3. 下载固定版本的回答和原始模型

两个训练集都有访问门槛。在对应数据卡登录 Hugging Face，按作者页面要求申请访问，再在下载机执行 `hf auth login`，使用该账号有读取权限的 token。可以在有网络的机器执行本节，再把 `models/`、`raw/` 和校验记录原样复制到训练机。开发集生成需要目标模型权重。8B 目标与草稿初值共用同一份权重，32B 目标另下载 Qwen3-32B。[8B 数据卡](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-8B-NoThink-Sampling) · [32B 数据卡](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-32B-NoThink-Sampling) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

```bash
unset HF_HUB_OFFLINE
hf auth login
"$PY" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

size = os.environ["TARGET_SIZE"]
data_rev = {"8B": "68bf7e238f4715bcf5a82e78d777716ced534ca1",
            "32B": "7301b26f05d282715b3ac9e202c04d520cc7e164"}[size]
model_rev = {"8B": "b968826d9c46dd6066d109eabc6255188de91218",
             "32B": "9216db5781bf21249d130ec9da846c4624c16137"}
raw = Path(os.environ["RAW"])
repo = f"fudan-generative-ai/PerfectBlend-Qwen3-{size}-NoThink-Sampling"
hf_hub_download(repo, raw.name, repo_type="dataset", revision=data_rev, local_dir=raw.parent, token=True)
assert raw.stat().st_size == {"8B": 6230953913, "32B": 6874226894}[size]
revisions = {repo: data_rev}
for model_size in dict.fromkeys(["8B", size]):
    name = f"Qwen3-{model_size}"
    revisions[f"Qwen/{name}"] = model_rev[model_size]
    snapshot_download(f"Qwen/{name}", revision=model_rev[model_size],
                      local_dir=Path(os.environ["D3D_WORK"]) / "models" / name,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
Path(os.environ["RUN"], "source-revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
PY
(cd "$(dirname "$RAW")" && sha256sum "$(basename "$RAW")") > "$RUN/source.sha256"
# 传输到训练机后也可用此命令校验
(cd "$(dirname "$RAW")" && sha256sum -c "$RUN/source.sha256")
export HF_HUB_OFFLINE=1
```

## 4. 准备训练集和两份独立开发集

发布文件的 `conversations` 已有 `role/content`，监督目标是最后一轮 assistant 回答。先在 CPU 上扫描 JSONL 一次。这里复用恢复训练的严格模板前缀检查，并应用第一阶段的长度条件，即 prompt 至多 4,096 token、整段至多 8,192 token、回答至少 33 token。索引按对话里的用户输入和系统上下文去重，保留首条符合条件的回答。SQLite 只存提示哈希和字节偏移，原始文件要保持原位。此步包含全量分词，可以单独执行。[离线训练读数与组批](../training/train/dataset.py) · [恢复训练读数](../training/pruning/data.py)

```bash
"$PY" - <<'PY'
import hashlib, json, os, sqlite3, tempfile
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from training.pruning.data import encode_example

run = Path(os.environ["RUN"])
dest = run / "index"
assert not dest.exists(), f"已有索引可直接进入下一段：{dest}"
tok = AutoTokenizer.from_pretrained(os.environ["TARGET_MODEL"], local_files_only=True)
stats = Counter()
with tempfile.TemporaryDirectory(dir=run) as tmp:
    stage = Path(tmp) / "index"
    stage.mkdir()
    db = sqlite3.connect(stage / "rows.sqlite")
    db.execute("CREATE TABLE rows (key TEXT PRIMARY KEY, pos INTEGER) WITHOUT ROWID")
    with open(os.environ["RAW"], "rb") as src:
        while True:
            pos = src.tell()
            line = src.readline()
            if not line:
                break
            if not line.strip():
                continue
            stats["release_rows"] += 1
            row = json.loads(line)
            ex = encode_example(tok, row, str(row["id"]), max_prompt_tokens=4096, max_sequence_tokens=8192)
            if ex is None or len(ex.response_ids) <= 32:
                stats["filtered"] += 1
                continue
            messages = row["conversations"]
            end = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
            prompt = [(m["role"], m["content"]) for m in messages[:end] if m["role"] != "assistant"]
            payload = json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode()
            key = hashlib.sha256(b"d3d-910b-v1\0" + payload).hexdigest()
            inserted = db.execute("INSERT OR IGNORE INTO rows VALUES (?, ?)", (key, pos)).rowcount
            stats["unique_eligible" if inserted else "duplicate"] += 1
    db.commit()
    db.close()
    assert stats["unique_eligible"] > 1024, dict(stats)
    (stage / "counts.json").write_text(json.dumps(stats, indent=2) + "\n")
    stage.rename(dest)
print(dict(stats))
PY
```

下面的开发集构造方式是本指南的复现选择。按稳定哈希遍历合格提示，从每份回答选一个固定位置，将此前 token 交给**原始目标模型**，以贪心生成连续 16 个 token。首 token 是草稿可见的锚点，后 15 个用于测接受长度。提前结束且不足 16 个的候选会被跳过。合格提示交替进入两份 512 条的开发集。其余提示按固定哈希顺序写为训练集，五轮恢复都复用该文件。若有作者实验的原始开发集，可直接使用其 token 文件并按 prompt 从训练集排除。[选层实际读取的字段](../training/pruning/select_layer.py)

```bash
"$PY" - <<'PY'
import json, os, sqlite3, tempfile
from pathlib import Path
import torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from training.pruning.data import encode_example

run, model_dir = Path(os.environ["RUN"]), os.environ["TARGET_MODEL"]
dest = run / "data"
assert not dest.exists(), f"已有划分可直接进入训练：{dest}"
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
placement = {"": "npu:0"} if os.environ["TARGET_SIZE"] == "8B" else "balanced"
target = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=placement,
                                            attn_implementation="sdpa", local_files_only=True).eval()
input_device = target.get_input_embeddings().weight.device
generation = GenerationConfig(do_sample=False, max_new_tokens=16, use_cache=True,
                              eos_token_id=target.generation_config.eos_token_id, pad_token_id=tok.pad_token_id)
splits = ("layer_selection", "layer_validation")
per_split = 512
counts = json.loads((run / "index/counts.json").read_text())
held = {name: [] for name in splits}
dev_keys = set()
db = sqlite3.connect(run / "index/rows.sqlite")
with open(os.environ["RAW"], "rb") as src, tempfile.TemporaryDirectory(dir=run) as tmp:
    stage = Path(tmp) / "data"
    stage.mkdir()
    with torch.inference_mode():
        for key, pos in db.execute("SELECT key, pos FROM rows ORDER BY key"):
            src.seek(pos)
            row = json.loads(src.readline())
            ex = encode_example(tok, row, str(row["id"]), max_prompt_tokens=4096, max_sequence_tokens=8192)
            offset = int(key[-16:], 16) % (len(ex.response_ids) - 16 + 1)
            prefix = ex.prompt_ids + ex.response_ids[:offset]
            ids = torch.tensor([prefix], device=input_device)
            result = target.generate(ids, attention_mask=torch.ones_like(ids), generation_config=generation,
                                     use_model_defaults=False, do_sample=False)
            block = result[0, len(prefix):].tolist()
            if len(block) < 16:
                continue
            split = splits[len(dev_keys) % 2]
            held[split].append(dict(analysis_id=key, source_id=row["id"], split=split,
                                    prefix_ids=prefix, target_ids=block))
            dev_keys.add(key)
            if len(dev_keys) == 2 * per_split:
                break
    assert all(len(rows) == per_split for rows in held.values()), "合格的完整开发块不足"
    for name, rows in held.items():
        with (stage / f"{name}.jsonl").open("w") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    counts["train"] = 0
    with (stage / "train.jsonl").open("wb") as f:
        for key, pos in db.execute("SELECT key, pos FROM rows ORDER BY substr(key, 33), key"):
            if key not in dev_keys:
                src.seek(pos)
                f.write(src.readline().rstrip(b"\r\n") + b"\n")
                counts["train"] += 1
    assert counts["train"] + len(dev_keys) == counts["unique_eligible"]
    (stage / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    stage.rename(dest)
db.close()
print(counts)
PY
(cd "$RUN/data" && sha256sum *.jsonl > files.sha256)
cat "$RUN/data/counts.json"
```

## 5. 训练完整的 36 层 D3D

启动器按 **micro-step** 计算离线学习率和保存间隔。下面从过滤后的实际行数计算 3 个 epoch 和 1% warmup，每个 epoch 保存一次。目标模型的回答和 tokenizer 已在本地，CE 训练进程只加载草稿模型。训练前可单独用短样本和 `--max-micro-steps` 测显存，用独立的输出目录存放短测结果。[训练入口和默认参数](../scripts/train.sh) · [步数及保存逻辑](../training/train/trainer.py)

```bash
ROWS=$("$PY" -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN"],"data/counts.json")))["train"])')
MICRO_PER_EPOCH=$(( (ROWS + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
TOTAL_MICRO=$((3 * MICRO_PER_EPOCH))
WARMUP=$(( (TOTAL_MICRO + 99) / 100 ))
export FULL="$OUTPUT_DIR/step-$TOTAL_MICRO"
test ! -e "$OUTPUT_DIR"
bash scripts/train.sh --device npu --attn-implementation sdpa --dtype bf16 \
    --epochs 3 --batch-size 1 --gradient-accumulation-steps "$ACC" \
    --objective ce --ce-weight 1 --ar-loss-weight 1 --block-size 16 \
    --parallel-anchors-per-example 128 --loss-decay-gamma 7.75 --disable-thinking-template \
    --lr 1e-5 --min-lr-ratio 0.1 --warmup-steps "$WARMUP" --weight-decay 0 --grad-clip 1 \
    --max-prompt-len 4096 --max-seq-len 8192 --gradient-checkpointing \
    --fsdp --fsdp-reshard-after-forward --seed 42 --disable-wandb \
    --save-every "$MICRO_PER_EPOCH" --log-every 20 2>&1 | tee "$RUN/alignment.log"
"$PY" - <<'PY'
import json, os
from pathlib import Path
ckpt = Path(os.environ["FULL"])
cfg = json.loads((ckpt / "config.json").read_text())
assert cfg["num_hidden_layers"] == 36 and list(ckpt.glob("*.safetensors")), ckpt
print("五轮恢复使用的固定教师", ckpt)
PY
printf '%s\n' "$FULL" > "$RUN/full-depth-teacher.txt"
```

启动日志应显示 `device=npu`、`attention=sdpa`，多芯片时还应显示 `FSDP` 和 `hccl`。检查日志中的分块损失和 AR 损失均为有限数。正式训练请保留 `$FULL` 的绝对路径，后面五轮都用它作为冻结教师。

## 6. 跑满五轮选层、剪枝、恢复和选模

每轮分别测量移去每一层后的接受长度，物理删除选中的一层，在同一训练集上恢复 10,000 步，然后用另一份 512 条数据选出本轮检查点。恢复损失实现含 AR、分块 CE、教师 KL 和隐藏状态匹配。标准 AdamW 和 HBM 开销要按 910B 预算。默认每 500 次更新保存一次，即每轮 20 个候选，五轮共 100 个模型快照。请按首个实际检查点大小规划 TB 级容量，另留全深度教师、剪枝初值和写入临时空间。调大 `--save-steps` 会减少参加选模的候选数。[剪枝入口](../scripts/prune.sh) · [恢复训练](../training/pruning/train_recovery.py) · [检查点选择](../training/pruning/select_checkpoint.py)

```bash
json_field() {
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
parent="$FULL"
cursor=0
for round in 1 2 3 4 5; do
    cycle="$RUN/pruning/cycle-$round"
    mkdir -p "$cycle"
    if [[ ! -f "$cycle/selection/selection.json" ]]; then
        bash scripts/prune.sh select --parent "$parent" --examples "$RUN/data/layer_selection.jsonl" \
            --block-size 16 --expected-samples 512 --candidate-batch-size 1 \
            --devices "${devices[@]}" --output-dir "$cycle/selection"
    fi
    if [[ ! -f "$cycle/pruned/layer_prune.json" ]]; then
        bash scripts/prune.sh prune --selection "$cycle/selection/selection.json" --output "$cycle/pruned"
    fi
    if [[ ! -f "$cycle/recovery/training_summary.json" ]]; then
        bash scripts/prune.sh train --nproc-per-node "$NPROC_PER_NODE" --device npu --optimizer adamw \
            --student "$cycle/pruned" --dataset "$DATA_PATH" --output-dir "$cycle/recovery" \
            --data-cursor "$cursor" --max-steps 10000 --gradient-accumulation-steps "$ACC" \
            --anchors-per-example 128 --learning-rate 1e-5 --min-lr 1e-6 --warmup-ratio 0.01 \
            --weight-decay 0 --gradient-clip 1 --save-steps 500 --seed 42 \
            2>&1 | tee "$cycle/recovery.log"
    fi
    if [[ ! -f "$cycle/chosen/selected_checkpoint.json" ]]; then
        bash scripts/prune.sh choose --run-dir "$cycle/recovery" --examples "$RUN/data/layer_validation.jsonl" \
            --expected-samples 512 --block-size 16 --devices "${devices[@]}" --output-dir "$cycle/chosen"
    fi
    cursor=$(json_field "$cycle/recovery/training_summary.json" next_data_cursor)
    parent=$(json_field "$cycle/chosen/selected_checkpoint.json" selected_checkpoint)
    test "$(json_field "$parent/config.json" num_hidden_layers)" -eq "$((36 - round))"
done
export FINAL_DRAFT="$parent"
printf '%s\n' "$FINAL_DRAFT" > "$RUN/final-draft.txt"
```

跨轮传递的是验证集选出的检查点和恢复脚本写出的 `next_data_cursor`。在同一次实验内重跑循环会跳过已完成的阶段。重新登录后，激活原 Python 环境，恢复第 2 节中的路径、卡号与并行变量，执行 `export FULL="$(cat "$RUN/full-depth-teacher.txt")"` 即可从本节继续。单阶段中断时，先把该阶段的未完成输出目录移到旁边保留，再从该阶段开头重跑。离线训练和恢复训练保存的是模型权重，当前入口没有完整优化器断点续训功能。

本地磁盘紧张时，可在每轮选模完成后把该轮未选中的候选检查点归档到其他盘，并保留固定教师、各轮被选中的检查点和 JSON 记录。HBM 不足时先核对卡上其他进程及实际可见芯片数。多卡可增加分片芯片并重算 `ACC`，也可在训练入口启用 `--fsdp-cpu-offload` 并核对主机内存和吞吐。改变实际长度上限需要重新生成划分并记录配置。

## 7. 验收最终产物

下面逐轮检查数据游标、教师、层数、10,000 次更新、损失和 512 个选模样本，再从最终目录加载 tokenizer、模型代码及权重，在可见的第一颗 NPU 做一次前向。部署时使用 `final-draft.txt` 指向的完整目录。

```bash
"$PY" - <<'PY'
import json, math, os
from pathlib import Path
import torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

run, teacher = Path(os.environ["RUN"]), Path(os.environ["FULL"]).resolve()
parent, cursor = teacher, 0
for i in range(1, 6):
    cycle = run / f"pruning/cycle-{i}"
    sel = json.loads((cycle / "selection/selection.json").read_text())
    summary = json.loads((cycle / "recovery/training_summary.json").read_text())
    best = json.loads((cycle / "chosen/selected_checkpoint.json").read_text())
    assert Path(sel["parent_checkpoint"]).resolve() == parent
    assert Path(summary["teacher"]).resolve() == teacher and sel["child_depth"] == 36 - i
    assert summary["completed_steps"] == 10000 and summary["data_window"]["cursor_start"] == cursor
    assert sel["selection_samples"] == best["samples"] == 512 and math.isfinite(best["mean_accept_length"])
    logged_steps = 0
    with (cycle / "recovery/training_metrics.jsonl").open() as f:
        for logged_steps, line in enumerate(f, 1):
            metric = json.loads(line)
            assert metric["optimizer_step"] == logged_steps and math.isfinite(metric["loss"])
    assert logged_steps == summary["completed_steps"]
    cursor, parent = summary["next_data_cursor"], Path(best["selected_checkpoint"]).resolve()
    assert json.loads((parent / "config.json").read_text())["num_hidden_layers"] == 36 - i
    print(i, 36 - i, best["selected_step"], best["mean_accept_length"])
assert parent == Path((run / "final-draft.txt").read_text().strip()).resolve()
tok = AutoTokenizer.from_pretrained(parent, local_files_only=True, trust_remote_code=True)
model, info = AutoModelForCausalLM.from_pretrained(
    parent, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
    local_files_only=True, output_loading_info=True)
assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), info
assert type(model).__name__ == "D3DForCausalLM" and model.config.num_hidden_layers == 31
assert 0 <= model.config.mask_token_id < model.config.vocab_size
model = model.to("npu:0").eval()
ids = tok("Hello", return_tensors="pt").input_ids.to("npu:0")
with torch.inference_mode():
    logits = model(ids, use_cache=False, draft_causal=True).logits
assert logits.shape[-1] == model.config.vocab_size and torch.isfinite(logits).all().item()
print(type(model).__name__, model.config.num_hidden_layers, "层，前向通过", parent)
PY
```

选模输出的接受长度是在上述固定开发块上算的。论文表 1 的接受长度和加速比还涉及七个推理基准、目标模型及解码配置，属于训练完成后的另一项评测。[论文表 1](https://arxiv.org/html/2611.06425v1#S4.T1)

**实机验证范围（2026-11-19）。** 本机为 Ascend 910_9382（910C）。原始 Qwen3-8B 用 4 个芯片和短合成样本完成 1 次 FSDP2/AdamW 更新。开发集脚本用真实 Qwen3-8B 生成两组各 2 个块，共 64 个 token，逐个通过独立贪心前向核对。另用隐藏维度 64 的 36 层小模型完成 3 个 epoch 和全部 5 轮剪枝，每轮执行 2 次恢复更新，最终 31 层检查点通过重载与前向验收。910B 共用这一实现，其全量发布数据训练、长序列峰值显存和收敛结果仍待实机测量。
