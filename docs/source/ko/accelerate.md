# Accelerate

[Accelerate](https://hf.co/docs/accelerate/index)는 PyTorch로 모든 종류의 설정에서 분산 학습을 단순화하도록 설계된 라이브러리이며, 가장 일반적인 프레임워크([Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 및 [DeepSpeed](https://www.deepspeed.ai/))를 단일 인터페이스로 통합합니다. [`Trainer`]는 Accelerate로 구동되며, 대규모 모델을 로드하고 분산 학습을 가능하게 합니다.

이 가이드는 FSDP를 백엔드로 사용하여 Accelerate를 Transformers와 함께 사용하는 두 가지 방법을 보여줍니다. 첫 번째 방법은 [`Trainer`]를 사용한 분산 학습을 보여주고, 두 번째 방법은 PyTorch 학습 루프를 적응시키는 것을 보여줍니다. Accelerate에 대한 자세한 정보는 [설명서](https://hf.co/docs/accelerate/index)를 참조하세요.

```bash
pip install accelerate
```

명령줄에서 [accelerate config](https://hf.co/docs/accelerate/main/en/package_reference/cli#accelerate-config)를 실행하여 학습 시스템에 대한 일련의 프롬프트에 답변함으로써 시작하세요. 이렇게 하면 설정에 따라 Accelerate가 학습을 올바르게 설정하는 데 도움이 되는 구성 파일이 생성되고 저장됩니다.

```bash
accelerate config
```

설정과 제공한 답변에 따라, 한 대의 머신에서 두 개의 GPU로 FSDP 학습을 분산하기 위한 예제 구성 파일은 다음과 같이 보일 수 있습니다.

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Trainer

저장된 구성 파일의 경로를 [`TrainingArguments`]에 전달한 다음, [`TrainingArguments`]를 [`Trainer`]에 전달하세요.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    fsdp_config="path/to/fsdp_config",
    fsdp="full_shard",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Native PyTorch

Accelerate는 모든 PyTorch 학습 루프에 추가되어 분산 학습을 가능하게 할 수 있습니다. [`~accelerate.Accelerator`]는 PyTorch 코드를 Accelerate와 함께 작동하도록 적응시키기 위한 주요 진입점입니다. 분산 학습 설정을 자동으로 감지하고 학습에 필요한 모든 구성 요소를 초기화합니다. 모델을 장치에 명시적으로 배치할 필요가 없습니다. [`~accelerate.Accelerator`]는 모델을 어느 장치로 이동할지 알고 있습니다.

```py
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
```

이제 모든 PyTorch 객체(모델, 옵티마이저, 스케줄러, 데이터로더)를 [`~accelerate.Accelerator.prepare`] 메소드에 전달해야 합니다. 이 메소드는 모델을 적절한 장치 또는 장치로 이동하고, 옵티마이저와 스케줄러를 [`~accelerate.optimizer.AcceleratedOptimizer`] 및 [`~accelerate.scheduler.AcceleratedScheduler`]를 사용하도록 적응시키며, 새로운 샤드 가능한 데이터로더를 생성합니다.

```py
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

학습 루프에서 `loss.backward`를 Accelerate의 [`~accelerate.Accelerator.backward`] 메소드로 바꾸어 그레이디언트를 스케일링하고 프레임워크(예: DeepSpeed 또는 Megatron)에 따라 적절한 `backward` 메소드를 결정하세요.

```py
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

모든 것을 함수로 결합하고 스크립트로 호출할 수 있도록 만드세요.

```py
from accelerate import Accelerator
  
def main():
  accelerator = Accelerator()

  model, optimizer, training_dataloader, scheduler = accelerator.prepare(
      model, optimizer, training_dataloader, scheduler
  )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      accelerator.backward(loss)
      optimizer.step()
      scheduler.step()

if __name__ == "__main__":
    main()
```

명령줄에서 [accelerate launch](https://hf.co/docs/accelerate/main/en/package_reference/cli#accelerate-launch)를 호출하여 학습 스크립트를 실행하세요. 추가 인수나 매개변수를 여기에 전달할 수 있습니다.

학습 스크립트를 두 개의 GPU에서 시작하려면 `--num_processes` 인수를 추가하세요.

```bash
accelerate launch --num_processes=2 your_script.py
```

자세한 내용은 [Launching Accelerate scripts](https://hf.co/docs/accelerate/main/en/basic_tutorials/launch)를 참조하세요.