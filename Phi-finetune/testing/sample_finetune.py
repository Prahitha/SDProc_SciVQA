import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

DEFAULT_INSTSRUCTION = "Answer the question based on the information in the image. Do not hallucinate or infer information from general knowledge."
_IGNORE_INDEX = -100
# _TRAIN_SIZE = 15100
# _EVAL_SIZE = 1680
_TRAIN_SIZE = 1000
_EVAL_SIZE = 200
_MAX_TRAINING_LENGTH = 15200


class PmcVqaTrainDataset(Dataset):
    def __init__(self, processor, data_size, instruction=DEFAULT_INSTSRUCTION):
        # Download the file
        file_path = hf_hub_download(
            repo_id='katebor/SciVQA',  # repository name
            filename='images_train.zip',  # file to download
            repo_type='dataset',  # specify it's a dataset repo
            force_download=True,
        )

        # file_path will be the local path where the file was downloaded
        print(f'File downloaded to: {file_path}')

        # unzip to temp folder
        self.image_folder = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.image_folder)

        data_files = {
            'train': 'https://huggingface.co/datasets/katebor/SciVQA/resolve/main/train_2025-03-27_18-34-44.json',
        }
        split = 'train' if data_size is None else f'train[:{data_size}]'
        self.annotations = load_dataset(
            'katebor/SciVQA', data_files=data_files, split=split)
        self.processor = processor
        self.instruction = instruction

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        {'index': 35,
         'Figure_path': 'PMC8253797_Fig4_11.jpg',
         'Caption': 'A slightly altered cell . (c-c‴) A highly altered cell as seen from 4 different angles . Note mitochondria/mitochondrial networks (green), Golgi complexes (red), cell nuclei (light blue) and the cell outline (yellow).',
         'Question': ' What color is used to label the Golgi complexes in the image?',
         'Choice A': ' A: Green ',
         'Choice B': ' B: Red ',
         'Choice C': ' C: Light blue ',
         'Choice D': ' D: Yellow',
         'Answer': 'B',
         'split': 'train'}
        """
        annotation = self.annotations[idx]
        image = Image.open(self.image_folder / "images_train" / annotation['image_file'])
        question = annotation['question']
        choices = [f"{list(choice.keys())[0]}: {list(choice.values())[0]}" for choice in annotation['answer_options']]
        caption = annotation['caption'] 
        figures = annotation['figs_numb'] # if there are 2 figures or multiple plots, we should compare the scales of the axes
        figure_type = annotation['figure_type']
        qa_pair_type = annotation['qa_pair_type']

        if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
            if "non-binary" in qa_pair_type and choices:
                final_prompt = self.instruction + " " + (
                    f"Match the answer to one or more of the provided answer options: {{{choices}}}. "
                    "Return only the corresponding letter(s) of the correct answer(s). "
                    "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                    "Only output the letter(s) corresponding to the correct choice. "
                    "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                    "If all options are correct, return A,B,C,D. "
                    "Do not add anything else."
                )
            elif "binary" in qa_pair_type:
                final_prompt = self.instruction + " " + "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
            else:
                final_prompt = self.instruction + " " + "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined or that the information is insufficient, \
                return exactly: 'It is not possible to answer this question based only on the provided data.'"
        else:
            final_prompt = self.instruction + " " + "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined or that the information is insufficient, \
                return exactly: 'It is not possible to answer this question based only on the provided data.'"


        user_message = {
            'role': 'user',
            'content': '<|image_1|>' + '\n'.join([question] + choices + [final_prompt]),
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{annotation["answer"]}<|end|><|endoftext|>'
        # inputs = self.processor(prompt, images=[image], return_tensors='pt')

        # answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids

        # input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        # labels = torch.full_like(input_ids, _IGNORE_INDEX)
        # labels[:, -answer_ids.shape[1]:] = answer_ids

        # if input_ids.size(1) > _MAX_TRAINING_LENGTH:
        #     input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
        #     labels = labels[:, :_MAX_TRAINING_LENGTH]
        #     if torch.all(labels == _IGNORE_INDEX).item():
        #         # workaround to make sure loss compute won't fail
        #         labels[:, -1] = self.processor.tokenizer.eos_token_id

        return {
            # 'input_ids': input_ids,
            # 'labels': labels,
            'prompt': prompt,
            'image': image.convert("RGB"),
            'answer': answer,
            # 'input_image_embeds': inputs.input_image_embeds,
            # 'image_attention_mask': inputs.image_attention_mask,
            # 'image_sizes': inputs.image_sizes,
        }

    def __del__(self):
        __import__('shutil').rmtree(self.image_folder)


class DataCollatorWithProcessor:
    def __init__(self, processor, ignore_index=_IGNORE_INDEX, max_length=_MAX_TRAINING_LENGTH):
        self.processor = processor
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        prompts = [x['prompt'] for x in batch]
        images = [x['image'] for x in batch]
        answers = [f"{x['answer']}<|end|><|endoftext|>" for x in batch]

        # Tokenize prompt + image
        inputs = self.processor(prompts, images=images, return_tensors='pt', padding=True, truncation=True)

        # Tokenize answers
        answer_ids = self.tokenizer(answers, return_tensors='pt', padding=True, truncation=True).input_ids

        labels = torch.full_like(inputs.input_ids, _IGNORE_INDEX)
        for i in range(len(answer_ids)):
            answer_len = answer_ids[i].shape[0]
            labels[i, -answer_len:] = answer_ids[i]

        inputs['labels'] = labels
        inputs['input_mode'] = torch.ones(len(images), dtype=torch.long)  # vision mode

        return BatchFeature(inputs)


class EvalCollatorWithProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        ids = [x['id'] for x in batch]
        prompts = [x['prompt'] for x in batch]
        images = [x['image'] for x in batch]
        answers = [x['answer'] for x in batch]

        inputs = self.processor(prompts, images=images, return_tensors='pt', padding=True, truncation=True)
        inputs['input_mode'] = torch.ones(len(images), dtype=torch.long)

        return ids, answers, BatchFeature(inputs)


class PmcVqaEvalDataset(Dataset):
    def __init__(
        self, processor, data_size, instruction=DEFAULT_INSTSRUCTION, rank=0, world_size=1
    ):
        # Download the file
        file_path = hf_hub_download(
            repo_id='katebor/SciVQA',  # repository name
            filename='images_validation.zip',  # file to download
            repo_type='dataset',  # specify it's a dataset repo
            force_download=True,
        )

        # file_path will be the local path where the file was downloaded
        print(f'File downloaded to: {file_path}')

        # unzip to temp folder
        self.image_folder = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.image_folder)

        data_files = {
            'validation': 'https://huggingface.co/datasets/katebor/SciVQA/resolve/main/validation_2025-03-27_18-34-44.json',
        }
        split = 'validation' if data_size is None else f'validation[:{data_size}]'
        self.annotations = load_dataset(
            'katebor/SciVQA', data_files=data_files, split=split
        ).shard(num_shards=world_size, index=rank)
        self.processor = processor
        self.instruction = instruction

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        {'index': 62,
         'Figure_path': 'PMC8253867_Fig2_41.jpg',
         'Caption': 'CT pulmonary angiogram reveals encasement and displacement of the left anterior descending coronary artery ( blue arrows ).',
         'Question': ' What is the name of the artery encased and displaced in the image? ',
         'Choice A': ' A: Right Coronary Artery ',
         'Choice B': ' B: Left Anterior Descending Coronary Artery ',
         'Choice C': ' C: Circumflex Coronary Artery ',
         'Choice D': ' D: Superior Mesenteric Artery ',
         'Answer': 'B',
         'split': 'test'}
        """
        annotation = self.annotations[idx]
        image = Image.open(self.image_folder / "images_validation" / annotation['image_file'])
        question = annotation['question']
        choices = [f"{list(choice.keys())[0]}: {list(choice.values())[0]}" for choice in annotation['answer_options']]
        caption = annotation['caption'] 
        figures = annotation['figs_numb'] # if there are 2 figures or multiple plots, we should compare the scales of the axes
        figure_type = annotation['figure_type']
        qa_pair_type = annotation['qa_pair_type']

        if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
            if "non-binary" in qa_pair_type and choices:
                final_prompt = self.instruction + " " + (
                    f"Match the answer to one or more of the provided answer options: {{{choices}}}. "
                    "Return only the corresponding letter(s) of the correct answer(s). "
                    "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                    "Only output the letter(s) corresponding to the correct choice. "
                    "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                    "If all options are correct, return A,B,C,D. "
                    "Do not add anything else."
                )
            elif "binary" in qa_pair_type:
                final_prompt = self.instruction + " " + "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
            else:
                final_prompt = self.instruction + " " + "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined or that the information is insufficient, \
                return exactly: 'It is not possible to answer this question based only on the provided data.'"
        else:
            final_prompt = self.instruction + " " + "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined or that the information is insufficient, \
                return exactly: 'It is not possible to answer this question based only on the provided data.'"


        user_message = {
            'role': 'user',
            'content': '<|image_1|>' + '\n'.join([question] + choices + [final_prompt]),
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = annotation['answer']
        # inputs = self.processor(prompt, images=[image], return_tensors='pt')

        unique_id = annotation["instance_id"]
        return {
            'id': unique_id,
            # 'input_ids': inputs.input_ids,
            'prompt': prompt,
            'image': image.convert("RGB"),
            # 'input_image_embeds': inputs.input_image_embeds,
            # 'image_attention_mask': inputs.image_attention_mask,
            # 'image_sizes': inputs.image_sizes,
            'answer': answer,
        }

    def __del__(self):
        __import__('shutil').rmtree(self.image_folder)


## do we need this?
def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


## do we need this?
def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def pmc_vqa_collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'][0] for x in batch], padding_side='right')
    labels = pad_sequence([x['labels'][0] for x in batch], padding_side='right')
    images = [x['image'] for x in batch]
    attention_mask = (input_ids != 0).long()
    return BatchFeature({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'images': images,
        'input_mode': 1,
    })


def pmc_vqa_eval_collate_fn(batch):
    input_ids_list = []
    images = []
    all_unique_ids = []
    all_answers = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        images.append(inputs['image'])  # raw image
        all_unique_ids.append(inputs['id'])
        all_answers.append(inputs['answer'])

    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    attention_mask = (input_ids != 0).long()

    return (
        all_unique_ids,
        all_answers,
        BatchFeature({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'input_mode': 1,
        }),
    )


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    )
    
    original_audio_module = model.model.embed_tokens_extend.audio_embed
    
    class SmartDummyAudioEmbed(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            
        def forward(self, *args, **kwargs):
            if 'audio_inputs' in kwargs and kwargs['audio_inputs'] is not None:
                batch_size = kwargs['audio_inputs'].shape[0]
                seq_length = kwargs['audio_inputs'].shape[1] if len(kwargs['audio_inputs'].shape) > 1 else 1
                device = kwargs['audio_inputs'].device
            else:
                batch_size = 1
                seq_length = 1
                device = next(model.parameters()).device
                
            return torch.zeros((batch_size, seq_length, self.hidden_size), 
                              device=device, 
                              dtype=next(model.parameters()).dtype)
    
    model.model.embed_tokens_extend.audio_embed = SmartDummyAudioEmbed()
    
    class DummyLoRAAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return torch.zeros_like(x)
    
    for layer in model.model.layers:
        layer.mlp.down_proj.lora_A.speech = DummyLoRAAdapter()
        layer.mlp.down_proj.lora_B.speech = DummyLoRAAdapter()
        layer.mlp.gate_up_proj.lora_A.speech = DummyLoRAAdapter()
        layer.mlp.gate_up_proj.lora_B.speech = DummyLoRAAdapter()
        layer.self_attn.o_proj.lora_A.speech = DummyLoRAAdapter()
        layer.self_attn.o_proj.lora_B.speech = DummyLoRAAdapter()
        layer.self_attn.qkv_proj.lora_A.speech = DummyLoRAAdapter()
        layer.self_attn.qkv_proj.lora_B.speech = DummyLoRAAdapter()

    for param in model.parameters():
        param.requires_grad = False
        
    model.gradient_checkpointing_enable()
    
    model.set_lora_adapter('vision')
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable_params:,} parameters out of {total_params:,} total parameters")
    print(f"Percentage of parameters being trained: {trainable_params/total_params*100:.2f}%")
    
    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Get the actual model if it's wrapped in DDP
    if hasattr(model, 'module'):
        inference_model = model.module
    else:
        inference_model = model
        
    inference_model.eval()
    all_answers = []
    all_generated_texts = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=EvalCollatorWithProcessor(processor),
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    for ids, answers, inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        all_answers.extend({'id': i, 'answer': a.strip().lower()} for i, a in zip(ids, answers))

        inputs = inputs.to(inference_model.device)
        generated_ids = inference_model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64, num_logits_to_keep=1
        )

        input_len = inputs.input_ids.size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_generated_texts.extend(
            {'id': i, 'generated_text': g.strip().lower()} for i, g in zip(ids, generated_texts)
        )

    # gather outputs from all ranks
    all_answers = gather_object(all_answers)
    all_generated_texts = gather_object(all_generated_texts)

    if rank == 0:
        assert len(all_answers) == len(all_generated_texts)
        acc = sum(
            a['answer'] == g['generated_text'] for a, g in zip(all_answers, all_generated_texts)
        ) / len(all_answers)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers_unique': all_answers,
                    'generated_texts_unique': all_generated_texts,
                    'accuracy': acc,
                }
                json.dump(save_dict, f)

        return acc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--dynamic_hd',
        type=int,
        default=36,
        help='Number of maximum image crops',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no_tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--full_run', action='store_true', help='Run the full training and eval')
    args = parser.parse_args()

    accelerator = Accelerator()
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # First load the processor on all ranks
    with accelerator.main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            dynamic_hd=args.dynamic_hd,
        )

    # Then create and prepare the model
    # Important: create_model now contains the gradient_checkpointing_enable and other setup
    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
    )

    # Create datasets
    train_dataset = PmcVqaTrainDataset(processor, data_size=None if args.full_run else _TRAIN_SIZE)
    eval_dataset = PmcVqaEvalDataset(
        processor,
        data_size=None if args.full_run else _EVAL_SIZE,
        rank=rank,
        world_size=world_size,
    )

    # Prepare with accelerator (this wraps model with DDP if using distributed training)
    model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

    # Calculate gradient accumulation steps
    num_gpus = accelerator.num_processes
    print(f'Training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    # Set precision flags
    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # Training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=2,
        save_only_model=True,
        eval_strategy='steps',
        eval_steps=200,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
    )

    # Make output directory
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Evaluate before fine-tuning
    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    
    if accelerator.is_main_process:
        print(f'Accuracy before finetuning: {acc}')

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithProcessor(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train the model
    trainer.train()
    
    # Save model
    trainer.save_model()
    accelerator.wait_for_everyone()

    # Free memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # Reload the model for inference evaluation
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    )

    # Prepare model for evaluation
    model = accelerator.prepare(model)
    
    # Evaluate after fine-tuning
    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    
    if accelerator.is_main_process:
        print(f'Accuracy after finetuning: {acc}')


if __name__ == '__main__':
    main()
