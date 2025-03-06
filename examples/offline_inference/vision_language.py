# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
import random

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Aria
def run_aria(questions: list[str], modality: str):
    assert modality == "image"
    model_name = "rhymes-ai/Aria"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=2,
              dtype="bfloat16",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompts = [(f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>{question}"
                "<|im_end|>\n<|im_start|>assistant\n")
               for question in questions]

    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
    return llm, prompts, stop_token_ids


# BLIP-2
def run_blip2(questions: list[str], modality: str):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompts = [f"Question: {question} Answer:" for question in questions]
    llm = LLM(model="Salesforce/blip2-opt-2.7b",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Chameleon
def run_chameleon(questions: list[str], modality: str):
    assert modality == "image"

    prompts = [f"{question}<image>" for question in questions]
    llm = LLM(model="facebook/chameleon-7b",
              max_model_len=4096,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Deepseek-VL2
def run_deepseek_vl2(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "deepseek-ai/deepseek-vl2-tiny"

    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
              hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]})

    prompts = [
        f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
        for question in questions
    ]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Florence2
def run_florence2(question: str, modality: str):
    assert modality == "image"

    llm = LLM(model="microsoft/Florence-2-large",
              tokenizer="facebook/bart-large",
              max_num_seqs=8,
              trust_remote_code=True,
              dtype="bfloat16",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompt = "<MORE_DETAILED_CAPTION>"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Fuyu
def run_fuyu(questions: list[str], modality: str):
    assert modality == "image"

    prompts = [f"{question}\n" for question in questions]
    llm = LLM(model="adept/fuyu-8b",
              max_model_len=2048,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Gemma 3
def run_gemma3(questions: list[str], modality: str):
    assert modality == "image"
    model_name = "google/gemma-3-4b-it"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompts = [("<bos><start_of_turn>user\n"
                f"<start_of_image>{question}<end_of_turn>\n"
                "<start_of_turn>model\n") for question in questions]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# GLM-4v
def run_glm4v(questions: list[str], modality: str):
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True,
              hf_overrides={"architectures": ["GLM4VForCausalLM"]},
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    prompts = [
        f"<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>\
        {question}<|assistant|>" for question in questions
    ]

    stop_token_ids = [151329, 151336, 151338]
    return llm, prompts, stop_token_ids


# H2OVL-Mississippi
def run_h2ovl(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-800m"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-800m
    stop_token_ids = [tokenizer.eos_token_id]
    return llm, prompts, stop_token_ids


# Idefics3-8B-Llama3
def run_idefics3(questions: list[str], modality: str):
    assert modality == "image"
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    prompts = [(
        f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    ) for question in questions]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# InternVL
def run_internvl(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompts, stop_token_ids


# LLaVA-1.5
def run_llava(questions: list[str], modality: str):
    assert modality == "image"

    prompts = [
        f"USER: <image>\n{question}\nASSISTANT:" for question in questions
    ]

    llm = LLM(model="llava-hf/llava-1.5-7b-hf",
              max_model_len=4096,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(questions: list[str], modality: str):
    assert modality == "image"

    prompts = [f"[INST] <image>\n{question} [/INST]" for question in questions]
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(questions: list[str], modality: str):
    assert modality == "video"

    prompts = [
        f"USER: <video>\n{question} ASSISTANT:" for question in questions
    ]
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf",
              max_model_len=8192,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(questions: list[str], modality: str):

    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n" for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n" for question in questions
        ]

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384,
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Mantis
def run_mantis(questions: list[str], modality: str):
    assert modality == "image"

    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'  # noqa: E501
    prompts = [
        llama3_template.format(f"{question}\n<image>")
        for question in questions
    ]

    llm = LLM(
        model="TIGER-Lab/Mantis-8B-siglip-llama3",
        max_model_len=4096,
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    stop_token_ids = [128009]
    return llm, prompts, stop_token_ids


# MiniCPM-V
def run_minicpmv_base(questions: list[str], modality: str, model_name):
    assert modality in ["image", "video"]
    # If you want to use `MiniCPM-o-2_6` with audio inputs, check `audio_language.py` # noqa

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    # model_name = "openbmb/MiniCPM-V-2_6"
    # o2.6

    # modality supports
    # 2.0: image
    # 2.5: image
    # 2.6: image, video
    # o2.6: image, video, audio
    # model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6 / o2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    prompts = [
        tokenizer.apply_chat_template(
            [{
                'role': 'user',
                'content': f"{modality_placeholder[modality]}\n{question}"
            }],
            tokenize=False,
            add_generation_prompt=True) for question in questions
    ]
    return llm, prompts, stop_token_ids


def run_minicpmo(questions: list[str], modality: str):
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(questions: list[str], modality: str):
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-V-2_6")


# LLama 3.2
def run_mllama(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [[{
        "role":
        "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": question
        }]
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=False)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Molmo
def run_molmo(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [
        f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n" for question in questions
    ]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# NVLM-D
def run_nvlm_d(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# PaliGemma
def run_paligemma(question: str, modality: str):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = ["caption en"]
    llm = LLM(model="google/paligemma-3b-mix-224",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma 2
def run_paligemma2(question: str, modality: str):
    assert modality == "image"

    # PaliGemma 2 has special prompt format for VQA
    prompt = ["caption en"]
    llm = LLM(model="google/paligemma2-3b-ft-docci-448",
              disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(questions: list[str], modality: str):
    assert modality == "image"

    prompts = [
        f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        for question in questions
    ]

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Phi-4-multimodal-instruct
def run_phi4mm(questions: list[str], modality: str):
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process image inputs.
    """
    assert modality == "image"
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    vision_lora_path = os.path.join(model_path, "vision-lora")
    prompts = [
        f"<|user|><|image_1|>{question}<|end|><|assistant|>"
        for question in questions
    ]
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        enable_lora=True,
        max_lora_rank=320,
        lora_extra_vocab_size=0,
    )
    lora_request = LoRARequest("vision", 1, vision_lora_path)
    # To maintain code compatibility in this script, we add LoRA here.
    llm.llm_engine.add_lora(lora_request=lora_request)
    # You can also add LoRA using:
    # llm.generate(prompts, lora_request=lora_request,...)

    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(questions: list[str], modality: str):
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Qwen
def run_qwen_vl(questions: list[str], modality: str):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [f"{question}Picture 1: <img></img>\n" for question in questions]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(questions: list[str], modality: str):

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str):

    model_name = "/root/wl/cache/modelscope/models/Qwen/Qwen2.5-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]
    stop_token_ids = None
    return llm, prompts, stop_token_ids


model_example_map = {
    "aria": run_aria,
    "blip-2": run_blip2,
    "chameleon": run_chameleon,
    "deepseek_vl_v2": run_deepseek_vl2,
    "florence2": run_florence2,
    "fuyu": run_fuyu,
    "gemma3": run_gemma3,
    "glm4v": run_glm4v,
    "h2ovl_chat": run_h2ovl,
    "idefics3": run_idefics3,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "mantis": run_mantis,
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "mllama": run_mllama,
    "molmo": run_molmo,
    "NVLM_D": run_nvlm_d,
    "paligemma": run_paligemma,
    "paligemma2": run_paligemma2,
    "phi3_v": run_phi3v,
    "phi4_mm": run_phi4mm,
    "pixtral_hf": run_pixtral_hf,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5_vl": run_qwen2_5_vl,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_questions = ["Why is this video funny?"]

        return {
            "data": video,
            "questions": vid_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(image_repeat_prob, num_prompts, data,
                       prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: cur_image
            }
        })

    return inputs


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    llm, prompts, stop_token_ids = model_example_map[model](questions,
                                                            modality)
    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = prompts if args.use_different_prompt_per_request else [
        prompts[0]
    ]

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {
                modality: data
            },
        }
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs = apply_image_repeat(args.image_repeat_prob,
                                        args.num_prompts, data, prompts,
                                        modality)
        else:
            # Use the same image for all prompts
            inputs = [{
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {
                    modality: data
                },
            } for i in range(args.num_prompts)]

    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen2_5_vl",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=4,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')

    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=None,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--disable-mm-preprocessor-cache',
        action='store_true',
        help='If True, disables caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')

    parser.add_argument(
        '--use-different-prompt-per-request',
        action='store_true',
        help='If True, then use different prompt (with the same multi-modal '
        'data) for each request.')

    args = parser.parse_args()
    main(args)
