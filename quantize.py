import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path

def load_model():
    model_id = 'meta-llama/Llama-3.1-8B-Instruct'

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def load_calibration_dataset(tokenizer):
    DATASET_ID = "mit-han-lab/pile-val-backup"
    DATASET_SPLIT = "validation"

    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 2048 + 128

    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["text"]}],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )
    
    ds = ds.map(tokenize, remove_columns=ds.column_names) # type: ignore

    return ds, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH

def main(args: argparse.Namespace):
    
    method = args.method

    print("=" * 30)
    print(f"Quantization Method:  {method}")
    print("=" * 30)

    model, tokenizer = load_model()

    ds, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH = load_calibration_dataset(tokenizer)

    # if method.startswith("awq"):
    #     from llmcompressor import oneshot
    #     from llmcompressor.utils import dispatch_for_generation
    #     from llmcompressor.modifiers.awq import AWQModifier
        
    #     recipe = [
    #         AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    #     ]

    #     oneshot(
    #         model=model,
    #         dataset=ds, # type: ignore
    #         recipe=recipe, # type: ignore
    #         max_seq_length=MAX_SEQUENCE_LENGTH,
    #         num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    #     )

    #     print("\n\n")
    #     print("========== SAMPLE GENERATION ==============")
    #     dispatch_for_generation(model)
    #     input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    #     output = model.generate(input_ids, max_new_tokens=100)
    #     print(tokenizer.decode(output[0]))
    #     print("==========================================\n\n")

    #     # Save to disk compressed.
    #     quantized_model_id = 'Llama-3.1-8B-Instruct-AWQ-INT4'
    #     SAVE_DIR = (Path("./quantized_model") / quantized_model_id).absolute().as_posix()
    #     model.save_pretrained(SAVE_DIR, save_compressed=True)
    #     tokenizer.save_pretrained(SAVE_DIR)


    if method.startswith("awq"):
        from awq import AutoAWQForCausalLM
        model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        model = AutoAWQForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if method == 'awq_official':
            quantized_model_id = 'Llama-3.1-8B-Instruct-AWQ-Official-INT4'
            quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        elif method == 'awq_marlin':
            quantized_model_id = 'Llama-3.1-8B-Instruct-AWQ-Marlin-INT4'
            quant_config = { "zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "Marlin" }
        else:
            raise Exception("NotImplemented")

        SAVE_DIR = (Path("quantized_model") / quantized_model_id).absolute().as_posix()

        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        print(f"Model is quantized and saved at '{SAVE_DIR}'")
        return
    
    if method == 'gptq_exllamav2':
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        # GPTQ ExLlamaV2 Quantization with AutoGPTQ
        quantized_model_id = 'Llama-3.1-8B-Instruct-GPTQ-ExLlamaV2-INT4'
        SAVE_DIR = (Path("quantized_model") / quantized_model_id).absolute().as_posix()
        model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        quantize_config = BaseQuantizeConfig(bits=4, group_size=128) # type: ignore
        model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        model.quantize(examples)

        model.save_quantized(SAVE_DIR)

        return
    
    if method == 'gptq_marlin':
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor import oneshot
        from llmcompressor.utils import dispatch_for_generation
        # GPTQ Marlin Quantization with LLM-Compressor
        quantized_model_id = 'Llama-3.1-8B-Instruct-GPTQ-INT4'

        model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        ds, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH = load_calibration_dataset(tokenizer)

        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

        oneshot(
            model=model,
            dataset=ds, # type: ignore
            recipe=recipe, # type: ignore
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )

        print("\n\n")
        print("========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to("cuda") for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=100)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

        # Save to disk compressed.
        SAVE_DIR = (Path("./quantized_model") / quantized_model_id).absolute().as_posix()
        model.save_pretrained(SAVE_DIR, save_compressed=True)
        tokenizer.save_pretrained(SAVE_DIR)

        return
    
    if method == 'gptq_machete':
        # GPTQ Machete Quantization with LLM-Compressor
        quantized_model_id = 'Llama-3.1-8B-Instruct-GPTQ-Machete-INT4'
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='Quantization Method', choices=['awq_official', 'awq_marlin', 'gptq_exllamav2', 'gptq_marlin', 'gptq_machete'])

    args = parser.parse_args()
    main(args)