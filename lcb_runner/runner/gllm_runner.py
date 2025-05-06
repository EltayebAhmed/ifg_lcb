try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

import dataclasses
from dataclasses import dataclass
from vllm import outputs
from lcb_runner.runner.base_runner import BaseRunner
import gllm
import json
import os

class GLLMMRunnerIFG(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        # self.llm = LLM(
        #     model=model_tokenizer_path,
        #     tokenizer=model_tokenizer_path,
        #     tensor_parallel_size=args.tensor_parallel_size,
        #     dtype=args.dtype,
        #     enforce_eager=True,
        #     disable_custom_all_reduce=True,
        #     enable_prefix_caching=args.enable_prefix_caching,
        #     trust_remote_code=args.trust_remote_code,
        # )
        self._model_identifier = model_tokenizer_path
        api_key = os.environ.get("GLLM_API_KEY", None)
        self.llm = gllm.GLLM(
            server_address=args.server_address,
            api_key=api_key,
        )

        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop,
        )

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[prompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        if remaining_prompts:
            vllm_outputs = self._generate_batch(remaining_prompts, self.sampling_params)
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs

    def _generate_batch(self, prompts: list[str], sampling_params) -> list[list[str]]:
        # self.llm.get_completions(

        # )
        log_path = "/data/debug/lcb/gllm.log"
        with open(log_path, "a") as f:
            json.dump(prompts, f)
            f.write("\n")
        results = []
        for prompt in prompts:
            generation = VLLMLookingOutput(
                ["Welcome to Rotterdam! How can I help you today?"]*
                sampling_params.n
            )
            results.append(generation)
        print(sampling_params)

        return results

    
class VLLMLookingOutput:
    # Looks like a VLLM LLM.generate output.
    # Duck typing says Quack, Quack!
    @dataclasses.dataclass
    class VLLMSingleOutput:
        text: str
    outputs: list[VLLMSingleOutput]

    def __init__(self, outputs: list[str]):
        self.outputs = [self.VLLMSingleOutput(output) for output in outputs]
