try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

import dataclasses
import itertools
import json
from math import exp
from multiprocessing.pool import ThreadPool
import os
from re import I

from vllm import SamplingParams

import gllm
from lcb_runner.runner.base_runner import BaseRunner


class VLLMLookingOutput:
    # Looks like a VLLM LLM.generate output.
    # Duck typing says Quack, Quack!
    @dataclasses.dataclass
    class VLLMSingleOutput:
        text: str

    outputs: list[VLLMSingleOutput]

    def __init__(self, outputs: list[str]):
        self.outputs = [self.VLLMSingleOutput(output) for output in outputs]


MODEL = "Qwen/Qwen2.5-Coder-32B"


class GLLMMRunnerIFG(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )

        self._model_identifier = model_tokenizer_path
        if self._model_identifier.endswith("-IFG"):
            self.mode = "IFG"
            self._model_identifier = self._model_identifier[:-4]

        elif self._model_identifier.endswith("-GLLM"):
            self.mode = "GLLM"
            self._model_identifier = self._model_identifier[:-5]
        else:
            raise ValueError(
                f"Model identifier {self._model_identifier} does not end with -IFG or -GLLM"
            )

        api_key = os.environ.get("GLLM_API_KEY", None)
        self.llm = gllm.GLLM(
            server_address=args.server_address,
            api_key=api_key,
        )
        try:
            self.llm.load_model(
                model_identifier=MODEL,
            )
            self.llm.wait_for_live()
            self.llm.wait_for_health()
        except Exception as e:
            print(f"Error loading model {MODEL}: {e}")

        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop,
        )

        self.use_ifg = args.use_ifg
        self.ifg_even_temperature = args.ifg_even_temperature
        self.ifg_odd_temperature = args.ifg_odd_temperature
        self.ifg_separator = args.ifg_separator
        self.ifg_termination_str = args.ifg_termination_str

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
        print("Batch of %d prompts completed" % len(prompts))
        breakpoint()
        return outputs

    def _generate_batch(
        self, prompts: list[str], sampling_params
    ) -> list[VLLMLookingOutput]:
        results = []
        n_prompts = len(prompts)
        expanded_prompts = [[prompt] * sampling_params.n for prompt in prompts]
        expanded_prompts = sum(expanded_prompts, [])

        # prepare arguments for each prompt
        args_list = [
            (
                sampling_params,
                prompt,
                self.llm,
                self.ifg_separator,
                self.ifg_even_temperature,
                self.ifg_odd_temperature,
                self._model_identifier,
                self.ifg_termination_str,
            )
            for prompt in expanded_prompts
        ]

        # run in parallel threads
        with ThreadPool() as pool:
            results = pool.starmap(ifg_sample_single, args_list)

        results = [
            results[i : i + sampling_params.n]
            for i in range(0, len(results), sampling_params.n)
        ]
        assert (
            len(results) == n_prompts
        ), f"Expected {n_prompts} prompts, but got {len(results)}"
        for i in range(len(results)):
            assert (
                len(results[i]) == sampling_params.n
            ), f"Expected {sampling_params.n} outputs for prompt {i}, but got {len(results[i])}"

        results = [VLLMLookingOutput(result) for result in results]
        return results


def ifg_sample_single(
    sampling_params: SamplingParams,
    prompt: str,
    model: gllm.GLLM,
    separator: str,
    even_temperature: float,
    odd_temperature: float,
    model_name: str,
    ifg_termination_str: str,
) -> str:
    token_budget = sampling_params.max_tokens
    response = ""

    temperatures = [even_temperature, odd_temperature]
    temperatures = itertools.cycle(temperatures)

    for temperature in temperatures:

        gllm_response = model.get_completions(
            # model=model_name,
            model=MODEL,
            prompt=prompt + response,
            max_tokens=token_budget,
            temperature=temperature,
            top_p=sampling_params.top_p,
            frequency_penalty=sampling_params.frequency_penalty,
            presence_penalty=sampling_params.presence_penalty,
            stop=separator,
            n=1,
            return_mode="raw",
        )

        response += gllm_response.choices[0].text
        response += separator
        used_tokens = gllm_response.usage.completion_tokens
        token_budget -= used_tokens
        if token_budget <= 0 or ifg_termination_str in response:
            break

    return response


if __name__ == "__main__":
    from lcb_runner.runner import parser
    from lcb_runner.lm_styles import LanguageModelStore

    args = parser.get_args()
    model = LanguageModelStore[args.model]
    runner = GLLMMRunnerIFG(args, model)
    prompts_path = "/data/debug/lcb/gllm2.jsonl"
    prompts = []
    with open(prompts_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    prompts = sum(prompts, [])

    GLLMMRunnerIFG.run_batch(runner, prompts[:3])
