# IFG on LiveCodeBench

This is a fork of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) that applies the [Intent Factored Generation](https://github.com/FLAIROx/IFG) inference time method from our [paper](https://arxiv.org/abs/2506.09659) to LiveCodeBench. 

## Introduction
Our method Intent Factored Generation (IFG) is an inference time method that increases the diversity of repeated samples from an LLM. This improved explorations leads to higher `pass@k`, particularly for values of `k > 1`. To apply to IFG to LiveCodeBench we prompt the LLM to generate a comment before every one or two lines of code detailing what the next lines of code will do. We then sample these comments at higher temperatures than the temperatures we use for the actual code.

We those wanting to see the crux without navigating a large codebase to the following [function](https://github.com/EltayebAhmed/ifg_lcb/blob/69128ebcd804e7485fe54e3744f0ed032219e1a3/lcb_runner/runner/ifg_runner.py#L177) where the crux of out method is implemented.

## Installation
You can clone the repository using the following command:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
```

We recommend using [uv](https://github.com/astral-sh/uv)
for managing dependencies, which can be installed a [number of ways](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

Verify that `uv` is installed on your system by running:

```bash
uv --version
```

Once `uv` has been installed, use it to create a virtual environment for
LiveCodeBench and install its dependencies with the following commands:

```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .
```


# Reproducing the Experiments
To reproduce the results in Table 1 in our ArXiv paper you need to carry out the following steps.

## 1. Launch an OpenAI compatible server
Set launch a OpenAI API compatible server running IFG running `Qwen/Qwen2.5-Coder-32B`.
You can do this either using a `vllm` ([instructions here](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)) or you can use our packaged tool, gllm, to launch a server from the command line as follows.
To launch a gllm server to use GPUs 0-7 run the following command
```bash
>> gllm start-cluster -w 0:7 -wp 12000 --port 8181
```
This will start a gllm server with one worker using the GPUs 0-7. 
The  the worker will use a random port between 12000 and 12000 and the main server (which is OpenAI API compatible) will be accessible at `localhost:8181` and `0.0.0.0:8181`.

If you bring your own server and it is not at `localhost:8181` please change the `.yaml` files in `tuning` to point to your server. If you server requires an API key please the environment variable GLLM_API_KEY as follows
```bash
>> export GLLM_API_KEY=<your_api_key>
```

## 2. Launching the Experiments
First set the working directory.
```bash
>> cd ifg_lcb
```

To replicate the baseline run
```bash
>> ./experiments/baseline.sh
```
To replicate the IFG experiment run
```bash
>> ./experiments/ifg.sh
```
To replicate the ablation `IFG - Equal` in Appendix G.1 of the paper 
```bash
>> ./experiments/ifg_shared_temp_ablation.sh
```

If you use this code or the techniques described in the paper please cite
```bibtex
@misc{ahmed2025intentfactoredgenerationunleashing,
      title={Intent Factored Generation: Unleashing the Diversity in Your Language Model}, 
      author={Eltayeb Ahmed and Uljad Berdica and Martha Elliott and Danijela Horak and Jakob N. Foerster},
      year={2025},
      eprint={2506.09659},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.09659}, 
}
```
