import dataclasses
import time
import tyro
import os
import numpy as np
import logging
import subprocess
import json
import glob
from collections import namedtuple
import ifg_utils
import datetime

@dataclasses.dataclass(kw_only=True)
class TuneTemperature:
    output_dir: str
    temp_min: float = 0.1
    temp_max: float = 0.7
    temp_comment_max: float = 1.2
    run_budget: int = 10
    mode: str = "ifg"
    seed: int = 42
    start_date_tune: str = "2023-10-1"#"2023-10-01"
    end_date_tune: str = "2023-11-1" #"2023-10-31"

    start_date_eval: str = "2025-01-01"#"2024-10-01"
    end_date_eval: str = "2025-02-01"
    default_args: str = (
        "--scenario codegeneration --evaluate --multiprocess=12 --cache_batch_size=2 --n=10"
    )
    ifg_use_global_temperature: bool = False
    metric: str = "pass@10"
    server_address: str =  "http://52.23.181.222:8181"

    def __post_init__(self):
        if self.mode not in ["ifg", "direct"]:
            raise ValueError("mode must be either 'ifg' or 'direct'")
        if self.temp_min < 0 or self.temp_max < 0:
            raise ValueError("temp_min and temp_max must be non-negative")
        if self.temp_min >= self.temp_max:
            raise ValueError("temp_min must be less than temp_max")
        if self.run_budget <= 0:
            raise ValueError("run_budget must be a positive integer")

        # TODO why is post_init called multiple times. Feels sketchy.
        if "server_address" not in self.default_args:
            self.default_args = (
                f"{self.default_args} --server-address={self.server_address}"
        )


def run_for_given_params(
    mode: str,
    sampled_args: str,
    start_date: str,
    end_date: str,
    default_args: str,
    output_dir: str,
):
    if mode == "ifg":
        model_arg = "--model generic-ifg-model"
    elif mode == "direct":
        model_arg = "--model generic-vanilla-gllm"
    else:
        raise ValueError("Invalid mode")
    custom_args = f" {model_arg} --start_date {start_date} --end_date {end_date}"
    custom_args += f" {sampled_args} --output_dir {output_dir}"

    command = f"python -m lcb_runner.runner.main {default_args} {custom_args}"
    logging.info(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


def sample_hps(min_temp: float, max_temp: float, temp_comment_max:float, mode: str, rng: np.random.RandomState,
ifg_use_global_temperature: bool = False):
    if mode == "ifg":
        if ifg_use_global_temperature:
            temp_even = rng.uniform(low=min_temp, high=max_temp)
            temp_odd = temp_even
        else:
            temp_even = rng.uniform(low=min_temp, high=max_temp)
            temp_odd = rng.uniform(low=temp_even, high=temp_comment_max)

        directory_slug = f"ifg_temp_even_{temp_even:.2f}_odd_{temp_odd:.2f}"
        sampled_args = f"--ifg-even-temperature={temp_even:.2f} --ifg-odd-temperature={temp_odd:.2f}"
    elif mode == "direct":
        temp = rng.uniform(low=min_temp, high=max_temp)
        directory_slug = f"temp_{temp:.2f}"
        sampled_args = f"--temperature={temp:.2f}"
    else:
        raise ValueError("Invalid mode")
    return sampled_args, directory_slug


def load_results(run_output_dir: str):
    glob_pattern = os.path.join(run_output_dir, "*eval.json")
    files = glob.glob(glob_pattern)
    if not files:
        return None
    assert len(files) == 1, f"Multiple eval files found in {run_output_dir}"
    with open(files[0], "r") as f:
        results = json.load(f)
    return results[0]


if __name__ == "__main__":
    cfg = ifg_utils.tyro_cli_with_yaml_support(TuneTemperature)
    os.makedirs(cfg.output_dir, exist_ok=True)
    # assert not os.listdir(
    #     cfg.output_dir
    # ), "Output directory is not empty. Please specify a different output directory."
    # Use London time (Europe/London) which handles BST/GMT changes automatically
    # Timezone logic comes from claude, doubt it handles BST/GMT changes correctly.
    london_tz = datetime.timezone(datetime.timedelta(hours=0), name="Europe/London")
    timestamp = datetime.datetime.now(london_tz).strftime("%Y%m%d_%H%M%S")

    # Set up logging to both file and stdout
    log_file = os.path.join(cfg.output_dir, f"tune_temperature_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    config_file = os.path.join(cfg.output_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)
    logging.info("Configuration saved to %s", config_file)
    
    logging.info("Starting tuning process...")
    logging.info(f"Configuration: {cfg}")

    rng = np.random.RandomState(cfg.seed)

    results = {}
    ordered_results = None
    for i in range(cfg.run_budget):
        sampled_args, directory_slug = sample_hps(
            min_temp=cfg.temp_min, max_temp=cfg.temp_max, mode=cfg.mode, rng=rng,
            temp_comment_max=cfg.temp_comment_max,
            ifg_use_global_temperature=cfg.ifg_use_global_temperature
        )
        logging.info("Step %d/%d", i + 1, cfg.run_budget)
        logging.info(f"Sampled args: {sampled_args}")
        logging.info(f"Directory slug: {directory_slug}")

        run_output_dir = os.path.join(cfg.output_dir, directory_slug)
        logging.info(f"Output directory: {run_output_dir}")

        run_results = load_results(run_output_dir)

        if run_results is None:
            logging.info("Running tune run <%s> for the first time", directory_slug)
            subprocess.run(["rm", "-rf", run_output_dir], check=False)
            os.makedirs(run_output_dir, exist_ok=True)
            start_time = time.time()
            run_for_given_params(
                mode=cfg.mode,
                sampled_args=sampled_args,
                start_date=cfg.start_date_tune,
                end_date=cfg.end_date_tune,
                default_args=cfg.default_args,
                output_dir=run_output_dir,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Log elapsed time using timedelta for better formatting
            elapsed_td = datetime.timedelta(seconds=elapsed_time)
            logging.info(f"Run took {str(elapsed_td)}")

            logging.info("Run completed for <%s>", directory_slug)
            logging.info("Loading results for <%s>", directory_slug)
            run_results = load_results(run_output_dir)
        else:
            logging.info("Results already exist at <%s>", run_output_dir)
            logging.info("Loading results for <%s>", directory_slug)


        assert run_results is not None, f"Results for {directory_slug} are None"

        logging.info("Results: %s", run_results)

        Result = namedtuple("Result", ["metric_value", "sampled_args"])
        results[directory_slug] = Result(run_results[cfg.metric], sampled_args)

        ordered_results = sorted(results.items(), key=lambda x: x[1].metric_value, reverse=True)
        logging.info("End of step %d of %d", i + 1, cfg.run_budget)
        logging.info("Ordered results: \n%s", json.dumps(ordered_results, indent=4))

        results_path = os.path.join(cfg.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(ordered_results, f, indent=4)
        logging.info("Results saved to %s", results_path)

    assert ordered_results is not None, "No results found"
    
    best_args = ordered_results[0][1].sampled_args

    logging.info("Best args: %s", best_args)
    logging.info("Best result: %s", ordered_results[0][1].metric_value)
    test_output_dir = os.path.join(cfg.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)
    run_for_given_params(
        mode=cfg.mode,
        sampled_args=best_args,
        start_date=cfg.start_date_eval,
        end_date=cfg.end_date_eval,
        default_args=cfg.default_args,
        output_dir=test_output_dir,
    )
    logging.info("Test run completed")
    logging.info("Test run output directory: %s", test_output_dir)
    results_for_test = load_results(test_output_dir)  
    assert results_for_test is not None, "Test results are None"
    logging.info("Test results: %s", results_for_test)
    logging.info("Final metric: %s", results_for_test[cfg.metric])
    logging.info("Done")
