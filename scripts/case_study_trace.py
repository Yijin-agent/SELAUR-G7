"""Case-study tracer: run full ALFWorld episodes and log every step.

Reproduces the training-time agent loop exactly (same prompt templates, same
2-step sliding-window memory, same <think>/<action> projection, same sampling
temperature as validation), and for every step records the observation, the
model's reasoning, the parsed action, validity, environment feedback and the
SELAUR uncertainty signals computed with the repo's own functions.

Usage (on a GPU node):
    python scripts/case_study_trace.py --model PATH --tag NAME \
        --games-filter pick_cool_then_place_in_recep-Apple-None-CounterTop \
        --max-games 4 --out-dir logs/case_study
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment  # noqa: E402
from agent_system.environments.env_package.alfworld.projection import alfworld_projection  # noqa: E402
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS  # noqa: E402
from agent_system.memory import SimpleMemory  # noqa: E402
from selaur.uncertainty.aggregation import discounted_smooth  # noqa: E402
from selaur.uncertainty.signals import combined_uncertainty, step_entropy, step_nll, step_top2_margin  # noqa: E402

ALF_CONFIG = os.path.join(REPO, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
HISTORY_LENGTH = 2   # env.history_length used in training
MAX_STEPS = 50       # env.max_steps used in training
LAMBDA_UQ = 0.95     # selaur.lambda_uq
STEP_WEIGHT = 0.85   # selaur.step_weight


def build_prompt(tokenizer, task, memory, obs, admissible, init):
    acts = "\n ".join(f"'{s}'" for s in admissible if s != "help")
    if init:
        text = ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=obs, admissible_actions=acts)
    else:
        ctx, lens = memory.fetch(HISTORY_LENGTH, obs_key="text_obs", action_key="action")
        text = ALFWORLD_TEMPLATE.format(
            task_description=task, step_count=len(memory[0]), history_length=lens[0],
            action_history=ctx[0], current_step=len(memory[0]) + 1,
            current_observation=obs, admissible_actions=acts)
    chat = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False), text


def uncertainty_of(output):
    """Top-1 / top-2 logprobs per generated token, exactly as the rollout collects them."""
    ids = output.token_ids
    top1, top2 = [], []
    for i, lp in enumerate(output.logprobs):
        top1.append(lp[ids[i]].logprob)
        rank2 = [v for v in lp.values() if v.rank == 2]
        top2.append(rank2[0].logprob if rank2 else lp[ids[i]].logprob - 20.0)
    pair = (torch.tensor(top1), torch.tensor(top2))
    return {
        "entropy": step_entropy(pair, reduction="mean").item(),
        "least_confidence": step_nll(pair, reduction="mean").item(),
        "margin": step_top2_margin(pair, reduction="mean").item(),
        "combined": combined_uncertainty(pair),
        "n_tokens": len(ids),
    }


def run_episode(llm, tokenizer, env, sampling):
    obs, infos = env.reset()
    obs = obs[0]
    gamefile = infos["extra.gamefile"][0]
    task = obs[obs.find("Your task is to: ") + len("Your task is to: "):].strip()
    admissible = infos["admissible_commands"][0]
    memory = SimpleMemory()
    memory.reset(batch_size=1)
    steps, won = [], False
    for t in range(1, MAX_STEPS + 1):
        prompt, user_text = build_prompt(tokenizer, task, memory, obs, admissible, init=(t == 1))
        out = llm.generate([prompt], sampling, use_tqdm=False)[0].outputs[0]
        response = out.text
        actions, valids = alfworld_projection([response], [admissible])
        action = actions[0]
        unc = uncertainty_of(out)
        new_obs, _, dones, new_infos = env.step([action])
        memory.store({"text_obs": [obs], "action": [action]})
        won = bool(new_infos["won"][0])
        think = response.split("<think>")[-1].split("</think>")[0].strip() if "<think>" in response else ""
        steps.append({
            "step": t,
            "observation": obs,
            "n_admissible": len([a for a in admissible if a != "help"]),
            "memory_window": memory.fetch(HISTORY_LENGTH)[0][0] if t > 1 else "",
            "think": think,
            "raw_response": response,
            "action": action,
            "action_valid_format": bool(valids[0]),
            "action_admissible": action in admissible,
            "env_feedback": new_obs[0],
            "won_after_step": won,
            "uncertainty": unc,
            "prompt_user_text": user_text if t in (1, 2) else None,
        })
        obs, admissible = new_obs[0], new_infos["admissible_commands"][0]
        if dones[0]:
            break
    raw = np.array([s["uncertainty"]["combined"] for s in steps])
    smoothed = discounted_smooth(raw, LAMBDA_UQ)
    for s, u in zip(steps, smoothed):
        s["uncertainty"]["smoothed"] = float(u)
        # linear transform: failed episodes get +u (step level scaled by step_weight)
        s["uncertainty"]["step_shaping_if_failed"] = float(u * STEP_WEIGHT)
    return {
        "gamefile": gamefile,
        "task": task,
        "won": won,
        "n_steps": len(steps),
        "episode_uncertainty": float(np.mean(smoothed)),
        "episode_shaping_if_failed": float(np.mean(smoothed)),
        "steps": steps,
    }


def write_readable(ep, path, tag):
    with open(path, "w") as f:
        f.write(f"===== [{tag}] {ep['task']} =====\n")
        f.write(f"game: {ep['gamefile']}\n")
        f.write(f"result: {'SUCCESS' if ep['won'] else 'FAIL'} in {ep['n_steps']} steps | "
                f"episode uncertainty (mean smoothed) = {ep['episode_uncertainty']:.4f}\n\n")
        for s in ep["steps"]:
            u = s["uncertainty"]
            f.write(f"--- step {s['step']} ---\n")
            f.write(f"OBS     : {s['observation']}\n")
            if s["memory_window"]:
                f.write(f"MEMORY  : {s['memory_window']}\n")
            f.write(f"THINK   : {s['think']}\n")
            f.write(f"ACTION  : {s['action']}   (format ok={s['action_valid_format']}, "
                    f"admissible={s['action_admissible']}, choices={s['n_admissible']})\n")
            f.write(f"FEEDBACK: {s['env_feedback']}\n")
            f.write(f"UNCERT  : entropy={u['entropy']:.3f} least_conf={u['least_confidence']:.3f} "
                    f"margin={u['margin']:.3f} combined={u['combined']:.3f} smoothed={u['smoothed']:.3f}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--games-filter", nargs="+", required=True,
                    help="substrings of gamefile paths; up to --max-games per filter")
    ap.add_argument("--max-games", type=int, default=3)
    ap.add_argument("--split", default="eval_in_distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(ALF_CONFIG) as fh:
        config = yaml.safe_load(fh)
    base_env = get_environment(config["env"]["type"])(config, train_eval=args.split)
    chosen = []
    for flt in args.games_filter:
        hits = sorted(g for g in base_env.game_files if flt in g)[: args.max_games]
        print(f"filter {flt!r}: {len(hits)} games")
        chosen += hits
    base_env.game_files = chosen
    base_env.num_games = len(chosen)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=0.5, max_model_len=4096,
              enforce_eager=True, seed=args.seed)
    # Same as validation: temperature 0.4, sampled; logprobs=2 gives top-1/top-2 like the rollout.
    sampling = SamplingParams(temperature=0.4, max_tokens=512, logprobs=2, seed=args.seed)

    summary = []
    for g in chosen:
        base_env.game_files = [g]
        env = base_env.init_env(batch_size=1)
        env.seed(args.seed)
        t0 = time.time()
        ep = run_episode(llm, tokenizer, env, sampling)
        ep["wall_s"] = round(time.time() - t0, 1)
        name = os.path.basename(os.path.dirname(os.path.dirname(g))) + "__" + os.path.basename(os.path.dirname(g))
        with open(os.path.join(args.out_dir, f"{args.tag}__{name}.json"), "w") as fh:
            json.dump(ep, fh, indent=1)
        write_readable(ep, os.path.join(args.out_dir, f"{args.tag}__{name}.txt"), args.tag)
        summary.append({"tag": args.tag, "game": name, "task": ep["task"], "won": ep["won"],
                        "n_steps": ep["n_steps"], "episode_uncertainty": ep["episode_uncertainty"],
                        "invalid_steps": sum(not s["action_admissible"] for s in ep["steps"]),
                        "wall_s": ep["wall_s"]})
        print(json.dumps(summary[-1]), flush=True)
        env.close() if hasattr(env, "close") else None
    base_env.game_files = chosen
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
