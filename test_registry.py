from agents.planner import PlannerAgent
from agents.coder import CoderAgent
from sandbox.executor import SandboxExecutor
from core.evaluator import Evaluator
from tools import global_registry
from openai import Client

import numpy as np
import json
import optuna
import random

if __name__ != "__main__":
    exit()

def test_planner():
    print(global_registry.get_all_schemas_for_llm())
    planner = PlannerAgent(
        Client(api_key='', base_url='http://10.147.18.7:1234/v1'),
        model_name="gemma-4-e4b-it"
    )

    print(global_registry.get_all_schemas_for_llm_short())
    res = planner.execute("我有一张ISO 6400的照片，我想对它进行降噪处理")
    print(json.dumps(res, ensure_ascii=False, indent=2))

def test_coder():
    coder = CoderAgent(
        Client(api_key='', base_url='http://10.147.18.7:1234/v1'),
        model_name="gemma-4-e4b-it"
    )

    code_str = coder.execute("我用手机拍摄了一份文稿，我希望让其更加易读")

    base_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    executor = SandboxExecutor()

    def objective(trial):
        try:
            executor.execute_pipeline(code_str, base_img, trial)
            return random.randint(1, 200)
        except Exception as e:
            print(e)
            raise optuna.TrialPruned() # 代码执行错误，修剪该 trial

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

test_coder()