import sys
import requests
import re
import subprocess
import time
import os
from colorama import init, Fore, Style
import torch
from pathlib import Path

# 获取当前文件所在目录（deeplearning目录）
DEEPLEARNING_DIR = os.path.dirname(os.path.abspath(__file__))

# 导入prepare.py的CFG配置
from deeplearning.prepare import CFG

init(autoreset=True)

# ====================== 【配置区】 ======================
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY = "81a0ee981c284443b52cff49963865c2.9OqmVKnkU8ST3Aiy"  # 请妥善保管
MODEL_NAME = "glm-4.5-air"
MAX_DEBUG_RETRIES = 5  # 最大修复次数


# ==========================================================

def print_status(msg): print(f"\n{Fore.BLUE}[📌 状态] {msg}{Style.RESET_ALL}")


def print_success(msg): print(f"{Fore.GREEN}[✅ 成功] {msg}{Style.RESET_ALL}")


def print_error(msg): print(f"{Fore.RED}[❌ 错误] {msg}{Style.RESET_ALL}")


def print_warning(msg): print(f"{Fore.YELLOW}[⚠️ 警告] {msg}{Style.RESET_ALL}")


def init_environment():
    """初始化目录结构，并放置一张测试图以便测试"""
    print_status("执行环境初始化...")
    for d in [CFG["input_dir"], CFG["output_dir"], CFG["gt_dir"]]:
        os.makedirs(d, exist_ok=True)

    # 检查是否有测试图片，如果没有，给出警告（不阻断，因为用户可以随时放入图片）
    input_files = list(Path(CFG["input_dir"]).glob("*.*"))
    if not input_files:
        print_warning(f"目录 {CFG['input_dir']} 为空。建议在里面放入几张图片，以便 LLM 生成代码后能进行实际测试。")


def read_file(filepath):
    if not os.path.exists(filepath): return ""
    with open(filepath, 'r', encoding='utf-8') as f: return f.read()


def extract_python_code(text):
    pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def run_infer_script():
    print_status(f"开始执行 infer.py 测试...")
    log_file_path = os.path.join(DEEPLEARNING_DIR, "infer_run.log")
    infer_path = os.path.join(DEEPLEARNING_DIR, "infer.py")
    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            process = subprocess.run(
                ["python", infer_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=1800,  # 30分钟超时（包含下载模型时间）
                cwd=DEEPLEARNING_DIR
            )
        log_content = read_file(log_file_path)

        # 将终端输出打印出来供用户查看
        print(Fore.CYAN + "--- infer.py 输出 ---" + Style.RESET_ALL)
        print(log_content.strip())
        print(Fore.CYAN + "---------------------" + Style.RESET_ALL)

        if process.returncode == 0:
            return "success", log_content
        else:
            return "crash", log_content
    except subprocess.TimeoutExpired:
        return "timeout", "执行超时（超过30分钟）。"
    except Exception as e:
        return "error", str(e)


def call_llm_api_stream(chat_history, output_callback=None):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL_NAME, "messages": chat_history, "temperature": 0.7, "stream": True}
    # 输出开始信息
    start_msg = f"\n{Fore.MAGENTA}{Style.BRIGHT}🤖 LLM 正在编写/修复代码中：{Style.RESET_ALL}\n"
    if output_callback:
        output_callback(start_msg)
    else:
        print(start_msg)
    full_content = ""
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=100, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "): continue
            chunk = line[6:]
            if chunk == "[DONE]": break
            try:
                import json
                j = json.loads(chunk)
                delta = j["choices"][0]["delta"].get("content", "")
                if delta:
                    full_content += delta
                    # 输出到回调函数或控制台
                    if output_callback:
                        output_callback(delta)
                    else:
                        print(delta, end="", flush=True)
            except Exception as e:
                if output_callback:
                    error_msg = f"解析响应时出错: {str(e)}\n"
                    output_callback(error_msg)
                else:
                    print(f"解析响应时出错: {str(e)}\n")
                continue
    except Exception as e:
        error_msg = f"LLM API调用失败：{str(e)}\n"
        if output_callback:
            output_callback(error_msg)
        else:
            print_error(error_msg)
        return ""
    end_msg = "\n" + Fore.MAGENTA + "📝 代码生成完毕" + Style.RESET_ALL + "\n"
    if output_callback:
        output_callback(end_msg)
    else:
        print(end_msg)
    
    return full_content


def main():
    print(Fore.MAGENTA + "=" * 60)
    print("🚀 Hugging Face 图像增强工具自动生成器")
    print("=" * 60 + Style.RESET_ALL)

    init_environment()
    prepare_code = read_file(os.path.join(DEEPLEARNING_DIR, "prepare.py"))
    program_prompt = read_file(os.path.join(DEEPLEARNING_DIR, "program.md"))

    chat_history = [{
        "role": "user",
        "content": (
            f"【前置条件】\n这是无法修改的 prepare.py 内容：\n```python\n{prepare_code}\n```\n\n"
            f"【开发任务】\n{program_prompt}"
        )
    }]

    for attempt in range(1, MAX_DEBUG_RETRIES + 1):
        print_status(f"尝试回合: {attempt}/{MAX_DEBUG_RETRIES}")

        reply = call_llm_api_stream(chat_history, None)  # 保持原有行为，传入None作为回调
        if not reply:
            print_warning("LLM返回空内容，重试中...")
            continue

        chat_history.append({"role": "assistant", "content": reply})
        infer_code = extract_python_code(reply)

        if not infer_code:
            print_warning("未检测到有效的 Python 代码块。正在要求 LLM 重新提供...")
            chat_history.append({"role": "user",
                                 "content": "没有检测到 Python 代码块，请严格使用 ```python ... ``` 包裹你编写的 infer.py 代码。"})
            continue

        # 保存代码
        infer_path = os.path.join(DEEPLEARNING_DIR, "infer.py")
        with open(infer_path, "w", encoding="utf-8") as f:
            f.write(infer_code)
        print_success("infer.py 写入完成，准备测试。")

        # 运行测试
        status, log = run_infer_script()

        if status == "success":
            print_success("🎉 infer.py 执行成功！代码已完善。")
            print_status("现在你可以随时往 input_images 文件夹中放入图片，然后手动运行 `python infer.py` 来处理它们了。")
            return True  # 表示成功
            break
        else:
            print_error(f"执行失败，触发 Bug 修复机制。错误类型: {status}")
            error_feedback = (
                f"你编写的 infer.py 在执行时失败了。请分析以下错误日志，并输出修复后的完整 infer.py 代码：\n\n"
                f"【错误日志】\n{log[-1500:]}\n\n"
                "请直接给出修改后的完整代码。"
            )
            chat_history.append({"role": "user", "content": error_feedback})
            time.sleep(3)

    else:
        print_error(f"达到最大尝试次数 ({MAX_DEBUG_RETRIES})，未能生成完全可用的代码，请人工介入检查 infer.py。")
        return False  # 表示失败


def main_with_output(output_callback=None):
    """带输出回调的主函数，用于在Web界面中显示过程"""
    if output_callback:
        output_callback(Fore.MAGENTA + "=" * 60 + "\n")
        output_callback("🚀 Hugging Face 图像增强工具自动生成器\n")
        output_callback("=" * 60 + Style.RESET_ALL + "\n")
    else:
        print(Fore.MAGENTA + "=" * 60)
        print("🚀 Hugging Face 图像增强工具自动生成器")
        print("=" * 60 + Style.RESET_ALL)

    init_environment()
    prepare_code = read_file(os.path.join(DEEPLEARNING_DIR, "prepare.py"))
    program_prompt = read_file(os.path.join(DEEPLEARNING_DIR, "program.md"))

    chat_history = [{
        "role": "user",
        "content": (
            f"【前置条件】\n这是无法修改的 prepare.py 内容：\n```python\n{prepare_code}\n```\n\n"
            f"【开发任务】\n{program_prompt}"
        )
    }]

    for attempt in range(1, MAX_DEBUG_RETRIES + 1):
        attempt_msg = f"尝试回合: {attempt}/{MAX_DEBUG_RETRIES}\n"
        if output_callback:
            output_callback(attempt_msg)
        else:
            print_status(attempt_msg.rstrip())

        reply = call_llm_api_stream(chat_history, output_callback)
        if not reply:
            warn_msg = "LLM返回空内容，重试中...\n"
            if output_callback:
                output_callback(warn_msg)
            else:
                print_warning(warn_msg.rstrip())
            continue

        chat_history.append({"role": "assistant", "content": reply})
        infer_code = extract_python_code(reply)

        if not infer_code:
            warn_msg = "未检测到有效的 Python 代码块。正在要求 LLM 重新提供...\n"
            if output_callback:
                output_callback(warn_msg)
            else:
                print_warning(warn_msg.rstrip())
            
            chat_history.append({"role": "user",
                                 "content": "没有检测到 Python 代码块，请严格使用 ```python ... ``` 包裹你编写的 infer.py 代码。"})
            continue

        # 保存代码
        infer_path = os.path.join(DEEPLEARNING_DIR, "infer.py")
        with open(infer_path, "w", encoding="utf-8") as f:
            f.write(infer_code)
        success_msg = "infer.py 写入完成，准备测试。\n"
        if output_callback:
            output_callback(success_msg)
        else:
            print_success(success_msg.rstrip())

        # 运行测试
        status, log = run_infer_script()

        if status == "success":
            success_msg = "🎉 infer.py 执行成功！代码已完善。\n"
            if output_callback:
                output_callback(success_msg)
            else:
                print_success(success_msg.rstrip())
            
            status_msg = "现在你可以随时往 input_images 文件夹中放入图片，然后手动运行 `python infer.py` 来处理它们了。\n"
            if output_callback:
                output_callback(status_msg)
            else:
                print_status(status_msg.rstrip())
            
            return True  # 表示成功
            break
        else:
            error_msg = f"执行失败，触发 Bug 修复机制。错误类型: {status}\n"
            if output_callback:
                output_callback(error_msg)
            else:
                print_error(error_msg.rstrip())
            
            error_feedback = (
                f"你编写的 infer.py 在执行时失败了。请分析以下错误日志，并输出修复后的完整 infer.py 代码：\n\n"
                f"【错误日志】\n{log[-1500:]}\n\n"
                "请直接给出修改后的完整代码。"
            )
            chat_history.append({"role": "user", "content": error_feedback})
            time.sleep(3)

    else:
        error_msg = f"达到最大尝试次数 ({MAX_DEBUG_RETRIES})，未能生成完全可用的代码，请人工介入检查 infer.py。\n"
        if output_callback:
            output_callback(error_msg)
        else:
            print_error(error_msg.rstrip())
        
        return False  # 表示失败


if __name__ == "__main__":
    main()