"""
查看训练记录中的参数配置 (全量版)
用于确定每个检查点模型对应的所有训练参数，无遗漏显示所有配置。
"""
import argparse
import os
import pickle
import json
import pprint
import numpy as np

def make_serializable(obj):
    """辅助函数：处理无法直接 JSON 序列化的对象 (如 numpy 类型)"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj

def format_params(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """
    格式化参数为可读的字典。
    修改：不再手动筛选字段，而是直接返回完整字典。
    """
    # 将 numpy 类型转换为原生类型，防止 JSON 序列化报错
    env_cfg = make_serializable(env_cfg)
    obs_cfg = make_serializable(obs_cfg)
    reward_cfg = make_serializable(reward_cfg)
    command_cfg = make_serializable(command_cfg)
    train_cfg = make_serializable(train_cfg)

    params = {
        "PPO训练配置 (Train Config)": train_cfg,
        "环境配置 (Env Config)": env_cfg,
        "奖励配置 (Reward Config)": reward_cfg,
        "观测配置 (Obs Config)": obs_cfg,
        "命令配置 (Command Config)": command_cfg
    }
    return params

def main():
    parser = argparse.ArgumentParser(description="查看训练记录中的完整参数配置")
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo", 
                       help="实验名称 (logs/ 下的文件夹名)")
    parser.add_argument("--ckpt", type=int, default=None, 
                       help="检查点迭代数（可选，仅用于确认文件是否存在）")
    parser.add_argument("--format", type=str, choices=["pretty", "json", "markdown"], 
                       default="pretty", help="输出格式")
    parser.add_argument("--output", type=str, default=None, 
                       help="输出文件路径（可选，如果不指定则打印到控制台）")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"
    
    if not os.path.exists(cfg_path):
        print(f"错误：配置文件不存在: {cfg_path}")
        print(f"请确认实验名称 '{args.exp_name}' 是否正确")
        return
    
    # 加载配置
    try:
        # 注意：这里的加载顺序必须与 multi_drone_mappo_train.py 保存时的顺序一致
        with open(cfg_path, "rb") as f:
            cfgs = pickle.load(f)
            
        # 兼容不同长度的配置保存 (防止因版本差异导致的解包错误)
        if len(cfgs) == 5:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs
        else:
            print(f"警告: cfgs.pkl 包含 {len(cfgs)} 个对象，预期是 5 个。尝试按顺序解包...")
            # 尝试尽最大努力解包，如果结构完全不同可能会出错
            env_cfg = cfgs[0] if len(cfgs) > 0 else {}
            obs_cfg = cfgs[1] if len(cfgs) > 1 else {}
            reward_cfg = cfgs[2] if len(cfgs) > 2 else {}
            command_cfg = cfgs[3] if len(cfgs) > 3 else {}
            train_cfg = cfgs[4] if len(cfgs) > 4 else {}

    except Exception as e:
        print(f"错误：无法加载配置文件: {e}")
        return
    
    # 格式化参数 (获取全量数据)
    params = format_params(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    
    # 检查检查点是否存在
    ckpt_msg = ""
    if args.ckpt is not None:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        if os.path.exists(model_path):
            ckpt_msg = f"✓ 目标检查点 model_{args.ckpt}.pt 存在"
        else:
            ckpt_msg = f"⚠ 警告：目标检查点 model_{args.ckpt}.pt 不存在"
            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if model_files:
                # model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
                pass # 保持输出简洁，不列出所有文件
    
    # === 生成输出文本 ===
    output_text = ""
    
    if args.format == "pretty":
        output_text += "=" * 80 + "\n"
        output_text += f"全量训练参数 - {args.exp_name}\n"
        if ckpt_msg:
            output_text += f"{ckpt_msg}\n"
        output_text += "=" * 80 + "\n"
        
        for section, content in params.items():
            output_text += f"\n## {section}\n"
            output_text += "-" * 80 + "\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    # 如果值是字典，进行缩进显示
                    if isinstance(value, dict):
                        output_text += f"\n### {key}\n"
                        for k, v in value.items():
                            val_str = pprint.pformat(v, indent=2) # 处理复杂对象如列表
                            if '\n' in val_str: # 如果是多行，换行显示
                                output_text += f"  - **{k}**:\n{val_str}\n"
                            else:
                                output_text += f"  - **{k}**: {val_str}\n"
                    else:
                        # 顶层键值对
                        val_str = pprint.pformat(value, indent=2, width=120)
                        if len(val_str) > 100 or '\n' in val_str: # 长内容换行
                             output_text += f"  - **{key}**:\n{val_str}\n"
                        else:
                             output_text += f"  - **{key}**: {value}\n"
            else:
                output_text += f"  {content}\n"
        
        # 结尾说明
        output_text += "\n" + "=" * 80 + "\n"
        output_text += "注：以上参数来自 logs 目录下的 cfgs.pkl 文件。\n"
        output_text += "如果是通过 --resume --update_config 恢复训练的，这里显示的是最新的配置。\n"
        output_text += "=" * 80 + "\n"
    
    elif args.format == "json":
        output_text = json.dumps(params, indent=2, ensure_ascii=False)
    
    elif args.format == "markdown":
        output_text = f"# 全量训练参数 - {args.exp_name}\n\n"
        if ckpt_msg:
            output_text += f"**状态**: {ckpt_msg}\n\n"
        
        for section, content in params.items():
            output_text += f"## {section}\n\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        output_text += f"### {key}\n\n"
                        output_text += "```python\n"
                        output_text += pprint.pformat(value)
                        output_text += "\n```\n"
                    else:
                        val_str = str(value)
                        if len(val_str) > 100 or '\n' in val_str:
                             output_text += f"- **{key}**:\n  ```python\n  {value}\n  ```\n"
                        else:
                             output_text += f"- **{key}**: `{value}`\n"
            else:
                output_text += f"{content}\n"
            output_text += "\n"
    
    # 输出
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"参数已保存到: {args.output}")
            if ckpt_msg: print(ckpt_msg)
        except Exception as e:
            print(f"写入文件失败: {e}")
            print("尝试打印到控制台:")
            print(output_text)
    else:
        print(output_text)

if __name__ == "__main__":
    main()