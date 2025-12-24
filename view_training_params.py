"""
查看训练记录中的参数配置
用于确定每个检查点模型对应的训练参数
"""
import argparse
import os
import pickle
import json
from pprint import pprint


def format_params(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """格式化参数为可读的字典"""
    params = {
        "PPO超参数": {
            "算法参数": train_cfg.get("algorithm", {}),
            "策略网络参数": train_cfg.get("policy", {}),
            "训练配置": {
                "num_steps_per_env": train_cfg.get("num_steps_per_env"),
                "save_interval": train_cfg.get("save_interval"),
                "seed": train_cfg.get("seed"),
                "max_iterations": train_cfg.get("runner", {}).get("max_iterations"),
            }
        },
        "环境参数": {
            "num_drones": env_cfg.get("num_drones"),
            "episode_length_s": env_cfg.get("episode_length_s"),
            "at_target_threshold": env_cfg.get("at_target_threshold"),
            "obstacle_safe_distance": env_cfg.get("obstacle_safe_distance"),
            "obstacle_collision_distance": env_cfg.get("obstacle_collision_distance"),
            "drone_safe_distance": env_cfg.get("drone_safe_distance"),
            "drone_collision_distance": env_cfg.get("drone_collision_distance"),
            "obstacle_radius": env_cfg.get("obstacle_radius"),
            "obstacle_height": env_cfg.get("obstacle_height"),
        },
        "奖励参数": reward_cfg.get("reward_scales", {}),
        "观测配置": {
            "num_obs": obs_cfg.get("num_obs"),
            "num_obs_per_drone": obs_cfg.get("num_obs_per_drone"),
            "obs_scales": obs_cfg.get("obs_scales", {}),
        }
    }
    return params


def main():
    parser = argparse.ArgumentParser(description="查看训练记录中的参数配置")
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-ppo-v2", 
                       help="实验名称")
    parser.add_argument("--ckpt", type=int, default=None, 
                       help="检查点迭代数（可选，用于确认该检查点使用的参数）")
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
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    except Exception as e:
        print(f"错误：无法加载配置文件: {e}")
        return
    
    # 格式化参数
    params = format_params(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    
    # 检查检查点是否存在
    if args.ckpt is not None:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        if os.path.exists(model_path):
            print(f"✓ 检查点 model_{args.ckpt}.pt 存在")
        else:
            print(f"⚠ 警告：检查点 model_{args.ckpt}.pt 不存在")
            # 列出可用的检查点
            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if model_files:
                model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
                print(f"可用的检查点: {[f.replace('model_', '').replace('.pt', '') for f in model_files]}")
        print()
    
    # 输出参数
    output_text = ""
    
    if args.format == "pretty":
        output_text += "=" * 80 + "\n"
        output_text += f"训练参数配置 - {args.exp_name}\n"
        if args.ckpt is not None:
            output_text += f"检查点: model_{args.ckpt}.pt\n"
        output_text += "=" * 80 + "\n\n"
        
        for section, content in params.items():
            output_text += f"\n## {section}\n"
            output_text += "-" * 80 + "\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        output_text += f"\n### {key}\n"
                        for k, v in value.items():
                            output_text += f"  - **{k}**: {v}\n"
                    else:
                        output_text += f"  - **{key}**: {value}\n"
            else:
                output_text += f"  {content}\n"
        
        # 添加重要说明
        output_text += "\n" + "=" * 80 + "\n"
        output_text += "重要说明：\n"
        output_text += "=" * 80 + "\n"
        output_text += "1. 这些参数是从训练开始时保存的 cfgs.pkl 文件中读取的\n"
        output_text += "2. 如果训练过程中使用了 --update_config 恢复训练，参数可能已更新\n"
        output_text += "3. 如果训练是从头开始的，这些参数就是所有检查点使用的参数\n"
        output_text += "4. 如果训练是恢复的且没有使用 --update_config，参数与初始训练时相同\n"
        output_text += "=" * 80 + "\n"
    
    elif args.format == "json":
        output_text = json.dumps(params, indent=2, ensure_ascii=False)
    
    elif args.format == "markdown":
        output_text = f"# 训练参数配置 - {args.exp_name}\n\n"
        if args.ckpt is not None:
            output_text += f"**检查点**: model_{args.ckpt}.pt\n\n"
        
        for section, content in params.items():
            output_text += f"## {section}\n\n"
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        output_text += f"### {key}\n\n"
                        for k, v in value.items():
                            output_text += f"- **{k}**: {v}\n"
                        output_text += "\n"
                    else:
                        output_text += f"- **{key}**: {value}\n"
            else:
                output_text += f"{content}\n"
            output_text += "\n"
    
    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"参数已保存到: {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()


# 查看 1400pt 的参数
#python view_training_params.py -e multi-drone-ppo-v2 --ckpt 1400

# 导出为 Markdown 文档
#python view_training_params.py -e multi-drone-ppo-v2 --ckpt 1400 --format markdown --output params_1400pt.md


