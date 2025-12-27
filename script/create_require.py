import os

def process_vul_func(root_dir):
    # 遍历 vul_func 下各 proX 文件夹
    for pro_name in os.listdir(root_dir):
        pro_path = os.path.join(root_dir, pro_name)

        if not os.path.isdir(pro_path):
            continue

        # 遍历 proX 下的 fY 文件夹
        for fx_name in os.listdir(pro_path):
            fx_path = os.path.join(pro_path, fx_name)

            if not os.path.isdir(fx_path):
                continue

            # 1. 删除旧的 requirements.txt / setup.py / setup.cfg
            for filename in ["requirements.txt", "setup.py", "setup.cfg"]:
                file_path = os.path.join(fx_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            # 2. 创建新的 requirements.txt，内容为当前 pro 文件夹名
            req_path = os.path.join(fx_path, "requirements.txt")
            with open(req_path, "w", encoding="utf-8") as f:
                f.write(pro_name.lower() + "\n")

            print(f"Created: {req_path} (content: {pro_name.lower()})")


if __name__ == "__main__":
    root = r"E:\001_some_AI_code\003_AutoExtension\STAR\vulnerable_fun"  # 修改为你的根目录
    process_vul_func(root)
