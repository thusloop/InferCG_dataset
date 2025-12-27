import json
import os

f1 = "../ground-truth-cgs"
f2 = "../ground-truth-cgs-after"
f3 = "../ground-truth-cgs-differ"

os.makedirs(f3, exist_ok=True)
project_name_list_1 = ["asciinema","autojump","fabric","face_classification","Sublist3r"]
project_name_list_2 = ['bpytop','furl','rich_cli','sqlparse','sshtunnel','textrank4zh']
project_all = project_name_list_1 + project_name_list_2
for filename in os.listdir(f1):
    if filename.endswith(".json") and filename[:-5] in project_all:
        f1_path = os.path.join(f1, filename)
        f2_path = os.path.join(f2, filename)

        # f2 中不存在同名文件则跳过
        if not os.path.exists(f2_path):
            continue

        # 加载文件
        with open(f1_path, "r", encoding="utf-8") as f:
            data1 = json.load(f)
        with open(f2_path, "r", encoding="utf-8") as f:
            data2 = json.load(f)

        diff = {}

        # 找键不同 或 val不同 的项
        all_keys = set(data1.keys()) | set(data2.keys())
        for key in all_keys:
            v1 = set(data1.get(key, []))
            v2 = set(data2.get(key, []))

            if v1 != v2:
                diff[key] = {
                    "only_in_f1": list(v1 - v2),
                    "only_in_f2": list(v2 - v1),
                }

        # 存到 f3
        if diff:
            out_path = os.path.join(f3, filename)
            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(diff, out, indent=2, ensure_ascii=False)

print("差异 JSON 已生成在 f3 文件夹 ✔")
