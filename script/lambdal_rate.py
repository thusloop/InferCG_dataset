import json
import os

def calc_lambda_ratio(json_files):

    rate_lst = 0
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            lambda_count = 0
            total_count = 0
            for caller, callees in data.items():
                for callee in callees:
                    total_count += 1
                    if "lambda" in callee or "lambda" in caller:
                        lambda_count += 1
            rate_lst+=(lambda_count / total_count if total_count else 0)

    #ratio = lambda_count / total_count if total_count else 0
    return rate_lst/len(json_files)


project_name_list_1 = ["asciinema","autojump","fabric","face_classification","Sublist3r"]
#project_name_list_2 = ['bpytop','furl','rich_cli','sqlparse','sshtunnel','textrank4zh']
# 示例输入
json_files = []
for name in project_name_list_1 :#+ project_name_list_2:
    path = os.path.join('../ground-truth-cgs', str(name)+'.json')
    json_files.append(path)

ratio = calc_lambda_ratio(json_files)
print(f"被调用者为 lambda 的占比: {ratio:.4f}")
