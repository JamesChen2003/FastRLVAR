from huggingface_hub import list_repo_files

repo_id = "playgroundai/MJHQ-30K"
files = list_repo_files(repo_id=repo_id, repo_type="dataset")

print("Repo 內的前 20 個檔案路徑如下：")
for f in files[:20]:
    print(f)

# 檢查是否有包含我們想要的關鍵字
selected = [f for f in files if any(k in f for k in ["landscape", "people", "food"])]
print(f"\n匹配到的相關檔案數量: {len(selected)}")