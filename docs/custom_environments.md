# 创建自定义环境

我们提供的环境是使用 ZED 相机扫描得到的，但重建流程本身并不依赖特定相机。

拍摄一段视角密集、没有运动模糊的场景视频，然后使用 [COLMAP](https://colmap.github.io/install.html) 处理该视频。

得到 COLMAP 数据集后，按照 [2DGS](https://github.com/hbb1/2d-gaussian-splatting) 的说明生成 splat，并导出对应的 mesh。

将 `fuse_post.ply` mesh 转换为 USD，并创建如下结构的 asset 目录。
```
new_asset/
├── mesh.usd
├── splat.ply
├── textures/ (可选，如果 USD 需要纹理)
└── config.yaml (可选，USD 参数配置)
```

使用 [在线场景组合 GUI](https://polaris-evals.github.io/compose-environments/) 创建一个 USD stage，用来组合场景中的对象。导出 USD 后，使用下面的命令解压。
```
unzip scene.zip -d PolaRiS-Hub/new_env/
```

此时你应该得到类似下面的目录结构：
```
PolaRiS-Hub/
└── new_env/
    ├── assets/
    │   ├── object_1/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   ├── object_2/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   └── scene_splat/
    │       ├── config.yaml
    │       └── splat.ply
    ├── scene.usda             # 主 USD stage 文件
    └── initial_conditions.json  (通过 GUI 定义)
```

按照默认 6 个环境的写法，将新环境添加到 [environments 文件](../src/polaris/environments/__init__.py) 中。你也可以参考其中的示例，用几行代码定义一个 rubric 来给 rollout 打分。之后，在 eval 脚本中修改 `--environment` 参数即可使用该环境。

测试环境后，建议提交 PR，将它上传到 [PolaRiS-Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub)。具体说明见下文。

## 上传环境到 HuggingFace

你可以将自定义环境上传到 [PolaRiS-Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub) 数据集，与社区共享。**所有上传都会自动以 pull request 的形式提交**，不会直接提交到主分支，以便进行审核和质量控制。

### 环境结构

你的环境目录应类似下面这样：
```
PolaRiS-Hub/
└── new_env/
    ├── assets/
    │   ├── object_1/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   ├── object_2/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   └── scene_splat/
    │       ├── config.yaml
    │       └── splat.ply
    ├── scene.usda             # 主 USD stage 文件
    └── initial_conditions.json  (通过 GUI 定义)
```

### 上传命令

```bash
uv run scripts/upload_env_to_hf.py ./PolaRiS-Hub/new_env --pr-title "Add new_env" --pr-description "Description of the environment"
```

### CLI 选项

| 参数 | 说明 |
|------|-------------|
| `--dry-run` | 只进行验证，不上传 |
| `--pr-title` | pull request 标题 |
| `--pr-description` | PR 描述/正文 |
| `--repo-id` | 目标 HF 数据集，默认是 `owhan/PolaRiS-Hub` |
| `--branch` | 目标分支，默认是 `main` |
| `--token` | HF token，也可以使用 `HF_TOKEN` 环境变量 |
| `--strict` | 将验证警告视为错误 |
| `--require-pxr` | 如果无法打开 USD 文件则失败，需要 pxr |
| `--skip-validation` | 跳过验证，不推荐 |

### HuggingFace 数据集 PR 的工作方式

运行 `polaris upload` 时，该工具会自动：

1. 在本地验证你的环境结构
2. 向目标数据集创建 pull request，而不是直接提交
3. 返回 PR URL，或返回查看 PR 的说明

**查看你的 PR：**

- 上传完成后，CLI 会打印 PR URL，例如 `https://huggingface.co/datasets/owhan/PolaRiS-Hub/discussions/<PR_NUMBER>`
- 你也可以在这里查看所有 PR：`https://huggingface.co/datasets/owhan/PolaRiS-Hub/discussions`
- PR 必须由数据集维护者审核并合并后，你的环境才会出现在数据集中

**合并你的 PR：**

- 在浏览器中打开 PR URL
- 在 "Files" 标签页中检查变更
- 准备合并时点击 "Publish"，这需要你拥有该数据集的写权限

### 在本地管理你的 PR

创建 PR 后，你可以在本地 checkout 该 PR 并继续修改：

```bash
# 克隆数据集仓库
git clone https://huggingface.co/datasets/owhan/PolaRiS-Hub
cd PolaRiS-Hub

# 获取并 checkout PR，将 <PR_NUMBER> 替换为上传输出中的 PR 编号
git fetch origin refs/pr/<PR_NUMBER>:pr/<PR_NUMBER>
git checkout pr/<PR_NUMBER>

# 修改后推送回 PR 分支
git add .
git commit -m "Update environment"
git push origin pr/<PR_NUMBER>:refs/pr/<PR_NUMBER>
```
