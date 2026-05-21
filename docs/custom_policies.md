# 联合训练
拉取 RLDS 仿真联合训练数据集：
```bash
uvx hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir [path/to/rlds/datasets]
```

如果要基于现成策略进行联合训练，请使用 [openpi](https://github.com/Physical-Intelligence/openpi) 中的 PolaRiS 训练配置。运行前，请确保更新每个配置中的 `rlds_data_dir`。下面是一个运行示例：
```bash
cd third_party/openpi
uv run --group rlds scripts/train.py  pi05_droid_jointpos_polaris --exp-name=polaris-pi05-droid --overwrite
```


# 评估自定义策略

PolaRiS 提供了一个简单接口，用于评估自定义策略。为简化流程，我们采用 server-client 架构：策略运行在与评估流程不同的进程中。当策略需要大量资源，或存在依赖冲突时，这种方式尤其有用。

PolaRiS 通过 [openpi 的 WebsockeClientPolicy](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py) 与策略交互。你可以用任意方式托管策略服务器。若要定义一个客户端，需要实现 [InferenceClient](src/polaris/policy/abstract_client.py) 抽象类。可参考 [DroidJointPosClient](src/polaris/policy/droid_jointpos_client.py) 中的可运行示例。

最小示例：
```py
@InferenceClient.register(client_name="CustomPolicy")
class CustomPolicy(InferenceClient):
    def __init__(self, args: PolicyArgs):
        # 初始化所需状态，例如观测历史、动作块等
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )

    @property
    def rerender(self) -> bool:
        """
        策略是否请求重新渲染可视化结果。对于 chunked policies，
        这可以减少 splat 渲染开销。如果不需要该优化，可以默认始终返回 True。
        """
        return True

    def infer(self, obs, instruction, return_viz: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """
        对观测进行推理，并返回动作和可视化结果。如果不需要可视化结果，则返回 None。
        """
        request_data = {
            "external_image": obs["splat"]["external_cam"],
            "wrist_image": obs["splat"]["wrist_cam"],
            "instruction": instruction,
        }
        server_response= self.client.infer(request_data)
        return server_response["action"], None 

    def reset(self):
        """
        重置客户端以开始新的 episode。如果策略有状态，这会很有用。
        """
        pass
```

运行策略时，指定客户端名称和端口：
```bash
uv run scripts/eval.py --environment DROID-FoodBussing --policy.client CustomPolicy --policy.port 8000 --run-folder runs/test
```
