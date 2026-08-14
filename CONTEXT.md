# 领域词汇

**场景（scenario）**：一次评测所需的地图、交通条件和场景随机种子的完整组合。场景名称只标识该组合，不代表运行结果。

**回合（episode）**：场景从 reset 到到达、时间截断、碰撞、出界或运行错误的单次闭环运行。预热属于回合的初始化条件，但不计入正式评测指标。

**规划周期（planning cycle）**：规划器基于当前观测生成一次联合未来预测，并由闭环执行其中一段后再次规划的高层决策单位。

**评测作业（evaluation job）**：由一份 resolved config、一个推理运行时和一个独立产物目录组成的评测执行单位；一个作业可以依次包含多个回合。

**作业级并行（job-level parallel evaluation）**：在相互隔离的进程中并发运行多个评测作业。它不表示在同一评测作业内启动多个 MetaDrive 引擎，也不表示对多个回合进行集中式 batch inference。

**交通历史（traffic history）**：以当前交通状态结束、按时间排序的一组交通快照，用于构造动态参与者的过去状态。

**预热（history warmup）**：正式评测前为建立交通历史而推进背景交通的初始化阶段。预热状态必须与正式评测状态和指标分开记录。

**地图 seed（map seed）**：决定程序化地图和场景生成状态的随机种子。它与控制规划器噪声的噪声 seed 是不同实验变量。

**噪声 seed（noise seed）**：决定规划周期扩散初始噪声序列的随机种子。比较策略时，配对的运行使用相同噪声 seed。

**policy action seed（策略动作 seed）**：只决定 Exploration Policy 的 Beta base action 抽样序列。
它与地图 seed、扩散噪声 seed 和 PyTorch 全局 RNG 分离；相同 seed 在相同设备和策略参数下应可重放。

**base action（基础动作）**：Exploration Policy 从两个独立 Beta distributions 抽取并供未来 PPO
rollout 保存的 `u in (0,1)^2`。它不等于 DDIM transition，也不包含候选选择概率。

**guidance action（引导动作）**：由 `2u-1` 得到的横向、纵向 scale，严格位于 `(-1,1)^2`。
该仿射映射及其概率记账是本项目复现决定；边界值不得用 clamp 伪造为有效策略动作。

**参考轨迹（reference trajectory）**：同一规划周期内，由冻结 planner 在共享观测、scene encoding
和扩散随机流下先生成的完整联合未来预测。它是 guidance 的审计基准，不是道路中心线、专家轨迹
或回退控制器。

**正交 guidance（orthogonal guidance）**：相对参考轨迹切向与左法向定义的横纵向扩散样本更新。
本项目阶段 2 的 guidance 只修改 ego future noisy channels，不平滑、投影或裁剪最终轨迹。

**运动学执行条件（kinematic execution condition）**：规划轨迹点被直接写入仿真车辆状态的评测条件。该条件隔离轨迹规划行为，不表示低层 steering、throttle 或 brake 的动力学可执行性。

**稳定能耗场景（stable energy scenario）**：能够重复表达巡航、曲率、变道或合流、限速变化及有限交通交互等能耗因素，并支持可比重复运行的短程或中程场景。

**代理能耗指标（proxy energy metric）**：在固定仿真环境和车辆配置下用于相对比较的能耗量，不能直接解释为真实车辆燃油或电能消耗。

**精细能耗模型（high-fidelity energy model）**：使用车辆参数和实际执行轨迹复核能耗的外部车辆模型，例如 FASTSim；其结果不得与代理能耗指标静默混用。

**道路预瞄（road preview）**：当前局部路线几何之外，明确描述前方曲率、限速变化、拓扑、交通或可用道路属性的条件信息。

**终止类型（termination type）**：回合结束原因的规范分类，包括到达、时间截断、碰撞、出界和运行错误。不同终止类型不能在缺少标注时合并解释。
