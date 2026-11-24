# 作业 3：简易 CUDA 渲染器

**截止时间：10 月 30 日（星期四）晚 11:59，PST**

**总分 100 分**

![示例图像](handout/teaser.jpg?raw=true)

## 概述

本作业要求用 CUDA 编写绘制彩色圆的并行渲染器。渲染器本身虽然简单，但要将其并行化，需要设计并实现能够高效并行构建和操作的数据结构。本作业难度较高，建议尽早开始。**认真地说，请尽早开始。**祝顺利！

## 环境配置

1. 本作业需要在 Amazon Web Services（AWS）配有 GPU 的虚拟机上收集结果，即运行性能测试。请按照 [云端配置说明](cloud_readme.zh-CN.md) 配置机器。
2. 从课程 GitHub 下载起始代码：

`git clone https://github.com/stanford-cs149/asst3`

《CUDA C 编程指南》的 [PDF 版](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)或[网页版](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)是学习 CUDA 编程的优秀参考资料。网上以及 [NVIDIA 开发者网站](http://docs.nvidia.com/cuda/)上也有大量 CUDA 教程和 SDK 示例。还可以看看 Udacity 的免费课程 [Introduction to Parallel Programming in CUDA](https://www.udacity.com/blog/2014/01/update-on-udacity-cs344-intro-to.html)。

[CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)的表 21 可用于查阅本作业使用的 NVIDIA T4 GPU 的限制，例如每个线程块的最大 CUDA 线程数、线程块大小和共享内存容量等。NVIDIA T4 支持 CUDA 计算能力 7.5。

关于 C++ 的问题，例如 `virtual` 关键字的含义，可以查阅 [C++ Super-FAQ](https://isocpp.org/faq)。它详细且易于理解，并由 C++ 创始人 Bjarne Stroustrup 参与编写。

## 第 1 部分：CUDA 热身 1——SAXPY（5 分）

为练习 CUDA 编程，首先用 CUDA 重新实现作业 1 的 SAXPY 函数。本部分起始代码在仓库的 `/saxpy` 目录中，进入该目录执行 `make` 和 `./cudaSaxpy` 即可编译、运行。

请完成 `saxpy.cu` 中 `saxpyCuda` 函数的 SAXPY 实现。需要分配设备全局内存数组，并在计算前把主机输入数组 `X`、`Y` 和 `result` 的内容复制到 CUDA 设备内存。计算完成后，必须将结果复制回主机内存。有关 `cudaMemcpy`，参见编程指南网页版第 3.2.2 节，或起始代码中链接的教程。

请在 `saxpyCuda` 的 CUDA 内核调用前后增加计时。完成后，程序应测量两种时间：

* 起始代码已有计时器，测量将数据复制到 GPU、运行内核、再把数据复制回 CPU 的**整个过程**。
* 你增加的计时器应**只测量内核运行时间**，不包括 CPU 到 GPU 的输入传输，也不包括 GPU 到 CPU 的结果传输。

**只测内核时间时要特别小心：**默认情况下，GPU 上的 CUDA 内核相对于 CPU 主应用线程异步执行。例如：

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

这样测出的“内核时间”会快得惊人，因为你只测到了 API 调用本身的开销，没有测到 GPU 实际计算的开销。

因此，应在内核调用后执行 `cudaDeviceSynchronize()`，等待 GPU 上此前提交的全部 CUDA 工作完成后再停止计时。把数据传到 GPU 的 `cudaMemcpy()` 之后不必再调用它，因为在本作业使用的条件下，`cudaMemcpy()` 是同步的。更多说明见[同步行为文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)。

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

对于包含 CPU 与 GPU 之间往返传输的总时间测量，在最后一次将数据复制回 CPU 的 `cudaMemcpy()` 之后、停止计时之前，**无须**额外调用 `cudaDeviceSynchronize()`：该复制完成后，`cudaMemcpy()` 才会向调用线程返回。

**问题 1：**与 CPU 串行 SAXPY 实现相比，观察到的性能如何？回顾作业 1 程序 5 的 SAXPY 结果。

**问题 2：**比较并解释两组计时结果的差异：一组只测内核执行，另一组测量包括 GPU 数据往返传输在内的整个过程。观察到的带宽是否与机器各部件的标称带宽大致一致？请上网查找 NVIDIA T4 的显存带宽，提示：[T4 数据手册](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)。原文给出的 AWS 内存总线预期带宽为 5.3 GB/s，与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的带宽不同。多种因素会阻碍达到峰值，包括 CPU 主板芯片组性能，以及作为传输源的主机 CPU 内存是否为页锁定内存（pinned memory）；页锁定内存允许 GPU 直接访问，无须经过虚拟地址转换。感兴趣可参阅[主机与设备通信优化：页锁定内存](https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory)。

## 第 2 部分：CUDA 热身 2——并行前缀和（10 分）

熟悉 CUDA 程序的基本结构后，第二个练习是并行实现 `find_repeats`：给定整数数组 `A`，返回所有满足 `A[i] == A[i+1]` 的索引 `i`。

例如，输入 `{1,2,2,1,1,1,3,5,3,3}`，输出应为 `{1,3,4,8}`。

#### 排他前缀和（Exclusive Prefix Sum）

请先实现并行排他前缀和，再用它实现 `find_repeats`。

排他前缀和将数组 `A` 转换为数组 `output`，其中 `output[i]` 是 `A[i]` 之前所有元素之和，**不包括 `A[i]` 本身**。例如，`A={1,4,6,8,2}` 的输出为 `output={0,1,5,11,19}`。

下面的类 C 伪代码是 scan 的迭代版本，`parallel_for` 表示可并行执行的循环。这正是[课堂讨论的算法](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_17)：

```
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

请用 CUDA 实现该算法，完成 `scan/scan.cu` 中的 `exclusive_scan` 函数。实现同时包含主机代码与设备代码，需要多次启动 CUDA 内核：上述伪代码中每执行一次 `parallel_for`，就启动一次内核。

**注意：**上述参考 scan 算法假定数组长度 `N` 是 2 的幂。起始代码的 `cudaScan` 在 GPU 上分配缓冲区时，会将长度向上取整为不小于原长度的 2 的幂，但只将原始的 `N` 个元素复制回 CPU。这一处理可以简化你的 CUDA 实现。

编译后生成可执行文件 `cudaScan`，命令行用法如下：

```
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

#### 使用前缀和实现“查找重复项”

完成 `exclusive_scan` 后，实现 `scan/scan.cu` 中的 `find_repeats`。除了调用一次或多次 `exclusive_scan()`，还需要编写其他设备代码。将相邻重复元素对应的索引列表写入提供的输出指针（位于设备内存），并返回输出列表长度。

调用自己的 `exclusive_scan` 时，记住 `start` 数组的内容会复制到 `output`；传入的数组都应位于设备内存中。

**评分：**我们会使用随机输入数组测试正确性和性能。

下表展示简单 CUDA 实现在 K80 GPU 上的 scan 性能，供参考。分别执行 **`./checker.py scan`** 和 **`./checker.py find_repeats`**，检查两个实现的正确性和性能分。工具会生成类似下表的结果，分数按实现性能计算。获得满分要求运行时间不超过参考实现的 120%。

```
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

本部分主要用于练习 CUDA 编程和数据并行思维，并不以性能调优为重点。通常直接将伪代码移植为 CUDA 就能拿到全部性能分，几乎不需要调优。但有一个关键点：朴素实现可能在每轮并行循环中启动 N 个 CUDA 线程，再在内核中通过条件判断决定哪些线程真正工作；这样性能不会好。想想 upsweep 最外层循环最后一轮，有用工作只剩极少数线程。满分实现应当只为最内层并行循环的每次迭代启动一个 CUDA 线程。

**测试工具：**默认使用每次程序运行时都相同的伪随机数组，以便调试。传入 `-i random` 可使用随机数组，评分时会这样做。鼓励设计其他输入检查程序，也可用 `-n <size>` 改变输入长度。

`--thrust` 选项使用 [Thrust 库](http://thrust.github.io/)的[排他 scan 实现](https://docs.nvidia.com/cuda/archive/12.2.2/thrust/index.html?highlight=group%20prefix%20sums#prefix-sums)。**实现性能可与 Thrust 媲美者，最多可获得 2 分加分。**

## 第 3 部分：简易圆形渲染器（85 分）

现在进入主要任务！

起始代码的 `/render` 目录包含彩色圆渲染器。编译后执行 `./render -r cpuref rgb`，会生成包含三个圆的图像 `output_0000.ppm`。再执行 `./render -r cpuref snow`，输出就是飘雪图像。macOS 可直接用“预览”打开 PPM；Windows 可能需要另装查看器。

也可使用 `-i` 将输出显示在屏幕上，而不是写入文件；对于 snow 场景，会显示飘雪动画。交互模式需要配置到本地机器的 X Window 转发，可参考[这篇说明](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html)或[这个问答](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine)。

起始代码包含两个版本：`refRenderer.cpp` 中串行、单线程的 C++ 参考实现，以及 `cudaRenderer.cu` 中**不正确的**并行 CUDA 实现。

### 渲染器概览

建议先阅读 `refRenderer.cpp`，熟悉代码结构。`setup` 在第一帧渲染前调用，CUDA 加速版本中的初始化工作（如分配缓冲区）通常放在这里。`render` 每帧调用，负责将所有圆绘制到输出图像。另一个主要函数 `advanceAnimation` 也每帧调用一次，更新圆的位置和速度；本作业不需要修改它。

输入是一个圆数组，每个圆包含三维位置、速度、半径和颜色。逐帧渲染的基本串行算法如下：

    Clear image
    for each circle
        update position and velocity
    for each circle
        compute screen bounding box
        for all pixels in bounding box
            compute pixel center point
            if center point is within the circle
                compute color of circle at point
                blend contribution of circle into image for this pixel

下图展示用“点是否位于圆内”测试计算圆对像素覆盖的基本算法。只有像素中心位于圆内时，该圆才会为该输出像素贡献颜色。

![点在圆内测试](handout/point_in_circle.jpg?raw=true "测试圆的包围盒中的全部像素。像素中心（黑点）位于圆内即视为被覆盖；图中圆内的像素中心标成红色。只为被覆盖的像素计算圆的颜色贡献。")

一个重要细节是圆为**半透明**。因此，像素颜色并非来自单个圆，而是覆盖该像素的所有半透明圆的颜色混合结果，对应上面伪代码的 blend contribution。圆的颜色用红、绿、蓝和不透明度四元组 RGBA 表示。alpha = 1 表示完全不透明，alpha = 0 表示完全透明。将颜色为 `(C_r, C_g, C_b, C_alpha)` 的圆叠加到颜色为 `(P_r, P_g, P_b)` 的像素上，计算公式为：

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

注意，合成操作不满足交换律：X 覆盖 Y 与 Y 覆盖 X 的外观不同。因此必须按应用程序提供的顺序绘制圆，可以假定圆已按深度排序。下图是蓝圆叠在绿圆上、绿圆再叠在红圆上的例子。左图按正确顺序绘制，右图顺序不同，结果就不正确。

![绘制顺序](handout/order.jpg?raw=true "渲染器必须产生与按应用程序给定顺序逐个绘制所有圆相同的结果。")

### CUDA 渲染器

熟悉参考实现的算法后，阅读 `cudaRenderer.cu` 中的 CUDA 版本，使用 `--renderer cuda`（或 `-r cuda`）运行。

起始 CUDA 实现按输入圆并行，每个 CUDA 线程负责一个圆。虽然数学运算完整，但它存在几个严重错误，需要你修复：图像更新不是原子操作，也没有保持必要的更新顺序，具体要求如下。

### 渲染器要求

你的并行 CUDA 渲染器必须保持以下两个性质，它们在串行实现中自然成立：

1. **原子性：**所有图像更新必须是原子的。临界区包含：读出像素 RGBA 的四个 32 位浮点数、将当前圆的贡献与现有颜色混合、将新颜色写回内存。
2. **顺序：**对同一像素的更新必须遵循**输入圆的顺序**。若圆 1 和圆 2 都影响像素 P，圆 1 对 P 的更新必须先于圆 2。这样才能正确绘制透明圆；它对图形系统还有其他好处，感兴趣可与 Kayvon 交流。**关键观察：顺序要求仅约束对同一个像素的更新。**因此，不影响同一像素的圆之间没有顺序要求，可以独立处理。

![依赖关系](handout/dependencies.jpg?raw=true "圆 1、2、3 对重叠像素的贡献，必须按输入渲染器的顺序应用。")

起始 CUDA 实现两项要求都不满足。在 rgb 和 circles 场景中运行它，就能看到违反顺序或原子性的后果：输出出现水平条纹，而且每帧都会变化，如下图所示。

![顺序错误](handout/bug_example.jpg?raw=true "帧缓冲更新缺乏原子性导致输出错误，注意图像底部的条纹。")

### 需要完成的工作

**你的任务是写出尽可能快且正确的 CUDA 渲染器。**可以采用任何合适的方法，但必须满足原子性和顺序要求。未同时满足两项要求的解法，第 3 部分最多得 12 分；起始代码已经是这样的解法了！

建议首先通读 `cudaRenderer.cu`，确认它为什么不正确。特别注意 `CudaRenderer::render` 如何启动 `kernelRenderCircles`，所有主要工作都在此内核中。执行 `make` 后，运行 `./render -r cuda rand10k`，查看包含 1 万个圆的图像，对应上图底行。再运行 `./render -r cpuref rand10k`，与正确的串行输出比较。

建议按以下步骤推进：

1. 先重写 CUDA 起始实现，使其在并行运行时逻辑正确；建议考虑不需要锁或同步的方法。
2. 确定该解法的性能瓶颈。
3. 然后才进入真正需要深入思考的阶段。提示：`circleBoxTest.cu_inl` 提供的圆与矩形相交测试很有用，鼓励使用。

`./render` 的命令行选项如下：

```
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**检查程序：**`render` 的 `--check` 选项会同时运行串行 CPU 参考渲染器和你的 CUDA 渲染器，比较输出图像以检查正确性，并打印 CUDA 实现耗时。

评分使用 8 个圆形数据集，但满分要求通过全部正确性测试。在 `/render` 目录执行 **`./checker.py`**（注意 `.py` 后缀），即可检查正确性和性能分。对起始代码运行时，会显示全部测试结果以及类似下面的表格：

```
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2622           | (F)             | 0               |
| rand10k         | 3.0658           | (F)             | 0               |
| rand100k        | 29.6144          | (F)             | 0               |
| pattern         | 0.4043           | (F)             | 0               |
| snowsingle      | 19.7155          | (F)             | 0               |
| biglittle       | 15.2422          | (F)             | 0               |
| rand1M          | 230.478          | (F)             | 0               |
| micro2M         | 439.9369         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：有时你**可能**在部分场景得到分数，因为起始渲染器的行为不确定，偶尔会恰好得到正确结果。但这不改变其 CUDA 实现通常不正确这一事实。

“Ref time”是所提供 `render_ref` 参考实现在当前机器上的耗时；“Your time”是你的 CUDA 实现耗时，`(F)` 表示结果错误。成绩取决于你的性能与参考实现的比较，见“评分指南”。

除了代码，还需提交清楚的高层次实现说明，并简述如何得到最终解法。尤其要说明尝试过哪些方法，以及如何判断该怎样优化，例如做了哪些测量来指导优化。

报告应包含：

1. 顶部注明两位组员的姓名和 SUNet ID。
2. 复制你的评分结果表，并注明运行机器。
3. 描述问题如何分解、工作如何分配给 CUDA 线程块和线程，必要时也说明 warp。
4. 描述哪里进行了同步。
5. 说明为减少通信需求采取了哪些措施，例如减少同步或主存带宽需求；若无，也请说明。
6. 简述如何得到最终方案、途中尝试了哪些方法，以及它们有什么问题。

### 评分指南

* 作业报告占 18 分。
* 并行前缀和实现占 10 分。
* 渲染器实现占 72 分，8 个场景各 9 分：每个场景正确性 2 分，只测试图像尺寸为 256 倍数的情况；性能 7 分，只有结果正确才能获得。以参考渲染器耗时 T<sub>ref</sub> 为基准：当耗时 T 达到其 10 倍时，无性能分；当 T <= 1.20 \* T<sub>ref</sub> 时，获得全部性能分；对于 1.20 T<sub>ref</sub> < T < 10 \* T<sub>ref</sub>，性能分按 `7 * T_ref / T` 计算，原文将其描述为 1 至 7 分。
* 性能显著超过要求的方案，可由教师酌情加最多 5 分，报告必须充分说明方法。
* 高质量的纯 CPU 并行渲染器若能充分利用所有核心及 SIMD 向量单元，可由教师酌情加最多 5 分。可使用 SIMD intrinsics、ISPC、pthreads 等工具。要获得加分，需分析 GPU 和 CPU 方案的性能，并讨论实现选择不同的原因。

项目总分分配如下：

* 第 1 部分：5 分。
* 第 2 部分：10 分。
* 第 3 部分报告：13 分。
* 第 3 部分实现：72 分。
* 可能的额外加分：最多 10 分。

> 译注：原文分别在前缀和部分列出最多 2 分加分、在此处列出最多 10 分总加分；以上均按原文保留。

## 作业提示与建议

以下建议来自往年经验。渲染器有多种实现方式，并非每条建议都适用于你的方案。

* 本作业有两个潜在并行维度：**像素之间的并行**和**圆之间的并行**，后者必须满足重叠圆的顺序要求。解法需要利用两类并行，可能分别在计算的不同阶段使用。
* `circleBoxTest.cu_inl` 提供的圆与矩形相交测试很有用，鼓励使用。
* `exclusiveScan.cu_inl` 提供共享内存前缀和，可按需使用。[这里](https://docs.nvidia.com/cuda/archive/12.2.1/thrust/index.html#prefix-sums)有前缀和的简要说明。提供的实现要求数组长度为 **2 的幂**；**不支持其他长度，而且线程块线程数必须等于数组长度。务必阅读代码注释。**
* 查看被调用的 `shadePixel` 方法，它更新像素颜色时执行许多全局内存操作。可以考虑在 `kernelRenderCircles` 中使用局部累加器，在寄存器中累加最终像素值，最后只写一次全局内存。
* 允许使用 [Thrust 库](http://thrust.github.io/)，但达到优化参考实现的性能并不一定需要它。一种常见方案使用提供的共享内存前缀和，另一种使用 Thrust 的前缀和，两者都可行。
* 渲染器中是否存在数据复用？如何利用？
* CUDA 没有能原子地完成整个像素更新逻辑的语言原语，你如何保证原子性？用全局内存原子操作构造锁是一种方案，但即便更新原子化，仍需遵守指定顺序。**建议先考虑如何保证顺序，再判断方案是否仍有原子性问题。**
* `rand1M` 和 `micro2M` 的圆很多，分配全局内存临时结构时需特别小心。分配过多会耗尽设备内存。若不检查 `cudaMalloc` 返回的 `cudaError_t`，程序仍可能继续执行，你却不知道分配失败，最后因临时结构未创建成功而无法通过正确性检查。因此，建议使用下节的 CUDA API 包装宏，在显存不足时报告错误。
* 有余力的话，可以自己设计场景！

### 捕获 CUDA 错误

数组越界、内存分配过多等错误通常不会自动显示；CUDA 往往只返回错误码。可以使用以下宏包装 CUDA 调用，也可按需修改：

```
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

代码正确后，可以取消定义 `DEBUG`，关闭检查以提高性能。

随后用如下方式包装 CUDA API 调用，处理返回的错误：

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

内核启动不能直接这样包装；其错误会在下一次被包装的 CUDA 调用处捕获：

```
kernel<<<1,1>>>(a); // suppose kernel causes an error!
cudaCheckError( cudaDeviceSynchronize() ); // error is printed on this line
```

`cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset` 等所有 CUDA API 函数都可包装。

**重要：**如果先前的 CUDA 函数发生错误但未被捕获，下一次检查会报告该错误，即使检查包装的是另一个函数。例如：

```
...
line 742: cudaMalloc(&a, -1); // executes, then continues
line 743: cudaCheckError(cudaMemcpy(a,b)); // prints "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此，调试时建议包装**全部** CUDA API 调用，至少要覆盖你编写的代码。

来源：改编自[这篇 Stack Overflow 问答](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)。

## 3.4 提交说明

通过 Gradescope 提交。若与他人合作，请记得在 Gradescope 上关联队友。

1. **报告文件命名为 `writeup.pdf`。**
2. **执行 `sh create_submission.sh` 生成 ZIP 后提交到 Gradescope。**脚本会在代码目录执行 `make clean`，之后若要运行代码需重新执行 `make`。若报 `Permission denied`，执行 `chmod +x create_submission.sh` 后重试。

评分脚本会重新运行检查程序，核验成绩是否与 `writeup.pdf` 中一致，也可能使用其他数据集进一步检查正确性。
