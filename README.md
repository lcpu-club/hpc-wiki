# HPC Wiki

HPC 比赛，通常指的是高性能计算相关的比赛，主要形式包括以 `SCC`、`ISC`和`ASC`为代表的学生集群竞赛，和以 `PKU HPCGame`为代表的高性能计算挑战赛。比赛要求选手在规定时间、规定功耗或成本范围内解决高性能计算相关问题，并尽可能提高问题的解决效率。比赛对选手在并行程序设计、软硬件协同优化、计算机体系结构理解与运用、临场问题处理以及团队协作等诸多方面都有很高的要求。

全国高校范围内，有大约200所学校建有超算队。大多数队伍都有自己的文档库用来培养新队员，这些文档库中的内容大多数是相同的，而且限于超算队的规模，文档库的内容也很难得到及时的更新。为此，我们共同建设 **HPC Wiki**，提高文档质量和内容丰富度，让更多的同学能够更快地学习到高性能计算相关的知识，从而更好地参与到 HPC 比赛中。希望能够减少“重复建造轮子”的现象，让大家能够更好地利用时间做更有意义的事情。

**HPC Wiki** 源于社区，由北京大学学生 Linux 俱乐部长期运营和维护，将始终保持**独立自由**的性质，采取`cc-by-nc-sa`的知识共享许可协议，绝不会商业化。

## How to build？

本文档目前采用 [mkdocs](https://github.com/mkdocs/mkdocs) 部署在 [https://hpcwiki.io](https://hpcwiki.io)。

本项目可以直接部署在本地，具体方式如下：

```shell
# 1. clone
git clone https://github.com/lcpu-club/hpc-wiki.git
# 2. requirements
pip install -r requirements.txt
# generate static file in site/
python3 scripts/docs.py build-all
# deploy at http://127.0.0.1:8008
python3 scripts/docs.py serve
```

**mkdocs 本地部署的网站是动态更新的，即当你修改并保存 md 文件后，刷新页面就能随之动态更新。**

## What can you get?

在阅读 Wiki 之前，这里有一些小建议：

- 学习 [提问的智慧](https://github.com/ryanhanwu/How-To-Ask-Questions-The-Smart-Way)
- 善用 Google 搜索能帮助你更好地提升自己
- 至少掌握一门编程语言，比如 Python
- 动手实践比什么都要管用
- 保持对技术的好奇与渴望并坚持下去

## 特别鸣谢

本项目受 [CTF Wiki](https://ctf-wiki.org/) 和 [OI Wiki](https://oi-wiki.org/) 的启发，同时在编写过程中参考了很多资料，特别鸣谢以下项目：
- 上海科技大学 GeekPie 社区的 [GeekPie_HPC Wiki](https://hpc.geekpie.club/wiki/index.html)
- 东南大学超算团队的 [asc-wiki](https://asc-wiki.com)
- 北京大学学生 Linux 俱乐部的 [HPC from Scratch 项目](https://wiki.lcpu.dev/zh/hpc/from-scratch/arrange)

## Copyleft
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。