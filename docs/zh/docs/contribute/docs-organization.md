# 文档组织方式

## 部署方式

HPC Wiki 使用 [MkDocs](https://www.mkdocs.org/) 作为文档生成工具，使用 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 作为主题。文档源码托管在 [GitHub](https://github.com/lcpu-club/hpc-wiki)，使用 Cloudflare Pages 进行自动部署。所有对`main`分支的更改都会在几分钟内同步到网站上。

部署方式如下：

```bash
# 1. clone
git clone https://github.com/lcpu-club/hpc-wiki.git
# 2. requirements
pip install -r requirements.txt
# generate static file in site/
python3 scripts/docs.py build-all
# deploy at http://127.0.0.1:8008
python3 scripts/docs.py serve # or just python3 -m http.server  --directory site
```

## 文档组织方式

本项目的仓库结构如下：（省略了不重要的部分）

```
-- docs
 |-- zh
 | |--overrides
 | |-- docs
 | | |-- mkdocs.yml
 | | |-- index.md
 | | |-- contribute
 | | | |-- doc-orgnazation.md
 | | | |-- images
 | | |   |-- doc-orgnazation.png
 | | |-- other topics
 | | |-- ...
 |-- en
 |-- missing-translations.md
-- scripts
```

具体来说，只需要关注 `docs` 和 `scripts`两个文件夹。`docs`中是按语言分类的内容，`scripts`中是一些辅助脚本。我们目前没有多语言支持的计划，所以只有`zh`文件夹。每个语言文件夹中，`mkdocs.yml`是MkDocs的配置文件，`docs`文件夹中是文档源码，`overrides`文件夹中是一些覆盖文件，用于修改主题的一些默认设置，但我们目前也没有启用。

`docs`中的文档按照主题分类，每个主题一个文件夹，每个文件夹下的一个`md`文件都是一篇文章。同时，我们约定，将所有文档的图片放置于同文件夹下`image`文件夹的`文档标题`子文件夹中。

## 举例说明

### 如何增加一篇新文章

假设我们要增加一篇关于 FPGA 硬件特性介绍的文章，我们需要做以下几步：

1. 分类与定位：我们需要将这篇文章放置于哪个主题下？我们可以将其放置于`Hardware`主题下，也可以新建一个主题，比如`FPGA`，然后将其放置于`FPGA`主题下。这里我们选择将其放置于`Hardware`主题下。
2. 创建新文件：在`docs/zh/docs/hardware`文件夹下，创建一个新的`md`文件，命名为`fpga.md`。
3. 编写文章：在`fpga.md`中，编写文章内容。文章的格式使用Markdown语法，具体请参考[Markdown 语法说明](https://www.markdown.xyz/basic-syntax/)。文章在导航栏中的标题是文章第一个一级标题决定的，所以请在文章开头使用一级标题。
4. 添加图片：如果文章中需要插入图片，请将图片放置于`docs/zh/docs/hardware/image`文件夹下，并在文章中使用相对路径引用图片。例如，如果我们在`fpga.md`中需要引用`fpga.png`，则可以使用`![fpga](image/fpga.png)`来引用图片。
5. 添加索引：在`docs/zh/docs/hardware/index.md`中，添加一行`- fpga.md`，这样就可以在导航栏中添加对`fpga.md`的链接了。
6. 本地预览：在`docs`文件夹下，运行部署命令（参考上文），在本地预览效果。
7. 提交更改：将更改以PR的形式提交到仓库，在由两位同学审读后，即可合并到`main`分支，网站将在几分钟内自动更新。
8. 完成：至此，我们就完成了一篇新文章的添加。

### 如何修改一篇文章

假设我们要修改`docs/zh/docs/hardware/fpga.md`这篇文章，我们需要做以下几步：

1. 修改文章：在`fpga.md`中，修改文章内容。
2. 本地预览：在`docs`文件夹下，运行部署命令（参考上文），在本地预览效果。
3. 提交更改：将更改以PR的形式提交到仓库，在由两位同学审读后，即可合并到`main`分支，网站将在几分钟内自动更新。
4. 完成：至此，我们就完成了一篇文章的修改。


