# Personal Academic Homepage

正式站点目录为 preview，入口为 index.html。默认英文，使用 ?lang=zh 打开中文；无需构建步骤、npm 运行依赖或后端。

## 文件与维护

- preview/index.html：正式入口。
- preview/site-data.js：个人资料、双语介绍、论文、荣誉、学术服务和更新时间。仅 biography.researchHtml 允许主动写 HTML，用于 strong 标记；其他字段由 app.js 转义。
- preview/app.js：渲染、导航、语言和头像切换、论文图片浮层。
- preview/style.css：字体、排版和响应式布局。
- preview/assets：页面实际使用的图片、字体及 PT Serif 字体许可证。
- scripts/check-assets.mjs：本地维护检查，不参与发布。

修改后在项目根目录检查：

```powershell
node --check preview/app.js
node --check preview/site-data.js
node scripts/check-assets.mjs
git diff --check
```

资源检查覆盖整个 preview 目录，报告缺失引用、闲置文件和重复资源；字体许可证属于必要文件。

## 发布范围

线上仓库为 lee-plus-plus/lee-plus-plus.github.io，地址为 https://lee-plus-plus.github.io/。本地 pages 分支用于后续维护，对应远端 main；本地 main 分支保留原项目历史，不上传到公开仓库。

GitHub Pages 使用 GitHub Actions 发布。推送 main 分支或手动运行发布流程时，先检查文件，再仅上传 preview 目录；没有构建步骤。发布配置位于 .github/workflows/pages.yml。

仅发布 preview 目录内的全部文件，并将该目录作为站点根目录。仓库根目录的维护说明、检查脚本和 Git 文件不属于站点发布内容。

设计稿、备用入口、比较工具、验证截图及历史审计和来源记录已从当前目录清理，可从本地 main 分支的 Git 历史恢复。旧 Codex 工作副本不再作为编辑或同步来源，后续直接维护本仓库。

学术服务仍待本人补充；中文字体目前使用访客系统中的 Noto / 思源宋体或后备字体。发布结果以 GitHub Actions 的运行状态为准。
