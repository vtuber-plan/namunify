# Namunify

JavaScript 代码反混淆工具，使用大语言模型 (LLM) 对混淆的变量名进行重命名。

## 功能特性

- **Webpack 解包**: 使用 webcrack 解包 Webpack 打包的代码
- **AST 解析**: 使用 tree-sitter 精确解析 JavaScript AST
- **智能分析**: 自动识别混淆的变量名并按作用域分组
- **LLM 重命名**: 调用 OpenAI 或 Anthropic API 进行智能重命名
- **插件系统**: 支持通过插件链扩展功能

## 安装

```bash
# 使用 pip 安装
pip install -e .

# 或者使用 pip 安装开发依赖
pip install -e ".[dev]"
```

### 系统要求

- Python 3.10+
- Node.js (可选，用于 webcrack 解包和 prettier 格式化)

## 快速开始

### 1. 设置 API Key

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# 或者 Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

### 2. 反混淆 JavaScript 文件

```bash
# 基本用法
namunify deobfuscate input.js -o output.js

# 使用 Anthropic
namunify deobfuscate input.js --provider anthropic -o output.js

# 先解包 webpack 再反混淆
namunify deobfuscate bundle.js --unpack -o output_dir/
```

### 3. 只分析不反混淆

```bash
namunify analyze input.js
```

### 4. 只解包 webpack

```bash
namunify unpack bundle.js -o unpacked/
```

## CLI 命令

### `namunify deobfuscate`

反混淆 JavaScript 代码。

```
Usage: namunify deobfuscate [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH          JavaScript 文件或目录路径

Options:
  -o, --output PATH   输出文件/目录路径
  --provider          LLM 提供商: openai 或 anthropic (默认: openai)
  --model             LLM 模型名称
  --api-key           API key (或设置环境变量)
  --base-url          自定义 API 基础 URL
  --max-symbols       每次调用的最大符号数 (默认: 50)
  --context-padding   符号周围的上下文行数 (默认: 500)
  --no-prettier       禁用 prettier 格式化
  --unpack            先解包 webpack bundle
  --install-webcrack  如果需要则安装 webcrack
```

### `namunify unpack`

使用 webcrack 解包 webpack bundle。

```
Usage: namunify unpack [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH          Webpack bundle 文件路径

Options:
  -o, --output PATH   输出目录
  --install           如果需要则安装 webcrack
```

### `namunify analyze`

分析 JavaScript 文件并显示混淆的符号。

```
Usage: namunify analyze INPUT_PATH
```

## 配置

可以通过环境变量或 `.env` 文件配置：

```bash
# .env 文件示例
NAMUNIFY_LLM_PROVIDER=openai
NAMUNIFY_LLM_MODEL=gpt-4o
NAMUNIFY_LLM_API_KEY=your-api-key
NAMUNIFY_LLM_BASE_URL=https://api.openai.com/v1
NAMUNIFY_MAX_CONTEXT_SIZE=32000
NAMUNIFY_MAX_SYMBOLS_PER_BATCH=50
NAMUNIFY_PRETTIER_FORMAT=true
```

## 工作原理

1. **Webcrack 解包**: 解析 Webpack 打包的代码，提取原始模块文件
2. **插件链处理**: 对每个文件应用插件链（如 beautify 格式化）
3. **Babel AST 解析**: 将代码解析为抽象语法树，识别所有绑定的标识符
4. **标识符分组**: 按作用域位置分组，合并小的嵌套作用域
5. **LLM 重命名**: 将上下文发送给 LLM，获取新的变量名映射
6. **代码生成**: 应用重命名，使用 prettier 格式化输出

## 作为库使用

```python
import asyncio
from namunify import Config, analyze_identifiers, parse_javascript
from namunify.core.generator import CodeGenerator
from namunify.llm import OpenAIClient

async def deobfuscate(code: str):
    # 解析代码
    parse_result = parse_javascript(code)

    # 分析标识符
    scopes = analyze_identifiers(parse_result)

    # 创建 LLM 客户端
    config = Config(llm_provider="openai")
    client = OpenAIClient(
        api_key=config.llm_api_key,
        model=config.llm_model,
    )

    # 代码生成器
    generator = CodeGenerator(code)

    # 处理每个作用域
    for scope in scopes:
        symbols = [id.name for id in scope.identifiers]
        context = scope.identifiers[0].context_before

        renames = await client.rename_symbols(context, symbols)
        generator.apply_renames(renames)

    await client.close()
    return generator.get_current_source()

# 运行
code = "var a = 1, b = 2; function c() { return a + b; }"
result = asyncio.run(deobfuscate(code))
print(result)
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black .
isort .

# 类型检查
mypy namunify
```

## License

MIT
