# aimix

[English](README.md) | 简体中文

**aimix** 是一个统一调用 OpenAI 和 Anthropic 模型的 Go 流式客户端。Anthropic Messages API、OpenAI Responses API 和 OpenAI Chat Completions API（包括各种 OpenAI 兼容服务）共用同一套 API、同一种事件流和同一种对话记录格式，底层全部使用官方 SDK。

## 介绍

每家 LLM 厂商的请求格式、流式事件、推理参数和工具调用约定都不一样。aimix 用一小组类型把这些差异屏蔽掉：同一份代码可以调用 Claude、GPT 或任何 OpenAI 兼容服务，甚至可以在一段对话中途换模型。

主要特性：

- **统一的流式协议**：所有响应都转换成同一种事件序列（`start`、`text_*`、`thinking_*`、`tool_call_*`、`done`/`error`），每个事件都带一份当前消息的快照。
- **统一的思考档位**：`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`，自动映射到 Anthropic 的思考预算或自适应 effort，以及 OpenAI 的 `reasoning_effort`。模型不支持的档位会自动调整到最接近的可用档位。
- **工具调用**：
  - 工具参数在流式输出过程中始终是合法 JSON。
  - 最终参数会严格解析，并修复模型常犯的 JSON 错误。
  - 格式错误的参数会明确报告出来，不会被悄悄丢掉。
- **跨模型接力**：一个模型生成的对话记录可以交给另一个模型继续。系统会丢弃对方的思考内容、把工具调用 ID 改写成目标 API 的格式，并给没有结果的工具调用补上结果。
- **推理内容回放**：Anthropic 的 thinking 签名和 OpenAI 的加密推理内容都保存在消息里，下一轮会原样传回。OpenAI Responses 请求是无状态的（`store=false`）。
- **Prompt 缓存**：Anthropic 用 `cache_control` 断点，OpenAI 用 `prompt_cache_key`，支持短期和长期两种保留时间。
- **OpenAI 兼容服务**：DeepSeek、vLLM、Ollama 等通过 Chat Completions API 加上 base URL 即可使用，也能读取它们流式返回的 `reasoning_content`。
- **持久化**：消息可以序列化为 JSON，反序列化时还原成具体类型。
- **只用 API Key，行为可预期**：不支持 OAuth。SDK 自带的环境变量默认值（如 `ANTHROPIC_BASE_URL`、`OPENAI_ORG_ID`）都被禁用了，请求发到哪里只由你的 `Model` 和选项决定。

aimix **不做**的事：OAuth 登录、内置模型目录、费用计算、执行工具。模型信息由你自己描述，工具由你自己的代码执行（以后也可以由 agent 层负责）。

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [包结构](#包结构)
- [核心概念](#核心概念)
  - [Model](#model)
  - [上下文与消息](#上下文与消息)
  - [流式事件](#流式事件)
  - [停止原因与用量](#停止原因与用量)
- [用法](#用法)
  - [流式输出](#流式输出)
  - [等待最终消息](#等待最终消息)
  - [思考档位](#思考档位)
  - [工具调用](#工具调用)
    - [严格模式](#严格模式)
  - [图片输入](#图片输入)
  - [多轮对话与切换模型](#多轮对话与切换模型)
  - [请求选项](#请求选项)
  - [协议专属选项](#协议专属选项)
  - [API Key 与 Base URL](#api-key-与-base-url)
  - [OpenAI 兼容服务](#openai-兼容服务)
  - [Prompt 缓存](#prompt-缓存)
  - [取消请求](#取消请求)
  - [保存与恢复对话](#保存与恢复对话)
- [错误处理](#错误处理)
- [测试](#测试)

## 安装

```bash
go get github.com/yoneyy/aimix
```

需要 Go 1.27.3 或更高版本。

## 快速开始

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/providers"
)

func main() {
	// 从环境变量读取 ANTHROPIC_API_KEY / OPENAI_API_KEY
	client := aimix.NewClient(
		providers.Anthropic(),
		providers.OpenAI(),
	)

	model := aimix.Model{
		ID:            "claude-sonnet-4-5",
		Api:           aimix.ApiAnthropicMessages,
		Provider:      aimix.ProviderAnthropic,
		Reasoning:     true,
		ContextWindow: 200000,
		MaxTokens:     64000,
	}

	conv := aimix.Context{
		SystemPrompt: "你是一个简洁的助手。",
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					aimix.TextContent{
						Type: aimix.ContentTypeText,
						Text: "用一句话解释 goroutine。",
					},
				},
			},
		},
	}

	ctx := context.Background()
	s, err := client.StreamSimple(ctx, model, conv, aimix.SimpleStreamOptions{
		Reasoning: aimix.ThinkingLow,
	})
	if err != nil {
		log.Fatal(err)
	}
	for ev := range s.Events() {
		if e, ok := ev.(aimix.StreamTextDeltaEvent); ok {
			fmt.Print(e.Delta)
		}
	}
	msg, err := s.Result(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\n[%s] %d tokens\n", msg.StopReason, msg.Usage.TotalTokens)
}
```

换成 GPT 只需要换模型：

```go
model := aimix.Model{
	ID:            "gpt-5",
	Api:           aimix.ApiOpenAIResponses,
	Provider:      aimix.ProviderOpenAI,
	Reasoning:     true,
	ContextWindow: 400000,
	MaxTokens:     128000,
}
```

## 包结构

| 包 | 作用 |
|---|---|
| `github.com/yoneyy/aimix` | 与厂商无关的类型：`Model`、`Context`、消息、内容块、流式事件、选项，以及 `Client` |
| `github.com/yoneyy/aimix/api` | 每种协议一个实现：`AnthropicMessages()`、`OpenAIResponses()`、`OpenAICompletions()`，以及接收协议专属选项的函数 |
| `github.com/yoneyy/aimix/providers` | 厂商定义：`OpenAI()`、`Anthropic()`，以及自定义厂商用的 `New(Config)`（负责解析 API Key 和 base URL，并按 `Model.Api` 分发） |
| `github.com/yoneyy/aimix/infra` | 基础组件：泛型 `EventStream`、增量补全不完整的 JSON、JSON 修复 |

一个请求的流转路径：`Client` → `Provider`（按 `Model.Provider` 选择）→ `ApiStreamer`（按 `Model.Api` 选择）→ 官方 SDK。

## 核心概念

### Model

aimix 不内置模型目录，每个模型用 `aimix.Model` 自己描述：

| 字段 | 含义 |
|---|---|
| `ID` | 发给 API 的模型 ID，例如 `claude-sonnet-4-5`、`gpt-5` |
| `Name` | 显示名称（可选） |
| `Api` | 协议：`ApiAnthropicMessages`、`ApiOpenAIResponses` 或 `ApiOpenAICompletions` |
| `Provider` | `Client` 路由用的厂商 ID：`ProviderAnthropic`、`ProviderOpenAI` 或自定义 ID |
| `BaseURL` | 覆盖厂商默认的接口地址 |
| `Reasoning` | 模型是否支持思考/推理。为 false 时忽略思考档位 |
| `Input` | 支持的输入类型（`ModalityText`、`ModalityImage`）。`nil` 表示不限制；不含 `ModalityImage` 时，图片会被替换成文字占位 |
| `ContextWindow` | 输入加输出的 token 窗口，用来保证 `max_tokens` 不超出窗口（0 表示未知） |
| `MaxTokens` | 默认的输出上限（0 表示未知） |
| `Headers` | 发给这个模型的每个请求都附带的 HTTP 头 |
| `ThinkingLevels` | 统一思考档位到厂商取值的映射，见[思考档位](#思考档位) |
| `Compat` | 模型偏离默认协议时需要打开的开关（见下表） |

`ModelCompat` 开关：

| 字段 | 作用 |
|---|---|
| `AdaptiveThinking` | Anthropic：用自适应思考加 `output_config.effort`，不发 `budget_tokens`（较新的 Claude 模型） |
| `NoTemperature` | 从不发送 `temperature` |
| `LegacyMaxTokens` | Chat Completions：发 `max_tokens`，不发 `max_completion_tokens` |
| `NoReasoningEffort` | Chat Completions：从不发送 `reasoning_effort` |

### 上下文与消息

请求接收一个 `aimix.Context`：

```go
type Context struct {
	SystemPrompt string
	Messages     Messages
	Tools        []Tool
}
```

`Messages` 是封闭接口 `Message` 的切片，共有四种实现：

| 类型 | Role | 内容 |
|---|---|---|
| `SystemMessage` | `system` | 追加的指令；可以在对话中途增删工具（`ToolsAdded`、`ToolsRemoved`） |
| `UserMessage` | `user` | `TextContent`、`ImageContent` |
| `AssistantMessage` | `assistant` | `TextContent`、`ThinkingContent`、`ToolCallContent`，以及用量、停止原因和响应元数据 |
| `ToolResultMessage` | `toolResult` | 一次工具调用的结果：`TextContent`、`ImageContent`、`IsError` |

内容块实现封闭接口 `Content`，共四种：`TextContent`、`ThinkingContent`、`ImageContent`、`ToolCallContent`。序列化时，`Role` 和 `Type` 字段为空会自动补上，手动构造的值也能正确往返。

用类型断言读取内容：

```go
for _, c := range msg.Content {
	switch b := c.(type) {
	case aimix.TextContent:
		fmt.Println(b.Text)
	case aimix.ThinkingContent:
		fmt.Println("思考：", b.Thinking)
	case aimix.ToolCallContent:
		fmt.Println("调用：", b.Name, string(b.Arguments))
	}
}
```

### 流式事件

`Stream` 和 `StreamSimple` 返回 `*aimix.AssistantMessageEventStream`，`Events()` 会依次产出这些事件：

| 事件（`Event()`） | Go 类型 | 字段 |
|---|---|---|
| `start` | `StreamStartEvent` | `Partial` |
| `text_start` / `text_delta` / `text_end` | `StreamTextStartEvent` / `StreamTextDeltaEvent` / `StreamTextEndEvent` | `ContentIndex`、`Delta`（delta 事件）、`Content`（end 事件）、`Partial` |
| `thinking_start` / `thinking_delta` / `thinking_end` | `StreamThinking*Event` | 同上 |
| `tool_call_start` / `tool_call_delta` / `tool_call_end` | `StreamToolCall*Event` | `ContentIndex`、`Delta`（参数原文片段）、`ToolCall`（end 事件）、`Partial` |
| `done` | `StreamDoneEvent` | `Reason`、`Message` |
| `error` | `StreamErrorEvent` | `Reason`（`error` 或 `aborted`）、`Message`（保留已生成的内容）、`Err` |

- `ContentIndex` 是该内容块在 `Partial.Content` 中的下标。
- `Partial` 是当前消息的不可变快照，可以放心保存。
- 流式过程中，`Partial` 里工具调用的 `Arguments` 始终是合法 JSON：没写完的文本会被补齐，不完整的 key 或值会被丢掉。
- 最后一个事件一定是 `done` 或 `error`。之后调用 `Result(ctx)` 可以拿到最终的 `AssistantMessage`。

> **一定要把 `Events()` 读到关闭为止，或者取消 context。** 事件缓冲区满了以后，生产端会阻塞等待。如果只关心最终结果，请用 `Complete`/`CompleteSimple` 或 `aimix.Collect`。

### 停止原因与用量

| `StopReason` | 含义 |
|---|---|
| `stop` | 模型正常结束本轮回答 |
| `length` | 输出达到 token 上限 |
| `tool_use` | 模型在等待工具结果 |
| `error` | 请求或流失败，详见 `ErrorReason` |
| `aborted` | context 被取消 |

厂商返回的原始值（如 `end_turn`、`incomplete.max_output_tokens`）保存在 `AssistantMessage.RawStopReason`。

`Usage` 包含以下字段：

- `Input`：不含缓存部分的输入 token。
- `Output`：输出 token。
- `CacheRead`、`CacheWrite`：缓存读取和写入的 token。
- `CacheWrite1h`：缓存写入中保留 1 小时的部分，只有 Anthropic 提供。
- `Reasoning`：推理 token，属于 `Output` 的一部分。
- `TotalTokens`：总计。

## 用法

### 流式输出

```go
s, err := client.StreamSimple(ctx, model, conv, aimix.SimpleStreamOptions{
	Reasoning: aimix.ThinkingMedium,
})
if err != nil {
	return err // 配置问题：没有对应的 provider、缺少 API Key、工具 schema 不合法……
}
for ev := range s.Events() {
	switch e := ev.(type) {
	case aimix.StreamThinkingDeltaEvent:
		fmt.Print(e.Delta)
	case aimix.StreamTextDeltaEvent:
		fmt.Print(e.Delta)
	case aimix.StreamToolCallEndEvent:
		fmt.Printf("\n-> %s(%s)\n", e.ToolCall.Name, e.ToolCall.Arguments)
	case aimix.StreamErrorEvent:
		fmt.Println("\n出错：", e.Message.ErrorReason)
	}
}
msg, err := s.Result(ctx)
```

`Client` 有两组方法：

- `Stream`/`Complete` 接收 `aimix.StreamOptions`，思考设置保持模型默认。
- `StreamSimple`/`CompleteSimple` 接收 `aimix.SimpleStreamOptions`，多了一个统一的 `Reasoning` 档位。

### 等待最终消息

```go
msg, err := client.CompleteSimple(ctx, model, conv, aimix.SimpleStreamOptions{})
if err != nil {
	var se *aimix.StreamError
	if errors.As(err, &se) {
		// 本轮失败或被取消；msg 里仍然保留已生成的部分内容
		log.Printf("%s: %s", se.Reason, se.Message)
	}
	return err
}
fmt.Println(msg.Content)
```

如果手上已经有一个 stream，可以用 `aimix.Collect(ctx, stream)` 达到同样效果。

### 思考档位

`SimpleStreamOptions.Reasoning` 可取 `ThinkingOff`、`ThinkingMinimal`、`ThinkingLow`、`ThinkingMedium`、`ThinkingHigh`、`ThinkingXhigh`、`ThinkingMax`。留空表示关闭。

档位会先调整到模型支持的范围内（见 `aimix.SupportedThinkingLevels`、`aimix.ClampThinkingLevel`）：

- `Reasoning: false` 的模型只支持 `off`。
- 在 `Model.ThinkingLevels` 里映射为 `""` 的档位视为不支持。
- `xhigh` 和 `max` 必须显式映射才算支持。
- 不支持的档位先往高找最近的可用档位，找不到再往低找。

调整后的档位按下表发送：

| 档位 | Anthropic（预算模式） | Anthropic（`AdaptiveThinking`） | OpenAI（两种 API） |
|---|---|---|---|
| `off` | `thinking: disabled` | `thinking: disabled` | 映射的 `off` 值（如 `none`），没有映射则不发 |
| `minimal` | 预算 1024 | effort `low` | `minimal` |
| `low` | 预算 2048 | effort `low` | `low` |
| `medium` | 预算 8192 | effort `medium` | `medium` |
| `high` | 预算 16384 | effort `high` | `high` |
| `xhigh` / `max`（已映射时） | 预算 16384 | 映射的 effort | 映射的值 |

`ThinkingLevels` 里有映射时，一律以映射值为准。

预算模式下，`max_tokens` 会自动调大来容纳思考预算，最多到 `Model.MaxTokens`，并且始终给回答留至少 1024 个 token。`SimpleStreamOptions.ThinkingBudgets` 可以覆盖默认预算。

```go
// 使用自适应思考、并支持 "max" effort 的 Claude 模型
opus := aimix.Model{
	ID:        "claude-opus-4-6",
	Api:       aimix.ApiAnthropicMessages,
	Provider:  aimix.ProviderAnthropic,
	Reasoning: true,
	MaxTokens: 128000,
	ThinkingLevels: aimix.ThinkingLevelMap{
		aimix.ThinkingMax: "max",
	},
	Compat: aimix.ModelCompat{
		AdaptiveThinking: true,
	},
}

// "off" 对应 effort "none"、不支持 "minimal" 的 OpenAI 模型
gpt := aimix.Model{
	ID:        "gpt-5.1",
	Api:       aimix.ApiOpenAIResponses,
	Provider:  aimix.ProviderOpenAI,
	Reasoning: true,
	ThinkingLevels: aimix.ThinkingLevelMap{
		aimix.ThinkingOff:     "none",
		aimix.ThinkingMinimal: "",
	},
}
```

### 工具调用

用 JSON Schema 声明工具，然后按常规循环进行：请求一轮，执行工具，把结果追加进对话，再请求下一轮。

```go
weather := aimix.Tool{
	Name:        "get_weather",
	Description: "查询某个城市的当前天气",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
}

conv := aimix.Context{
	SystemPrompt: "需要时使用工具。",
	Tools: []aimix.Tool{
		weather,
	},
	Messages: aimix.Messages{
		aimix.UserMessage{
			Role: aimix.RoleUser,
			Content: aimix.Contents{
				aimix.TextContent{
					Type: aimix.ContentTypeText,
					Text: "巴黎现在天气怎么样？",
				},
			},
		},
	},
}

for {
	msg, err := client.CompleteSimple(ctx, model, conv, aimix.SimpleStreamOptions{})
	if err != nil {
		return err
	}
	conv.Messages = append(conv.Messages, msg)
	if msg.StopReason != aimix.StopReasonToolUse {
		break
	}

	for _, c := range msg.Content {
		call, ok := c.(aimix.ToolCallContent)
		if !ok {
			continue
		}
		output, isError := runTool(call) // 你自己的代码
		if call.ArgumentsError != "" {
			output, isError = call.ArgumentsError, true // 模型给出的参数不是合法 JSON
		}
		conv.Messages = append(conv.Messages, aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: call.ID,
			ToolName:   call.Name,
			IsError:    isError,
			Content: aimix.Contents{
				aimix.TextContent{
					Type: aimix.ContentTypeText,
					Text: output,
				},
			},
		})
	}
}
```

说明：

- `ToolCallContent.Arguments` 是一个 JSON 对象（`json.RawMessage`）。模型给出的参数不是合法对象时，`Arguments` 为 `{}`，`ArgumentsError` 会说明原因并附上模型的原始输出。
- `StreamOptions.ToolChoice` 可以设为 `ToolChoiceAuto`、`ToolChoiceNone` 或 `ToolChoiceRequired`。
- 模型一轮里可能返回多个工具调用，要逐个处理每个 `ToolCallContent`。如果希望每轮最多只调用一个工具，把 `StreamOptions.ParallelToolCalls` 设为 `false`：OpenAI 会收到 `parallel_tool_calls: false`，Anthropic 会收到 `tool_choice.disable_parallel_tool_use: true`。
- 工具结果可以包含图片（`ImageContent`），各协议会按自己支持的方式发送。
- 没有结果的工具调用会被自动补一条错误结果（"No result provided"），循环中途被打断也不会产生不合法的对话记录。

#### 严格模式

工具默认是非严格模式：会要求模型按 `Parameters` 生成参数，但不保证完全符合。给工具设置 `Strict: true` 后，由 API 强制参数严格符合 schema：

```go
weather := aimix.Tool{
	Name:        "get_weather",
	Description: "查询某个城市的当前天气",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
	Strict:      true,
}
```

| 协议 | `Strict: true` | `Strict: false`（默认） |
|---|---|---|
| Anthropic Messages | `strict: true` | 不发送（非严格） |
| OpenAI Responses | `strict: true` | `strict: false`（显式发送，因为这个 API 默认是严格模式） |
| OpenAI Chat Completions | `function.strict: true` | 不发送（非严格） |

严格模式的 schema 必须符合厂商的规则：
- 每个对象都要写 `"additionalProperties": false`。
- 所有字段都要列进 `required`；可选字段用可为 null 的类型表示，例如 `"type": ["string", "null"]`。
- 部分关键字不支持。

不符合规则时请求会返回 400，aimix 会以流的 `error` 事件报告。

### 图片输入

```go
img, _ := os.ReadFile("chart.png")
msg := aimix.UserMessage{
	Role: aimix.RoleUser,
	Content: aimix.Contents{
		aimix.TextContent{
			Type: aimix.ContentTypeText,
			Text: "这张图表说明了什么？",
		},
		aimix.ImageContent{
			Type:     aimix.ContentTypeImage,
			Data:     base64.StdEncoding.EncodeToString(img),
			MimeType: "image/png",
		},
	},
}
```

如果设置了 `Model.Input` 且其中不含 `ModalityImage`，图片不会发送，而是替换成一段简短的文字占位。

### 多轮对话与切换模型

继续对话时，把返回的 `AssistantMessage`（以及工具结果）追加到 `Context.Messages`，再发起下一次请求。思考签名和加密推理内容都保存在消息上，同一个模型可以接着之前的推理继续。

对话记录随时可以交给另一个模型或厂商：

```go
first, _ := client.CompleteSimple(ctx, claude, conv, opts)
conv.Messages = append(conv.Messages, first, nextUserMessage)

// 同一段对话换成 GPT 继续
second, _ := client.CompleteSimple(ctx, gpt, conv, opts)
```

每次请求前，对话记录会按目标模型做如下调整：

- 丢弃其他模型的思考内容（在目标模型上无法回放）。
- 把工具调用 ID 改写成目标 API 接受的格式，并同步改写对应的工具结果。
- 跳过以 `error` 或 `aborted` 结束的回合。

你保存的原始对话记录不会被修改。

### 请求选项

`aimix.StreamOptions` 是所有协议通用的选项：

| 字段 | 含义 |
|---|---|
| `APIKey` | 直接指定 API Key，优先于 provider 的解析 |
| `Headers` | 额外的 HTTP 头（在 `Model.Headers` 之后应用） |
| `Temperature` | `*float64`。为 nil、Anthropic 开启思考或设置了 `Compat.NoTemperature` 时都不发送 |
| `MaxTokens` | 输出上限，0 表示用 `Model.MaxTokens`。Simple 系列方法还会把它限制在 `ContextWindow` 以内 |
| `CacheRetention` | `CacheRetentionShort`（默认）、`CacheRetentionLong`、`CacheRetentionNone` |
| `SessionID` | Prompt 缓存键（OpenAI） |
| `ToolChoice` | `auto`、`none` 或 `required` |
| `ParallelToolCalls` | `*bool`。`false` 表示每轮最多调用一个工具，`true` 表示允许多个；nil 保持厂商默认（允许多个）。只在请求带工具时发送 |
| `MaxRetries` | 请求失败时 SDK 的重试次数，默认 0，即重试交给调用方 |
| `HTTPClient` | 自定义 `*http.Client`（代理、超时、链路追踪等） |

`aimix.SimpleStreamOptions` 内嵌 `StreamOptions`，另外增加了 `Reasoning` 和 `ThinkingBudgets`。

指针类型的选项可以直接用 `new` 写在一行里（Go 1.26+）：

```go
opts := aimix.SimpleStreamOptions{
	StreamOptions: aimix.StreamOptions{
		Temperature:       new(0.2),
		ParallelToolCalls: new(false),
	},
	Reasoning: aimix.ThinkingLow,
}
```

### 协议专属选项

需要完全控制时，可以直接调用 `api` 包里对应协议的函数，传入它专属的选项。这时中间没有 provider，API Key 需要自己传；base URL 默认是官方地址。

```go
enabled := true
s, err := api.StreamAnthropicMessages(ctx, model, aimix.NormalizeContext(conv), api.AnthropicOptions{
	StreamOptions: aimix.StreamOptions{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	},
	ThinkingEnabled:      &enabled,
	ThinkingBudgetTokens: 4096,
})
```

| 函数 | 选项 |
|---|---|
| `api.StreamAnthropicMessages` | `AnthropicOptions{ThinkingEnabled, ThinkingBudgetTokens, Effort}` |
| `api.StreamOpenAIResponses` | `OpenAIResponsesOptions{ReasoningEffort, ReasoningSummary}` |
| `api.StreamOpenAICompletions` | `OpenAICompletionsOptions{ReasoningEffort}` |

每个函数都有对应的 `StreamSimple…` 版本，接收 `aimix.SimpleStreamOptions`。

### API Key 与 Base URL

provider 按以下顺序确定 API Key：

1. `StreamOptions.APIKey`
2. `providers.Config.APIKey`
3. `Config.APIKeyEnv` 里列出的环境变量：`Anthropic()` 读 `ANTHROPIC_API_KEY`，`OpenAI()` 读 `OPENAI_API_KEY`

都没有设置时，`Stream` 会在发出请求前直接返回错误，例如 `aimix: missing API key for openai (set OPENAI_API_KEY)`。

Base URL 优先用 `Model.BaseURL`，否则用 provider 的默认值：

| Provider | 默认值 | 说明 |
|---|---|---|
| `OpenAI()` | `https://api.openai.com/v1` | 自定义地址必须带 `/v1`（SDK 会拼上 `/responses`、`/chat/completions`） |
| `Anthropic()` | `https://api.anthropic.com` | 末尾的 `/v1` 会被自动去掉（SDK 会自己拼 `/v1/messages`） |

想在代码里指定 key，可以修改配置：

```go
cfg := providers.OpenAIConfig()
cfg.APIKey = "sk-..."
cfg.BaseURL = "https://my-gateway.example.com/v1"
client := aimix.NewClient(providers.New(cfg))
```

aimix 刻意忽略 SDK 自己读取的环境变量（`ANTHROPIC_BASE_URL`、`OPENAI_BASE_URL`、`OPENAI_ORG_ID` 等）。接口地址请通过 `Model.BaseURL` 或 `Config.BaseURL` 配置。

### OpenAI 兼容服务

任何支持 Chat Completions API 的服务都可以通过 `providers.New` 接入：

```go
deepseek := providers.New(providers.Config{
	ID:      "deepseek",
	Name:    "DeepSeek",
	BaseURL: "https://api.deepseek.com",
	APIKeyEnv: []string{
		"DEEPSEEK_API_KEY",
	},
	Apis: map[aimix.Api]aimix.ApiStreamer{
		aimix.ApiOpenAICompletions: api.OpenAICompletions(),
	},
})
client := aimix.NewClient(providers.OpenAI(), providers.Anthropic(), deepseek)

model := aimix.Model{
	ID:        "deepseek-reasoner",
	Api:       aimix.ApiOpenAICompletions,
	Provider:  "deepseek",
	Reasoning: true,
	Compat: aimix.ModelCompat{
		NoReasoningEffort: true,
	},
}
```

- 通过 `reasoning_content`、`reasoning` 或 `reasoning_text` 字段流式返回的推理内容会变成 `ThinkingContent`；同一个模型的后续轮次会按原字段传回。
- OpenAI 专有的字段（`store`、`prompt_cache_key`）只发给 `api.openai.com`。
- 只认 `max_tokens` 的服务，打开 `Compat.LegacyMaxTokens`。
- 不校验 key 的本地服务也需要填一个占位 key，例如 `Config.APIKey: "local"`。

### Prompt 缓存

- **Anthropic**：缓存断点打在系统提示、最后一个工具定义和最后一条用户消息上。`CacheRetentionLong` 会请求 1 小时的缓存时长。
- **OpenAI**：给每段对话设置一个固定的 `StreamOptions.SessionID`，它会作为 `prompt_cache_key` 发送。`CacheRetentionLong` 会请求 24 小时的保留时间。
- `CacheRetentionNone` 关闭缓存。

缓存命中情况体现在 `Usage.CacheRead` 和 `Usage.CacheWrite` 里。

### 取消请求

取消 context 即可中止响应。流会以一个 `Reason` 为 `aborted` 的 `error` 事件结束，消息里保留已经生成的内容。

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
msg, err := client.CompleteSimple(ctx, model, conv, opts)
// 超时时：msg.StopReason == aimix.StopReasonAborted，msg.Content 里是已生成的部分回答
```

### 保存与恢复对话

`Messages`、`Contents` 和 `Context` 都可以序列化为 JSON，反序列化时还原成具体类型：

```go
raw, err := json.Marshal(conv)

var restored aimix.Context
err = json.Unmarshal(raw, &restored)
```

## 错误处理

错误分为两类：

1. **配置错误**：由 `Stream`/`StreamSimple` 在发出请求前直接返回。例如没有为 `Model.Provider` 注册 provider、provider 不支持 `Model.Api`、缺少 API Key、工具参数 schema 不是合法 JSON。
2. **请求和流错误**：以 `error` 事件结束流。例如 HTTP 错误、限流、模型拒绝回答、内容过滤、流被截断、base URL 返回了 HTML 页面。最终消息的 `StopReason` 为 `error` 或 `aborted`，保留已生成的内容，`ErrorReason` 是可读的错误信息，比如 `Anthropic API error (429): ...`。`Collect`/`Complete` 会以 `*aimix.StreamError` 的形式返回它。

## 测试

```bash
go test ./...          # 单元测试（本地测试服务器，不访问网络）
go test -race ./...
```

调用真实接口的 live 测试需要加 build tag：

```bash
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... go test -tags live -run Live -v .
```

可选的环境变量：

- `ANTHROPIC_BASE_URL`、`OPENAI_BASE_URL`：自定义接口地址。
- `AIMIX_ANTHROPIC_MODEL`、`AIMIX_OPENAI_MODEL`：要测试的模型。
- `AIMIX_OPENAI_APIS`：`responses`、`completions` 或两者都测。

每个模型会跑一段带思考的两回合工具调用对话，然后把每段对话交给其他每个模型继续。
