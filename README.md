# aimix

English | [简体中文](README_zh.md)

**aimix** is a unified, streaming Go client for OpenAI and Anthropic models. One API, one event stream and one transcript format cover the Anthropic Messages API, the OpenAI Responses API and the OpenAI Chat Completions API (including OpenAI-compatible servers), all built on the official SDKs.

## Introduction

Every LLM provider has its own request format, streaming events, reasoning settings and tool-calling conventions. aimix hides those differences behind a small set of types, so the same code can stream from Claude, GPT or any OpenAI-compatible server, and a conversation can even switch models halfway through.

Highlights:

- **One streaming protocol.** Every response becomes the same sequence of events (`start`, `text_*`, `thinking_*`, `tool_call_*`, `done`/`error`). Every event carries a snapshot of the message built so far.
- **Unified thinking levels.** `off`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max` map to Anthropic thinking budgets or adaptive effort and to OpenAI `reasoning_effort`. Levels a model does not support are clamped to the nearest supported one.
- **Tool calling.** Tool arguments stream as always-valid partial JSON. Final arguments are parsed strictly, with repair for common model mistakes. Malformed arguments are reported instead of silently lost.
- **Cross-model hand-off.** A transcript produced by one model can be continued by another. Foreign thinking is dropped, tool call ids are rewritten to the target's format, and tool calls left without results are closed.
- **Reasoning replay.** Anthropic thinking signatures and OpenAI encrypted reasoning are kept on the message and sent back on the next turn. OpenAI Responses requests are stateless (`store=false`).
- **Prompt caching.** Uses Anthropic `cache_control` breakpoints and the OpenAI `prompt_cache_key`, with short or long retention.
- **OpenAI-compatible servers.** DeepSeek, vLLM, Ollama and others work through the Chat Completions API and a base URL, including their `reasoning_content` streams.
- **Persistence.** Messages marshal to JSON and decode back into their concrete types.
- **API key only and predictable.** No OAuth. The SDKs' own environment defaults (such as `ANTHROPIC_BASE_URL` or `OPENAI_ORG_ID`) are disabled, so only your `Model` and options decide where requests go.

What aimix does **not** do: OAuth logins, a built-in model catalog, cost calculation, or executing tools. You describe models yourself, and running tools is up to your code (or a future agent layer).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Packages](#packages)
- [Core Concepts](#core-concepts)
  - [Model](#model)
  - [Context and Messages](#context-and-messages)
  - [Stream Events](#stream-events)
  - [Stop Reasons and Usage](#stop-reasons-and-usage)
- [Usage](#usage)
  - [Streaming](#streaming)
  - [Waiting for the Final Message](#waiting-for-the-final-message)
  - [Thinking Levels](#thinking-levels)
  - [Tool Calling](#tool-calling)
    - [Strict Mode](#strict-mode)
  - [Images](#images)
  - [Multi-turn Conversations and Switching Models](#multi-turn-conversations-and-switching-models)
  - [Request Options](#request-options)
  - [Provider-specific Options](#provider-specific-options)
  - [API Keys and Base URLs](#api-keys-and-base-urls)
  - [OpenAI-compatible Servers](#openai-compatible-servers)
  - [Prompt Caching](#prompt-caching)
  - [Cancellation](#cancellation)
  - [Saving and Restoring Conversations](#saving-and-restoring-conversations)
- [Error Handling](#error-handling)
- [Testing](#testing)

## Installation

```bash
go get github.com/yoneyy/aimix
```

aimix requires Go 1.27.3 or later.

## Quick Start

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
	// Reads ANTHROPIC_API_KEY / OPENAI_API_KEY from the environment.
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
		SystemPrompt: "You are a concise assistant.",
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					aimix.TextContent{
						Type: aimix.ContentTypeText,
						Text: "Explain goroutines in one sentence.",
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

Switching to GPT only changes the model:

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

## Packages

| Package | Purpose |
|---|---|
| `github.com/yoneyy/aimix` | Provider-neutral types: `Model`, `Context`, messages, content blocks, stream events, options, and `Client` |
| `github.com/yoneyy/aimix/api` | One implementation per wire protocol: `AnthropicMessages()`, `OpenAIResponses()`, `OpenAICompletions()`, plus functions taking protocol-specific options |
| `github.com/yoneyy/aimix/providers` | Vendors: `OpenAI()`, `Anthropic()`, and `New(Config)` for your own (API key and base URL resolution, dispatch by `Model.Api`) |
| `github.com/yoneyy/aimix/infra` | Building blocks: the generic `EventStream`, incremental partial-JSON completion, and JSON repair |

A request flows like this: `Client` → `Provider` (picked by `Model.Provider`) → `ApiStreamer` (picked by `Model.Api`) → official SDK.

## Core Concepts

### Model

aimix ships no model catalog. You describe each model with an `aimix.Model`:

| Field | Meaning |
|---|---|
| `ID` | Model id sent to the API, e.g. `claude-sonnet-4-5`, `gpt-5` |
| `Name` | Display name (optional) |
| `Api` | Wire protocol: `ApiAnthropicMessages`, `ApiOpenAIResponses` or `ApiOpenAICompletions` |
| `Provider` | Provider id used by `Client` for routing: `ProviderAnthropic`, `ProviderOpenAI` or your own |
| `BaseURL` | Overrides the provider's default endpoint |
| `Reasoning` | Whether the model has a thinking/reasoning channel. Thinking levels are ignored when false |
| `Input` | Accepted modalities (`ModalityText`, `ModalityImage`). `nil` means no restriction. Without `ModalityImage`, images are replaced by a text placeholder |
| `ContextWindow` | Input + output token window, used to keep `max_tokens` inside the window (0 = unknown) |
| `MaxTokens` | Default output cap (0 = unknown) |
| `Headers` | Extra HTTP headers for every request to this model |
| `ThinkingLevels` | Maps unified thinking levels to provider values; see [Thinking Levels](#thinking-levels) |
| `Compat` | Protocol switches for models that deviate from the defaults (below) |

`ModelCompat` switches:

| Field | Effect |
|---|---|
| `AdaptiveThinking` | Anthropic: use adaptive thinking with `output_config.effort` instead of a `budget_tokens` budget (newer Claude models) |
| `NoTemperature` | Never send `temperature` |
| `LegacyMaxTokens` | Chat Completions: send `max_tokens` instead of `max_completion_tokens` |
| `NoReasoningEffort` | Chat Completions: never send `reasoning_effort` |

### Context and Messages

A request takes an `aimix.Context`:

```go
type Context struct {
	SystemPrompt string
	Messages     Messages
	Tools        []Tool
}
```

`Messages` is a slice of the sealed `Message` interface, implemented by four types:

| Type | Role | Content |
|---|---|---|
| `SystemMessage` | `system` | Extra instructions; can add or remove tools mid-conversation (`ToolsAdded`, `ToolsRemoved`) |
| `UserMessage` | `user` | `TextContent`, `ImageContent` |
| `AssistantMessage` | `assistant` | `TextContent`, `ThinkingContent`, `ToolCallContent`, plus usage, stop reason and response metadata |
| `ToolResultMessage` | `toolResult` | The result of one tool call: `TextContent`, `ImageContent`, `IsError` |

Content blocks implement the sealed `Content` interface: `TextContent`, `ThinkingContent`, `ImageContent` and `ToolCallContent`. When marshaling, an empty `Role` or `Type` field is filled in automatically, so hand-built values always round-trip.

Use a type switch to read them:

```go
for _, c := range msg.Content {
	switch b := c.(type) {
	case aimix.TextContent:
		fmt.Println(b.Text)
	case aimix.ThinkingContent:
		fmt.Println("thinking:", b.Thinking)
	case aimix.ToolCallContent:
		fmt.Println("call:", b.Name, string(b.Arguments))
	}
}
```

### Stream Events

`Stream` and `StreamSimple` return an `*aimix.AssistantMessageEventStream`. `Events()` yields these events:

| Event (`Event()`) | Go type | Fields |
|---|---|---|
| `start` | `StreamStartEvent` | `Partial` |
| `text_start` / `text_delta` / `text_end` | `StreamTextStartEvent` / `StreamTextDeltaEvent` / `StreamTextEndEvent` | `ContentIndex`, `Delta` (delta), `Content` (end), `Partial` |
| `thinking_start` / `thinking_delta` / `thinking_end` | `StreamThinking*Event` | same as text |
| `tool_call_start` / `tool_call_delta` / `tool_call_end` | `StreamToolCall*Event` | `ContentIndex`, `Delta` (raw argument text), `ToolCall` (end), `Partial` |
| `done` | `StreamDoneEvent` | `Reason`, `Message` |
| `error` | `StreamErrorEvent` | `Reason` (`error` or `aborted`), `Message` (partial content kept), `Err` |

- `ContentIndex` is the index of the block in `Partial.Content`.
- `Partial` is an immutable snapshot of the message built so far, so it is safe to keep.
- During streaming, `Partial`'s tool call `Arguments` are always valid JSON. Incomplete text is closed, and unfinished keys or values are dropped.
- The last event is always `done` or `error`. After it, `Result(ctx)` returns the final `AssistantMessage`.

> **Always drain `Events()` until it closes, or cancel the context.** The producer waits while the event buffer is full. If you only need the final message, use `Complete`/`CompleteSimple` or `aimix.Collect`.

### Stop Reasons and Usage

| `StopReason` | Meaning |
|---|---|
| `stop` | The model finished its turn |
| `length` | The output hit the token limit |
| `tool_use` | The model is waiting for tool results |
| `error` | The request or stream failed; see `ErrorReason` |
| `aborted` | The context was canceled |

The provider's original value (e.g. `end_turn`, `incomplete.max_output_tokens`) is kept in `AssistantMessage.RawStopReason`.

`Usage` reports `Input` (excluding cached tokens), `Output`, `CacheRead`, `CacheWrite`, `CacheWrite1h` (Anthropic only), `Reasoning` (a subset of `Output`) and `TotalTokens`.

## Usage

### Streaming

```go
s, err := client.StreamSimple(ctx, model, conv, aimix.SimpleStreamOptions{
	Reasoning: aimix.ThinkingMedium,
})
if err != nil {
	return err // configuration problem: no provider, missing API key, invalid tool schema...
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
		fmt.Println("\nerror:", e.Message.ErrorReason)
	}
}
msg, err := s.Result(ctx)
```

`Client` has two pairs of methods:

- `Stream`/`Complete` take `aimix.StreamOptions` and leave thinking at the model default.
- `StreamSimple`/`CompleteSimple` take `aimix.SimpleStreamOptions`, which adds a unified `Reasoning` level.

### Waiting for the Final Message

```go
msg, err := client.CompleteSimple(ctx, model, conv, aimix.SimpleStreamOptions{})
if err != nil {
	var se *aimix.StreamError
	if errors.As(err, &se) {
		// The turn failed or was aborted; msg still holds the partial content.
		log.Printf("%s: %s", se.Reason, se.Message)
	}
	return err
}
fmt.Println(msg.Content)
```

`aimix.Collect(ctx, stream)` does the same for a stream you already have.

### Thinking Levels

`SimpleStreamOptions.Reasoning` takes `ThinkingOff`, `ThinkingMinimal`, `ThinkingLow`, `ThinkingMedium`, `ThinkingHigh`, `ThinkingXhigh` or `ThinkingMax`. An empty value means off.

The level is first clamped to the levels the model supports (`aimix.SupportedThinkingLevels`, `aimix.ClampThinkingLevel`):

- A model with `Reasoning: false` only supports `off`.
- A level mapped to `""` in `Model.ThinkingLevels` is unsupported.
- `xhigh` and `max` are only supported when they are mapped explicitly.
- An unsupported level moves to the nearest supported level above it, then below it.

How the clamped level is sent:

| Level | Anthropic (budget) | Anthropic (`AdaptiveThinking`) | OpenAI (both APIs) |
|---|---|---|---|
| `off` | `thinking: disabled` | `thinking: disabled` | the mapped `off` value (e.g. `none`), otherwise nothing |
| `minimal` | budget 1024 | effort `low` | `minimal` |
| `low` | budget 2048 | effort `low` | `low` |
| `medium` | budget 8192 | effort `medium` | `medium` |
| `high` | budget 16384 | effort `high` | `high` |
| `xhigh` / `max` (when mapped) | budget 16384 | the mapped effort | the mapped value |

A mapped value in `ThinkingLevels` always wins over the defaults above.

For budget thinking, `max_tokens` is raised to make room for the budget, up to `Model.MaxTokens`. At least 1024 tokens are always left for the answer. `SimpleStreamOptions.ThinkingBudgets` overrides the default budgets.

```go
// An adaptive-thinking Claude model that also accepts the "max" effort.
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

// An OpenAI model where "off" means effort "none" and "minimal" is not available.
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

### Tool Calling

Declare tools with a JSON Schema. Then run the usual loop: stream a turn, execute the calls, append the results, and stream again.

```go
weather := aimix.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
}

conv := aimix.Context{
	SystemPrompt: "Use tools when they help.",
	Tools: []aimix.Tool{
		weather,
	},
	Messages: aimix.Messages{
		aimix.UserMessage{
			Role: aimix.RoleUser,
			Content: aimix.Contents{
				aimix.TextContent{
					Type: aimix.ContentTypeText,
					Text: "What's the weather in Paris?",
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
		output, isError := runTool(call) // your code
		if call.ArgumentsError != "" {
			output, isError = call.ArgumentsError, true // the model sent malformed JSON
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

Notes:

- `ToolCallContent.Arguments` is a `json.RawMessage` holding a JSON object. If the model's arguments were not a valid object, `Arguments` is `{}` and `ArgumentsError` explains why and quotes what the model sent.
- `StreamOptions.ToolChoice` can be `ToolChoiceAuto`, `ToolChoiceNone` or `ToolChoiceRequired`.
- Tool results may contain images (`ImageContent`). Each protocol sends them in the form it supports.
- A tool call left without a result gets an automatic error result ("No result provided"), so an interrupted loop never produces an invalid transcript.

#### Strict Mode

Tools are non-strict by default: the model is asked to follow `Parameters`, but its arguments are not guaranteed to match. Set `Strict: true` on a tool to have the API enforce the schema exactly:

```go
weather := aimix.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
	Strict:      true,
}
```

| Protocol | `Strict: true` | `Strict: false` (default) |
|---|---|---|
| Anthropic Messages | `strict: true` | not sent (non-strict) |
| OpenAI Responses | `strict: true` | `strict: false` (sent explicitly, because the API defaults to strict) |
| OpenAI Chat Completions | `function.strict: true` | not sent (non-strict) |

A strict schema must follow the provider's rules. Every object needs `"additionalProperties": false`. Every property must be listed in `required`; express optional fields as nullable, e.g. `"type": ["string", "null"]`. Some keywords are not supported. A schema that breaks these rules makes the request fail with a 400 error, which aimix reports as a stream `error` event.

### Images

```go
img, _ := os.ReadFile("chart.png")
msg := aimix.UserMessage{
	Role: aimix.RoleUser,
	Content: aimix.Contents{
		aimix.TextContent{
			Type: aimix.ContentTypeText,
			Text: "What does this chart show?",
		},
		aimix.ImageContent{
			Type:     aimix.ContentTypeImage,
			Data:     base64.StdEncoding.EncodeToString(img),
			MimeType: "image/png",
		},
	},
}
```

If `Model.Input` is set and lacks `ModalityImage`, images are replaced by a short text placeholder instead of being sent.

### Multi-turn Conversations and Switching Models

To continue a conversation, append the returned `AssistantMessage` (and any tool results) to `Context.Messages`, then send the next request. Thinking signatures and encrypted reasoning travel on the message, so the same model can continue its reasoning.

The transcript can be sent to a different model or provider at any point:

```go
first, _ := client.CompleteSimple(ctx, claude, conv, opts)
conv.Messages = append(conv.Messages, first, nextUserMessage)

// Continue the same conversation on GPT.
second, _ := client.CompleteSimple(ctx, gpt, conv, opts)
```

Before each request, the transcript is adapted to the target model:

- Thinking from other models is dropped, because it cannot be replayed there.
- Tool call ids are rewritten to the format the target API accepts, and the matching tool results are rewritten too.
- Turns that ended with `error` or `aborted` are skipped.

Your stored transcript is never modified.

### Request Options

`aimix.StreamOptions`, shared by every protocol:

| Field | Meaning |
|---|---|
| `APIKey` | Overrides the provider's key resolution |
| `Headers` | Extra headers (applied after `Model.Headers`) |
| `Temperature` | `*float64`. Omitted when nil, when Anthropic thinking is on, or when `Compat.NoTemperature` is set |
| `MaxTokens` | Output cap; 0 uses `Model.MaxTokens`. The simple variants also keep it inside `ContextWindow` |
| `CacheRetention` | `CacheRetentionShort` (default), `CacheRetentionLong`, `CacheRetentionNone` |
| `SessionID` | Prompt cache key (OpenAI) |
| `ToolChoice` | `auto`, `none` or `required` |
| `MaxRetries` | SDK retries for failed requests. The default of 0 leaves retrying to you |
| `HTTPClient` | Custom `*http.Client` (proxies, timeouts, tracing) |

`aimix.SimpleStreamOptions` embeds `StreamOptions` and adds `Reasoning` and `ThinkingBudgets`.

### Provider-specific Options

For full control, call a protocol directly from the `api` package with its own options. There is no provider in between, so you pass the API key yourself. The base URL defaults to the official endpoint.

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

| Function | Options |
|---|---|
| `api.StreamAnthropicMessages` | `AnthropicOptions{ThinkingEnabled, ThinkingBudgetTokens, Effort}` |
| `api.StreamOpenAIResponses` | `OpenAIResponsesOptions{ReasoningEffort, ReasoningSummary}` |
| `api.StreamOpenAICompletions` | `OpenAICompletionsOptions{ReasoningEffort}` |

Each function also has a `StreamSimple…` variant that takes `aimix.SimpleStreamOptions`.

### API Keys and Base URLs

A provider resolves the API key in this order:

1. `StreamOptions.APIKey`
2. `providers.Config.APIKey`
3. The environment variables in `Config.APIKeyEnv`: `ANTHROPIC_API_KEY` for `Anthropic()`, `OPENAI_API_KEY` for `OpenAI()`

If none is set, `Stream` returns an error such as `aimix: missing API key for openai (set OPENAI_API_KEY)` before any request is sent.

The base URL is `Model.BaseURL` if set, otherwise the provider default:

| Provider | Default | Notes |
|---|---|---|
| `OpenAI()` | `https://api.openai.com/v1` | A custom base URL must include `/v1` (the SDK appends `/responses`, `/chat/completions`) |
| `Anthropic()` | `https://api.anthropic.com` | A trailing `/v1` is removed automatically (the SDK appends `/v1/messages`) |

To set a key in code, change the config:

```go
cfg := providers.OpenAIConfig()
cfg.APIKey = "sk-..."
cfg.BaseURL = "https://my-gateway.example.com/v1"
client := aimix.NewClient(providers.New(cfg))
```

aimix deliberately ignores the SDKs' environment variables (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, ...). Configure endpoints through `Model.BaseURL` or `Config.BaseURL`.

### OpenAI-compatible Servers

Any server that speaks the Chat Completions API works through `providers.New`:

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

- Reasoning streamed in `reasoning_content`, `reasoning` or `reasoning_text` becomes `ThinkingContent`. It is sent back in the same field on later turns of the same model.
- OpenAI-only fields (`store`, `prompt_cache_key`) are only sent to `api.openai.com`.
- Use `Compat.LegacyMaxTokens` for servers that only understand `max_tokens`.
- Local servers that do not check keys still need a placeholder key, for example `Config.APIKey: "local"`.

### Prompt Caching

- **Anthropic:** cache breakpoints are placed on the system prompt, the last tool definition and the last user message. `CacheRetentionLong` asks for a 1-hour TTL.
- **OpenAI:** set `StreamOptions.SessionID` to a stable id per conversation; it becomes the `prompt_cache_key`. `CacheRetentionLong` requests 24-hour retention.
- `CacheRetentionNone` turns caching off.

Cache hits show up in `Usage.CacheRead` and `Usage.CacheWrite`.

### Cancellation

Cancel the context to stop a response. The stream ends with an `error` event whose `Reason` is `aborted`. The message keeps everything streamed so far.

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
msg, err := client.CompleteSimple(ctx, model, conv, opts)
// On timeout: msg.StopReason == aimix.StopReasonAborted, msg.Content holds the partial answer.
```

### Saving and Restoring Conversations

`Messages`, `Contents` and `Context` marshal to JSON and decode back into the concrete types:

```go
raw, err := json.Marshal(conv)

var restored aimix.Context
err = json.Unmarshal(raw, &restored)
```

## Error Handling

There are two kinds of failures:

1. **Configuration errors** are returned by `Stream`/`StreamSimple` before any request is sent. Examples: no provider registered for `Model.Provider`, the provider does not support `Model.Api`, a missing API key, or tool parameters that are not valid JSON.
2. **Request and stream errors** end the stream with an `error` event. Examples: HTTP errors, rate limits, refusals, content filters, truncated streams, and a base URL that returns an HTML page. The final message has `StopReason` `error` or `aborted`. It keeps the partial content, and `ErrorReason` carries a readable message such as `Anthropic API error (429): ...`. `Collect`/`Complete` return it as an `*aimix.StreamError`.

## Testing

```bash
go test ./...          # unit tests (local test servers, no network)
go test -race ./...
```

Live tests call the real APIs and are behind a build tag:

```bash
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... go test -tags live -run Live -v .
```

Optional environment variables:

- `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`: custom endpoints.
- `AIMIX_ANTHROPIC_MODEL`, `AIMIX_OPENAI_MODEL`: the models to test.
- `AIMIX_OPENAI_APIS`: `responses`, `completions`, or both.

Each model runs a two-turn tool-calling conversation with thinking. Every transcript is then continued on every other model.
