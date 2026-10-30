//go:build live

// Live tests call the real APIs:
//
//	ANTHROPIC_API_KEY=... OPENAI_API_KEY=... go test -tags live -run Live -v .
//
// Optional settings:
//
//	ANTHROPIC_BASE_URL, OPENAI_BASE_URL    endpoints (the library itself ignores these;
//	                                       the test copies them into Model.BaseURL)
//	AIMIX_ANTHROPIC_MODEL, AIMIX_OPENAI_MODEL  model ids
//	AIMIX_OPENAI_APIS                      "responses", "completions" or both
//	                                       (comma-separated, default both)

package aimix_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/providers"
)

var weatherTool = aimix.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
}

func liveModels(t *testing.T) map[string]aimix.Model {
	t.Helper()
	models := map[string]aimix.Model{}
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		models["anthropic"] = aimix.Model{
			ID:            envOr("AIMIX_ANTHROPIC_MODEL", "claude-haiku-4-5"),
			Api:           aimix.ApiAnthropicMessages,
			Provider:      aimix.ProviderAnthropic,
			BaseURL:       os.Getenv("ANTHROPIC_BASE_URL"),
			Reasoning:     true,
			ContextWindow: 200000,
			MaxTokens:     8192,
		}
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		apis := map[string]aimix.Api{
			"responses":   aimix.ApiOpenAIResponses,
			"completions": aimix.ApiOpenAICompletions,
		}
		for _, name := range strings.Split(envOr("AIMIX_OPENAI_APIS", "responses,completions"), ",") {
			name = strings.TrimSpace(name)
			api, ok := apis[name]
			if !ok {
				t.Fatalf("AIMIX_OPENAI_APIS: unknown api %q", name)
			}
			models["openai-"+name] = aimix.Model{
				ID:            envOr("AIMIX_OPENAI_MODEL", "gpt-5-mini"),
				Api:           api,
				Provider:      aimix.ProviderOpenAI,
				BaseURL:       os.Getenv("OPENAI_BASE_URL"),
				Reasoning:     true,
				ContextWindow: 400000,
				MaxTokens:     8192,
			}
		}
	}
	if len(models) == 0 {
		t.Skip("set ANTHROPIC_API_KEY and/or OPENAI_API_KEY")
	}
	return models
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func userText(s string) aimix.UserMessage {
	return aimix.UserMessage{
		Role: aimix.RoleUser,
		Content: aimix.Contents{
			aimix.TextContent{
				Type: aimix.ContentTypeText,
				Text: s,
			},
		},
		Timestamp: time.Now().UnixMilli(),
	}
}

// answerTools appends a result for every tool call of msg.
func answerTools(conv *aimix.Context, msg aimix.AssistantMessage) int {
	n := 0
	for _, c := range msg.Content {
		call, ok := c.(aimix.ToolCallContent)
		if !ok {
			continue
		}
		n++
		conv.Messages = append(conv.Messages, aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: call.ID,
			ToolName:   call.Name,
			Content: aimix.Contents{
				aimix.TextContent{
					Type: aimix.ContentTypeText,
					Text: `{"temperature_c": 18, "sky": "cloudy"}`,
				},
			},
			Timestamp: time.Now().UnixMilli(),
		})
	}
	return n
}

// TestLiveToolTurn runs a tool call round trip with thinking on every model,
// then continues each transcript on every other model.
func TestLiveToolTurn(t *testing.T) {
	models := liveModels(t)
	client := aimix.NewClient(providers.OpenAI(), providers.Anthropic())
	opts := aimix.SimpleStreamOptions{
		Reasoning: aimix.ThinkingLow,
	}

	transcripts := map[string]aimix.Context{}
	for name, model := range models {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			conv := aimix.Context{
				SystemPrompt: "Use tools when they help. Answer in one sentence.",
				Messages: aimix.Messages{
					userText("What is the weather in Paris right now?"),
				},
				Tools: []aimix.Tool{
					weatherTool,
				},
			}

			first, err := client.CompleteSimple(ctx, model, conv, opts)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("turn 1: stop=%s usage=%+v", first.StopReason, first.Usage)
			conv.Messages = append(conv.Messages, first)
			if first.StopReason != aimix.StopReasonToolUse || answerTools(&conv, first) == 0 {
				t.Fatalf("expected a tool call, got %+v", first.Content)
			}

			second, err := client.CompleteSimple(ctx, model, conv, opts)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("turn 2: stop=%s text=%q", second.StopReason, firstText(second))
			if second.StopReason != aimix.StopReasonStop || firstText(second) == "" {
				t.Fatalf("expected a final answer, got %+v", second)
			}
			conv.Messages = append(conv.Messages, second)
			transcripts[name] = conv
		})
	}

	for from, conv := range transcripts {
		for to, model := range models {
			if from == to {
				continue
			}
			t.Run(from+"->"+to, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
				defer cancel()
				conv := conv
				conv.Messages = append(append(aimix.Messages{}, conv.Messages...), userText("And in Rome? Use the tool again."))
				msg, err := client.CompleteSimple(ctx, model, conv, opts)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("stop=%s content=%d blocks", msg.StopReason, len(msg.Content))
			})
		}
	}
}

func firstText(m aimix.AssistantMessage) string {
	for _, c := range m.Content {
		if tc, ok := c.(aimix.TextContent); ok && tc.Text != "" {
			return tc.Text
		}
	}
	return ""
}
