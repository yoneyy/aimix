package aimix_test

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/providers"
)

func Example() {
	client := aimix.NewClient(
		providers.OpenAI(),
		providers.Anthropic(),
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
		SystemPrompt: "You are a helpful assistant.",
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					aimix.TextContent{
						Type: aimix.ContentTypeText,
						Text: "What is the weather in Paris?",
					},
				},
			},
		},
		Tools: []aimix.Tool{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a city",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
			},
		},
	}

	ctx := context.Background()
	s, err := client.StreamSimple(ctx, model, conv, aimix.SimpleStreamOptions{
		Reasoning: aimix.ThinkingMedium,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for ev := range s.Events() {
		switch e := ev.(type) {
		case aimix.StreamTextDeltaEvent:
			fmt.Print(e.Delta)
		case aimix.StreamToolCallEndEvent:
			fmt.Printf("\ncall %s(%s)\n", e.ToolCall.Name, e.ToolCall.Arguments)
		}
	}
	msg, err := s.Result(ctx)
	if err != nil {
		fmt.Println(err)
		return
	}
	// Append msg and the tool results to conv.Messages for the next turn.
	fmt.Println(msg.StopReason, msg.Usage.TotalTokens)
}
