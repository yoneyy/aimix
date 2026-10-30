package api

import (
	"encoding/json"
	"testing"

	"github.com/yoneyy/aimix"
)

func TestTransformMessages(t *testing.T) {
	model := aimix.Model{
		ID:       "m1",
		Api:      aimix.ApiAnthropicMessages,
		Provider: aimix.ProviderAnthropic,
		Input: []aimix.Modality{
			aimix.ModalityText,
		},
	}
	call := func(id string) aimix.ToolCallContent {
		return aimix.ToolCallContent{
			Type:      aimix.ContentTypeToolCall,
			ID:        id,
			Name:      "t",
			Arguments: json.RawMessage(`{}`),
		}
	}
	messages := []aimix.Message{
		&aimix.UserMessage{
			Role: aimix.RoleUser,
			Content: aimix.Contents{
				&aimix.ImageContent{
					Type: aimix.ContentTypeImage,
				},
				aimix.ImageContent{
					Type: aimix.ContentTypeImage,
				},
				text("caption"),
			},
		},
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAIResponses,
			Provider:   aimix.ProviderOpenAI,
			Model:      "gpt",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:     aimix.ContentTypeThinking,
					Thinking: "foreign",
				},
				aimix.TextContent{
					Type:          aimix.ContentTypeText,
					Text:          "x",
					TextSignature: "msg_1",
				},
				call("a|b"),
				call("c"),
			},
		},
		// A system message between a call and its results waits until after the results.
		aimix.SystemMessage{
			Role: aimix.RoleSystem,
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "a|b",
		},
		aimix.UserMessage{
			Role: aimix.RoleUser,
		},
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			StopReason: aimix.StopReasonAborted,
			Content: aimix.Contents{
				call("lost"),
			},
		},
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiAnthropicMessages,
			Provider:   aimix.ProviderAnthropic,
			Model:      "m1",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:     aimix.ContentTypeThinking,
					Thinking: "  ",
				},
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					ThinkingSignature: "sig",
				},
				call("d"),
			},
		},
	}
	normalize := func(id string) string {
		return sanitizeID(id, 64)
	}
	got := transformMessages(messages, model, normalize)

	var roles []string
	for _, m := range got {
		roles = append(roles, string(m.GetRole()))
	}
	want := "user,assistant,toolResult,toolResult,system,user,assistant,toolResult"
	if g := compactJSON(t, roles); g != compactJSON(t, splitComma(want)) {
		t.Fatalf("roles %s, want %s", g, want)
	}

	user := got[0].(aimix.UserMessage)
	if len(user.Content) != 2 || user.Content[0].(aimix.TextContent).Text != nonVisionUserImagePlaceholder {
		t.Fatalf("user content %+v", user.Content)
	}

	foreign := got[1].(aimix.AssistantMessage)
	if len(foreign.Content) != 3 {
		t.Fatalf("foreign content %+v", foreign.Content)
	}
	if foreign.Content[0].(aimix.TextContent).TextSignature != "" {
		t.Fatal("text signature kept across providers")
	}
	if id := foreign.Content[1].(aimix.ToolCallContent).ID; id != "a_b" {
		t.Fatalf("normalized id %q", id)
	}
	if id := got[2].(aimix.ToolResultMessage).ToolCallID; id != "a_b" {
		t.Fatalf("tool result id %q", id)
	}
	synthetic := got[3].(aimix.ToolResultMessage)
	if synthetic.ToolCallID != "c" || !synthetic.IsError {
		t.Fatalf("synthetic result %+v", synthetic)
	}

	same := got[6].(aimix.AssistantMessage)
	if len(same.Content) != 2 {
		t.Fatalf("same-model content %+v", same.Content)
	}
	if got[7].(aimix.ToolResultMessage).ToolCallID != "d" {
		t.Fatalf("trailing synthetic result %+v", got[7])
	}

	// The input is left untouched.
	if messages[1].(aimix.AssistantMessage).Content[2].(aimix.ToolCallContent).ID != "a|b" {
		t.Fatal("input modified")
	}
}

func splitComma(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

func TestSimpleOptions(t *testing.T) {
	model := aimix.Model{
		Reasoning:     true,
		ContextWindow: 20000,
		MaxTokens:     16000,
	}

	maxTokens, budget := adjustMaxTokensForThinking(0, 16000, aimix.ThinkingHigh, nil)
	if maxTokens != 16000 || budget != 14976 {
		t.Fatalf("model cap: max %d budget %d", maxTokens, budget)
	}
	maxTokens, budget = adjustMaxTokensForThinking(2000, 16000, aimix.ThinkingLow, nil)
	if maxTokens != 4048 || budget != 2048 {
		t.Fatalf("caller cap: max %d budget %d", maxTokens, budget)
	}
	maxTokens, budget = adjustMaxTokensForThinking(1000, 0, aimix.ThinkingXhigh, map[aimix.ThinkingLevel]int{
		aimix.ThinkingHigh: 3000,
	})
	if maxTokens != 4000 || budget != 3000 {
		t.Fatalf("custom budget: max %d budget %d", maxTokens, budget)
	}

	// 7000 chars ~ 2000 tokens: 20000 - 2000 - 4096 leaves 13904.
	messages := []aimix.Message{
		aimix.UserMessage{
			Role: aimix.RoleUser,
			Content: aimix.Contents{
				text(string(make([]byte, 7000))),
			},
		},
	}
	if got := baseMaxTokens(model, messages, aimix.StreamOptions{}); got != 13904 {
		t.Fatalf("baseMaxTokens %d", got)
	}
	// Reported usage replaces the estimate of the prefix it covers.
	messages = append(messages, aimix.AssistantMessage{
		Role: aimix.RoleAssistant,
		Usage: aimix.Usage{
			TotalTokens: 15000,
		},
	})
	if got := baseMaxTokens(model, messages, aimix.StreamOptions{}); got != 904 {
		t.Fatalf("baseMaxTokens with usage %d", got)
	}
	if got := baseMaxTokens(aimix.Model{}, nil, aimix.StreamOptions{}); got != 0 {
		t.Fatalf("unknown caps: %d", got)
	}
}
